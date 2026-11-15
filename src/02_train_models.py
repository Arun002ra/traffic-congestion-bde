"""
02_train_models.py
Spark MLlib: predict whether a junction will be congested 15 minutes from now.
Time-based split (no leakage): train = 1 Aug..18 Sep, test = 19 Sep..29 Sep.
Evaluates Logistic Regression, Random Forest, GBT, and Persistence Baseline.
Also fits a Random Forest Regressor to predict the future speed ratio.
"""
import os
import json
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.sql import functions as F
from common import (
    get_spark, FEATURE_PATH, MODEL_PATH, RESULTS_DIR, METRICS_DIR, PREDICTIONS_DIR, SPLIT_DATE
)

def run_train_models():
    print("=" * 60)
    print("STAGE 2: Machine Learning with Spark MLlib")
    print("=" * 60)

    spark = get_spark("Traffic-02-Train-Models")

    print(f"Reading feature table from: {FEATURE_PATH}")
    df = spark.read.parquet(FEATURE_PATH).withColumn("label", F.col("label_future_congested").cast("double"))
    df = df.withColumn("w", F.when(F.col("label") == 1.0, 3.0).otherwise(1.0))   # class weight for 13% imbalance

    train = df.filter(F.col("date") < SPLIT_DATE).cache()
    test  = df.filter(F.col("date") >= SPLIT_DATE).cache()

    info = {
        "train_rows": train.count(),
        "test_rows": test.count(),
        "train_pos_pct": round(train.agg(F.avg("label")).first()[0] * 100, 2),
        "test_pos_pct": round(test.agg(F.avg("label")).first()[0] * 100, 2)
    }
    print(f"Dataset Split Info: {json.dumps(info, indent=2)}")

    NUM = [
        "vehicle_count", "avg_speed_kmph", "occupancy_pct", "queue_length_m", "speed_ratio", "free_flow_kmph",
        "minute_of_day", "hour", "dow", "is_weekend", "is_holiday", "is_peak", "incident_flag",
        "speed_ratio_lag1", "speed_ratio_lag2", "speed_ratio_lag3", "count_lag1", "count_lag3",
        "count_roll30", "speed_roll30", "speed_trend"
    ]

    prep = [
        StringIndexer(inputCols=["weather", "zone"], outputCols=["weather_i", "zone_i"], handleInvalid="keep"),
        OneHotEncoder(inputCols=["weather_i", "zone_i"], outputCols=["weather_oh", "zone_oh"]),
        VectorAssembler(inputCols=NUM + ["weather_oh", "zone_oh"], outputCol="features", handleInvalid="skip")
    ]
    feature_names = NUM + ["weather_(onehot)", "zone_(onehot)"]

    models = {
        "Logistic Regression": LogisticRegression(
            featuresCol="features", labelCol="label", weightCol="w", maxIter=60, regParam=0.01
        ),
        "Random Forest": RandomForestClassifier(
            featuresCol="features", labelCol="label", weightCol="w", numTrees=60, maxDepth=8, seed=11
        ),
        "Gradient Boosted Trees": GBTClassifier(
            featuresCol="features", labelCol="label", weightCol="w", maxIter=50, maxDepth=5, seed=11
        ),
    }

    auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

    def calc_metrics(pred):
        c = {(r["label"], r["prediction"]): r["count"] for r in pred.groupBy("label", "prediction").count().collect()}
        tp = c.get((1.0, 1.0), 0)
        tn = c.get((0.0, 0.0), 0)
        fp = c.get((0.0, 1.0), 0)
        fn = c.get((1.0, 0.0), 0)
        n = tp + tn + fp + fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        acc = (tp + tn) / n if n else 0.0
        return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=round(acc, 4), precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4))

    results, fitted = {}, {}

    # --- 1. Naive Persistence Baseline: "future = current" ---
    print("\nEvaluating Persistence Baseline...")
    base = test.withColumn("prediction", F.col("congested").cast("double"))
    results["Baseline (persistence)"] = calc_metrics(base)
    results["Baseline (persistence)"].update(auc=None, pr_auc=None)
    print("Baseline (persistence):", results["Baseline (persistence)"])

    # --- 2. Train and Evaluate Classifiers ---
    for name, clf in models.items():
        print(f"\nFitting {name}...")
        pipeline = Pipeline(stages=prep + [clf])
        model = pipeline.fit(train)
        pred = model.transform(test).cache()
        m = calc_metrics(pred)
        m["auc"] = round(float(auc.evaluate(pred)), 4)
        m["pr_auc"] = round(float(pr.evaluate(pred)), 4)
        results[name] = m
        fitted[name] = (model, pred)
        print(f"{name}: {m}")

    best = max(fitted, key=lambda k: results[k]["f1"])
    print(f"\n>>> BEST MODEL SELECTED BY F1: {best} <<<")

    # Save best model
    print(f"Saving best model pipeline to: {MODEL_PATH}")
    fitted[best][0].write().overwrite().save(MODEL_PATH)

    # --- 3. Feature Importance (RF & GBT) ---
    print("\nExtracting feature importances...")
    imp = {}
    for nm in ("Random Forest", "Gradient Boosted Trees"):
        est = fitted[nm][0].stages[-1]
        imp[nm] = [float(x) for x in est.featureImportances.toArray()]

    n_extra = len(imp["Random Forest"]) - len(NUM)
    names = NUM + [f"weather_oh_{i}" for i in range(3)] + [f"zone_oh_{i}" for i in range(n_extra - 3)]
    fi_path = os.path.join(METRICS_DIR, "feature_importance.json")
    with open(fi_path, "w") as f:
        json.dump({"names": names, **imp}, f, indent=2)
    print(f"Saved: {fi_path}")

    # --- 4. Save Test Predictions for Visualizations ---
    print("\nExporting test predictions...")
    best_pred_df = fitted[best][1]
    # GBT probability vector: [P(0), P(1)]
    p_cols = best_pred_df.select("timestamp", "junction_name", "label", "prediction", "speed_ratio", "probability")
    pdf = p_cols.toPandas()
    pdf["prob"] = pdf["probability"].apply(lambda v: float(v[1]) if hasattr(v, "__getitem__") else float(v))
    pdf = pdf.drop(columns=["probability"])
    test_pred_path = os.path.join(PREDICTIONS_DIR, "test_predictions.csv")
    pdf.to_csv(test_pred_path, index=False)
    print(f"Saved test predictions: {test_pred_path}")

    # --- 5. Regression: Predict Future Speed Ratio (15 min ahead) ---
    print("\nFitting Random Forest Regressor for Future Speed Ratio...")
    rf_reg = Pipeline(stages=prep + [
        RandomForestRegressor(featuresCol="features", labelCol="label_future_ratio", numTrees=40, maxDepth=8, seed=5)
    ]).fit(train)

    rp = rf_reg.transform(test).cache()
    reg = {
        m: round(float(RegressionEvaluator(labelCol="label_future_ratio", predictionCol="prediction", metricName=m).evaluate(rp)), 4)
        for m in ("rmse", "mae", "r2")
    }
    b_rmse = round(float(RegressionEvaluator(labelCol="label_future_ratio", predictionCol="speed_ratio", metricName="rmse").evaluate(rp)), 4)
    b_mae  = round(float(RegressionEvaluator(labelCol="label_future_ratio", predictionCol="speed_ratio", metricName="mae").evaluate(rp)), 4)
    reg["baseline_rmse"] = b_rmse
    reg["baseline_mae"] = b_mae
    print(f"Regression Metrics: {reg}")

    # Export sample regression day (Central Bus Stand on 24 Sep)
    reg_sample_path = os.path.join(PREDICTIONS_DIR, "regression_sample.csv")
    rp.select("timestamp", "junction_name", "label_future_ratio", "prediction") \
      .filter("junction_name='Central Bus Stand' and date='2026-09-24'") \
      .toPandas().to_csv(reg_sample_path, index=False)
    print(f"Saved regression sample: {reg_sample_path}")

    # Save complete model results JSON
    model_results_path = os.path.join(METRICS_DIR, "model_results.json")
    with open(model_results_path, "w") as f:
        json.dump({
            "info": info,
            "classification": results,
            "best": best,
            "regression": reg,
            "features": feature_names
        }, f, indent=2, default=float)
    print(f"Saved model results JSON: {model_results_path}")

    spark.stop()
    print("Stage 2 completed successfully.")

if __name__ == "__main__":
    run_train_models()
