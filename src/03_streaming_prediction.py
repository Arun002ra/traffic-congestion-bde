"""
03_streaming_prediction.py
Spark Structured Streaming demo of the real-time processing layer.
* Source  : data/stream_input receives one micro-batch file per hour of the last day (2026-09-29)
* Query A : 15-minute tumbling-window aggregation per junction (30-min watermark) -> congestion alerts
* Query B : foreachBatch scoring with the saved GBT PipelineModel -> predicted congestion 15 min ahead
"""
import os
import shutil
import json
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F, types as T
from common import (
    get_spark, FEATURE_PATH, STREAM_IN, CHECKPOINT_DIR, MODEL_PATH,
    RESULTS_DIR, ALERTS_DIR, CONGESTION_RATIO, save_csv
)

def run_streaming_prediction():
    print("=" * 60)
    print("STAGE 3: Real-Time Spark Structured Streaming Simulation")
    print("=" * 60)

    spark = get_spark("Traffic-03-Streaming")

    print(f"Reading feature dataset from: {FEATURE_PATH}")
    feat = spark.read.parquet(FEATURE_PATH)
    day = feat.filter(F.col("date") == "2026-09-29").withColumn("hr", F.hour("timestamp"))

    # Prepare stream input folder
    local_in = STREAM_IN.replace("file:///", "").replace("file://", "")
    shutil.rmtree(local_in, ignore_errors=True)
    os.makedirs(local_in, exist_ok=True)

    print(f"Generating 24 hourly micro-batch files for 2026-09-29 into: {local_in}")
    pdf = day.toPandas()
    for h, g in pdf.groupby("hr"):
        batch_file = os.path.join(local_in, f"batch_{int(h):02d}.csv")
        g.drop(columns=["hr"]).to_csv(batch_file, index=False)

    schema = day.drop("hr").schema
    print(f"Configuring readStream with maxFilesPerTrigger=1 on {STREAM_IN}...")
    stream = (spark.readStream
              .schema(schema)
              .option("header", True)
              .option("maxFilesPerTrigger", 1)
              .csv(STREAM_IN))

    # ---------- Query A: Windowed Aggregation (Tumbling 15 min with 30 min Watermark) ----------
    print("\nStarting Query A: 15-Minute Tumbling Windows with 30-Minute Watermark...")
    agg = (stream.withWatermark("timestamp", "30 minutes")
           .groupBy(F.window("timestamp", "15 minutes"), "junction_name")
           .agg(
               F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed"),
               F.round(F.avg("speed_ratio"), 3).alias("avg_ratio"),
               F.sum("vehicle_count").alias("vehicles")
           ))

    qa = (agg.writeStream
          .outputMode("update")
          .format("memory")
          .queryName("win_agg")
          .trigger(availableNow=True)
          .start())
    qa.awaitTermination()

    w = spark.sql("select window.start as window_start, junction_name, avg_speed, avg_ratio, vehicles from win_agg")
    w = (w.withColumn("alert", (F.col("avg_ratio") < CONGESTION_RATIO).cast("int"))
          .orderBy("window_start", "junction_name"))

    win_agg_path = os.path.join(ALERTS_DIR, "stream_window_agg.csv")
    w.toPandas().to_csv(win_agg_path, index=False)
    print(f"Windows produced: {w.count():,} | Alert windows (ratio < 0.60): {w.filter('alert=1').count():,}")
    print(f"Saved window aggregates: {win_agg_path}")

    # ---------- Query B: Model Scoring inside foreachBatch ----------
    print(f"\nStarting Query B: Live Scoring with saved PipelineModel from {MODEL_PATH}...")
    model = PipelineModel.load(MODEL_PATH)
    log = []

    # Clear previous checkpoint
    chk_local = CHECKPOINT_DIR.replace("file:///", "").replace("file://", "")
    shutil.rmtree(chk_local, ignore_errors=True)

    stream_b = (spark.readStream
                .schema(schema)
                .option("header", True)
                .option("maxFilesPerTrigger", 1)
                .csv(STREAM_IN))

    def score_microbatch(batch_df, batch_id):
        out = model.transform(batch_df.withColumn("label", F.col("label_future_congested").cast("double")))
        n = out.count()
        if n == 0:
            return

        s = out.select("timestamp", "junction_name", "label", "prediction")
        stats = s.agg(
            F.count("*").alias("n"),
            F.sum("prediction").alias("pred_pos"),
            F.sum(F.when(F.col("prediction") == F.col("label"), 1).otherwise(0)).alias("correct")
        ).first()

        batch_info = {
            "batch_id": int(batch_id),
            "rows": int(stats["n"]),
            "predicted_congested": int(stats["pred_pos"]) if stats["pred_pos"] is not None else 0,
            "correct": int(stats["correct"]) if stats["correct"] is not None else 0,
            "batch_accuracy": round(float(stats["correct"]) / float(stats["n"]), 4) if stats["n"] else 1.0
        }
        log.append(batch_info)

        # Save individual batch predictions
        batch_out_path = os.path.join(ALERTS_DIR, f"stream_scores_{int(batch_id):02d}.csv")
        s.toPandas().to_csv(batch_out_path, index=False)

    qb = (stream_b.writeStream
          .foreachBatch(score_microbatch)
          .trigger(availableNow=True)
          .option("checkpointLocation", CHECKPOINT_DIR)
          .start())
    qb.awaitTermination()

    batches_df = pd.DataFrame(log)
    batches_path = os.path.join(ALERTS_DIR, "stream_batches.csv")
    batches_df.to_csv(batches_path, index=False)
    print(f"Saved stream micro-batch log: {batches_path}")
    print(batches_df.to_string(index=False))

    spark.stop()
    print("Stage 3 completed successfully.")

if __name__ == "__main__":
    run_streaming_prediction()
