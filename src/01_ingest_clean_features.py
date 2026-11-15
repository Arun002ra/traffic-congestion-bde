"""
01_ingest_clean_features.py
Stage 1-3 of the pipeline: ingest raw CSV -> clean -> feature engineering -> aggregates.
Executes using PySpark DataFrames, Window functions, and Spark SQL.
"""
import os
import json
from pyspark.sql import Window, functions as F, types as T
from common import (
    get_spark, RAW_PATH, CLEAN_PATH, FEATURE_PATH, RESULTS_DIR, SUMMARIES_DIR,
    CONGESTION_RATIO, HORIZON_SLOTS, save_csv
)

def run_ingest_clean_features():
    print("=" * 60)
    print("STAGE 1: Ingestion, Data Cleaning & Feature Engineering")
    print("=" * 60)

    spark = get_spark("Traffic-01-Clean-Features")

    schema = T.StructType([
        T.StructField("reading_id", T.LongType(), True),
        T.StructField("timestamp", T.TimestampType(), True),
        T.StructField("junction_id", T.IntegerType(), True),
        T.StructField("junction_name", T.StringType(), True),
        T.StructField("zone", T.StringType(), True),
        T.StructField("vehicle_count", T.IntegerType(), True),
        T.StructField("avg_speed_kmph", T.DoubleType(), True),
        T.StructField("occupancy_pct", T.DoubleType(), True),
        T.StructField("queue_length_m", T.DoubleType(), True),
        T.StructField("weather", T.StringType(), True),
        T.StructField("is_holiday", T.IntegerType(), True),
        T.StructField("incident_flag", T.IntegerType(), True),
        T.StructField("free_flow_kmph", T.IntegerType(), True),
    ])

    print(f"Reading raw sensor dataset from: {RAW_PATH}")
    raw = spark.read.option("header", True).schema(schema).csv(RAW_PATH)

    # ---------------- 1. Data-quality reporting ----------------
    rows_raw = raw.count()
    print(f"Raw rows loaded: {rows_raw:,}")

    dq = {"rows_raw": rows_raw}
    dedup_count = raw.dropDuplicates(["timestamp", "junction_id"]).count()
    dq["duplicates"] = rows_raw - dedup_count
    dq["null_speed"] = raw.filter(F.col("avg_speed_kmph").isNull()).count()
    dq["negative_count"] = raw.filter(F.col("vehicle_count") < 0).count()

    print(f"Duplicates detected: {dq['duplicates']:,}")
    print(f"Null speeds detected: {dq['null_speed']:,}")
    print(f"Negative vehicle counts detected: {dq['negative_count']:,}")

    # ---------------- 2. Data Cleaning & Repair ----------------
    dedup = raw.dropDuplicates(["timestamp", "junction_id"])

    # Invalid sensor values become NULL, then forward-filled per junction
    cleaned_sensors = (dedup
        .withColumn("vehicle_count", F.when(F.col("vehicle_count") < 0, None).otherwise(F.col("vehicle_count")))
        .withColumn("avg_speed_kmph", F.when(F.col("avg_speed_kmph") <= 0, None).otherwise(F.col("avg_speed_kmph")))
    )

    w_ff = Window.partitionBy("junction_id").orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0)
    clean = (cleaned_sensors
        .withColumn("vehicle_count",  F.last("vehicle_count",  True).over(w_ff))
        .withColumn("avg_speed_kmph", F.last("avg_speed_kmph", True).over(w_ff))
        .dropna(subset=["vehicle_count", "avg_speed_kmph"])
    )

    clean = (clean
        .withColumn("speed_ratio", F.round(F.col("avg_speed_kmph") / F.col("free_flow_kmph"), 4))
        .withColumn("congested", (F.col("speed_ratio") < CONGESTION_RATIO).cast("int"))
        .withColumn("congestion_level",
            F.when(F.col("speed_ratio") >= 0.8, "Free-flow")
             .when(F.col("speed_ratio") >= 0.6, "Moderate")
             .when(F.col("speed_ratio") >= 0.4, "Heavy").otherwise("Severe"))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("minute", F.minute("timestamp"))
        .withColumn("minute_of_day", F.col("hour") * 60 + F.col("minute"))
        .withColumn("dow", F.dayofweek("timestamp"))          # 1=Sun .. 7=Sat
        .withColumn("is_weekend", F.col("dow").isin(1, 7).cast("int"))
        .withColumn("date", F.to_date("timestamp"))
        .withColumn("is_peak", (((F.col("hour").between(8, 10)) | (F.col("hour").between(17, 19))) & (F.col("is_weekend") == 0)).cast("int"))
    )

    clean = clean.cache()
    rows_clean = clean.count()
    dq["rows_clean"] = rows_clean
    dq["min_ts"] = str(clean.agg(F.min("timestamp")).first()[0])
    dq["max_ts"] = str(clean.agg(F.max("timestamp")).first()[0])
    dq["junctions"] = clean.select("junction_id").distinct().count()
    dq["congested_rate_pct"] = round(clean.agg(F.avg("congested")).first()[0] * 100, 2)

    print(f"Clean rows: {rows_clean:,} (Grid completeness: {dq['junctions']} junctions)")
    print(f"Overall Congestion Rate: {dq['congested_rate_pct']}%")

    print(f"Writing clean Parquet to: {CLEAN_PATH}")
    clean.write.mode("overwrite").parquet(CLEAN_PATH)

    # ---------------- 3. Feature Engineering ----------------
    w = Window.partitionBy("junction_id").orderBy("timestamp")
    w_roll6 = w.rowsBetween(-5, 0)

    feat = (clean
        .withColumn("speed_ratio_lag1", F.lag("speed_ratio", 1).over(w))
        .withColumn("speed_ratio_lag2", F.lag("speed_ratio", 2).over(w))
        .withColumn("speed_ratio_lag3", F.lag("speed_ratio", 3).over(w))
        .withColumn("count_lag1", F.lag("vehicle_count", 1).over(w))
        .withColumn("count_lag3", F.lag("vehicle_count", 3).over(w))
        .withColumn("count_roll30", F.avg("vehicle_count").over(w_roll6))
        .withColumn("speed_roll30", F.avg("speed_ratio").over(w_roll6))
        .withColumn("speed_trend", F.col("speed_ratio") - F.col("speed_ratio_lag3"))
        .withColumn("label_future_congested", F.lead("congested", HORIZON_SLOTS).over(w))
        .withColumn("label_future_ratio", F.lead("speed_ratio", HORIZON_SLOTS).over(w))
        .dropna(subset=["speed_ratio_lag3", "count_lag3", "label_future_congested"])
    )

    rows_feat = feat.count()
    dq["rows_features"] = rows_feat
    print(f"Feature table rows (after warmup/horizon drop): {rows_feat:,}")

    print(f"Writing feature Parquet to: {FEATURE_PATH}")
    feat.write.mode("overwrite").parquet(FEATURE_PATH)

    # ---------------- 4. Analytical Aggregations ----------------
    print("Computing Spark SQL analytical aggregations...")
    cl = clean

    # 1. Hourly profile
    save_csv(cl.groupBy("hour").agg(
        F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed"),
        F.round(F.avg("vehicle_count"), 1).alias("avg_vehicles"),
        F.round(F.avg("speed_ratio"), 3).alias("avg_speed_ratio"),
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct")
    ).orderBy("hour"), "hourly_profile.csv")

    # 2. Junction summary
    save_csv(cl.groupBy("junction_id", "junction_name", "zone").agg(
        F.count("*").alias("records"),
        F.round(F.avg("vehicle_count"), 1).alias("avg_vehicles"),
        F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed"),
        F.round(F.avg("occupancy_pct"), 2).alias("avg_occupancy"),
        F.round(F.avg("queue_length_m"), 1).alias("avg_queue_m"),
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct")
    ).orderBy(F.desc("congestion_rate_pct")), "junction_summary.csv")

    # 3. Weather impact
    save_csv(cl.groupBy("weather").agg(
        F.count("*").alias("records"),
        F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed"),
        F.round(F.avg("speed_ratio"), 3).alias("avg_speed_ratio"),
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct")
    ), "weather_impact.csv")

    # 4. Day of week x Hour heatmap
    save_csv(cl.groupBy("dow", "hour").agg(
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct")
    ).orderBy("dow", "hour"), "dow_hour_heatmap.csv")

    # 5. Junction x Hour heatmap
    save_csv(cl.groupBy("junction_name", "hour").agg(
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct")
    ).orderBy("junction_name", "hour"), "junction_hour_heatmap.csv")

    # 6. Zone summary
    save_csv(cl.groupBy("zone").agg(
        F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed"),
        F.round(F.avg("queue_length_m"), 1).alias("avg_queue_m"),
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct")
    ).orderBy(F.desc("congestion_rate_pct")), "zone_summary.csv")

    # 7. Peak vs Off-peak
    save_csv(cl.groupBy("is_peak").agg(
        F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed"),
        F.round(F.avg("vehicle_count"), 1).alias("avg_vehicles"),
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct")
    ), "peak_vs_offpeak.csv")

    # 8. Incident impact
    save_csv(cl.groupBy("incident_flag").agg(
        F.count("*").alias("records"),
        F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed"),
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct")
    ), "incident_impact.csv")

    # 9. Congestion levels
    save_csv(cl.groupBy("congestion_level").count().orderBy("count"), "congestion_levels.csv")

    # 10. Daily trend
    save_csv(cl.groupBy("date").agg(
        F.round(F.avg("congested") * 100, 2).alias("congestion_rate_pct"),
        F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed")
    ).orderBy("date"), "daily_trend.csv")

    # 11. Sample for correlation
    save_csv(cl.select(
        "vehicle_count", "avg_speed_kmph", "occupancy_pct", "queue_length_m", "speed_ratio"
    ).sample(0.05, seed=7), "sample_for_corr.csv")

    # 12. Junction volume by hour (weekdays)
    save_csv(cl.filter("is_weekend=0").groupBy("junction_name", "hour").agg(
        F.round(F.avg("vehicle_count"), 1).alias("avg_vehicles")
    ).orderBy("junction_name", "hour"), "junction_hour_volume.csv")

    # 13. Worst junction hours
    save_csv(cl.groupBy("date", "hour", "junction_name").agg(
        F.round(F.avg("avg_speed_kmph"), 2).alias("avg_speed"),
        F.round(F.avg("congested") * 100, 1).alias("cong_pct")
    ).orderBy(F.desc("cong_pct")).limit(10), "worst_junction_hours.csv")

    # Save data quality JSON
    dq_file = os.path.join(RESULTS_DIR, "data_quality.json")
    with open(dq_file, "w") as f:
        json.dump(dq, f, indent=2)
    print(f"Saved: {dq_file}")

    print("Summary of Congestion Levels:")
    clean.groupBy("congestion_level").count().show()

    spark.stop()
    print("Stage 1 completed successfully.")

if __name__ == "__main__":
    run_ingest_clean_features()
