"""common.py - shared Spark session, paths, and constants."""
import os
import sys

# Ensure Java 17 (OpenJDK 17) is used by PySpark on Windows
JAVA17_DIR = r"C:\Program Files\Tableau\Tableau 2026.2\bin\jre"
if os.path.isdir(JAVA17_DIR):
    os.environ["JAVA_HOME"] = JAVA17_DIR
    java_bin = os.path.join(JAVA17_DIR, "bin")
    if java_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession

# Base project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BDE_BASE = os.environ.get("BDE_BASE", f"file:///{BASE_DIR.replace(chr(92), '/')}")

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "raw", "traffic_readings.csv").replace("\\", "/")
CLEAN_PATH = os.path.join(DATA_DIR, "processed", "clean").replace("\\", "/")
FEATURE_PATH = os.path.join(DATA_DIR, "processed", "features").replace("\\", "/")
STREAM_IN = os.path.join(DATA_DIR, "stream_input").replace("\\", "/")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "chk").replace("\\", "/")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_congestion_model").replace("\\", "/")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUMMARIES_DIR = os.path.join(RESULTS_DIR, "summaries")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
ALERTS_DIR = os.path.join(RESULTS_DIR, "alerts")
METRICS_DIR = os.path.join(RESULTS_DIR, "model_metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Ensure required directories exist
for path in [
    os.path.join(DATA_DIR, "raw"),
    os.path.join(DATA_DIR, "processed", "clean"),
    os.path.join(DATA_DIR, "processed", "features"),
    os.path.join(DATA_DIR, "stream_input"),
    MODELS_DIR,
    SUMMARIES_DIR,
    PREDICTIONS_DIR,
    ALERTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
]:
    os.makedirs(path, exist_ok=True)

# Project Constants
CONGESTION_RATIO = 0.60   # speed/free-flow below this => "congested"
HORIZON_SLOTS = 3         # 3 x 5 min = 15 minutes ahead
SPLIT_DATE = "2026-09-19" # time-based train/test split (Aug 1 - Sep 18 vs Sep 19 - Sep 29)

# 12 Tiruchirappalli Junctions Configuration
JUNCTIONS = [  # id, name, zone, capacity veh/5min, free-flow speed km/h, demand weight
    (1,  "Central Bus Stand",   "Central",     210, 38, 1.00),
    (2,  "Chathiram Bus Stand", "Central",     190, 36, 0.95),
    (3,  "Palakkarai",          "Central",     170, 32, 0.92),
    (4,  "Thillai Nagar",       "Commercial",  180, 40, 0.90),
    (5,  "Cantonment",          "Commercial",  200, 42, 0.88),
    (6,  "Srirangam Gate",      "Temple",      160, 34, 0.85),
    (7,  "Woraiyur",            "Residential", 150, 40, 0.70),
    (8,  "Tennur",              "Residential", 140, 42, 0.62),
    (9,  "Ariyamangalam",       "Industrial",  220, 48, 0.80),
    (10, "Ponmalai",            "Industrial",  200, 46, 0.72),
    (11, "Airport Road",        "Suburban",    180, 55, 0.55),
    (12, "Samayapuram Toll",    "Highway",     240, 60, 0.65),
]

def get_spark(app="TrafficCongestion", cores=2):
    """Initializes and returns a configured local PySpark session."""
    spark = (SparkSession.builder
             .appName(app)
             .master(f"local[{cores}]")
             .config("spark.driver.memory", "3g")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.sql.session.timeZone", "Asia/Kolkata")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def save_csv(sdf, filename, target_dir=SUMMARIES_DIR):
    """Converts a Spark DataFrame to Pandas and writes to CSV."""
    os.makedirs(target_dir, exist_ok=True)
    out_file = os.path.join(target_dir, filename)
    sdf.toPandas().to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
