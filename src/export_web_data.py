"""
export_web_data.py
Packages all Spark SQL summaries, MLlib metrics, data quality logs, and streaming alerts
into a single lightweight JSON file for the Vercel web application.
"""
import os
import json
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUMMARIES_DIR = os.path.join(RESULTS_DIR, "summaries")
METRICS_DIR = os.path.join(RESULTS_DIR, "model_metrics")
ALERTS_DIR = os.path.join(RESULTS_DIR, "alerts")

web_data = {}

# 1. Data Quality
dq_file = os.path.join(RESULTS_DIR, "data_quality.json")
if os.path.exists(dq_file):
    with open(dq_file) as f:
        web_data["data_quality"] = json.load(f)

# 2. Model Metrics
mr_file = os.path.join(METRICS_DIR, "model_results.json")
if os.path.exists(mr_file):
    with open(mr_file) as f:
        web_data["model_results"] = json.load(f)

# 3. Feature Importance
fi_file = os.path.join(METRICS_DIR, "feature_importance.json")
if os.path.exists(fi_file):
    with open(fi_file) as f:
        web_data["feature_importance"] = json.load(f)

# 4. Summaries
for fname in [
    "hourly_profile.csv", "junction_summary.csv", "weather_impact.csv",
    "daily_trend.csv", "dow_hour_heatmap.csv", "junction_hour_heatmap.csv",
    "zone_summary.csv", "peak_vs_offpeak.csv", "incident_impact.csv",
    "congestion_levels.csv"
]:
    fpath = os.path.join(SUMMARIES_DIR, fname)
    if os.path.exists(fpath):
        key = fname.replace(".csv", "")
        web_data[key] = pd.read_csv(fpath).to_dict(orient="records")

# 5. Streaming Batches
batches_file = os.path.join(ALERTS_DIR, "stream_batches.csv")
if os.path.exists(batches_file):
    web_data["stream_batches"] = pd.read_csv(batches_file).to_dict(orient="records")

out_path = os.path.join(BASE_DIR, "web_data.json")
with open(out_path, "w") as f:
    json.dump(web_data, f, indent=2)

print(f"Exported web data to {out_path} ({os.path.getsize(out_path):,} bytes)")
