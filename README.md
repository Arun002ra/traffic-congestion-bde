# Real-Time Traffic Data Analysis and Congestion Prediction Using Apache Spark and PySpark

## 📌 Project Overview
A complete, functional Big Data Engineering project developing an intelligent traffic monitoring and 15-minute congestion prediction pipeline for **12 major junctions in Tiruchirappalli**. The system ingests **207,774 raw readings** over a 60-day period (1 August 2026 to 29 September 2026), cleans and repairs sensor defects using PySpark window functions, computes spatio-temporal features, evaluates multiple Spark MLlib classifiers against a persistence baseline, replays real-time micro-batches using Spark Structured Streaming, and serves an interactive 10-page Streamlit analytics dashboard faithfully replicating **Slide 21** of the project presentation.

---

## 🏛️ Team & Academic Context
- **Institution**: SRM Institute of Science and Technology, Tiruchirappalli - 621105
- **Department**: Information Technology / Section A
- **Course**: Big Data Essentials (Apache Spark)
- **Mentor**: Dr. Yuvaraj D
- **Team Members**:
  - Vishal Kumar Yadav (RA2411008050010)
  - S Arunraj (RA2411008050005)
  - Edwin Joe M (RA2411008050034)
  - Sahil Chouhan (RA2411008050008)

---

## 🛠️ Technology Stack
- **Language**: Python 3.14
- **Distributed Big Data Engine**: Apache Spark & PySpark 4.3.0
- **Querying & Features**: Spark SQL & Window Functions
- **Machine Learning**: Spark MLlib (`LogisticRegression`, `RandomForestClassifier`, `GBTClassifier`, `RandomForestRegressor`)
- **Streaming**: Spark Structured Streaming (15-min tumbling windows, 30-min watermark)
- **Dashboard**: Python Streamlit
- **Visualizations**: Plotly & Matplotlib
- **Storage**: CSV & Apache Parquet

---

## 📂 Project Structure
```
traffic-congestion-bde/
├── app.py                         # Interactive 10-page Streamlit dashboard
├── requirements.txt               # Project Python dependencies
├── README.md                      # Detailed technical documentation
│
├── data/
│   ├── raw/
│   │   └── traffic_readings.csv   # 207,774 simulated raw sensor readings
│   ├── processed/
│   │   ├── clean/                 # 207,360 cleaned sensor readings (Parquet)
│   │   └── features/              # 207,288 feature-engineered records (Parquet)
│   └── stream_input/              # 24 hourly micro-batch CSVs (2026-09-29)
│
├── src/
│   ├── common.py                  # SparkSession builder, Java 17 configuration, paths
│   ├── generate_traffic_data.py   # Traffic simulator generating 207,774 rows
│   ├── 01_ingest_clean_features.py# PySpark ETL, validation, forward fill & 14 SQL summaries
│   ├── 02_train_models.py         # Spark MLlib model training (LogReg, RF, GBT, RF Regressor)
│   ├── 03_streaming_prediction.py # Spark Structured Streaming simulation & scoring
│   ├── 04_make_figures.py         # Publication-grade figure generation
│   └── 05_dashboard_static.py     # Slide 21 high-resolution Matplotlib dashboard rendering
│
├── models/
│   └── best_congestion_model/     # Serialized Spark MLlib GBT PipelineModel
│
├── results/
│   ├── summaries/                 # 14 Spark SQL summary tables (hourly, junction, weather, etc.)
│   ├── predictions/               # Test set predictions and regression evaluation samples
│   ├── alerts/                    # Streaming alert windows & micro-batch logs
│   ├── model_metrics/             # Model evaluations, confusion matrix, and feature importances
│   └── data_quality.json          # Data quality audit metrics
│
└── figures/                       # High-resolution charts and Slide 21 replica
```

---

## 🚀 Execution Guide

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Generate Simulated Traffic Dataset (207,774 rows)
```bash
python src/generate_traffic_data.py
```

### 3. Run PySpark Cleaning & Feature Engineering
```bash
python src/01_ingest_clean_features.py
```

### 4. Train Spark MLlib Models
```bash
python src/02_train_models.py
```

### 5. Run Spark Structured Streaming Simulation
```bash
python src/03_streaming_prediction.py
```

### 6. Render High-Resolution Figures & Slide 21 Dashboard Image
```bash
python src/04_make_figures.py
python src/05_dashboard_static.py
```

### 7. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Key Results
- **Clean Readings**: 207,360 across 12 Tiruchirappalli junctions (100% complete grid).
- **Overall Congestion Rate**: 13.32% (Weekday Peak: 61.65% vs Off-Peak: 3.07%).
- **Most Congested Junction**: Central Bus Stand (21.86%).
- **Best Classifier**: Gradient Boosted Trees (GBT).
  - **Accuracy**: 96.25%
  - **Precision**: 79.02%
  - **Recall**: 93.20% (caught 411 more traffic jams than persistence baseline)
  - **F1-Score**: 0.8553
  - **ROC-AUC**: 0.9925
  - **PR-AUC**: 0.9528
- **Speed Ratio Regressor (15-min ahead)**:
  - **RMSE**: 0.0571 (-18% error compared to persistence baseline)
  - **R²**: 0.9124
- **Real-Time Streaming**: 24 micro-batches scored with 164 alert windows triggered.
