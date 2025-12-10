# Churn Risk Service

End-to-end machine learning project that predicts customer churn and serves a production-ready churn risk score via API.

This repo shows how to go from raw data to:
- Reproducible training pipeline
- Packaged model with preprocessing
- FastAPI prediction service
- Experiment tracking with MLflow
- Basic monitoring scaffolding

---

## 1. Problem Overview

Many businesses run on recurring revenue (subscriptions, contracts). The key question:

> Which customers are at high risk of churn in the near future?

This project uses the Telco Customer Churn dataset to:
- Train a supervised model to predict churn (binary classification)
- Expose a REST endpoint to score individual customers
- Track experiments and model performance over time
- Prepare for drift detection on future data

---

## 2. Tech Stack

- **Language:** Python 3.13
- **ML & Data:** pandas, numpy, scikit-learn
- **API:** FastAPI, Uvicorn
- **Experiment Tracking:** MLflow
- **Serialization:** joblib

---

## 3. Project Structure

```text
churn-risk-service/
├── configs/
│   └── mlflow.yaml          # MLflow experiment + tracking config
├── data/
│   └── raw/
│       └── telco_churn.csv  # Raw Telco churn dataset (not for redistribution)
├── models/
│   └── churn_model.joblib   # Trained sklearn Pipeline (preprocess + model)
├── monitoring/
│   └── baseline_stats.parquet  # Baseline stats for drift detection
├── mlruns/                  # Local MLflow tracking directory
├── src/
│   └── churn_risk_service/
│       ├── __init__.py
│       ├── data.py          # Raw data loading
│       ├── features.py      # Cleaning + target creation
│       ├── model.py         # Model pipeline and training logic
│       ├── monitor.py       # Baseline stats + data drift utilities
│       ├── predict.py       # FastAPI app exposing /predict
│       └── train.py         # Training entrypoint with MLflow logging
├── LICENSE                  # MIT License
├── README.md
└── requirements.txt
