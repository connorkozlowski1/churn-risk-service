# Churn Risk Service
End-to-end machine learning project that predicts customer churn and exposes a production-style churn-risk scoring API.

The repo demonstrates a full ML pipeline:
- Reproducible data ingestion with automatic dataset download
- Deterministic preprocessing pipeline
- Trainable + tunable model with experiment tracking (MLflow)
- FastAPI prediction service
- Model promotion logic
- Basic monitoring scaffolding
- Automated tests (preprocessing + API contract)

## Problem Overview
Subscription-based businesses need to identify which customers are likely to churn. This project trains a supervised binary classifier on the Telco Customer Churn dataset and provides a real API that returns:
- churn_probability
- churn_prediction (0 or 1)

## Tech Stack
- Python 3.13
- pandas, numpy, scikit-learn
- FastAPI, Uvicorn
- MLflow
- joblib
- pytest

## Project Structure

churn-risk-service/
├── configs/
│   └── mlflow.yaml
├── src/
│   └── churn_risk_service/
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       ├── predict.py
│       ├── train.py
│       ├── tune.py
│       ├── promote.py
│       └── monitor.py
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── requirements.txt
├── LICENSE
└── README.md

Note: dataset, model artifacts, and MLflow runs are not committed — directories like data/, models/, and mlruns/ are created at runtime as needed.

## Setup
git clone https://github.com/connorkozlowski1/churn-risk-service.git

cd churn-risk-service

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

## Training the Model
Training automatically downloads the Telco churn dataset if missing.
python -m src.churn_risk_service.train

This will:
- Download and cache the raw CSV
- Preprocess data
- Train the model
- Log metrics to MLflow
- Save the model to models/churn_model.joblib

## Running the Prediction API
Start the API:
uvicorn src.churn_risk_service.predict:app --reload

Uvicorn prints something like:
Uvicorn running on http://127.0.0.1:8000

Open in your browser:
http://127.0.0.1:8000/docs

Use the interactive Swagger UI to submit a JSON payload to POST /predict.

## Hyperparameter Tuning
python -m src.churn_risk_service.tune

## Automated Model Promotion
python -m src.churn_risk_service.promote
Promotes the new model only if its validation AUC improves over the deployed one.

## Tests
pytest
Includes:
- Preprocessing tests
- API contract tests

## License
MIT License.
