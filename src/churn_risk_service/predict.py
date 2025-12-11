from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models") / "churn_model.joblib"

# Load model at startup
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Churn Risk Prediction Service")


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """
    Accepts a single customer's features and returns churn probability + prediction.
    """
    df = pd.DataFrame([features.model_dump()])
    proba = model.predict_proba(df)[0, 1]
    pred = model.predict(df)[0]
    return {
        "churn_probability": float(proba),
        "churn_prediction": int(pred),
    }
