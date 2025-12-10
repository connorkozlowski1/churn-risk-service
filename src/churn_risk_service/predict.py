from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models") / "churn_model.joblib"

# Load model at startup
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Churn Risk Prediction Service")


class CustomerData(BaseModel):
    data: Dict[str, Any]


@app.post("/predict")
def predict_churn(payload: CustomerData):
    """
    Accepts a single customer record as a dict of feature_name: value.
    Returns churn probability.
    """
    df = pd.DataFrame([payload.data])
    proba = model.predict_proba(df)[0, 1]
    pred = model.predict(df)[0]
    return {
        "churn_probability": float(proba),
        "churn_prediction": int(pred)
    }
