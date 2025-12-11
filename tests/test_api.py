import os
import sys
import pytest
from fastapi.testclient import TestClient

# Make src importable
sys.path.append(os.path.abspath("src"))

from churn_risk_service.predict import app


client = TestClient(app)


def test_predict_endpoint_returns_valid_response():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 300.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "churn_probability" in result
    assert "churn_prediction" in result

    assert isinstance(result["churn_probability"], float)
    assert result["churn_prediction"] in (0, 1)


def test_predict_endpoint_rejects_bad_payload():
    # Missing required fields
    bad_payload = {}

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # FastAPI validation error
