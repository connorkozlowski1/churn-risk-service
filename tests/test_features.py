import pandas as pd
import os
import sys

# Add the src directory to the import path
sys.path.append(os.path.abspath("src"))

from churn_risk_service.features import preprocess_telco_churn
from churn_risk_service.data import load_raw_telco_churn


def test_preprocess_creates_churnflag_and_drops_churn():
    raw_df = load_raw_telco_churn()
    clean_df = preprocess_telco_churn(raw_df)

    # ChurnFlag exists and is numeric
    assert "ChurnFlag" in clean_df.columns
    assert clean_df["ChurnFlag"].dtype.kind in ("i", "u")

    # Original Churn column is gone (no leakage)
    assert "Churn" not in clean_df.columns


def test_preprocess_drops_customerid():
    raw_df = load_raw_telco_churn()
    clean_df = preprocess_telco_churn(raw_df)

    assert "customerID" not in clean_df.columns
