from pathlib import Path
import pandas as pd

from .data import load_raw_telco_churn


def preprocess_telco_churn(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Basic cleaning for the Telco churn dataset.

    - Drop customerID (identifier).
    - Convert TotalCharges to numeric.
    - Drop rows with missing TotalCharges.
    - Create numeric churn target (ChurnFlag: 1 = churn, 0 = no churn).

    Parameters
    ----------
    df : pd.DataFrame or None
        Raw dataframe. If None, it will be loaded from disk.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering/modeling.
    """
    if df is None:
        df = load_raw_telco_churn()

    df = df.copy()

    # Drop identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges numeric conversion
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])

    # Create numeric target
    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' column in dataset.")

    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["ChurnFlag"].isna().any():
        raise ValueError("Unexpected values in 'Churn' column when creating ChurnFlag.")
    
    # Drop original Churn label from features
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])


    return df


if __name__ == "__main__":
    raw_df = load_raw_telco_churn()
    clean_df = preprocess_telco_churn(raw_df)
    print(clean_df.head())
    print(f"\nShape after cleaning: {clean_df.shape}")
    print("\nColumns:")
    print(clean_df.dtypes)
