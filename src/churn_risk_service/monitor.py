import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
import joblib

MODEL_PATH = Path("models") / "churn_model.joblib"
BASELINE_STATS_PATH = Path("monitoring") / "baseline_stats.parquet"


def compute_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {
        "feature": [],
        "mean": [],
        "std": [],
        "min": [],
        "max": [],
    }

    for col in df.columns:
        if df[col].dtype.kind in "if":  # numeric only
            stats["feature"].append(col)
            stats["mean"].append(df[col].mean())
            stats["std"].append(df[col].std())
            stats["min"].append(df[col].min())
            stats["max"].append(df[col].max())

    return pd.DataFrame(stats)


def save_baseline_stats(df: pd.DataFrame):
    stats = compute_feature_stats(df)
    BASELINE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(BASELINE_STATS_PATH, index=False)


def detect_data_drift(new_df: pd.DataFrame, p_threshold: float = 0.05):
    baseline = pd.read_parquet(BASELINE_STATS_PATH)

    drift_report = []

    for _, row in baseline.iterrows():
        col = row["feature"]
        if col not in new_df.columns:
            continue

        baseline_sample = np.random.normal(row["mean"], row["std"], size=len(new_df))
        new_sample = new_df[col].dropna().values

        if len(new_sample) < 2:
            continue

        stat, p_val = ks_2samp(baseline_sample, new_sample)

        drift_report.append({
            "feature": col,
            "p_value": float(p_val),
            "drift_detected": bool(p_val < p_threshold)
        })

    return drift_report


if __name__ == "__main__":
    # generate baseline stats after training
    df = pd.read_csv("data/raw/telco_churn.csv")
    save_baseline_stats(df)
    print("Baseline monitoring stats saved.")
