from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import preprocess_telco_churn


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # fraction of remaining data after test split


def get_feature_target(df: pd.DataFrame):
    """
    Split dataframe into features and target.
    """
    if "ChurnFlag" not in df.columns:
        raise ValueError("Expected 'ChurnFlag' column as target.")

    X = df.drop(columns=["ChurnFlag"])
    y = df["ChurnFlag"].astype(int)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles numeric and categorical columns.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build the full preprocessing + model pipeline.
    """
    preprocessor = build_preprocessor(X)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=None,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model


def train_baseline_model() -> dict:
    """
    Train a baseline churn model and save it to disk.

    Returns
    -------
    dict with basic metrics.
    """
    df = preprocess_telco_churn()
    X, y = get_feature_target(df)

    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Then split train/val from the remaining
    val_fraction_of_temp = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction_of_temp,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    model = build_model_pipeline(X_train)

    model.fit(X_train, y_train)

    # Eval on val and test
    def eval_split(X_split, y_split, split_name: str):
        y_pred = model.predict(X_split)
        y_proba = model.predict_proba(X_split)[:, 1]

        acc = accuracy_score(y_split, y_pred)
        auc = roc_auc_score(y_split, y_proba)

        print(f"\n=== {split_name.upper()} METRICS ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC : {auc:.4f}")
        print("Classification report:")
        print(classification_report(y_split, y_pred, digits=4))

        return {"accuracy": acc, "roc_auc": auc}

    metrics_val = eval_split(X_val, y_val, "validation")
    metrics_test = eval_split(X_test, y_test, "test")

    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH.resolve()}")

    return {
        "val": metrics_val,
        "test": metrics_test,
    }


if __name__ == "__main__":
    train_baseline_model()
