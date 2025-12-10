from pathlib import Path
import joblib
import json

from .model import MODEL_PATH
from .features import preprocess_telco_churn
from .model import get_feature_target
from .model import build_model_pipeline

PROMOTION_RECORD = Path("models/promotion_record.json")


def score_model(model, X_val, y_val):
    from sklearn.metrics import roc_auc_score

    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


def main():
    df = preprocess_telco_churn()
    X, y = get_feature_target(df)

    # Split off a validation block manually
    import numpy as np
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    val_idx = idx[:800]
    train_idx = idx[800:]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Load current deployed model
    if MODEL_PATH.exists():
        deployed = joblib.load(MODEL_PATH)
        deployed_auc = score_model(deployed, X_val, y_val)
    else:
        deployed_auc = -1  # Treat as no model deployed

    # Train candidate model
    candidate = build_model_pipeline(X_train)
    candidate.fit(X_train, y_train)
    candidate_auc = score_model(candidate, X_val, y_val)

    print(f"Deployed AUC: {deployed_auc}")
    print(f"Candidate AUC: {candidate_auc}")

    if candidate_auc > deployed_auc:
        # Promote new model
        joblib.dump(candidate, MODEL_PATH)
        PROMOTION_RECORD.write_text(json.dumps({
            "old_auc": deployed_auc,
            "new_auc": candidate_auc,
        }, indent=2))
        print("Promoted new model.")
    else:
        print("Kept existing model.")


if __name__ == "__main__":
    main()
