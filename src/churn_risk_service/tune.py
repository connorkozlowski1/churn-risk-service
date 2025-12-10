import numpy as np
from pathlib import Path
import mlflow
from sklearn.model_selection import RandomizedSearchCV

from .features import preprocess_telco_churn
from .model import build_model_pipeline, get_feature_target, RANDOM_STATE, MODEL_DIR


TUNED_MODEL_PATH = MODEL_DIR / "churn_model_tuned.joblib"


def main() -> None:
    df = preprocess_telco_churn()
    X, y = get_feature_target(df)

    pipeline = build_model_pipeline(X)

    param_dist = {
        "clf__learning_rate": np.linspace(0.01, 0.3, 10),
        "clf__max_iter": [100, 200, 300],
        "clf__max_depth": [None, 3, 5, 7],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    mlflow.set_experiment("churn_hyperparameter_search")

    with mlflow.start_run():
        search.fit(X, y)

        best_auc = search.best_score_
        best_params = search.best_params_

        mlflow.log_metric("best_cv_auc", best_auc)
        mlflow.log_params(best_params)

        # Save tuned model
        MODEL_DIR.mkdir(exist_ok=True)
        mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")

        print("\n=== BEST RESULTS ===")
        print("AUC:", best_auc)
        print("Params:", best_params)


if __name__ == "__main__":
    main()
