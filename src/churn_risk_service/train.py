from pathlib import Path
import yaml
import mlflow

from .model import train_baseline_model, MODEL_PATH


def load_mlflow_config() -> dict:
    with open("configs/mlflow.yaml", "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    print("Starting churn model training...")

    cfg = load_mlflow_config()

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run():
        metrics = train_baseline_model()

        mlflow.log_metrics({
            "val_accuracy": metrics["val"]["accuracy"],
            "val_auc": metrics["val"]["roc_auc"],
            "test_accuracy": metrics["test"]["accuracy"],
            "test_auc": metrics["test"]["roc_auc"],
        })

        mlflow.log_artifact(str(MODEL_PATH))

        print("\n=== TRAINING SUMMARY ===")
        print(f"Validation metrics: {metrics['val']}")
        print(f"Test metrics      : {metrics['test']}")
        print(f"Model artifact    : {Path(MODEL_PATH).resolve()}")


if __name__ == "__main__":
    main()
