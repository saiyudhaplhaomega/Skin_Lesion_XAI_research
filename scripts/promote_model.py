from __future__ import annotations

import argparse
import sys

import mlflow
from mlflow.tracking import MlflowClient


def promote_best_run(model_name: str, min_test_auc: float, tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    runs = mlflow.search_runs(
        experiment_names=[model_name],
        order_by=["metrics.test_auc DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError("No runs found. Run train_model.py first.")

    best_run = runs.iloc[0]
    test_auc = float(best_run.get("metrics.test_auc", 0.0))
    run_id = best_run["run_id"]

    print(f"Best run: {run_id}, test_auc: {test_auc:.4f}")

    if test_auc < min_test_auc:
        print(f"BLOCKED: test_auc {test_auc:.4f} < minimum {min_test_auc}. Do not promote.")
        sys.exit(1)

    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    model_version = result.version
    print(f"Registered as version {model_version}")

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version,
    )
    print(f"Assigned {model_name} v{model_version} to @champion")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="skin-lesion-resnet50")
    parser.add_argument("--min-test-auc", type=float, default=0.85)
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()
    promote_best_run(args.model_name, args.min_test_auc, args.mlflow_uri)


if __name__ == "__main__":
    main()
