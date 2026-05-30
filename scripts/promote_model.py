# Run this after training completes and test_auc >= MIN_TEST_AUC (0.85)
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "skin-lesion-resnet50"
MIN_TEST_AUC = 0.85

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# Find the best run for this model name
runs = mlflow.search_runs(
    experiment_names=[MODEL_NAME],
    order_by=["metrics.test_auc DESC"],
    max_results=1,
)

if runs.empty:
    raise RuntimeError("No runs found. Run train_model.py first.")

best_run = runs.iloc[0]
test_auc = best_run["metrics.test_auc"]
run_id = best_run["run_id"]

print(f"Best run: {run_id}, test_auc: {test_auc:.4f}")

if test_auc < MIN_TEST_AUC:
    raise ValueError(f"test_auc {test_auc:.4f} < minimum {MIN_TEST_AUC}. Do not promote.")

# Register
result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
model_version = result.version
print(f"Registered as version {model_version}")

# Assign the production alias (requires human sign-off in a real pipeline)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=model_version,
)
print(f"Assigned {MODEL_NAME} v{model_version} to @champion")
