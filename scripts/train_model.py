"""
Versioned training wrapper.
Usage:
  python scripts/train_model.py \
    --dataset-manifest manifests/dataset-v001.csv \
    --model-name skin-lesion-resnet50 \
    --model-version model-v001 \
    --output-dir outputs/model-v001 \
    --epochs 20 \
    --mlflow-tracking-uri http://localhost:5000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.models as tv_models
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast

# -- reuse the dataloader and dataset classes from the research notebooks --
# from notebooks.utils import HAM10000Dataset, get_transforms
# For now, assume these are importable from the research repo.


class EarlyStopping:
    """
    Stop training when val_auc stops improving.
    Saves the best checkpoint automatically.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, checkpoint_path: str = "best_model.pth") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_score: float = -1.0
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_auc: float, model: torch.nn.Module) -> None:
        if val_auc > self.best_score + self.min_delta:
            self.best_score = val_auc
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def build_model(device: torch.device) -> nn.Module:
    net = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net.to(device)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        with autocast(enabled=device.type == "cuda"):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device) -> float:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().squeeze().tolist()
            all_probs.extend(probs if isinstance(probs, list) else [probs])
            all_labels.extend(labels.tolist())
    return float(roc_auc_score(all_labels, all_probs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5000")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Wire up your dataset here from the manifest CSV --
    # train_loader, val_loader, test_loader = load_from_manifest(args.dataset_manifest)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.model_name)

    with mlflow.start_run(run_name=args.model_version):
        mlflow.log_param("model_version", args.model_version)
        mlflow.log_param("dataset_manifest", args.dataset_manifest)
        mlflow.log_param("epochs", args.epochs)

        model = build_model(device)
        pos_weight = torch.tensor([5.25]).to(device)   # HAM10000 class imbalance ratio
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler(enabled=device.type == "cuda")
        stopper = EarlyStopping(patience=5, checkpoint_path=str(output_dir / "best.pth"))

        for epoch in range(args.epochs):
            # train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            # val_auc = evaluate(model, val_loader, device)
            # mlflow.log_metrics({"train_loss": train_loss, "val_auc": val_auc}, step=epoch)
            # stopper.step(val_auc, model)
            # scheduler.step()
            # if stopper.should_stop:
            #     break
            pass  # replace pass with the above when loaders are ready

        # Save final checkpoint and log to MLflow
        final_ckpt = str(output_dir / "model.pth")
        torch.save(model.state_dict(), final_ckpt)
        mlflow.pytorch.log_model(model, artifact_path="model")

        # test_auc = evaluate(model, test_loader, device)
        test_auc = 0.0   # replace when test_loader is wired
        mlflow.log_metric("test_auc", test_auc)

        card = {
            "model_name": args.model_name,
            "model_version": args.model_version,
            "dataset_manifest": args.dataset_manifest,
            "metrics": {"test_auc": test_auc},
            "approved_for_production": False,
        }
        card_path = output_dir / "model-card.json"
        card_path.write_text(json.dumps(card, indent=2))
        mlflow.log_artifact(str(card_path))

        print(f"Training complete. test_auc={test_auc:.4f}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
