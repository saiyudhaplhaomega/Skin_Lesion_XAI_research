"""
Train EfficientNet-B2 and MobileNetV2 on HAM10000.
Uses the same data split, loss, and checkpoint format as resnet50_best.pth.

Usage:
    python train_backbones.py
    python train_backbones.py --models efficientnet_b2  # train one only
    python train_backbones.py --epochs 15               # override epoch count

Runtime: ~20-30 min per model on GPU (RTX 4070).
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from research_paths import METADATA_PATH, MODEL_DIR

# ─── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_MODELS = ["efficientnet_b2", "mobilenetv2_100"]
EPOCHS     = 2
BATCH_SIZE = 32
IMG_SIZE   = 224
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HAM10000Dataset(Dataset):
    def __init__(self, split_df, augment=False):
        self.df = split_df.dropna(subset=["filepath"]).reset_index(drop=True)
        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(30),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        return self.transform(img), torch.tensor(row["label"], dtype=torch.float32)


def build_splits():
    df = pd.read_csv(METADATA_PATH)
    np.random.seed(42)
    patients = df["patient_id"].unique()
    np.random.shuffle(patients)
    n = len(patients)
    train_pts = set(patients[: int(n * 0.65)])
    val_pts   = set(patients[int(n * 0.65) : int(n * 0.80)])
    test_pts  = set(patients[int(n * 0.80) :])
    df["_split"] = df["patient_id"].apply(
        lambda p: "train" if p in train_pts else ("val" if p in val_pts else "test")
    )
    return df[df["_split"] == "train"], df[df["_split"] == "val"], df[df["_split"] == "test"]


def train_model(model_name, train_df, val_df, test_df, epochs):
    out_path = MODEL_DIR / f"{model_name}_best.pth"
    if out_path.exists():
        print(f"\n[{model_name}] Already trained - skipping ({out_path})")
        return

    print(f"\n{'='*60}")
    print(f"Training {model_name} for {epochs} epochs on {DEVICE}")
    print(f"{'='*60}")

    # class weight for imbalanced data (same as resnet50 training)
    n_pos = train_df["label"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight: {pos_weight.item():.2f}  (neg={n_neg}, pos={n_pos})")

    train_loader = DataLoader(HAM10000Dataset(train_df, augment=True),  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(HAM10000Dataset(val_df,   augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(HAM10000Dataset(test_df,  augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = timm.create_model(model_name, pretrained=True, num_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_state   = None

    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validate
        model.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                logits_all.append(model(images.to(DEVICE)).cpu())
                labels_all.append(labels)
        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)
        val_auc = roc_auc_score(labels_all.numpy(), torch.sigmoid(logits_all).numpy())
        scheduler.step()

        marker = " *" if val_auc > best_val_auc else ""
        print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.4f}  val_auc={val_auc:.4f}{marker}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = copy.deepcopy(model.state_dict())

    # test eval
    model.load_state_dict(best_state)
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test eval", leave=False):
            preds_all.extend(torch.sigmoid(model(images.to(DEVICE))).cpu().numpy().flatten())
            labels_all.extend(labels.numpy())
    preds_all  = np.array(preds_all)
    labels_all = np.array(labels_all).astype(int)
    test_auc = roc_auc_score(labels_all, preds_all)
    test_acc = accuracy_score(labels_all, (preds_all > 0.5).astype(int))
    print(f"\nTest AUC={test_auc:.4f}  Acc={test_acc:.4f}")

    torch.save({
        "model_state_dict": best_state,
        "val_auc":          best_val_auc,
        "test_auc":         test_auc,
        "test_acc":         test_acc,
        "epochs":           epochs,
        "pos_weight":       pos_weight.item(),
    }, out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["efficientnet_b2", "mobilenetv2_100", "resnet50"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Models to train: {args.models}")
    print(f"Epochs: {args.epochs}")

    train_df, val_df, test_df = build_splits()
    print(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    for model_name in args.models:
        train_model(model_name, train_df, val_df, test_df, args.epochs)

    print("\nAll done. Models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
