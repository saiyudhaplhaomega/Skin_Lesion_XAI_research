"""
Train ResNet50 and save a checkpoint at every epoch.
Required by RQ5 to analyse how Grad-CAM attention evolves during training.

Checkpoints are saved as:
  ml/outputs/models/checkpoints/resnet50_epoch{N}.pth

Each checkpoint has the same format as resnet50_best.pth:
  { model_state_dict, val_auc, epoch, pos_weight }

Usage:
    skin-lesion-env/Scripts/python.exe train_epoch_checkpoints.py
    skin-lesion-env/Scripts/python.exe train_epoch_checkpoints.py --epochs 10
    make train-checkpoints
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from research_paths import CHECKPOINT_DIR, METADATA_PATH

EPOCHS     = 10
BATCH_SIZE = 32
IMG_SIZE   = 224
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 4


def dataloader_kwargs(num_workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


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
    df["_split"] = df["patient_id"].apply(
        lambda p: "train" if p in train_pts else ("val" if p in val_pts else "test")
    )
    return df[df["_split"] == "train"], df[df["_split"] == "val"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing checkpoints")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--limit-samples", type=int, help="Use a small balanced split for smoke testing.")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR, type=Path)
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # skip if all checkpoints already exist
    existing = sorted(args.checkpoint_dir.glob("resnet50_epoch*.pth"))
    if existing and not args.force:
        print(f"Found {len(existing)} existing checkpoints in {args.checkpoint_dir}")
        print("Use --force to retrain from scratch.")
        return

    print(f"Device: {DEVICE}")
    print(f"AMP enabled: {not args.no_amp and DEVICE.type == 'cuda'}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Saving {args.epochs} epoch checkpoints to {args.checkpoint_dir}")

    train_df, val_df = build_splits()
    if args.limit_samples:
        per_split = max(2, args.limit_samples)
        train_df = train_df.groupby("label", group_keys=False).head(per_split)
        val_df = val_df.groupby("label", group_keys=False).head(per_split)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    n_pos = train_df["label"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight: {pos_weight.item():.2f}")

    train_loader = DataLoader(HAM10000Dataset(train_df, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, **dataloader_kwargs(args.num_workers))
    val_loader   = DataLoader(HAM10000Dataset(val_df, augment=False),
                              batch_size=BATCH_SIZE * 2, shuffle=False, **dataloader_kwargs(args.num_workers))

    amp_enabled = not args.no_amp
    model = timm.create_model("resnet50", pretrained=not args.no_pretrained, num_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler("cuda", enabled=amp_enabled and DEVICE.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=amp_enabled and DEVICE.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validate
        model.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                with autocast(device_type=DEVICE.type, enabled=amp_enabled and DEVICE.type == "cuda"):
                    logits_all.append(model(images.to(DEVICE, non_blocking=True)).cpu())
                labels_all.append(labels)
        val_auc = roc_auc_score(
            torch.cat(labels_all).numpy(),
            torch.sigmoid(torch.cat(logits_all)).numpy()
        )
        scheduler.step()

        ckpt_path = args.checkpoint_dir / f"resnet50_epoch{epoch:02d}.pth"
        torch.save({
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "val_auc":          val_auc,
            "epoch":            epoch,
            "pos_weight":       pos_weight.item(),
            "amp_enabled":      amp_enabled and DEVICE.type == "cuda",
            "num_workers":      args.num_workers,
        }, ckpt_path)
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  val_auc={val_auc:.4f}  -> saved {ckpt_path.name}")

    print(f"\nDone. {args.epochs} checkpoints saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
