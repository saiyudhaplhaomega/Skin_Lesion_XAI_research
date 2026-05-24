r"""Collect validation logits for temperature calibration.

Run from Skin_Lesion_XAI_research:
  .\skin-lesion-env\Scripts\python.exe scripts\collect_calibration_logits.py `
    --checkpoint ..\Skin_Lesion_Classification_backend\ml\outputs\models\ham10000_resnet50_binary_best.pth `
    --output calibration_data\logits.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RESEARCH_DIR = Path(__file__).resolve().parents[1]
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from research_paths import METADATA_PATH, MODEL_DIR


DEFAULT_CHECKPOINT = MODEL_DIR / "ham10000_resnet50_binary_best.pth"


def get_val_transform():
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.7630, 0.5456, 0.5700], std=[0.1409, 0.1526, 0.1700]),
    ])


class CalibrationDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = get_val_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return self.transform(image), label


def build_validation_split(metadata_path: Path, limit: int | None = None) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    required = {"filepath", "patient_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Metadata is missing required column(s): {', '.join(sorted(missing))}")

    df = df.dropna(subset=["filepath", "patient_id", "label"]).copy()
    df = df[df["filepath"].map(lambda value: Path(str(value)).exists())].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No existing image files found from metadata: {metadata_path}")

    rng = np.random.default_rng(42)
    patients = df["patient_id"].drop_duplicates().to_numpy()
    rng.shuffle(patients)
    n_patients = len(patients)
    val_patients = set(patients[int(n_patients * 0.65) : int(n_patients * 0.80)])
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
    if val_df.empty:
        raise ValueError("Validation split is empty; check metadata patient_id values.")

    if limit is not None:
        positives = val_df[val_df["label"] == 1]
        negatives = val_df[val_df["label"] == 0]
        if not positives.empty and not negatives.empty and limit >= 2:
            n_pos = max(1, min(len(positives), limit // 2))
            n_neg = max(1, min(len(negatives), limit - n_pos))
            val_df = (
                pd.concat(
                    [
                        positives.sample(n=n_pos, random_state=42),
                        negatives.sample(n=n_neg, random_state=42),
                    ]
                )
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )
        else:
            val_df = val_df.head(limit).reset_index(drop=True)

    return val_df


def load_model(checkpoint_path: Path, device: torch.device, trust_local_checkpoint: bool = False) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = tv_models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 1)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        if not trust_local_checkpoint:
            raise RuntimeError(
                "PyTorch could not load this checkpoint in weights_only mode. "
                "If this checkpoint was created locally by this project, rerun with "
                "--trust-local-checkpoint. Do not use that flag for downloaded or unknown checkpoints."
            ) from exc
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state)
    net.eval()
    return net.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("calibration_data/logits.npz"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--trust-local-checkpoint",
        action="store_true",
        help="Allow loading an older local PyTorch checkpoint if weights_only=True is rejected.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, trust_local_checkpoint=args.trust_local_checkpoint)
    val_df = build_validation_split(args.metadata, args.limit)
    loader = DataLoader(CalibrationDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting validation logits"):
            logits = model(images.to(device)).squeeze(1).detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())

    logits_array = np.concatenate(all_logits).astype(np.float32)
    labels_array = np.concatenate(all_labels).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, logits=logits_array, labels=labels_array)
    print(f"Saved logits: {args.output}")
    print(f"Examples: {len(labels_array)}")
    print(f"Positive rate: {labels_array.mean():.4f}")


if __name__ == "__main__":
    main()
