r"""Fit temperature T on saved calibration logits.

Run from Skin_Lesion_XAI_research:
  .\skin-lesion-env\Scripts\python.exe scripts\fit_temperature.py --logits calibration_data\logits.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) - lower is better.
    Measures average gap between predicted confidence and actual accuracy.
    A perfectly calibrated model has ECE = 0.0.
    A ResNet50 without calibration typically has ECE = 0.08-0.15 on HAM10000.
    After temperature scaling, target: ECE < 0.03.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += mask.mean() * abs(avg_confidence - avg_accuracy)
    return float(ece)


def load_logits_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Generate it first with "
            "scripts\\collect_calibration_logits.py or scripts\\make_calibration_smoke_data.py."
        )

    data = np.load(path)
    missing = {"logits", "labels"} - set(data.files)
    if missing:
        raise KeyError(f"{path} is missing required array(s): {', '.join(sorted(missing))}")

    logits = np.asarray(data["logits"], dtype=np.float32).reshape(-1)
    labels = np.asarray(data["labels"], dtype=np.float32).reshape(-1)

    if logits.shape != labels.shape:
        raise ValueError(f"logits shape {logits.shape} does not match labels shape {labels.shape}")
    if logits.size < 2:
        raise ValueError("Need at least two examples to fit temperature.")
    if not np.isfinite(logits).all():
        raise ValueError("logits contains NaN or infinite values.")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("labels must be binary values: 0 or 1.")
    if np.unique(labels).size < 2:
        raise ValueError("Need both label classes, 0 and 1, to fit a useful temperature.")

    return logits, labels


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Returns the optimal temperature T."""
    logits_t = torch.tensor(logits, dtype=torch.float32).unsqueeze(1)
    labels_t = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # T is the only learnable parameter. Optimise log(T) so T stays positive.
    log_temperature = nn.Parameter(torch.log(torch.tensor([1.5], dtype=torch.float32)))
    optimizer = optim.LBFGS([log_temperature], lr=0.01, max_iter=100)
    criterion = nn.BCEWithLogitsLoss()

    def eval_step():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    return float(torch.exp(log_temperature).item())


def binary_cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32).unsqueeze(1)
    labels_t = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    loss = nn.BCEWithLogitsLoss()(logits_t, labels_t)
    return float(loss.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    logits, labels = load_logits_file(args.logits)

    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    ece_before = expected_calibration_error(raw_probs, labels)
    nll_before = binary_cross_entropy_from_logits(logits, labels)
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"NLL before calibration: {nll_before:.4f}")

    temperature = fit_temperature(logits, labels)
    print(f"Optimal temperature T = {temperature:.4f}")

    cal_probs = 1.0 / (1.0 + np.exp(-logits / temperature))
    ece_after = expected_calibration_error(cal_probs, labels)
    nll_after = binary_cross_entropy_from_logits(logits / temperature, labels)
    print(f"ECE after calibration:  {ece_after:.4f}")
    print(f"NLL after calibration:  {nll_after:.4f}")

    ece_worse = ece_after > ece_before
    if ece_worse:
        print(
            "WARNING: ECE increased after temperature scaling. "
            "Do not paste this temperature into the backend until you confirm the "
            "checkpoint, preprocessing transform, and validation split match production inference."
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "logits_file": str(args.logits),
            "n_examples": int(labels.size),
            "temperature": round(temperature, 6),
            "ece_before": round(ece_before, 6),
            "ece_after": round(ece_after, 6),
            "nll_before": round(nll_before, 6),
            "nll_after": round(nll_after, 6),
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report: {args.output_json}")

    if ece_worse:
        print("\nCandidate backend line after investigation:")
    else:
        print("\nPaste this into app/services/model_service.py:")
    print(
        f"_TEMPERATURE = {temperature:.4f}   "
        "# fitted on validation set, refit after every retraining run"
    )


if __name__ == "__main__":
    main()
