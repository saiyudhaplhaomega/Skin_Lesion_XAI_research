"""Create deterministic fake logits to test the calibration script wiring.

This file is only for local smoke testing. Do not use its output as the backend
temperature value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("calibration_data/smoke_logits.npz"))
    args = parser.parse_args()

    logits = np.array(
        [-4.0, -3.0, -2.0, -1.5, -1.0, -0.3, 0.2, 0.8, 1.2, 1.8, 2.5, 3.5],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, logits=logits, labels=labels)
    print(f"Saved smoke logits: {args.output}")
    print(f"Examples: {labels.size}")


if __name__ == "__main__":
    main()
