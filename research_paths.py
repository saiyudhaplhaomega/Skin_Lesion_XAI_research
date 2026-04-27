"""Path helpers for the standalone research repository."""

from __future__ import annotations

import os
from pathlib import Path


RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent


def backend_dir() -> Path:
    """Return the backend repo path, allowing an environment override."""
    override = os.environ.get("SKIN_LESION_BACKEND_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "Skin_Lesion_Classification_backend"


BACKEND_DIR = backend_dir()
ML_DIR = BACKEND_DIR / "ml"
METADATA_PATH = ML_DIR / "data" / "processed" / "metadata_with_paths.csv"
MODEL_DIR = ML_DIR / "outputs" / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
