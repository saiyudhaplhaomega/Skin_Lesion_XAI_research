# Skin Lesion XAI Research

Research notebooks and training utilities for the Skin Lesion XAI project.

This repository is intentionally separate from:

- `../Skin_Lesion_Classification_backend` - shared ML code, model artifacts, and future FastAPI service
- `../Skin_Lesion_Classification_frontend` - Next.js web app
- `../Skin_Lesion_GRADCAM_Classification` - parent architecture docs and Terraform infrastructure

## What This Repo Contains

- HAM10000 setup and sanity checks
- RQ1-RQ6 explainability notebooks
- Grad-CAM, Grad-CAM++, EigenCAM, and LayerCAM comparisons
- Faithfulness, agreement, focus area, entropy, and external-validation analyses
- Paper figures and CSV metrics under `notebooks/outputs/`
- Research training scripts for backbone comparison and temporal checkpoints

## Expected Folder Layout

The notebooks expect this repository to sit beside the backend repository:

```text
Skin_Lesion_GRADCAM_Classification/
  Skin_Lesion_Classification_backend/
    ml/
      data/
      outputs/
      src/
  Skin_Lesion_Classification_frontend/
  Skin_Lesion_XAI_research/
    notebooks/
    train_backbones.py
    train_epoch_checkpoints.py
```

You can override the backend location with:

```powershell
$env:SKIN_LESION_BACKEND_DIR="C:\path\to\Skin_Lesion_Classification_backend"
```

## Quick Start

```bash
make setup
make register-kernel
make run-notebook
```

Start with `notebooks/00_setup_and_sanity.ipynb`, then run the RQ notebooks.

## Training Helpers

```bash
make train-backbones
make train-backbones-full
make train-checkpoints
```

Model checkpoints and HAM10000 processed data are stored in the backend repo under `ml/outputs/` and `ml/data/processed/`.
