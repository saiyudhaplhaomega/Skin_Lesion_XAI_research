# Skin Lesion XAI Research

Research notebooks and training utilities for the Skin Lesion XAI project.

GitHub: https://github.com/saiyudhaplhaomega/Skin_Lesion_XAI_research

This repo is the source of truth for notebooks, RQ1-RQ6 experiments, metrics, figures, and research training scripts. It is intentionally separate from the frontend and backend repos.

## Repo Boundaries

| Repo | Responsibility |
| --- | --- |
| `Skin_Lesion_XAI_research` | HAM10000 setup, notebooks, RQ1-RQ6 experiments, figures, metrics, and training helpers |
| `Skin_Lesion_Classification_backend` | FastAPI inference API, Grad-CAM serving runtime, and backend model artifact loading |
| `Skin_Lesion_Classification_frontend` | Next.js web app |
| `Skin_Lesion_GRADCAM_Classification` | Workspace-level docs, Terraform, and production roadmap |

Do not add frontend application code or backend API code here.

## What This Repo Contains

- `notebooks/00_setup_and_sanity.ipynb` for dataset setup and baseline checks
- RQ1-RQ6 notebooks for explainability, faithfulness, backbone comparison, uncertainty, temporal CAMs, and external validation
- `notebooks/outputs/figures/` for generated figures
- `notebooks/outputs/metrics/` for generated CSV metrics
- `train_backbones.py` for backbone comparison training
- `train_epoch_checkpoints.py` for temporal checkpoint training
- `research_paths.py` for stable paths to the sibling backend repo

## Expected Workspace Layout

The easiest local setup keeps the repos as siblings:

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
    research_paths.py
    train_backbones.py
    train_epoch_checkpoints.py
```

The research scripts read and write shared ML data/artifacts through the backend repo's `ml/` directory by default. Override that location when needed:

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

## Outputs And Git Hygiene

Review generated outputs before pushing. Do not commit private datasets, patient-linked images, `.env` files, virtual environments, caches, or large checkpoints unless you intentionally decide that a file belongs in the research repo.

The `.gitignore` already excludes `skin-lesion-env/`, Python caches, notebook checkpoints, environment files, and `mlruns/`.
