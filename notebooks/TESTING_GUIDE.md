# Deep Learning Scientist Testing Guide
## XAI Skin Lesion Classification — Research Experiments

**Purpose**: Step-by-step guide for running the research notebooks to validate RQ1–RQ6.

**Prerequisites**: All notebooks assume the backend repo is at `../Skin_Lesion_Classification_backend/` relative to this directory.

---

## Quick Start

```bash
# 1. Navigate to notebooks
cd Skin_Lesion_XAI_research/notebooks

# 2. Start Jupyter
jupyter lab

# 3. Run notebooks in order below
```

---

## Notebook Order & Prerequisites

| Order | Notebook | What It Does | Time | Prerequisites |
|-------|----------|---------------|------|--------------|
| 0 | `00_setup_and_sanity.ipynb` | Verify environment, data, splits | 5 min | HAM10000 data downloaded |
| 1A | `01_training_ham10000_resnet50_baseline.ipynb` | Optional fresh HAM10000 binary ResNet50 training with patient-aware split, weighted sampling, OneCycleLR, ROC-AUC | 30 min-hours | HAM10000 metadata and images |
| 1B | `02_training_siim_isic2020_binary.ipynb` | Optional SIIM-ISIC 2020 binary melanoma training extension with patient-level split and imbalance handling | hours | SIIM-ISIC 2020 local data |
| 1C | `03_xai_lime_shap_extension.ipynb` | Optional LIME/SHAP research comparison plan beside Grad-CAM | 5 min setup; runtime varies | Optional `lime` and `shap` packages |
| 1 | `RQ1_cam_variant_comparison.ipynb` | Compare CAM methods by focus quality | 20–40 min | ResNet50 trained |
| 2 | `RQ3_backbone_xai_quality.ipynb` | Backbone architecture vs XAI quality | 30–50 min | All 3 models trained |
| 3 | `RQ4_agreement_vs_uncertainty.ipynb` | Does method agreement predict errors? | 20–30 min | ResNet50 trained |
| 4 | `RQ2_faithfulness.ipynb` | Insertion/deletion faithfulness | 30–60 min (GPU) | ResNet50 trained |
| 5 | `RQ5_temporal_xai.ipynb` | How attention evolves during training | 15–30 min | Training checkpoints saved |
| 6 | `RQ6_external_validation.ipynb` | External validation across diverse populations | 30–45 min | ISIC 2020 or external dataset |
| Final | `PAPER_RESULTS_TABLE.ipynb` | Compile all results into paper tables | 2 min | All above notebooks run |
| Addendum | `PAPER_RESULTS_TRAINING_ADDENDUM.ipynb` | Summarize optional `TRAINING_*.csv` outputs into a paper addendum table | 1 min | `01_` or `02_` training notebook outputs |

---

## Training Notebooks (`01_` and `02_`)

These notebooks are optional and separate from the existing RQ notebooks.

Use `01_training_ham10000_resnet50_baseline.ipynb` when you need a fresh HAM10000 binary baseline with:
- patient-aware splitting when `lesion_id` or `patient_id` exists
- `WeightedRandomSampler` for class imbalance
- ResNet50 with `BCEWithLogitsLoss`
- AdamW + OneCycleLR
- ROC-AUC, accuracy, balanced accuracy, and checkpoint export

Use `02_training_siim_isic2020_binary.ipynb` when you want to extend training evidence to SIIM-ISIC 2020:
- expects local data under `Skin_Lesion_XAI_research/data/external/siim_isic_2020/`
- requires `train-metadata.csv` with `isic_id`, `patient_id`, and `target`
- compatible Kaggle layout: `nischaydnk/isic-2020-jpg-224x224-resized`
- uses `GroupShuffleSplit` on `patient_id` to prevent patient leakage
- writes `outputs/metrics/TRAINING_siim_isic2020_resnet50_binary.csv`

Skip this notebook until the SIIM data is downloaded locally. Continue with `01_training_ham10000_resnet50_baseline.ipynb` and the RQ notebooks if you only need the HAM10000 research flow.

Do not compare HAM10000 and SIIM-ISIC metrics without clearly stating the dataset, label definition, and split protocol.

---

## Optional LIME/SHAP Extension (`03_xai_lime_shap_extension.ipynb`)

This notebook adds a research-only comparison path inspired by third-party XAI projects. Keep Grad-CAM as the primary production explanation method.

Use LIME/SHAP only to answer research questions such as:
- Do perturbation-based explanations agree with Grad-CAM?
- Are LIME/SHAP explanations stable enough for this image domain?
- How much slower are they per image?

Do not wire LIME/SHAP into the backend or frontend until runtime, stability, and safety-language constraints are validated.

---

## RQ0: Setup & Sanity Checks (`00_setup_and_sanity.ipynb`)

**Run this FIRST on any new machine.** It catches problems before they waste hours.

### What to look for

**CELL 1 — Environment:**
- All packages import without error
- GPU available (CUDA shown as True)

**CELL 2 — Data:**
- ~10,015 images total
- 7 lesion types (dx column)
- Binary labels: benign ~7,891 / malignant ~2,124 (3.7:1 ratio)
- No missing files

**CELL 3 — Leakage check:**
- All overlap counts = 0 (no patient appears in multiple splits)

**CELL 4 — DataLoader:**
- Tensor shape (8, 3, 224, 224)
- Value range ~[-2, 2] (normalized)
- Weighted sampler produces ~50/50 class balance in batches

**CELL 5 — Model checkpoints:**
- `resnet50_best.pth` exists
- If missing: train model first

### If something fails

| Problem | Fix |
|---------|-----|
| `metadata_with_paths.csv` not found | Run data pipeline in backend: `python ml/scripts/process_ham10000.py` |
| GPU not detected | Check CUDA: `nvidia-smi`; reinstall PyTorch with CUDA |
| Leakage detected | Fix `create_splits()` in `src/data/dataset.py` — use GroupShuffleSplit |
| No model checkpoints | Train model first |

---

## RQ1: CAM Variant Comparison (`RQ1_cam_variant_comparison.ipynb`)

**Research question**: Which CAM variant best localizes lesion regions?

### Hypothesis
Grad-CAM++ and Score-CAM produce more spatially focused attention maps than vanilla Grad-CAM.

### Metrics
- **Focus Area % (FAP)**: Lower = more focused. Target ~15–30%.
- **Entropy**: Lower = more concentrated attention.
- **Peak location**: Where does the model look most?

### Cell-by-cell guide

**CELL 1–2 (Load model + single image):**
- Pick a malignant image the model predicts correctly
- Wrong predictions → meaningless CAMs (skip them)

**CELL 3 (Visualize):**
- Always look at heatmaps before computing metrics
- What to expect:
  - **GradCAM**: broad, sometimes entire image
  - **GradCAM++**: more focused on specific regions
  - **EigenCAM**: may highlight different features
  - **LayerCAM**: finest-grained, sometimes patchy

**CELL 4 (Metrics):**
- Compare FAP@0.5, FAP@0.3, Entropy across methods
- Lower values = more focused

**CELL 5 (Scale to 150 images):**
- Uses only correctly classified images (CAMs on wrong predictions are meaningless)
- Takes 20–40 minutes on GPU
- Results saved to `outputs/metrics/RQ1_cam_comparison.csv`

**CELL 6 (Statistics):**
- **Kruskal-Wallis**: Tests if ANY method differs significantly from others
  - p < 0.05 → significant difference exists
  - p > 0.05 → report as "no significant difference" (still valid science)
- **Pairwise Wilcoxon**: Which specific pairs differ
- **Cohen's d**: How large is the difference (small < 0.5, medium < 0.8, large ≥ 0.8)

**CELL 7 (Visualization):**
- Paper-quality plots saved to `outputs/figures/`

### Expected results

| Method | FAP@0.5 | Entropy |
|--------|---------|---------|
| GradCAM | ~30–50% | higher |
| GradCAM++ | ~20–35% | lower |
| EigenCAM | varies | varies |
| LayerCAM | ~15–30% | varies |

### How to interpret

- **Significant difference found**: You can claim one method produces more focused explanations
- **No significant difference**: Report as negative result — all methods perform similarly for localization
- **More focused ≠ more clinically useful**: A diffuse CAM might reflect the model using distributed features

---

## RQ2: Faithfulness Metrics (`RQ2_faithfulness.ipynb`)

**Research question**: Do faithfulness metrics correlate with clinical trust?

### Hypothesis
A faithful explanation highlights regions that the model *actually uses* — not just visually plausible regions.

### Metrics
- **Insertion AUC**: High (>0.7) = confidence rises fast when important pixels revealed
- **Deletion AUC**: Low (<0.3) = confidence drops fast when important pixels removed
- **Faithfulness**: Insertion - Deletion. High (>0.4) = good separation

### Cell-by-cell guide

**CELL 1 (Concept):** Understand what insertion/deletion curves look like for good vs. poor CAMs.

**CELL 2–3 (Single image):** Verify implementation before scaling.

**CELL 4 (Curves):**
- Green curve (insertion) should rise quickly
- Red curve (deletion) should drop quickly
- Flat curves = poor faithfulness (model uses different features than CAM highlights)

**CELL 5 (Scale to 50 images):**
- SLOW: 50 images × 2 methods × 20 steps = 2000 forward passes
- Use GPU or run overnight
- 50 images is sufficient (more gives diminishing returns)

**CELL 6 (Statistics):**
- Compare GradCAM vs GradCAM++ on all three metrics
- Target: one method should score significantly better on faithfulness

### How to interpret

| Insertion AUC | Deletion AUC | Faithfulness | Interpretation |
|-------------|-------------|--------------|---------------|
| > 0.7 | < 0.3 | > 0.4 | Good CAM |
| ~0.5 | ~0.5 | ~0.0 | Poor CAM — highlights wrong regions |
| < 0.5 | > 0.5 | negative | CAM is anti-faithful |

---

## RQ3: Backbone Effect on XAI (`RQ3_backbone_xai_quality.ipynb`)

**Research question**: Does backbone architecture affect XAI quality independent of accuracy?

### Hypothesis
Models with similar AUC can produce significantly different XAI quality because different architectures learn different internal representations.

### Prerequisites
All 3 models must be trained:
- ResNet50
- EfficientNet-B2
- MobileNetV2

### Cell-by-cell guide

**CELL 1 (Load models):** Confirm all 3 models load successfully.

**CELL 2 (Classification metrics):**
- Check AUCs are similar (within 0.05 of each other)
- If one model is much better, you can't separate accuracy from XAI quality
- If AUCs are similar → fair comparison possible

**CELL 3 (XAI metrics):**
- Compute FAP and entropy for each backbone using GradCAM and GradCAM++
- Uses 100 test images (same images for all backbones for fairness)

**CELL 4 (Analysis):**
- Kruskal-Wallis per CAM method: do backbones differ significantly?
- Heatmap shows backbone × method interaction
- **KEY INSIGHT**: If backbones have similar AUC but different FAP → architecture affects how the model explains itself, not just its accuracy

### How to interpret

**Supports hypothesis**: Two models with similar AUC but different FAP/entropy.
This means XAI quality is not just a byproduct of accuracy — architecture matters independently.

**Contradicts hypothesis**: All backbones produce similar XAI quality regardless of architecture.

---

## RQ4: Inter-method Disagreement (`RQ4_agreement_vs_uncertainty.ipynb`)

**Research question**: Does inter-method disagreement predict misclassification?

### Hypothesis
When CAM methods disagree on which regions are important, the model is more likely to be uncertain or wrong.

### Metric
- **Jaccard Similarity**: Overlap between thresholded CAM maps
  - 1.0 = perfect agreement (same pixels highlighted)
  - 0.0 = complete disagreement

### Cell-by-cell guide

**CELL 1 (Metrics):** Understand Jaccard vs Spearman correlation.

**CELL 2 (Scale to 150 images):**
- Compute pairwise Jaccard and Spearman for all 3 method pairs per image
- Average gives one agreement score per image

**CELL 3 (Analysis):**
- Mann-Whitney U: Do correct predictions have higher agreement than incorrect?
  - p < 0.05 + correct > incorrect → SUPPORTS hypothesis
  - p > 0.05 → report as negative result
- Correlation: Does higher model confidence correlate with higher agreement?

**CELL 4 (Visualization):**
- Box plot: correct vs incorrect agreement
- Scatter: confidence vs agreement colored by correctness

### How to interpret

**Supports hypothesis**: Correct predictions show higher inter-method agreement.
This suggests disagreement could be used as an uncertainty signal.

**No significant result**: Agreement doesn't reliably predict correctness — still worth reporting as a finding.

---

## RQ5: Temporal XAI (`RQ5_temporal_xai.ipynb`)

**Research question**: How does Grad-CAM attention evolve during training?

### Hypothesis
Early training focuses on texture/color shortcuts; late training focuses on clinically meaningful morphological features.

### Prerequisites
Training checkpoints saved at multiple epochs (e.g., every 5 epochs).

### Cell-by-cell guide

**CELL 1 (Checkpoints):**
- Must find `resnet50_epoch5.pth`, `epoch10.pth`, etc.
- If missing: retrain with checkpoint saving enabled

**CELL 2 (Temporal CAMs):**
- Same image at every checkpoint
- Watch how attention changes as training progresses

**CELL 3 (Visualization):**
- Row of heatmaps + overlays + histograms per epoch
- How does the highlighted region change?
- Does it become more or less focused?

**CELL 4 (Metrics over time):**
- FAP and entropy plotted against epoch
- Val AUC as reference
- Correlation: does higher AUC = more focused attention?

### How to interpret

**Supports hypothesis**: Attention becomes more focused (lower FAP, lower entropy) as training progresses AND as AUC improves.
Early epochs = diffuse, late epochs = focused on lesion region.

**Contradicts hypothesis**: Attention remains diffuse or becomes more diffuse as training progresses.

---

## RQ6: External Validation — Distribution Shift (`RQ6_external_validation.ipynb`)

**Research question**: How do AI models trained on standard archives perform when subject to external validation on geographically and demographically diverse populations?

### Why This Matters
HAM10000 and ISIC were collected from specific clinical settings with particular patient demographics. Deploying in different regions exposes the model to different skin tone distributions, lesion presentations, imaging protocols, and lighting conditions. This is one of the most critical failure modes in clinical AI.

### Key Concept: Types of Distribution Shift
- **Covariate Shift**: Input distribution changes (different demographics, imaging) but label mapping stays the same
- **Prior Probability Shift**: Class prevalence changes (more/fewer malignant cases in one population)
- **Concept Drift**: The relationship between features and labels itself changes

### Datasets for External Validation

| Dataset | Images | Geography | Demographics |
|---------|--------|-----------|-------------|
| HAM10000 | 10,015 | Australia/USA | Predominantly fair skin |
| ISIC 2020 | 33,126 | Global | More diverse |
| PAD-UFES-20 | 22,000 | Brazil | Mixed skin tones |
| derm101 | 9,000+ | Mixed | Mixed |

### Cell-by-cell guide

**CELL 1 (Dataset overview):** Understand what external data is available and how it differs from training data.

**CELL 2 (Compare distributions):**
- Lesion type prevalence
- Binary label distribution
- Image characteristics

**CELL 3–4 (Internal baseline):** Establish baseline performance on held-out HAM10000 test set.

**CELL 5 (External evaluation):**
- Download ISIC 2020 and place metadata at `../Skin_Lesion_Classification_backend/ml/data/processed/isic2020_metadata.csv`
- Evaluate same model on external dataset
- **Key metrics**: AUC drop, accuracy drop, confidence drop

**CELL 6 (Quantify performance drop):**
- AUC drop > 0.05 = significant degradation
- Confidence drop = model recognizes uncertainty (good) or is overconfident (bad)

**CELL 7–8 (XAI analysis):**
- Compare CAM metrics (FAP, entropy) on internal vs external
- Do CAMs become more diffuse on OOD data?
- Statistical tests: Mann-Whitney U for each metric

**CELL 9–10 (RQ4 integration):**
- Does inter-method disagreement increase on external data?
- If YES: disagreement can serve as real-time OOD detection signal in production
- High disagreement → flag for human review

### How to interpret

| Finding | Interpretation |
|---------|----------------|
| Large AUC drop + high confidence | Model is overconfident on OOD — dangerous |
| Large AUC drop + low confidence | Model recognizes uncertainty — can flag for human review |
| Higher disagreement on external | RQ4 disagreement signal works for OOD detection |
| Broader/diffuse CAMs on external | Model cannot find coherent lesion signal |

### Prerequisites

```bash
# Download ISIC 2020
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
# Or from: https://challenge.isic-archive.com/data/2020
```

Place ISIC 2020 metadata at `../Skin_Lesion_Classification_backend/ml/data/processed/isic2020_metadata.csv`

---

## Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Insertion/deletion AUC ≈ 0.5 | CAMs are random/meaningless | Check model is trained; verify prediction is correct |
| FAP = 100% for all methods | Model hasn't learned to discriminate | Retrain model |
| Kruskal-Wallis p > 0.05 | Methods not significantly different | Report as negative result |
| ScoreCAM takes 10 min per image | Normal — N_channel forward passes | Skip ScoreCAM in loops |
| Jaccard = 0 for all pairs | Try lower threshold (0.3 instead of 0.5) | Adjust threshold |
| Temporal CAMs all look same | Checkpoints not saved properly | Verify checkpoint files; retrain with checkpoint saving |
| `metadata_with_paths.csv` not found | Data pipeline not run | Run `python ml/scripts/process_ham10000.py` in backend |
| No GPU detected | CUDA not available | Use Google Colab with GPU runtime |

---

## Output Files

All outputs are saved relative to `notebooks/`.

The optional training and addendum notebooks write:

- `outputs/metrics/TRAINING_ham10000_resnet50_baseline.csv`
- `outputs/metrics/TRAINING_siim_isic2020_resnet50_binary.csv`
- `outputs/metrics/XAI_lime_shap_extension_plan.csv`
- `outputs/metrics/PAPER_training_addendum.csv`

The RQ notebooks write:

```
notebooks/
├── outputs/
│   ├── metrics/
│   │   ├── RQ1_cam_comparison.csv
│   │   ├── RQ2_faithfulness.csv
│   │   ├── RQ3_backbone_comparison.csv
│   │   ├── RQ4_agreement_results.csv
│   │   └── RQ6_xai_external.csv
│   └── figures/
│       ├── sanity_training_batch.png
│       ├── RQ1_single_image_comparison.png
│       ├── RQ1_statistical_comparison.png
│       ├── RQ2_faithfulness_concept.png
│       ├── RQ2_single_ins_del.png
│       ├── RQ2_faithfulness_comparison.png
│       ├── RQ3_backbone_xai.png
│       ├── RQ4_agreement_analysis.png
│       ├── RQ5_temporal_cam.png
│       ├── RQ5_temporal_metrics.png
│       ├── RQ6_dataset_comparison.png
│       ├── RQ6_xai_comparison.png
│       └── RQ6_disagreement_ood.png
```

---

## Running on Google Colab (No GPU Locally)

1. Upload the notebooks to Colab
2. Clone the backend repo:
   ```python
   !git clone https://github.com/YOUR_USERNAME/skin-lesion-backend.git
   ```
3. Upload HAM10000 data to Colab
4. Update paths in notebooks to point to Colab data location
5. Enable GPU: Runtime → Change runtime type → GPU
6. Run notebooks — GPU accelerates CAM generation ~10x

---

## After All Notebooks Complete

Run `PAPER_RESULTS_TABLE.ipynb` to compile all results into paper-ready tables with LaTeX export.

If you ran the optional training notebooks, run `PAPER_RESULTS_TRAINING_ADDENDUM.ipynb` after `PAPER_RESULTS_TABLE.ipynb`. Keep it as a separate paper addendum so the main RQ1-RQ6 results remain unchanged.
