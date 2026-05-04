# Notebook-by-Notebook Detailed Summary

This file explains what every notebook in `Skin_Lesion_XAI_research/notebooks/` does, why it matters, and how it contributes to the paper.



## 00_setup_and_sanity.ipynb

Purpose: verify that the environment, paths, data, splits, DataLoader, model, training loop, evaluation, and checkpoints are correct before any research claims are made.

What it does:
- Detects notebook, repository, backend, data, output, and model paths.
- Adds backend ML code to the Python path.
- Checks PyTorch, CUDA, torchvision, timm, numpy, pandas, matplotlib, scikit-learn, albumentations, OpenCV, and pytorch-grad-cam.
- Downloads or verifies HAM10000 processed metadata and image paths.
- Builds patient-level train/validation/test splits.
- Checks that patient overlap is zero across splits.
- Creates a PyTorch Dataset and DataLoader using 224 x 224 normalized image tensors.
- Handles class imbalance through weighted sampling.
- Builds a ResNet50 binary classifier.
- Trains the model and saves the best checkpoint by validation AUC.
- Evaluates the model on a held-out test set.
- Verifies that checkpoint files exist for later RQ notebooks.

Why it matters: if paths, labels, data splits, or checkpoints are wrong, all later XAI metrics become invalid. This notebook is the scientific foundation.

Important result seen in the notebook output: the ResNet50 test evaluation reported AUC around 0.9125 and accuracy around 0.8596 in the captured run.

Paper use: describe this as the reproducibility and quality-control stage. Mention patient-level leakage checking and class-imbalance handling.



## 01_training_ham10000_resnet50_baseline.ipynb

Purpose: optional fresh training of a HAM10000 binary ResNet50 baseline.

What it does:
- Loads HAM10000 processed metadata.
- Creates benign/malignant binary labels.
- Uses patient-aware or lesion-aware splitting where possible.
- Uses image transforms and a custom Dataset.
- Uses WeightedRandomSampler to address class imbalance.
- Trains ResNet50 with binary classification loss.
- Monitors ROC-AUC, accuracy, balanced accuracy, loss, and train loss.
- Saves metrics to `TRAINING_ham10000_resnet50_baseline.csv`.

Current cached addendum result: best epoch 3, ROC-AUC 0.9136, accuracy 0.8710, balanced accuracy 0.8203.

Paper use: include as training addendum or implementation detail, not as the main novelty.



## 02_training_siim_isic2020_binary.ipynb

Purpose: optional SIIM-ISIC 2020 binary melanoma training extension.

What it does:
- Locates or downloads SIIM-ISIC 2020 resized JPG data.
- Resolves metadata rows to local image paths.
- Uses patient_id grouped splitting to avoid leakage.
- Handles severe class imbalance.
- Trains a binary ResNet50 classifier.
- Saves metrics to `TRAINING_siim_isic2020_resnet50_binary.csv`.

Current cached addendum result: best epoch 1, ROC-AUC 0.8668, accuracy 0.9252, balanced accuracy 0.7223.

Paper use: useful as additional evidence that dataset identity and class balance matter. Do not directly compare with HAM10000 without explaining split protocol and label definitions.



## 03_xai_lime_shap_extension.ipynb

Purpose: optional research scaffold for LIME and SHAP comparison.

What it does:
- Checks whether optional `lime` and `shap` imports are available.
- Records GradCAM as the primary production explanation method.
- Records LIME and SHAP as optional research comparisons.
- Proposes comparing methods by focus area, entropy, qualitative agreement, and runtime.

Paper use: mention as future-work or extension, not as completed evidence unless the LIME/SHAP experiments are actually run and quantified.



## RQ1_cam_variant_comparison.ipynb

Research question: which CAM variant best localizes lesion regions?

What it does:
- Loads the trained ResNet50 checkpoint.
- Starts with one image for visual sanity checking.
- Generates GradCAM, GradCAM++, EigenCAM, and LayerCAM heatmaps.
- Computes FAP@0.5, FAP@0.3, entropy, and peak activation location.
- Scales the analysis to a set of images and writes `RQ1_cam_comparison.csv`.
- Runs statistical comparisons and saves figures.

Key cached result: GradCAM has the lowest FAP@0.5 and entropy among the compared methods.

Scientific meaning: GradCAM is the most spatially concentrated method in this experiment. This supports using GradCAM as the primary explanation method, but it does not prove clinical correctness.



## RQ2_faithfulness.ipynb

Research question: do faithfulness metrics correlate with clinical trust?

What it does:
- Explains insertion/deletion tests.
- Runs a single-image implementation check.
- Implements insertion: add pixels from most important to least important.
- Implements deletion: remove pixels from most important to least important.
- Computes insertion AUC, deletion AUC, and faithfulness.
- Compares GradCAM and GradCAM++ across a sample.

Key cached result: GradCAM has higher faithfulness than GradCAM++.

Scientific meaning: GradCAM better reflects the model's own decision process in this setup. It still does not prove that the model relied on medically correct features.



## RQ3_backbone_xai_quality.ipynb

Research question: does backbone architecture affect XAI quality?

What it does:
- Loads ResNet50, EfficientNet-B2, and MobileNetV2.
- Computes classification metrics for each backbone.
- Computes GradCAM and GradCAM++ maps for each backbone.
- Compares FAP and entropy across architecture-method combinations.
- Discusses backbone-by-CAM interaction.

Key cached result: ResNet50 plus GradCAM is the most focused combination, while EfficientNet-B2 GradCAM is much more diffuse despite high correct rate in the sample.

Scientific meaning: explanation quality is not just a function of accuracy. Architecture affects explanation behavior.



## RQ4_agreement_vs_uncertainty.ipynb

Research question: does inter-method disagreement predict misclassification?

What it does:
- Defines Jaccard similarity between thresholded CAM masks.
- Defines Spearman agreement between heatmap rankings.
- Computes agreement scores across CAM methods.
- Compares correct and incorrect predictions.
- Visualizes agreement, confidence, and correctness.

Key cached result: incorrect predictions had higher CAM agreement than correct predictions.

Scientific meaning: high agreement among CAM methods is not a correctness guarantee. This is an important negative result.



## RQ5_temporal_xai.ipynb

Research question: how does GradCAM attention evolve during training?

What it does:
- Loads epoch checkpoints.
- Uses the same fixed image across checkpoints.
- Computes GradCAM for each checkpoint.
- Visualizes attention maps over time.
- Tracks validation AUC, malignant probability, FAP, entropy, and activation location.

Key cached result: attention appears to become more focused later in training for the studied example, but the correlation is not statistically significant.

Scientific meaning: RQ5 is exploratory. It is useful for visualizing learning dynamics but should not be presented as a universal law.



## RQ6_external_validation.ipynb

Research question: does a HAM10000-trained model remain reliable on external datasets?

What it does:
- Scans HAM10000, ISIC2020, MILK10K, Malignant-Benign, and PH2.
- Standardizes dataset rows into a common schema.
- Loads the HAM10000-trained ResNet50 model.
- Evaluates each dataset separately.
- Reports AUC, accuracy, confidence, Brier score, AUC drop, benign accuracy, and malignant accuracy.
- Computes GradCAM metrics and zero-CAM rates.
- Computes CAM agreement across datasets.
- Saves canonical CSV outputs for paper tables.

Key cached result: external AUC drops substantially from HAM10000, malignant accuracy is poor on several external datasets, and confidence remains high.

Scientific meaning: this is the strongest safety evidence. Internal performance does not guarantee external reliability.



## RQ6_extended_datasets.ipynb

Purpose: extended multi-dataset stress testing.

What it does:
- Scans all dataset folders.
- Evaluates separate datasets.
- Evaluates combined Top-3, dermoscopy-only, and All-5 scenarios.
- Computes XAI metrics across datasets.
- Analyzes MILK10K modality differences.
- Computes inter-method disagreement by dataset.

Paper use: use as supplementary robustness evidence if needed. Keep canonical RQ6 results from `RQ6_external_validation.ipynb` as the main manuscript table.



## PAPER_RESULTS_TABLE.ipynb

Purpose: compile all RQ metric CSVs into paper-ready result tables.

What it does:
- Loads RQ1-RQ6 outputs.
- Prints grouped summary tables.
- Adds optional extended scenario tables if available.
- Creates a final discussion summary.

Paper use: use this as the source for manuscript tables and numerical claims.



## PAPER_RESULTS_DEFENSE_NOTES.ipynb

Purpose: prepare oral defense and paper interpretation.

What it does:
- States the core thesis.
- Summarizes every RQ.
- Lists likely professor questions.
- Separates defensible claims from overclaims.

Paper use: this is the strongest source for framing limitations and avoiding exaggerated claims.



## PAPER_RESULTS_TRAINING_ADDENDUM.ipynb

Purpose: summarize optional training metric files.

What it does:
- Finds training CSVs.
- Selects best epochs.
- Saves `PAPER_training_addendum.csv`.

Paper use: include as an addendum or appendix table if you discuss training details.



## TESTING_GUIDE.md

Purpose: runbook for reproducing the research notebooks.

What it does:
- Gives notebook order.
- Lists prerequisites.
- Explains expected outputs.
- Provides troubleshooting advice.

Paper use: cite internally as reproducibility documentation, not as a scientific result.


## Appendix: Extracted Notebook Headings

### 00_setup_and_sanity.ipynb

- `# 00 - Environment Setup & Sanity Checks`

- `## XAI Skin Lesion Classification - Research Pipeline`

- `## Recommended Cell Order`

- `## Prerequisites: Python Environment`

- `# Git Bash / VS Code terminal`

- `# PowerShell`

- `## How to Run This Notebook`

- `### Option 1: Run from VS Code (Recommended)`

- `### Option 2: Run from PyCharm`

- `### Option 3: Standalone Jupyter Server`

- `# Navigate to notebooks directory`

- `# Start Jupyter`

- `# Or: jupyter notebook`

- `## CELL 1: Path Setup`

- `## CELL 2: Install CUDA PyTorch (Fix GPU)`

- `## CELL 3: Environment Sanity Check`

- `## CELL 4: Data Sanity Check + Download`

- `## CELL 5: Data Split Leakage Check`

- `## CELL 6: DataLoader Sanity Check`

- `## CELL 7: Training Setup - Model, Loss, Optimizer`

- `## CELL 8: Training Loop`

- `## CELL 9: Evaluate on Test Set and Save Model`

- `## CELL 10: Verify Model Checkpoints`

### 01_training_ham10000_resnet50_baseline.ipynb

- `# 01 - HAM10000 ResNet50 Training Baseline`

### 02_training_siim_isic2020_binary.ipynb

- `# 02 - SIIM-ISIC 2020 Binary Training Extension`

### 03_xai_lime_shap_extension.ipynb

- `# 03 - LIME and SHAP XAI Extension`

- `## Implementation Plan`

### PAPER_RESULTS_DEFENSE_NOTES.ipynb

- `# Paper Results Defense Notes`

- `## Core Thesis`

- `## RQ1 - CAM Variant Localization`

- `## RQ2 - Faithfulness`

- `## RQ3 - Backbone Architecture and XAI Quality`

- `## RQ4 - Inter-Method Agreement vs Correctness`

- `## RQ5 - Temporal XAI During Training`

- `## RQ6 - External Validation and Distribution Shift`

- `## RQ6 - Explanation Robustness and Zero-CAM Failures`

- `## Professor Q&A`

- `## Claims vs Limitations`

### PAPER_RESULTS_TABLE.ipynb

- `# Paper Results Table - Compile All RQ Results`

- `## Table 1 - RQ1 CAM Method Comparison`

- `## Table 2 - RQ2 Faithfulness`

- `## Table 3 - RQ3 Backbone x CAM Method`

- `## Table 4 - RQ4 Inter-Method Agreement vs Correctness`

- `## Table 5 - RQ6 External Validation Performance`

- `## Table 6 - RQ6 Combined Scenario Performance (Optional Extended Notebook)`

- `## Table 7 - RQ6 XAI Robustness and Zero-CAM Failures`

- `## Table 8 - RQ6 CAM Agreement Across Datasets`

- `## Final Results Summary and Discussion`

- `## Useful Sources for the Discussion Section`

### PAPER_RESULTS_TRAINING_ADDENDUM.ipynb

- `# Paper Results Training Addendum`

- `## How To Use In Paper Results`

### RQ1_cam_variant_comparison.ipynb

- `# RQ1 - Which CAM Variant Best Localizes Lesion Regions?`

- `## XAI Skin Lesion Classification - Research Question 1`

- `## CONCEPT: What IS a CAM and Why Do Variants Differ?`

- `## CELL 1: Load Model and Single Image`

- `## CELL 2: Generate All CAM Variants`

- `## CELL 5: Scale to N Images — Statistical Comparison`

- `## CELL 6: Statistical Testing — This Is What Makes It Science`

- `## CELL 7: Paper-Quality Visualizations`

### RQ2_faithfulness.ipynb

- `# RQ2 — Do Faithfulness Metrics Correlate with Clinical Trust?`

- `## XAI Skin Lesion Classification — Research Question 2`

- `## CONCEPT: Insertion/Deletion Tests`

- `## CELL 1: Visualize the Concept`

- `## CELL 2: Run Insertion/Deletion on a Single Image`

- `## CELL 3: Insertion & Deletion Implementation`

- `## CELL 4: Plot Insertion/Deletion Curves`

- `## CELL 5: Scale to Dataset and Compare Methods`

- `## CELL 6: Statistical Comparison and Visualization`

### RQ3_backbone_xai_quality.ipynb

- `# RQ3 — Does Backbone Architecture Affect XAI Quality?`

- `## XAI Skin Lesion Classification — Research Question 3`

- `# Quick (2 epochs, ~5 min on GPU):`

- `# Full quality (15 epochs, ~30 min on GPU):`

- `# Or directly with the venv Python:`

- `## CELL 1: Load All Three Trained Models`

- `## CELL 2: Compute Classification Metrics for Each Backbone`

- `## CELL 3: Compute XAI Metrics for Each Backbone`

- `## CELL 4: The Key Analysis — Does Backbone Predict XAI Quality?`

- `## RQ3 Results — Interpretation`

- `### Key Findings`

- `### Important Caveat`

- `### What This Means for the Paper`

### RQ4_agreement_vs_uncertainty.ipynb

- `# RQ4 - Does Inter-method Disagreement Predict Misclassification?`

- `## XAI Skin Lesion Classification - Research Question 4`

- `## CONCEPT: Jaccard Similarity`

- `## CELL 1: Define Agreement Metrics`

- `## CELL 2: Compute Agreement Scores for Every Image`

- `## CELL 3: THE KEY ANALYSIS — Does Agreement Predict Correctness?`

- `## RQ4 Results - What Do These Numbers Mean?`

- `### Jaccard Similarity - What It Measures`

- `### The Surprising Result (Negative Finding)`

- `### Confidence vs Agreement (Pearson r = -0.18, Spearman r = -0.39)`

- `## CELL 4: Visualize Results`

- `## RQ4 Graphs - What You're Looking At`

- `### Plot 1 (Left) - Box Plot: Correct vs Incorrect Agreement`

- `### Plot 2 (Middle) - Scatter: Confidence vs Agreement`

- `### Plot 3 (Right) - Histogram: Distribution of Agreement Scores`

### RQ5_temporal_xai.ipynb

- `# RQ5 — How Does Grad-CAM Attention Evolve During Training?`

- `## XAI Skin Lesion Classification — Research Question 5`

- `## CELL 1: Check Available Checkpoints`

- `## CELL 2: Compute CAM for Same Image at Each Checkpoint`

- `## CELL 3: Visualize Attention Over Time — Your Paper Figure`

- `## CELL 4: Metrics Over Epochs — Quantify the Evolution`

### RQ6_extended_datasets.ipynb

- `# RQ6 Extended - Multi-Dataset Validation (Separate + Combined)`

- `## CELL 2: Scan All Dataset Folders`

- `## CELL 3: Load Trained Model`

- `## CELL 4: Evaluate on Each Dataset SEPARATELY`

- `## CELL 6b: Evaluate on COMBINED Top-3 Dataset (ISIC2020 + HAM10000 + MILK10K)`

- `## CELL 6c: Evaluate on COMBINED Dermoscopy-Only (HAM10000 + ISIC2020 + PH2)`

- `## CELL 6d: Evaluate on COMBINED All-5 Dataset (ISIC2020 + HAM10000 + MILK10K + Malignant-Benign + PH2)`

- `## CELL 8: XAI Visualization — Box Plots`

- `## CELL 9: MILK10K Modality Analysis`

- `## CELL 10: Inter-Method Disagreement Per Dataset`

- `## CELL 11: Final Summary Table`

### RQ6_external_validation.ipynb

- `# RQ6 - External Validation and Distribution Shift`

- `## CELL 1: Dataset Scanners`

- `## CELL 2: Load Model`

- `## CELL 3: Evaluate Each Dataset Separately`

- `## CELL 4: Performance and Calibration Summary`

- `## CELL 5: Extended Scenarios Stay Separate`

- `## CELL 6: GradCAM Metrics and Zero-CAM Quality Check`

- `## CELL 7: XAI Visualization and Statistical Comparison`

- `## CELL 8: Inter-Method CAM Agreement`

- `## CELL 9: Save Canonical RQ6 Outputs`
