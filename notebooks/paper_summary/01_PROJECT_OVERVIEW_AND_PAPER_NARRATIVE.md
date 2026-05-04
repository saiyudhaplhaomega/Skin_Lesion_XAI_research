# Project Overview and Paper Narrative

## One-Sentence Thesis

This notebook directory shows that CAM-based explanations for skin-lesion classifiers must be evaluated as robustness artifacts, not merely displayed as attractive heatmaps, because method choice, model architecture, training stage, inter-method agreement, and external dataset shift all change how trustworthy the explanation appears.

## Main Research Problem

Medical image classifiers can produce impressive internal validation scores, but a clinical user needs more than a probability and a heatmap. The model may be accurate on a familiar dataset while failing on images from different cameras, clinics, populations, or labeling protocols. The heatmap may look medically plausible while not actually reflecting the model's decision process. The notebooks therefore ask a deeper question: under what conditions are explanations focused, faithful, stable, and robust?

## What the Pipeline Studies

The pipeline has three layers:

1. Model training and sanity checking: environment, data paths, patient-level splits, class imbalance, ResNet50 training, checkpoint export.
2. Explanation analysis: CAM variants, focus metrics, entropy, faithfulness, backbone effects, method agreement, temporal training behavior.
3. External robustness: performance, confidence, calibration, class-specific accuracy, zero-CAM failures, and CAM agreement across datasets.

## Core RQ Story

RQ1 asks which CAM method is most spatially focused. The cached results support GradCAM as the most focused method.

RQ2 asks whether the explanation is faithful to the model. The cached insertion/deletion results support GradCAM over GradCAM++.

RQ3 asks whether architecture matters. The cached results show that backbone choice changes explanation behavior, so model selection cannot rely on AUC alone.

RQ4 asks whether CAM method agreement predicts correctness. The cached results reject the simple trust assumption: incorrect predictions had higher agreement.

RQ5 asks whether attention changes during training. The result is exploratory: attention appears to become more focused later in training for the studied example, but it should not be overclaimed.

RQ6 asks whether the model generalizes externally. This is the strongest safety result: external datasets reveal AUC drops, class-specific malignant failures, persistent high confidence, and many zero-CAM explanation failures.

## Strongest Paper Claim

The strongest defensible claim is not "GradCAM explains skin cancer diagnosis." The stronger claim is:

> Internal skin-lesion classification performance and visually plausible CAMs are insufficient evidence for clinical reliability; explanation methods must be evaluated under method choice, architecture choice, temporal training behavior, inter-method agreement, and external distribution shift.

## What the Paper Should Emphasize

The paper should emphasize systematic evaluation. The value is not one single heatmap. The value is comparing explanation methods, quantifying faithfulness, showing architecture dependence, rejecting unsafe agreement assumptions, and exposing external validation failures.

## What the Paper Should Not Claim

Do not claim clinical readiness. Do not claim dermatologist-level explanation correctness. Do not claim that high confidence means reliability. Do not claim that high CAM agreement means the model is right. Do not claim that GradCAM is universally superior across all datasets and models. The cached results support GradCAM in this specific experimental setup.

## Why RQ6 Matters Most

RQ6 connects the research to real deployment risk. A model trained on HAM10000 can perform strongly internally while external datasets expose weaker discrimination and poor malignant accuracy. Mean confidence remains high in multiple shifted settings. That is a dangerous pattern because the model may appear certain exactly when it is less reliable.

## Recommended Manuscript Framing

Frame the work as a robustness study of XAI for dermatology AI. The central contribution is methodological and empirical: it tests explanation behavior from several angles and shows that explanation reliability is conditional. The paper should be cautious, scientifically honest, and clear that XAI is useful for auditing models but not a substitute for clinical validation.

## Defensible Contributions

- A reproducible notebook pipeline for CAM-based XAI analysis in skin-lesion classification.
- Quantitative comparison of GradCAM, GradCAM++, EigenCAM, and LayerCAM using focus and entropy.
- Faithfulness comparison using insertion/deletion AUC.
- Backbone-by-explanation analysis showing architecture effects.
- Negative result showing CAM agreement is not a correctness guarantee.
- Exploratory temporal XAI analysis across checkpoints.
- External validation across multiple datasets with performance, confidence, class-specific accuracy, XAI metrics, and zero-CAM failure reporting.

## Main Limitations

- No dermatologist segmentation masks are used to prove that highlighted regions are clinically correct.
- No prospective clinical reader study is included.
- Dataset label definitions and class balance differ across external datasets.
- RQ5 is exploratory and limited by checkpoint/image sample.
- RQ3 architecture effects may still be partly confounded by model performance differences.
- High zero-CAM rates limit claims about explanation reliability.
- Confidence is not fully calibrated for clinical use.

## Practical Conclusion

The notebooks support a cautious engineering conclusion: GradCAM with ResNet50 is the best current internal explanation choice in this project, but any clinical-facing system must add calibration, external validation, failure detection, doctor review, and explicit uncertainty handling before deployment.
