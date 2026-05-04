# Paper Writing Blueprint

## Suggested Title

Robustness Evaluation of CAM-Based Explanations for Skin Lesion Classification Under Architecture and Dataset Shift

## Abstract Structure

Background: Explainable AI is often proposed for medical image classifiers, but visual explanations can be misleading if not quantitatively evaluated.

Objective: Evaluate CAM explanation reliability for skin-lesion classification across method choice, faithfulness, backbone architecture, inter-method agreement, temporal training behavior, and external dataset shift.

Methods: Train/evaluate binary skin-lesion classifiers, generate CAM-family explanations, compute focus area percentage, entropy, insertion/deletion faithfulness, Jaccard/Spearman agreement, and external validation metrics.

Results: GradCAM was most focused and more faithful than GradCAM++ in the cached results. Backbone architecture changed explanation behavior. CAM agreement did not guarantee correctness. External datasets showed AUC drops, poor malignant accuracy in several settings, persistent confidence, and many zero-CAM failures.

Conclusion: CAM explanations can support model auditing but should not be treated as clinical evidence without external validation, calibration, and clinical annotation.

## Introduction Flow

Paragraph 1: Medical imaging AI can achieve high internal performance, but clinical deployment requires reliability under dataset shift.

Paragraph 2: XAI heatmaps are attractive because they appear to show model reasoning, but visual plausibility is not the same as faithfulness or clinical correctness.

Paragraph 3: Skin-lesion classification is a strong testbed because lesion localization and malignant detection are clinically important.

Paragraph 4: This work evaluates explanation robustness through six research questions.

## Methods Flow

Data: describe HAM10000 as the internal reference and external datasets as distribution-shift tests.

Model: describe ResNet50 baseline and additional backbones EfficientNet-B2 and MobileNetV2.

Splitting: emphasize patient-level split and leakage prevention.

Explanations: describe GradCAM, GradCAM++, EigenCAM, LayerCAM, and optional LIME/SHAP extension.

Metrics: define FAP, entropy, insertion AUC, deletion AUC, faithfulness, Jaccard, Spearman, ROC-AUC, accuracy, balanced accuracy, Brier score, class-specific accuracy, and zero-CAM rate.

Statistics: describe grouped comparisons and nonparametric tests where used.

## Results Flow

RQ1: GradCAM produced the most focused heatmaps by FAP and entropy.

RQ2: GradCAM was more faithful than GradCAM++ using insertion/deletion.

RQ3: Backbone architecture affected XAI quality, with ResNet50 plus GradCAM producing the most focused explanations.

RQ4: CAM agreement did not predict correctness; incorrect predictions had higher agreement.

RQ5: Temporal analysis suggested attention may become more focused later in training, but evidence is exploratory.

RQ6: External validation revealed performance degradation, class-specific malignant failures, overconfidence risk, and zero-CAM failures.

## Discussion Flow

Start with the main finding: explanation reliability is conditional and must be tested.

Discuss why GradCAM is currently preferred in this setup.

Discuss why architecture selection should include explanation behavior.

Discuss the negative result from RQ4 as a safety lesson.

Discuss RQ6 as the strongest deployment warning.

Discuss limitations honestly.

End with future work: dermatologist masks, calibration, prospective validation, subgroup analysis, saliency sanity checks, and production monitoring.

## Sentences You Can Use

"A visually plausible heatmap is not sufficient evidence that the model used clinically meaningful features."

"The study evaluates explanations as model-behavior artifacts rather than as direct clinical annotations."

"External validation revealed that internal performance substantially overestimated deployment reliability."

"Inter-method CAM agreement should be interpreted as a descriptive robustness signal, not as a correctness guarantee."

"Zero-CAM outputs were treated as explanation failures and reported explicitly."

## Sentences To Avoid

"The model diagnoses skin cancer."

"GradCAM proves the model looked at the lesion."

"High confidence means the prediction is reliable."

"CAM agreement validates model reasoning."

"The system is ready for clinical deployment."

## Recommended Tables

Table 1: Dataset overview and class balance.

Table 2: RQ1 CAM method comparison by FAP and entropy.

Table 3: RQ2 faithfulness comparison.

Table 4: RQ3 backbone-by-CAM comparison.

Table 5: RQ4 agreement by correctness.

Table 6: RQ6 external validation performance.

Table 7: RQ6 zero-CAM and XAI robustness.

## Recommended Figures

Figure 1: Pipeline diagram from image to prediction to CAM metrics.

Figure 2: RQ1 single-image CAM comparison.

Figure 3: RQ2 insertion/deletion curves.

Figure 4: RQ3 backbone-by-CAM heatmap or bar plot.

Figure 5: RQ4 agreement versus correctness plot.

Figure 6: RQ5 temporal attention over checkpoints.

Figure 7: RQ6 external dataset performance and confidence comparison.

## Limitations Paragraph

This study evaluates CAM explanations as model-behavior artifacts and does not establish clinical correctness of highlighted regions. Dermatologist segmentation masks and prospective reader studies are needed to validate whether highlighted regions correspond to medically meaningful lesion structures. External datasets differ in acquisition conditions, label definitions, and class prevalence, so cross-dataset comparisons must be interpreted as distribution-shift stress tests rather than controlled epidemiological comparisons. High zero-CAM rates also show that CAM generation can fail and should not be assumed reliable in every case.

## Future Work Paragraph

Future work should incorporate dermatologist-annotated lesion masks, calibration methods, subgroup analyses across skin tones and acquisition settings, repeated training seeds, formal saliency sanity checks, and prospective clinical evaluation. For deployment, the system should include uncertainty thresholds, model drift monitoring, explanation failure detection, and human review workflows.
