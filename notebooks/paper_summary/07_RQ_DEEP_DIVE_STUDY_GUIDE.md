# RQ Deep Dive Study Guide

This file is designed for paper writing, viva/oral defense, professor meetings, and reviewer responses. It explains every research question using what, why, how, when, where, who, interpretation, limitations, and likely challenges.

## Before the RQs: Setup and Training

### What was done?

The project first verifies the Python environment, GPU support, package imports, dataset paths, patient-level splits, DataLoader behavior, model training, held-out evaluation, and checkpoint availability.

### Why was it necessary?

Explanation metrics are only meaningful if the data and model are valid. A CAM generated from a bad checkpoint, wrong preprocessing, leaked split, or mislabeled image can look convincing while being scientifically invalid.

### How was it done?

The setup notebook checks imports, CUDA, HAM10000 metadata, image paths, train/validation/test splits, DataLoader tensor shape, class balance, ResNet50 training, test metrics, and saved checkpoints.

### What should be written in the paper?

Write that the experimental pipeline included sanity checks for environment reproducibility, patient-level leakage prevention, class-imbalance handling, and checkpoint verification before explainability experiments.

### Common professor challenge

Q: Why should we trust your XAI results?

A: Because the notebooks first validate the environment, data, split integrity, preprocessing, model checkpoint, and held-out performance. XAI is evaluated only after those conditions are satisfied.

## RQ1: Which CAM Variant Best Localizes Lesion Regions?

### What was asked?

RQ1 asks which CAM-family explanation method produces the most spatially focused heatmap for skin-lesion predictions.

### Why was it asked?

Different CAM methods can produce different heatmaps for the same image and same model. If a paper shows only one heatmap method without comparison, the explanation may be method-dependent and fragile.

### Where in the notebooks?

`RQ1_cam_variant_comparison.ipynb`, summarized numerically by `outputs/metrics/RQ1_cam_comparison.csv`.

### How was it tested?

The notebook loaded a trained ResNet50, generated GradCAM, GradCAM++, EigenCAM, and LayerCAM maps, then computed FAP@0.5, FAP@0.3, and entropy across images.

### Current result table

| Method | n | FAP@0.5 mean | FAP@0.5 std | FAP@0.3 mean | Entropy mean | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| EigenCAM | 125 | 0.0671 | 0.0292 | 0.1212 | 9.3640 | 0.9161 |
| GradCAM | 125 | 0.0442 | 0.0453 | 0.0868 | 7.0243 | 0.9161 |
| GradCAM++ | 125 | 0.0777 | 0.0382 | 0.1640 | 10.1709 | 0.9161 |
| LayerCAM | 125 | 0.0788 | 0.0372 | 0.1679 | 10.1857 | 0.9161 |

### What does the result mean?

GradCAM is the most focused method in the cached results because it has the lowest FAP@0.5 and lowest entropy. It highlights a smaller and less diffuse region than EigenCAM, GradCAM++, and LayerCAM.

### What does the result not mean?

It does not mean GradCAM is clinically correct. It does not mean GradCAM will be best for every architecture, dataset, or medical task. It only supports GradCAM as the strongest method in this experimental context.

### Paper sentence

"Among the evaluated CAM variants, GradCAM produced the most compact explanations on HAM10000 ResNet50 predictions, with the lowest FAP@0.5 and entropy."

### Likely professor questions

Q: Is a smaller heatmap always better?

A: No. A smaller heatmap is easier to inspect, but it may focus on an artifact. That is why RQ2 tests faithfulness and RQ6 tests robustness under dataset shift.

Q: Why include EigenCAM and LayerCAM?

A: They represent alternative CAM assumptions. Including them makes the comparison less dependent on a single explanation algorithm.

Q: Why not use ScoreCAM?

A: ScoreCAM is computationally expensive because it requires many forward passes. It may be valuable in future work, but the current pipeline prioritizes methods that can be scaled across datasets.

## RQ2: Are the Explanations Faithful to the Model?

### What was asked?

RQ2 asks whether the highlighted regions actually affect the model's prediction.

### Why was it asked?

A heatmap can look plausible while not reflecting what the model truly uses. Faithfulness testing moves beyond visual inspection.

### Where in the notebooks?

`RQ2_faithfulness.ipynb`, summarized by `outputs/metrics/RQ2_faithfulness.csv`.

### How was it tested?

The notebook uses insertion and deletion tests. In insertion, important pixels are added back first; a faithful explanation should make confidence rise quickly. In deletion, important pixels are removed first; a faithful explanation should make confidence drop quickly.

### Current result table

| Method | n | Insertion AUC | Deletion AUC | Faithfulness | Std | Positive rate | Correct rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradCAM | 50 | 0.2193 | 0.1364 | 0.0828 | 0.1506 | 0.6200 | 0.8600 |
| GradCAM++ | 50 | 0.1851 | 0.1610 | 0.0241 | 0.1472 | 0.4000 | 0.8600 |

### What does the result mean?

GradCAM has a larger faithfulness score than GradCAM++. This means GradCAM's highlighted pixels better align with the model's own confidence behavior in this run.

### What does the result not mean?

It does not prove the model is medically right. It proves relative model faithfulness, not clinical correctness.

### Paper sentence

"Insertion/deletion analysis showed higher faithfulness for GradCAM than GradCAM++, suggesting that GradCAM better reflected the model's decision process in this experiment."

### Likely professor questions

Q: Why is insertion AUC higher better?

A: If the heatmap identifies truly important pixels, adding those pixels back should rapidly restore confidence.

Q: Why is deletion AUC lower better?

A: If the heatmap identifies truly important pixels, removing them should rapidly reduce confidence.

Q: Can a faithful explanation still be clinically wrong?

A: Yes. A CAM can faithfully show that the model used a shortcut or artifact. That is why clinical masks and reader studies are future work.

## RQ3: Does Backbone Architecture Affect Explanation Quality?

### What was asked?

RQ3 asks whether different CNN backbones produce different explanation quality, even when used for the same task.

### Why was it asked?

In model selection, researchers often choose the architecture with the best AUC. But if explanations are part of the scientific or clinical claim, architecture must also be judged by explanation behavior.

### Where in the notebooks?

`RQ3_backbone_xai_quality.ipynb`, summarized by `outputs/metrics/RQ3_backbone_comparison.csv`.

### How was it tested?

The notebook compares ResNet50, EfficientNet-B2, and MobileNetV2 using GradCAM and GradCAM++. It computes correct rate, confidence, FAP@0.5, and entropy.

### Current result table

| Backbone | CAM | n | Correct rate | Confidence | FAP@0.5 | Entropy |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b2 | GradCAM | 100 | 0.9000 | 0.9498 | 0.1781 | 9.3330 |
| efficientnet_b2 | GradCAM++ | 100 | 0.9000 | 0.9498 | 0.0664 | 8.2239 |
| mobilenetv2_100 | GradCAM | 100 | 0.8400 | 0.9198 | 0.1061 | 7.1320 |
| mobilenetv2_100 | GradCAM++ | 100 | 0.8400 | 0.9198 | 0.0963 | 6.8454 |
| resnet50 | GradCAM | 100 | 0.8400 | 0.8912 | 0.0458 | 7.2723 |
| resnet50 | GradCAM++ | 100 | 0.8400 | 0.8912 | 0.0735 | 10.1184 |

### What does the result mean?

Backbone architecture affects explanation compactness. ResNet50 plus GradCAM is the most focused combination in the cached metrics. EfficientNet-B2 has high correct rate but much more diffuse GradCAM maps.

### What does the result not mean?

It does not prove ResNet50 is universally superior. It means ResNet50 plus GradCAM is the strongest explanation pair in this setup.

### Paper sentence

"Backbone architecture materially affected CAM compactness, indicating that explanation quality should be considered alongside predictive performance."

### Likely professor questions

Q: Is architecture effect independent of accuracy?

A: The notebook tries to reduce confounding by comparing the same task and image set, but performance differences can still influence explanation behavior. This should be reported as a limitation.

Q: Why can EfficientNet classify well but explain diffusely?

A: Different architectures organize internal spatial features differently. CAM methods depend on those feature maps, so high classification ability does not guarantee compact saliency.

## RQ4: Does CAM Agreement Predict Correctness or Uncertainty?

### What was asked?

RQ4 asks whether agreement among CAM methods can identify correct or incorrect predictions.

### Why was it asked?

A tempting safety idea is that if several explanation methods agree, the prediction is more trustworthy. RQ4 tests that assumption.

### Where in the notebooks?

`RQ4_agreement_vs_uncertainty.ipynb`, summarized by `outputs/metrics/RQ4_agreement_results.csv`.

### How was it tested?

The notebook computes Jaccard overlap and Spearman agreement between CAM maps, then compares agreement for correct versus incorrect predictions.

### Current result table

| Prediction group | n | Confidence | Avg Jaccard | Jaccard std | Avg Spearman | Min Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| Incorrect | 25 | 0.7584 | 0.5354 | 0.2166 | 0.6706 | 0.3852 |
| Correct | 125 | 0.9161 | 0.2905 | 0.2048 | 0.2722 | 0.1041 |

### What does the result mean?

Incorrect predictions had higher agreement than correct predictions. This means CAM agreement is not a correctness guarantee.

### What does the result not mean?

It does not mean agreement is useless. It means agreement should be interpreted descriptively and in context, not as a standalone trust score.

### Paper sentence

"Contrary to the initial hypothesis, inter-method CAM agreement did not indicate correctness; incorrect predictions showed higher average agreement."

### Likely professor questions

Q: Is a contradicted hypothesis a failure?

A: No. It is an important negative result. It prevents unsafe use of CAM agreement as a trust signal.

Q: Why might wrong predictions have high agreement?

A: Multiple CAM methods may converge on the same misleading cue or artifact. Agreement can mean consistent focus, not correct reasoning.

## RQ5: How Does Attention Evolve During Training?

### What was asked?

RQ5 asks whether GradCAM attention changes as model checkpoints progress through training.

### Why was it asked?

Models may initially learn broad shortcuts such as color, texture, or artifacts before learning more lesion-specific patterns. Temporal XAI can reveal this learning behavior.

### Where in the notebooks?

`RQ5_temporal_xai.ipynb`.

### How was it tested?

The notebook loads multiple epoch checkpoints, applies GradCAM to the same fixed image, and tracks validation AUC, malignant probability, FAP, entropy, and activation location.

### What did the cached run suggest?

The studied example showed improved validation AUC and later reduction in FAP, suggesting attention became more focused. The reported AUC-vs-FAP correlation was negative but not statistically significant.

### What does the result mean?

RQ5 provides exploratory evidence that attention can evolve during training. It is useful for model debugging and for generating hypotheses about shortcut learning.

### What does the result not mean?

It does not prove that all models always shift from shortcuts to clinically meaningful features.

### Paper sentence

"Temporal GradCAM analysis suggested a late-training concentration of attention for the examined case, but this result should be interpreted as exploratory."

### Likely professor questions

Q: Why include a non-significant result?

A: It provides useful qualitative insight into training dynamics and motivates future temporal XAI analysis. It should be framed honestly as exploratory.

Q: How would you make it stronger?

A: Use more images, more checkpoints, repeated seeds, lesion masks, and statistical analysis across the sample.

## RQ6: Does the Model Generalize Under External Dataset Shift?

### What was asked?

RQ6 asks whether a model trained on HAM10000 remains reliable on external datasets.

### Why was it asked?

Clinical AI often fails when deployed outside its development dataset. External validation is one of the most important tests of real-world safety.

### Where in the notebooks?

`RQ6_external_validation.ipynb`, with extended stress tests in `RQ6_extended_datasets.ipynb`.

### How was it tested?

The notebook evaluates the same HAM10000-trained ResNet50 on HAM10000, ISIC2020, MILK10K, Malignant-Benign, and PH2. It reports performance, confidence, Brier score, AUC drop, class-specific accuracy, GradCAM robustness, zero-CAM rate, and CAM agreement.

### Current performance table

| Dataset | n | AUC | Accuracy | Mean confidence | Brier score | AUC drop | Benign acc | Malignant acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAM10000 | 9815 | 0.9479 | 0.8699 | 0.8963 | 0.0917 | 0.0000 | 0.8676 | 0.8815 |
| ISIC2020 | 32926 | 0.6348 | 0.9126 | 0.8936 | 0.0704 | 0.3131 | 0.9245 | 0.2483 |
| MILK10K | 10034 | 0.6677 | 0.4202 | 0.8625 | 0.4598 | 0.2802 | 0.9219 | 0.2154 |
| Malignant-Benign | 34793 | 0.7799 | 0.6489 | 0.8655 | 0.2671 | 0.1681 | 0.9268 | 0.3708 |
| PH2 | 200 | 0.7450 | 0.5150 | 0.9266 | 0.4217 | 0.2029 | 0.9875 | 0.2000 |

### Current XAI robustness table

| Dataset | n XAI | Zero CAMs | Valid CAMs | FAP@0.5 | Entropy | Confidence | Correct rate | Zero-CAM rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAM10000 | 9785 | 9274 | 511 | 0.0573 | 9.2627 | 0.8690 | 0.8454 | 0.9478 |
| ISIC2020 | 32896 | 31026 | 1870 | 0.0517 | 9.3139 | 0.8799 | 0.9011 | 0.9432 |
| MILK10K | 10004 | 9384 | 620 | 0.0533 | 9.3952 | 0.8582 | 0.4032 | 0.9380 |
| Malignant-Benign | 34763 | 32749 | 2014 | 0.0573 | 9.4743 | 0.8504 | 0.6137 | 0.9421 |
| PH2 | 170 | 159 | 11 | 0.0806 | 9.7395 | 0.9087 | 0.7273 | 0.9353 |

### What does the result mean?

External validation reveals major generalization problems. AUC drops substantially outside HAM10000. Some datasets show very low malignant accuracy. Confidence remains high, which indicates overconfidence and calibration risk. Zero-CAM rates are high, which indicates explanation reliability problems.

### What does the result not mean?

It does not mean the model is useless. It means the model is not clinically ready without external validation, calibration, local adaptation, and human review.

### Paper sentence

"External validation exposed substantial degradation in discrimination and malignant-class accuracy despite persistently high confidence, highlighting the risk of relying on internal validation alone."

### Likely professor questions

Q: Why is high accuracy on ISIC2020 not enough?

A: Because malignant accuracy is low. Overall accuracy is inflated by class imbalance and benign dominance.

Q: Why report Brier score?

A: It gives calibration-related information. A model can rank cases reasonably but still produce unreliable probabilities.

Q: Why are zero-CAM failures important?

A: They show that explanation generation itself can fail. A clinical interface must detect and report this rather than displaying misleading empty explanations.

## Final Defense Summary

The safest final answer to most examiner questions is:

This work does not claim clinical readiness. It shows that explanation behavior in skin-lesion AI is conditional on method, model architecture, training stage, and dataset distribution. GradCAM is the strongest method in this project, but external validation and zero-CAM analysis reveal serious reliability limits. Therefore, CAM explanations should be used for auditing and research, not as standalone clinical evidence.
