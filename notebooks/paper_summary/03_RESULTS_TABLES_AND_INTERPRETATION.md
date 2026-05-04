# Results Tables and Interpretation

These tables are generated from the cached CSV outputs under `notebooks/outputs/metrics/`.

## RQ1: CAM Variant Comparison

| Method | n | FAP@0.5 mean | FAP@0.5 std | FAP@0.3 mean | Entropy mean | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| EigenCAM | 125 | 0.0671 | 0.0292 | 0.1212 | 9.3640 | 0.9161 |
| GradCAM | 125 | 0.0442 | 0.0453 | 0.0868 | 7.0243 | 0.9161 |
| GradCAM++ | 125 | 0.0777 | 0.0382 | 0.1640 | 10.1709 | 0.9161 |
| LayerCAM | 125 | 0.0788 | 0.0372 | 0.1679 | 10.1857 | 0.9161 |

Interpretation: GradCAM has the lowest FAP@0.5 and lowest entropy in the cached results. It is the most focused CAM method in this experiment.

Paper wording: "In the current HAM10000 ResNet50 experiment, GradCAM produced the most spatially concentrated explanations among the evaluated CAM variants."

Avoid wording: "GradCAM is clinically correct" or "GradCAM always works best."

## RQ2: Faithfulness

| Method | n | Insertion AUC | Deletion AUC | Faithfulness | Std | Positive rate | Correct rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradCAM | 50 | 0.2193 | 0.1364 | 0.0828 | 0.1506 | 0.6200 | 0.8600 |
| GradCAM++ | 50 | 0.1851 | 0.1610 | 0.0241 | 0.1472 | 0.4000 | 0.8600 |

Interpretation: GradCAM has higher insertion-minus-deletion faithfulness than GradCAM++. This supports GradCAM as more faithful to the model's own decision process.

Paper wording: "GradCAM showed stronger model-faithfulness behavior than GradCAM++ under insertion/deletion testing."

Avoid wording: "GradCAM proves the model used clinically meaningful features."

## RQ3: Backbone Architecture and XAI Quality

| Backbone | CAM | n | Correct rate | Confidence | FAP@0.5 | Entropy |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b2 | GradCAM | 100 | 0.9000 | 0.9498 | 0.1781 | 9.3330 |
| efficientnet_b2 | GradCAM++ | 100 | 0.9000 | 0.9498 | 0.0664 | 8.2239 |
| mobilenetv2_100 | GradCAM | 100 | 0.8400 | 0.9198 | 0.1061 | 7.1320 |
| mobilenetv2_100 | GradCAM++ | 100 | 0.8400 | 0.9198 | 0.0963 | 6.8454 |
| resnet50 | GradCAM | 100 | 0.8400 | 0.8912 | 0.0458 | 7.2723 |
| resnet50 | GradCAM++ | 100 | 0.8400 | 0.8912 | 0.0735 | 10.1184 |

Interpretation: model architecture changes explanation behavior. ResNet50 plus GradCAM is the most focused combination in the cached metrics, while EfficientNet-B2 plus GradCAM is much broader.

Paper wording: "Backbone choice affected explanation compactness, indicating that architecture should be considered when selecting models for explainable medical AI."

Avoid wording: "The highest-accuracy architecture is automatically the best explainable architecture."

## RQ4: Inter-Method Agreement vs Correctness

| Prediction group | n | Confidence | Avg Jaccard | Jaccard std | Avg Spearman | Min Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| Incorrect | 25 | 0.7584 | 0.5354 | 0.2166 | 0.6706 | 0.3852 |
| Correct | 125 | 0.9161 | 0.2905 | 0.2048 | 0.2722 | 0.1041 |

Interpretation: incorrect predictions had higher CAM agreement than correct predictions. This rejects the simple hypothesis that agreement means correctness.

Paper wording: "Inter-method agreement should be interpreted descriptively and cannot be used alone as a correctness or trust signal."

Avoid wording: "CAM agreement validates the prediction."

## RQ5: Temporal Grad-CAM Attention During Training

| Epoch | Val AUC | FAP@0.5 | Entropy | Malignant probability | Interpretation |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8211 | 0.1378 | 10.1757 | 0.7350 | Early checkpoint with broad, diffuse attention. |
| 2 | 0.8393 | 0.1537 | 10.2610 | 0.6920 | AUC improves, but attention remains scattered. |
| 3 | 0.8544 | 0.1997 | 10.1990 | 0.7220 | Attention becomes most diffuse despite higher AUC. |
| 4 | 0.8620 | 0.0838 | 10.1940 | 0.6080 | Sharp transition to more localized attention. |
| 5 | 0.8602 | 0.0735 | 10.0940 | 0.5110 | Most focused checkpoint, near the decision threshold. |

Interpretation: Grad-CAM attention became more focused late in training for the studied image. FAP@0.5 increased from epochs 1-3, then dropped sharply from 0.1997 to 0.0838 at epoch 4 and to 0.0735 at epoch 5. This supports the exploratory hypothesis that later checkpoints may rely on more localized lesion evidence, but the result should not be treated as a general law.

Paper wording: "Temporal Grad-CAM analysis suggested a late-training concentration of attention for the examined case, with FAP@0.5 decreasing from 0.1378 at epoch 1 to 0.0735 at epoch 5 as validation AUC improved from 0.8211 to 0.8602."

Avoid wording: "RQ5 proves that training always makes attention clinically meaningful." This result is exploratory because it uses one fixed image and five checkpoints.

## RQ6: External Validation Performance

| Dataset | n | AUC | Accuracy | Mean confidence | Brier score | AUC drop | Benign acc | Malignant acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAM10000 | 9815 | 0.9479 | 0.8699 | 0.8963 | 0.0917 | 0.0000 | 0.8676 | 0.8815 |
| ISIC2020 | 32926 | 0.6348 | 0.9126 | 0.8936 | 0.0704 | 0.3131 | 0.9245 | 0.2483 |
| MILK10K | 10034 | 0.6677 | 0.4202 | 0.8625 | 0.4598 | 0.2802 | 0.9219 | 0.2154 |
| Malignant-Benign | 34793 | 0.7799 | 0.6489 | 0.8655 | 0.2671 | 0.1681 | 0.9268 | 0.3708 |
| PH2 | 200 | 0.7450 | 0.5150 | 0.9266 | 0.4217 | 0.2029 | 0.9875 | 0.2000 |

Interpretation: external datasets show major AUC drops from HAM10000. Overall accuracy can be misleading, especially when malignant accuracy is low.

Paper wording: "External validation revealed substantial degradation in discrimination and malignant-class performance, despite persistently high confidence."

Avoid wording: "The model generalizes clinically because accuracy remains high on one external dataset."

## RQ6: XAI Robustness and Zero-CAM Failure

| Dataset | n XAI | Zero CAMs | Valid CAMs | FAP@0.5 | Entropy | Confidence | Correct rate | Zero-CAM rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAM10000 | 9785 | 9274 | 511 | 0.0573 | 9.2627 | 0.8690 | 0.8454 | 0.9478 |
| ISIC2020 | 32896 | 31026 | 1870 | 0.0517 | 9.3139 | 0.8799 | 0.9011 | 0.9432 |
| MILK10K | 10004 | 9384 | 620 | 0.0533 | 9.3952 | 0.8582 | 0.4032 | 0.9380 |
| Malignant-Benign | 34763 | 32749 | 2014 | 0.0573 | 9.4743 | 0.8504 | 0.6137 | 0.9421 |
| PH2 | 170 | 159 | 11 | 0.0806 | 9.7395 | 0.9087 | 0.7273 | 0.9353 |

Interpretation: zero-CAM rates are high and should be reported as explanation failures. Valid CAM summaries should not hide how many CAMs failed.

Paper wording: "Zero-CAM outputs were treated as explanation failures and reported explicitly."

Avoid wording: "The absence of a CAM means the model found no suspicious region."

## RQ6: CAM Agreement Across Datasets

| Dataset | n | Avg Jaccard | Min Jaccard | Max Jaccard | Correct rate | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| HAM10000 | 9795 | 0.4651 | 0.1627 | 0.8912 | 0.8699 | 0.8963 |
| ISIC2020 | 32906 | 0.3813 | 0.0465 | 0.8737 | 0.9126 | 0.8936 |
| MILK10K | 10014 | 0.4317 | 0.1075 | 0.8980 | 0.4198 | 0.8625 |
| Malignant-Benign | 34773 | 0.4501 | 0.1346 | 0.8974 | 0.6488 | 0.8655 |
| PH2 | 180 | 0.3868 | 0.0681 | 0.8826 | 0.5111 | 0.9277 |

Interpretation: CAM agreement changes across datasets but is not a correctness guarantee. It should be interpreted with performance and confidence.

## Optional Training Addendum

| Source | Best epoch | ROC-AUC | Accuracy | Balanced accuracy | Loss |
| --- | --- | --- | --- | --- | --- |
| TRAINING_ham10000_resnet50_baseline.csv | 3 | 0.9136 | 0.8710 | 0.8203 | 0.3028 |
| TRAINING_siim_isic2020_resnet50_binary.csv | 1 | 0.8668 | 0.9252 | 0.7223 | 0.1704 |

Interpretation: these are optional training summaries. Use them to describe model development, not as the main XAI contribution.

## Top-Level Numerical Takeaways

- GradCAM is strongest in RQ1 by FAP and entropy.
- GradCAM is stronger in RQ2 by faithfulness.
- ResNet50 plus GradCAM is the most focused architecture-method pair in RQ3.
- RQ4 is a negative result: agreement does not imply correctness.
- RQ5 is exploratory: attention becomes more focused late in training for the studied checkpoint sequence.
- RQ6 is the main safety result: external validation exposes performance and explanation reliability risks.
