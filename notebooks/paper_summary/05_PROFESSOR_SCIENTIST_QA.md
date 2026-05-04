# Professor and Scientist Q&A

## Core Defense

Q: What is the single strongest result?

A: RQ6. The model performs strongly on internal HAM10000 data but degrades on external datasets while confidence often remains high. That combination is a major safety concern and supports the need for external validation and explanation robustness checks.

Q: What is the main novelty?

A: The novelty is not simply using GradCAM. The novelty is systematically testing explanation reliability across CAM method, faithfulness, architecture, inter-method agreement, training time, and external dataset shift.

Q: Is this clinically deployable?

A: No. The notebooks are a research pipeline. Clinical deployment would require prospective validation, dermatologist review, calibration, monitoring, regulatory review, and robust failure handling.

Q: Are the heatmaps clinically validated?

A: No. They are evaluated as model explanations through focus, entropy, faithfulness, agreement, and robustness. Clinical validation would require dermatologist annotations or reader studies.

## Data and Splitting

Q: Why is patient-level splitting important?

A: Without patient-level splitting, images from the same patient can appear in both train and test sets. That causes leakage and inflated performance.

Q: Why not use random image splitting?

A: Random image splitting can leak patient-specific visual cues. Medical ML should split by patient or lesion identity when metadata allows it.

Q: Why does class imbalance matter?

A: If malignant cases are rare, a model can achieve high accuracy by predicting benign too often. That is dangerous because malignant misses are clinically important.

Q: Why report malignant accuracy separately?

A: Overall accuracy can hide failure on malignant cases. Malignant accuracy directly measures performance on the high-risk class.

## Model Questions

Q: Why ResNet50?

A: ResNet50 is a strong, stable, widely understood CNN baseline. It also works well with GradCAM because it has clear convolutional feature maps.

Q: Why compare EfficientNet-B2 and MobileNetV2?

A: They represent different architecture tradeoffs. EfficientNet emphasizes efficiency and scaling; MobileNet emphasizes lightweight inference. RQ3 tests whether those choices affect explanations.

Q: Why not choose only the highest-AUC model?

A: For explainable medical AI, the model must be judged by both prediction performance and explanation behavior. RQ3 shows these can diverge.

## XAI Questions

Q: What is GradCAM actually showing?

A: GradCAM estimates which spatial regions in the final convolutional feature maps most influenced the target class score.

Q: Does GradCAM prove the model looked at the lesion?

A: No. It shows model-relevant regions, not necessarily clinically correct lesion regions. Expert masks would be needed for that claim.

Q: Why is GradCAM favored in this project?

A: In the cached results, GradCAM is most focused in RQ1 and more faithful in RQ2 than GradCAM++.

Q: Is focused always better?

A: No. Focused means concentrated. A concentrated map could still focus on an artifact. That is why faithfulness and external validation matter.

Q: Why compare CAM methods?

A: Explanation methods can disagree. If conclusions depend heavily on method choice, the explanation is less robust.

Q: Why not trust a heatmap that looks medically plausible?

A: Visual plausibility is subjective. A heatmap can look plausible while not being faithful to the model's actual decision process.

## RQ1 Questions

Q: What did RQ1 show?

A: GradCAM produced the lowest FAP and entropy among the compared CAM methods in the cached results.

Q: What does FAP mean?

A: FAP is the percentage of heatmap pixels above a threshold. Lower FAP means the explanation highlights a smaller area.

Q: What does entropy mean?

A: Entropy measures how diffuse the heatmap is. Lower entropy means more concentrated attention.

## RQ2 Questions

Q: What did RQ2 show?

A: GradCAM had higher insertion-minus-deletion faithfulness than GradCAM++.

Q: What does insertion test?

A: It adds pixels back in importance order and checks whether confidence rises quickly.

Q: What does deletion test?

A: It removes important pixels and checks whether confidence drops quickly.

Q: Does faithfulness equal clinical trust?

A: No. Faithfulness means the heatmap reflects model behavior. Clinical trust also requires medically correct reasoning and clinical validation.

## RQ3 Questions

Q: What did RQ3 show?

A: Backbone architecture affects explanation quality. ResNet50 plus GradCAM produced the most focused maps in the cached results.

Q: Why can architecture affect heatmaps?

A: Different architectures learn different internal spatial representations, and CAM methods depend on those internal feature maps.

Q: What is the limitation of RQ3?

A: Architecture effects may be partly confounded by different classification performance. The notebook reduces but does not fully eliminate this issue.

## RQ4 Questions

Q: What did RQ4 show?

A: Incorrect predictions had higher CAM agreement than correct predictions in the cached results.

Q: Does that contradict the hypothesis?

A: Yes. The hypothesis expected disagreement to signal errors. The result shows agreement is not a reliable correctness signal.

Q: Is this negative result useful?

A: Yes. It prevents an unsafe production assumption: multiple explanation methods agreeing does not mean the prediction is correct.

## RQ5 Questions

Q: What did RQ5 show?

A: For the studied example, attention appeared to become more focused later in training, while validation AUC improved.

Q: Why is RQ5 exploratory?

A: It is limited by sample size and checkpoint coverage. It should motivate future work rather than serve as a definitive claim.

Q: What would strengthen RQ5?

A: More images, more checkpoints, repeated seeds, lesion masks, and statistical analysis across samples.

## RQ6 Questions

Q: What did RQ6 show?

A: The model's external AUC dropped substantially across several datasets, malignant accuracy was poor in multiple cases, and confidence remained high.

Q: Why is high external confidence concerning?

A: It suggests the model may be overconfident under distribution shift, which is dangerous in clinical settings.

Q: Why can ISIC2020 accuracy be high but malignant accuracy low?

A: Class imbalance. If most images are benign, benign-heavy predictions can yield high overall accuracy while missing malignant cases.

Q: What are zero-CAM failures?

A: Cases where the CAM output has no useful activation. They are explanation failures and should be reported honestly.

Q: Do zero-CAM failures invalidate the whole project?

A: They weaken claims about reliable explanations but strengthen the robustness argument because the study identifies and reports a real failure mode.

## Reviewer Challenges

Q: Why should we trust your results without clinical masks?

A: The paper should not claim clinical localization. It claims model-explanation behavior under quantitative XAI metrics. Clinical masks are future work.

Q: Are the external datasets directly comparable?

A: Not perfectly. They differ in acquisition, labels, class balance, and population. That is the point of distribution-shift testing, but comparisons must be interpreted cautiously.

Q: Are the thresholds arbitrary?

A: Thresholds such as FAP@0.5 are operational choices. The notebooks also use FAP@0.3 and entropy, and conclusions should be checked for threshold sensitivity.

Q: Could preprocessing differences explain external drops?

A: Yes, preprocessing and acquisition differences are part of distribution shift. The paper should acknowledge this and frame external drops as real-world transfer risk.

Q: What would be required for clinical deployment?

A: Calibration, threshold selection, local validation, prospective evaluation, dermatologist review, monitoring, audit logs, data governance, failure detection, and regulatory review.
