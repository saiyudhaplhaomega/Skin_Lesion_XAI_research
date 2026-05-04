# Glossary and Concepts

## Skin Lesion

A skin lesion is an abnormal region of skin. In this project, lesion images are mapped to benign or malignant labels for binary classification.

## Benign

Benign means non-cancerous. A benign lesion may still require medical attention, but in this binary classification task it is the negative class.

## Malignant

Malignant means cancerous or cancer-suspicious. In the binary task it is the positive and clinically high-risk class.

## Dermoscopy

Dermoscopy is a skin imaging technique that uses magnification and lighting to visualize lesion structures not easily visible to the naked eye.

## HAM10000

HAM10000 is the internal reference dataset in the notebooks. It contains roughly 10,015 dermoscopic skin-lesion images and is used for training and internal evaluation.

## SIIM-ISIC 2020

SIIM-ISIC 2020 is a large melanoma classification dataset. It is used in an optional training extension and in external validation analysis.

## PH2

PH2 is a smaller dermoscopy dataset. It is useful for external validation because it differs from HAM10000 in collection and population.

## MILK10K

MILK10K is an external dataset with diverse images and modalities. It is useful for stress testing generalization.

## Distribution Shift

Distribution shift means test data differs from training data. In dermatology AI, shift can come from skin tone, camera type, lighting, clinic, geography, lesion prevalence, or labeling protocol.

## Binary Classification

Binary classification means predicting one of two classes. Here the classes are benign and malignant.

## Class Imbalance

Class imbalance means one class is much more common than the other. Skin-lesion datasets often contain many more benign cases than malignant cases.

## WeightedRandomSampler

WeightedRandomSampler is a PyTorch sampler that increases the probability of drawing minority-class samples during training. It helps prevent the model from ignoring rare malignant examples.

## ResNet50

ResNet50 is a convolutional neural network with residual skip connections. It is the main baseline backbone in the notebooks.

## EfficientNet-B2

EfficientNet-B2 is an efficient convolutional architecture. It can be accurate, but the notebooks show its explanations can behave differently from ResNet50.

## MobileNetV2

MobileNetV2 is a lightweight neural network designed for efficient inference, often on mobile or edge devices.

## Backbone

The backbone is the feature extractor of a neural network. It produces internal feature maps that the classifier head uses for prediction.

## Feature Map

A feature map is an internal activation grid in a convolutional neural network. CAM methods use feature maps to build heatmaps.

## CAM

CAM means Class Activation Map. A CAM is a heatmap that estimates which image regions influenced a class prediction.

## GradCAM

GradCAM uses gradients of the target class with respect to feature maps to weight spatial features. In these notebooks it is the strongest method by focus and faithfulness.

## GradCAM++

GradCAM++ extends GradCAM with higher-order gradient weighting. It can help in some localization settings but is less faithful than GradCAM in the cached results.

## EigenCAM

EigenCAM is gradient-free and uses principal components of feature activations to create a heatmap.

## LayerCAM

LayerCAM uses layer-wise activations and gradients to produce finer-grained maps.

## LIME

LIME perturbs parts of an input and observes how predictions change. For images, it often perturbs superpixels.

## SHAP

SHAP estimates feature contributions using Shapley values. For images, it is often expensive.

## Focus Area Percentage

Focus Area Percentage, or FAP, is the fraction of heatmap pixels above a threshold. Lower FAP means the map is more concentrated.

## Entropy

Entropy measures how spread out the heatmap activation is. Lower entropy means more concentrated attention.

## Insertion AUC

Insertion AUC measures how quickly model confidence rises when important pixels are added back into a blank or blurred image. Higher is better.

## Deletion AUC

Deletion AUC measures how model confidence behaves when important pixels are removed from the original image. Lower is better if the highlighted pixels are truly important.

## Faithfulness

Faithfulness is insertion AUC minus deletion AUC. Higher values mean the explanation better reflects the model's own decision process.

## Jaccard Similarity

Jaccard similarity compares two highlighted regions: intersection divided by union. It ranges from 0 to 1.

## Spearman Agreement

Spearman agreement compares ranked heatmap values rather than raw values. It asks whether two maps rank pixels similarly.

## ROC-AUC

ROC-AUC measures how well the model ranks positives above negatives across thresholds. It is useful under class imbalance.

## Accuracy

Accuracy is the fraction of correct predictions. It can be misleading when classes are imbalanced.

## Balanced Accuracy

Balanced accuracy averages class-wise accuracy. It is more informative when one class is rare.

## Brier Score

Brier score measures probability calibration. Lower is better.

## Calibration

Calibration asks whether predicted confidence matches real correctness frequency. A calibrated 90 percent confidence prediction should be correct about 90 percent of the time.

## Zero-CAM

A zero-CAM is a failed explanation where the CAM contains no useful activation. In this project it is reported as an explanation failure.

## Patient Leakage

Patient leakage happens when images from the same patient appear in both training and testing. It can artificially inflate performance.

## External Validation

External validation tests a model on data from outside the training source. It is essential for clinical AI because deployment data often differs from training data.
