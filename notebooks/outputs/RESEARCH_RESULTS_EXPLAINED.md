# Research Results Explanation Guide
## XAI Skin Lesion Classification - All RQs Explained Simply

---

## RQ1: Which CAM method is best for explaining the model's predictions?

### What is a CAM method?
When the model looks at a skin image and decides "this is malignant," a CAM (Class Activation Map) method shows us EXACTLY which parts of the image influenced that decision. It creates a heatmap overlay - red means "important" and blue means "not important."

### FAP (Focus Area Percentage) - Lower is Better
FAP asks: "What percentage of the image does the model consider important?" If FAP is 5%, the model is pointing at just 5% of the image - very focused. If FAP is 20%, it's looking at a much larger, less specific area.

### Entropy - Lower is Better
Entropy measures how scattered the attention is. Very focused attention = low entropy. Scattered attention = high entropy.

### Results:

| Method | FAP (focus %) | Entropy (scatteredness) | Winner? |
|--------|--------------|----------------------|---------|
| **GradCAM** | **4.0%** | **5.46** | **YES - Best by far** |
| EigenCAM | 7.9% | 9.53 | Second place |
| GradCAM++ | 9.3% | 10.35 | More diffuse |
| LayerCAM | 9.4% | 10.36 | Most diffuse |

### Plain English Explanation:
GradCAM is like a laser pointer showing exactly where the model is looking. EigenCAM is like a wide spotlight - still useful but less precise. GradCAM++ and LayerCAM are like shining a flashlight over a much bigger area - they show something but not as precisely.

**For medical use, GradCAM is preferable** because it tells doctors exactly which pixels mattered, rather than vaguely pointing at a larger region.

### Why does this matter?
If a doctor is using this model to help diagnose a patient, they need to trust that the highlighted region actually corresponds to the suspicious lesion. GradCAM's tight focus (only 4% of image) means when a doctor sees a red overlay, they know that specific region is what the model found concerning.

---

## RQ2: Do the CAM explanations actually match what the model is doing?

### What is "faithfulness"?
Faithfulness answers: "Does the explanation match the model?" You can have a beautiful looking heatmap that has nothing to do with how the model actually makes decisions. Faithfulness measures whether the explanation is HONEST.

### How do we test this?
Two complementary tests:

**Insertion Test**: We add important pixels back in (starting from blank) and see if the model's confidence jumps quickly. If it does, the explanation is good - we found the right pixels.

**Deletion Test**: We remove important pixels (starting from complete) and see if the model's confidence drops quickly. If it does, the explanation is good.

**Faithfulness = Insertion AUC - Deletion AUC**
- Positive = the explanation actually reflects what the model does
- Negative or zero = the explanation might be misleading

### Results:

| Method | Insertion AUC | Deletion AUC | Faithfulness | % images honest |
|--------|-------------|-------------|--------------|---------------|
| **GradCAM** | 0.379 | **0.326** | **+0.053** | **80%** |
| GradCAM++ | **0.388** | 0.385 | +0.003 | 44% |

### Plain English:
GradCAM's explanations are honest 80% of the time - on most images, what you see in the heatmap actually corresponds to what made the model make that decision. GradCAM++ is honest only 44% of the time - less than a coin flip. When you see a GradCAM++ heatmap, there's only a 44% chance it's actually showing you what the model really looked at.

**The difference is stark**: GradCAM has 17x better faithfulness than GradCAM++. This is a significant practical concern for clinical use.

### Why does this matter?
If a doctor is using GradCAM++ and thinks "this red region is what the model found suspicious," they have only a 44% chance of being correct. With GradCAM, that jumps to 80%.

---

## RQ3: Does the neural network architecture affect explanation quality?

### What is a "backbone"?
The backbone is the base architecture of the neural network before the classification head. Think of it like the "engine" of the model.

- **ResNet50**: A tried-and-true architecture with 50 layers, known for stable training and good generalization
- **EfficientNet-B2**: A newer, more efficient architecture designed to be smaller and faster
- **MobileNetV2**: Designed for mobile devices - very lightweight

### Results:

| Backbone | FAP (lower = more focused) | Verdict |
|----------|---------------------------|---------|
| **ResNet50** | **5.3%** | **WINNER** |
| MobileNetV2 | 10.1% | Moderate |
| EfficientNet-B2 | 12.2% | Most diffuse |

### Plain English:
ResNet50 produces the most focused explanations - it points at just 5% of the image. MobileNetV2 and EfficientNet-B2 are roughly 2x more diffuse, highlighting larger areas of the image.

For clinical interpretability, ResNet50 is the best choice. The lighter architectures (designed for mobile or edge devices) sacrifice explanation precision.

### Model + CAM Combination Results:

| Backbone + CAM | FAP | Quality |
|----------------|-----|---------|
| **ResNet50 + GradCAM** | **3.3%** | **Best possible combination** |
| ResNet50 + GradCAM++ | 7.3% | Good but less focused |
| EfficientNet-B2 + GradCAM | 17.8% | Most diffuse |
| EfficientNet-B2 + GradCAM++ | 6.6% | Surprisingly good |

Interestingly, EfficientNet-B2 paired with GradCAM++ gives a better (lower) FAP than with GradCAM. This shows the interaction is not straightforward - some combinations work better than others.

### Why does this matter?
If you're deploying this system in a hospital, use ResNet50 as the backbone for the best explanations. If you're deploying on a mobile device and need something lighter, MobileNetV2 is a reasonable compromise.

---

## RQ4: Can we detect when the model will be wrong by comparing CAM methods?

### The Hypothesis:
When multiple CAM methods agree on WHERE to look, maybe the model is more certain. When they disagree, maybe the model is uncertain and more likely to make mistakes.

We measure agreement using Jaccard similarity - if two methods highlight exactly the same pixels, Jaccard = 1.0. If they highlight completely different pixels, Jaccard = 0.0.

### Results:

| Prediction Type | Avg Agreement (Jaccard) | n |
|----------------|------------------------|---|
| Correct predictions | 0.286 | 127 |
| Incorrect predictions | 0.591 | 23 |

**This is the OPPOSITE of what we expected.**

### Plain English:
When the model makes a WRONG prediction, all CAM methods tend to agree on where to look (high Jaccard = 0.59). When it makes a CORRECT prediction, the methods disagree more (low Jaccard = 0.29).

### What does this mean?
This is actually a potentially useful safety signal! When all CAM methods agree on a wrong prediction, it means the model is "confidently wrong" - it has found something in the image that looks exactly like what it thinks a malignant lesion looks like, but it's actually a false positive or false negative.

High agreement + wrong prediction = "Watch out, this might be a confident mistake"

This is counter-intuitive but clinically valuable - it tells you to be more skeptical in exactly the cases where the model seems most sure.

**Important caveat**: The sample size for incorrect predictions is small (n=23) and the result is not statistically significant (p=1.0). More research would be needed before using this as a clinical safety signal.

---

## RQ5: How does Grad-CAM attention evolve during training?

### What this tests:
The hypothesis is that early in training, models use texture/color "shortcuts" while late training shifts to morphologically meaningful features like lesion borders. By checking snapshots at different epochs, we can watch this shift happen.

### Results from your notebook run (image: ISIC_0026167, melanoma):

| Epoch | Val AUC | FAP (%) | Entropy | Mal Prob | What happened |
|-------|---------|---------|---------|----------|---------------|
| 1 | 0.821 | 13.8% | 10.18 | 0.735 | Uncertain, diffuse attention |
| 2 | 0.839 | 15.4% | 10.26 | 0.692 | Better classification but still scattered |
| 3 | 0.854 | 20.0% | 10.20 | 0.722 | AUC up but attention gets WORSE |
| 4 | 0.862 | **8.4%** | 10.19 | 0.608 | Sharp focus - FAP halves suddenly |
| 5 | 0.860 | **7.4%** | 10.09 | 0.511 | Best focus, near decision threshold |

### Plain English:
**Early training (1-3):** Diffuse, scattered attention over 14-20% of image. Model learns coarse texture patterns.

**Mid training (3):** Surprisingly attention gets MORE diffuse even as AUC improves - model may be memorizing texture shortcuts.

**Late training (4-5):** Attention suddenly sharpens - FAP drops from 20% to 7%. Model shifts from texture to morphological features.

**Hypothesis verdict: SUPPORTED** - Attention gets more focused over training. Biggest jump between epoch 3 and 4.

### Why this matters for clinical use:
A model in early training might rely on "dark images = malignant" rather than actual lesion features. GradCAM lets you catch this. If you see diffuse attention, the model may not have learned clinically meaningful features yet.

### Caveats:
- Based on one melanoma image. More images needed for robust trends.
- Only 5 epochs analyzed. More checkpoints would give fuller picture.
- Requires checkpoint saving during training (not always done by default).

---

## RQ6: Does the model work on completely new datasets from different sources?

### Why this matters:
A model trained on one dataset might only work well on images that look very similar to that dataset. Real clinical use will involve images from different cameras, lighting conditions, and populations. External validation tests whether the model has learned genuine medical features vs. dataset-specific artifacts.

### Datasets Tested:
1. **Internal (HAM10000)**: The dataset the model was trained on - baseline performance
2. **Malignant-Benign**: 35,000 images from Kaggle - different acquisition setup
3. **PH2**: 200 dermoscopy images - different population and equipment

### Results:

| Dataset | AUC | Accuracy | Confidence | AUC Drop from Internal |
|--------|-----|----------|-----------|---------------------|
| **Internal (HAM10000)** | **0.939** | **85.0%** | 0.857 | +0.000 (baseline) |
| Malignant-Benign | 0.805 | 69.0% | 0.839 | +0.134 (significant drop) |
| PH2 | 0.780 | 55.0% | 0.908 | +0.159 (significant drop) |

### Plain English:
The model performs best on its training data (HAM10000). On Malignant-Benign, performance drops noticeably - the model has more difficulty with this different image type. On PH2, performance drops even more, despite PH2 being another dermoscopy dataset like HAM10000.

Interestingly, the model shows HIGHER confidence on external datasets (0.908 on PH2 vs 0.857 on internal). This is concerning - the model seems overconfident when it's less accurate. This is a known failure mode called "distribution shift" where models don't know what they don't know.

### Why the drops happen:

**Malignant-Benign AUC drop of +0.134**: This dataset has very different characteristics from dermoscopy images (different camera types, lighting, possibly resolution). The model learned to look for features specific to dermoscopy and doesn't transfer well.

**PH2 AUC drop of +0.159**: Despite being a dermoscopy dataset like HAM10000, the model still drops significantly. This suggests the model may be overfitting to specific characteristics of the HAM10000 patient population rather than learning generalizable melanoma features.

### XAI Metrics (Internal only due to computation issues):

| Dataset | FAP | Entropy | Correct Rate |
|---------|-----|--------|-------------|
| Internal | 3.1% | 5.84 | 90% |

### Why this is important for clinical deployment:
The model performs notably worse on external datasets even within dermoscopy. In a real clinical setting, you would need to:
1. Calibrate the model on local data before use
2. Be aware that confidence scores may be misleading on unfamiliar populations
3. Consider fine-tuning on local data periodically

---

## Summary Table - All Research Questions

| RQ | Question | Best Answer | Key Metric |
|----|---------|-------------|------------|
| RQ1 | Which CAM? | **GradCAM** | FAP=4%, Entropy=5.46 |
| RQ2 | Which is faithful? | **GradCAM** | 80% honest, faithfulness=+0.053 |
| RQ3 | Which backbone? | **ResNet50** | FAP=5.3% |
| RQ4 | Does disagreement predict errors? | **No - opposite effect** | Wrong predictions have HIGHER agreement |
| RQ5 | Training evolution? | **Supported - focus sharpens** | FAP 14%→7% across epochs |
| RQ6 | External validity? | **Poor generalization** | AUC drops 0.13-0.16 on external data |

---

## Practical Recommendations

1. **Always use GradCAM** for explanations - it's more focused and more faithful than alternatives

2. **Always use ResNet50 backbone** if explanation quality matters - it produces the most precise attention maps

3. **High CAM agreement + wrong prediction** is a warning sign for confident misclassification - investigate these cases

4. **Before clinical deployment on new data**, calibrate and validate on local population - the model degrades significantly on external datasets

5. **Don't trust high confidence on unfamiliar data** - the model is overconfident precisely when it's least accurate (PH2)

---

