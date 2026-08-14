# Photography Composition Analyzer

*Practical ML project specification focused on photographic composition-technique detection*

| **PRIMARY STACK**                  | **CORE DATASET**    | **TARGET SCOPE**                    |
|------------------------------------|---------------------|-------------------------------------|
| TensorFlow / Keras, Python, OpenCV | CADB (9,497 images) | Resume-ready multi-label vision MVP |

# 1. Project Overview

Goal. Build a computer-vision application that analyzes an uploaded
photograph and predicts which photographic composition techniques are
present. The project is intentionally focused on composition rather than
trying to classify every property of a photo.

The core ML problem is multi-label classification. A single photograph
can use several composition techniques at the same time, such as rule of
thirds, diagonals, symmetry, and a vanishing point.

| **Input**                | **Output**                                                  | **ML formulation**         |
|--------------------------|-------------------------------------------------------------|----------------------------|
| Photograph               | Rule of thirds: 0.84; diagonal: 0.71; vanishing point: 0.63 | Multi-label classification |
| Optional later extension | Overall composition / aesthetic score                       | Regression or ranking      |

## Recommended MVP

Use one pretrained EfficientNet backbone with a single multi-label
output head. Train it on CADB composition annotations, evaluate each
composition class, then expose the trained model through a simple upload
interface that shows the predicted techniques and confidence scores.

## What is intentionally out of scope

- Scene or genre classification such as landscape, portrait,
  architecture, or wildlife.

- Photo search, facial recognition, and object identification.

- Automatic photo editing or crop suggestions.

- Aesthetic scoring until the composition classifier is working and
  evaluated.

# 2. Dataset and Labels

Primary dataset: CADB. The Image Composition Assessment Database
contains 9,497 images and provides composition-related annotations,
including common composition classes. This makes it a strong starting
point because the labels directly match the project goal.

## Composition labels to predict

| **Group**            | **Labels**                                               | **Why it matters**                               |
|----------------------|----------------------------------------------------------|--------------------------------------------------|
| Placement / balance  | center, rule of thirds, golden ratio, symmetric          | Where major subjects or visual weight are placed |
| Geometry / direction | triangle, horizontal, vertical, diagonal, curved, radial | Dominant visual structure and directional flow   |
| Depth / structure    | vanishing point                                          | Perspective and depth cues                       |
| Pattern / framing    | pattern, fill the frame                                  | Repeated structure or tight subject framing      |
| Fallback             | none                                                     | No listed composition rule is strongly present   |

## Why multi-label matters

Do not force the model to choose one technique. A single image might be
labeled both rule of thirds and diagonal, or symmetric and vanishing
point. The model should output one probability per technique using
sigmoid activations.

## Optional second dataset: AADB

AADB can be used later if you want to expand beyond composition into
broader photographic attributes such as shallow depth of field, motion
blur, lighting, color harmony, object emphasis, repetition, rule of
thirds, and symmetry. Keep this outside the MVP.

# 3. System Design

## End-to-end flow

| **Stage**        | **What happens**                                                        |
|------------------|-------------------------------------------------------------------------|
| Dataset          | Load CADB images and composition labels                                 |
| Preprocessing    | Resize and normalize each image; apply light training augmentation      |
| Vision backbone  | EfficientNet extracts visual features from the photograph               |
| Composition head | A sigmoid output predicts a probability for every composition technique |
| Inference        | Apply per-class thresholds and return the most likely techniques        |
| Demo             | Display the uploaded image, predicted labels, and confidence bars       |

## Model architecture

| **Component** | **Recommended choice**                                    | **Reason**                                                           |
|---------------|-----------------------------------------------------------|----------------------------------------------------------------------|
| Backbone      | EfficientNetB0 pretrained on ImageNet                     | Strong transfer-learning baseline and small enough to train quickly  |
| Output head   | Dense layer with one sigmoid output per composition label | Allows multiple composition techniques at once                       |
| Loss          | Binary cross-entropy                                      | Standard objective for multi-label classification                    |
| Optimizer     | Adam                                                      | Simple, reliable starting point for transfer learning                |
| Image size    | 224 x 224                                                 | Keeps training practical while preserving enough spatial information |

## Why transfer learning

CADB is large enough to fine-tune a pretrained vision model but not
large enough to justify training a modern CNN from scratch. Start with
ImageNet-pretrained weights, train only the new output layer, then
unfreeze the last part of EfficientNet for a short fine-tuning stage.

# 4. Training Pipeline

1.  Download CADB and parse the composition annotations into one table
    keyed by image ID.

2.  Convert each image's composition labels into a multi-hot target
    vector, for example \[0, 1, 0, 1, ...\].

3.  Create a deterministic train, validation, and test split at the
    image level. Keep the test set untouched until final evaluation.

4.  Resize images to the backbone input size and use the matching Keras
    preprocessing function.

5.  Apply light augmentation such as horizontal flips when appropriate,
    small rotations, and modest brightness or contrast changes. Avoid
    aggressive crops that can change the composition itself.

6.  Load EfficientNetB0 with pretrained ImageNet weights and replace its
    original classifier with the multi-label composition head.

7.  Train the new head with the backbone frozen, then unfreeze the last
    several backbone blocks and fine-tune with a lower learning rate.

8.  Save the best validation checkpoint and evaluate it once on the
    held-out test set.

## Suggested data representation

| **image_id** | **composition vector**    | **image_path**   |
|--------------|---------------------------|------------------|
| 10482        | \[0, 1, 0, 0, 0, 1, ...\] | images/10482.jpg |
| 10483        | \[1, 0, 0, 1, 0, 0, ...\] | images/10483.jpg |

## Handling label imbalance

- Measure the frequency of every composition label before changing the
  loss function.

- If rare techniques are consistently ignored, try per-class weights or
  weighted binary cross-entropy.

- Do not rely on ordinary accuracy. A multi-label model can look
  accurate by predicting common negatives.

# 5. Evaluation

The evaluation should answer one question: how well does the model
identify composition techniques on photographs it has not seen before?
Because the task is multi-label, report metrics that account for both
common and rare techniques.

| **Metric**                       | **Use**                                                                  |
|----------------------------------|--------------------------------------------------------------------------|
| Macro F1                         | Weights each composition technique equally, so rare labels matter        |
| Micro F1                         | Measures overall multi-label precision and recall across all predictions |
| Mean Average Precision (mAP)     | Evaluates ranking quality across confidence scores and labels            |
| Per-class precision / recall     | Shows which techniques the model detects well and which it confuses      |
| Confusion / co-occurrence review | Useful for understanding labels that often appear together               |

## Baseline experiment

- Baseline A: freeze the entire pretrained EfficientNet backbone and
  train only the composition head.

- Model B: fine-tune the last part of EfficientNet with a lower learning
  rate.

- Compare validation and test F1 / mAP to show whether fine-tuning
  improves composition recognition.

- Include correct predictions, ambiguous predictions, and failure cases
  in the README.

## Threshold selection

A default 0.50 threshold is easy to start with, but composition labels
may have different class frequencies and confidence distributions. Use
the validation set to choose one threshold per class if that noticeably
improves F1. Store those thresholds with the model for inference.

# 6. Simple Application

Keep the product layer small. The app only needs to prove that the
trained model can be loaded and used on a new photograph.

| **Screen / component** | **Contents**                                                   |
|------------------------|----------------------------------------------------------------|
| Upload                 | Drag-and-drop or file picker for JPG / PNG                     |
| Analysis               | Photo preview plus the top predicted composition techniques    |
| Confidence display     | Probability bar or percentage for each predicted technique     |
| Optional overlay       | Rule-of-thirds grid or simple geometric guide when appropriate |
| Model note             | Short disclaimer that composition labels can be subjective     |

## Recommended stack

| **Layer**        | **Tool**                              | **Scope**                                    |
|------------------|---------------------------------------|----------------------------------------------|
| Training         | TensorFlow / Keras                    | Transfer learning and multi-label classifier |
| Image processing | Python + OpenCV / Pillow              | Resize, validation, optional overlays        |
| Inference API    | FastAPI                               | POST image and return JSON predictions       |
| Frontend         | Simple React / Next.js page, optional | Upload image and display results             |

# 7. Build Plan

| **Milestone** | **Deliverable**                     | **Done when...**                                              |
|---------------|-------------------------------------|---------------------------------------------------------------|
| 1\. Data      | Download + annotation parser        | You can load an image and its multi-label composition target  |
| 2\. Baseline  | Frozen-backbone model               | Training / validation curves and first F1 / mAP metrics exist |
| 3\. Fine-tune | Improved model                      | Best checkpoint is evaluated on the test set                  |
| 4\. Inference | Single-image prediction script      | Any local JPG returns composition labels and confidences      |
| 5\. Demo      | Upload UI / API                     | A reviewer can test the model without opening notebooks       |
| 6\. Overlay   | Composition visualization, optional | Selected rules can be illustrated on the image                |

## Recommended stopping point

The project is complete enough for a resume once you have a trained
composition model, trustworthy test metrics, inference on new photos,
and a simple demo. Do not add more features just to make the project
sound larger.

# 8. What Makes the Project Resume-Worthy

- It uses a real composition-assessment dataset rather than a toy
  classification dataset.

- It demonstrates transfer learning with a modern computer-vision model.

- It uses multi-label learning, which matches the fact that several
  photographic techniques can coexist in one image.

- It requires appropriate evaluation with F1, mAP, and per-class metrics
  instead of a single accuracy number.

- It can be deployed as a small inference service and demonstrated
  visually in a browser.

## Draft resume entry after implementation

**Photography Composition Analyzer \| TensorFlow, Python, OpenCV**

- Fine-tuned an EfficientNet vision model on 9,000+ photographs to
  identify composition techniques including rule of thirds, symmetry,
  diagonals, golden ratio, and vanishing points.

- Developed a multi-label classification pipeline for overlapping
  composition techniques and evaluated performance using macro / micro
  F1 and mean average precision.

- Built an image inference application that analyzes uploaded
  photographs and returns predicted composition techniques with
  confidence scores.

Important: replace the sample wording with the exact dataset size,
architecture, metrics, and features you actually implement.

# 9. Optional Extensions

| **Extension**            | **What it adds**                                                                  | **Priority**                    |
|--------------------------|-----------------------------------------------------------------------------------|---------------------------------|
| Grad-CAM explainability  | Heatmaps showing which image regions influenced a composition prediction          | Good after MVP                  |
| Geometric overlays       | Rule-of-thirds grid, diagonal guides, or approximate vanishing-line visualization | Good demo polish                |
| Aesthetic score          | Predict an overall composition-quality / aesthetic score                          | Separate follow-up task         |
| Broader technique labels | Depth of field, motion blur, lighting, color harmony, object emphasis             | Use AADB later                  |
| Photo search             | Search a photo library by detected composition techniques                         | Interesting extension, not core |

# 10. Suggested Repository Structure

```text
photo-composition-analyzer/
  data/                 # local dataset path; do not commit image dataset
  src/data.py           # annotation parsing and tf.data pipeline
  src/model.py          # EfficientNet backbone + multilabel head
  src/train.py          # training / checkpointing
  src/evaluate.py       # metrics and plots
  src/inference.py      # single-image prediction
  api/main.py           # optional FastAPI endpoint
  app/                  # optional frontend
  notebooks/            # exploration only; core logic stays in src/
  README.md             # examples, metrics, failure cases, setup
```

# 11. Sources and Dataset References

CADB - Image Composition Assessment Database (official repository)

https://github.com/bcmi/Image-Composition-Assessment-Dataset-CADB

AADB - Photo Aesthetics Ranking Network / dataset repository

https://github.com/aimerykong/deepImageAestheticsAnalysis
