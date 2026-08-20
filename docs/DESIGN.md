# Design Notes

This document goes over the design of the composition analyzer and the reasoning behind the main decisions.
The README has the results, this is more about how the system is put together and why.

## The problem

The goal is to take a photo and predict which composition techniques it uses.
The important thing is that this is multi-label, not multi-class.
A photo can be rule of thirds and diagonal at the same time, so the model can't be forced to pick one.
That means one sigmoid output per class instead of a softmax, and binary cross entropy style losses instead of categorical.

There are 14 labels total: center, rule of thirds, golden ratio, triangle, horizontal, vertical, diagonal, symmetric, curved, radial, vanishing point, pattern, fill the frame, and none.
The "none" label is for photos where the annotators didn't think any rule clearly applied.

## Dataset

I used CADB (Composition Assessment Database), which has 9,497 photos.
The main reason I picked it is that it actually has composition class annotations that match the problem, most photo datasets only have aesthetic scores or scene categories.

The composition classes live in composition_elements.json.
The format is a dict from image filename to another dict, where the keys are the class names and the values are the annotated line segments or boxes for that class.
I only use the keys (the class names) for training.
The element geometry could be useful later for something like weak localization supervision but I didn't go there.

One thing worth knowing is the class distribution is really skewed.
In the training split, center has 3,902 positives and radial has 32.
The "none" class only has 15 training examples, so any metric on it is basically noise.
This imbalance drove a lot of the loss function decisions (see the model decisions doc).

## Splits

CADB ships with an official train/test split, so I used their test set (950 images) untouched.
For validation I carve 10 percent out of the training set.
Instead of a random shuffle I hash each image id with a fixed seed and bucket it.
The nice thing about hashing is the split is stable no matter what order the data loads in, and adding or removing images doesn't reshuffle everybody else.
Final counts are 7,648 train / 899 val / 950 test.

The test set is only touched once per experiment, after thresholds are already picked on validation.
The evaluate script actually refuses to run threshold selection on the test split so I can't do it by accident.

## Data pipeline

The pipeline is tf.data: read file, decode, resize to 224x224, batch, prefetch.
Pixels stay in 0 to 255 because the Keras EfficientNet has its own normalization layer inside, so normalizing twice would be wrong.

Augmentation is done with Keras layers inside the model, so it's automatically on during training and off at inference.
I use horizontal flips, very small rotations, and mild brightness/contrast changes.
The important constraint is no random cropping.
Cropping is the standard augmentation everywhere else, but here it would actually change the label.
If you crop a rule of thirds photo, the subject moves in the frame and it might become a center composition, so the augmentation would be teaching the model wrong things.
Flips are safe because none of the 14 classes care about left versus right.
I kept rotations tiny (about 3 degrees max) because big rotations would mess with the horizontal and vertical classes.

## Thresholds

A sigmoid output needs a threshold to become a yes/no prediction, and 0.5 is not a good default when classes are this imbalanced.
So after training I sweep thresholds from 0.05 to 0.95 per class on the validation set and keep whichever maximizes that class's F1.
The thresholds get saved to a json next to the model, and inference and the API load them.
This gave a real improvement on macro F1 compared to a flat 0.5.
The downside is that for rare classes the threshold is picked from a handful of validation positives, so it's noisy, which showed up when some rare-class thresholds didn't generalize great to test.

## Serving

The serving layer is deliberately small.
FastAPI with two endpoints: /api/predict takes an uploaded image and returns the per-class probabilities, thresholds, and which classes cleared them, and /api/gradcam takes an image plus a class name and returns a heatmap PNG.
The model loads once at startup, not per request.

The frontend is one static HTML file with no build step.
It does drag and drop upload, shows the photo, confidence bars for all 14 classes, a rule of thirds grid overlay you can toggle, and a dropdown to pick a class for Grad-CAM.
I thought about React but for one page with two fetch calls it wasn't worth the tooling.

Uploads are validated (JPEG/PNG only, 20 MB cap) and the image is decoded server side with Pillow, so weird files get rejected with a proper HTTP error instead of crashing the model.

## Artifacts and experiments layout

Every experiment writes into its own directory: artifacts/ for the main model, artifacts_focal/, artifacts_frozen/, artifacts_dinov2/ and so on.
Each one contains the model file, labels.json, thresholds.json, and the evaluation reports (json and markdown) plus plots.
Keeping them separate means experiments can't clobber each other and the comparison script just reads report_test.json from each folder.

The dataset and all artifacts directories are gitignored, only code, docs, and the README images get committed.

## Environment notes

Everything runs in a Python 3.12 venv.
TensorFlow is pinned to 2.19 because the tensorflow-metal plugin (which gives GPU acceleration on Apple Silicon) doesn't load against newer versions.
That pin matters, without it training falls back to CPU and takes way longer.
PyTorch is in the requirements only for the DINOv2 embedding extraction, the two frameworks don't interact otherwise.
