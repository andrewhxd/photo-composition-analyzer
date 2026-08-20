# Model Decisions

This covers the modeling choices: the backbone, the training recipe, the loss functions, the DINOv2 comparison, and Grad-CAM.
Results tables are in the README, here I'm mostly explaining why things are the way they are and what I learned.

## Why EfficientNetB0

I wanted a pretrained CNN that transfers well and trains fast on a laptop GPU.
EfficientNetB0 is small (about 4M backbone params), has good ImageNet accuracy for its size, and Keras ships it with pretrained weights and the preprocessing built in.
With 9,497 images there's no way training from scratch makes sense, the dataset is way too small for that, so transfer learning was the obvious route.
A bigger backbone like B3 or a ResNet50 might squeeze out a bit more, but B0 kept the experiment loop fast (a full training run is about 13 minutes), and being able to iterate quickly mattered more to me than a point or two of mAP.

The head is deliberately boring: global average pooling (built into the backbone), dropout 0.3, then one dense layer with 14 sigmoid outputs.
I tried to resist adding more layers there, with this little data a fancy head mostly just overfits.

## Two-stage training

Training happens in two stages.

Stage 1 freezes the whole backbone and only trains the new head at learning rate 1e-3.
The point is that the head starts from random weights, and if the backbone were trainable from the start, the big random gradients from the head would trash the pretrained features before the head settles down.

Stage 2 unfreezes the last two EfficientNet blocks (block6 and block7 plus the top conv) and continues at 1e-4.
Only the last blocks because the early layers hold generic edge and texture features that transfer fine, it's the later, more semantic layers that need to adapt to composition.
The lower learning rate is for the same reason, fine-tuning should nudge the features, not rewrite them.

One detail that matters: the BatchNormalization layers stay frozen even in stage 2.
If you let BN update its running statistics on a smallish dataset the activations shift under the trained layers and validation performance falls apart.
This is a known EfficientNet fine-tuning gotcha and I hit it via the Keras docs rather than the hard way, but it's real.

Checkpointing keeps the single best model by validation PR-AUC, and early stopping (patience 4) cuts things off when validation stops improving.
In the final run stage 2 peaked at epoch 4 and early stopping restored those weights.
The fine-tuning stage was clearly worth it: validation PR-AUC went from about 0.44 frozen to 0.57 fine-tuned, and on test it's the difference between 0.474 and 0.561 mAP.

## Loss functions and class imbalance

Plain BCE has a problem here: with 14 outputs and most labels absent on most images, the loss is dominated by easy negatives.
The model can get a nice low loss by just never predicting radial or pattern, and early on that's exactly what happened.

The default loss is weighted BCE.
Each class gets a positive weight equal to its negative-to-positive ratio in the training split, capped at 10.
So center gets weight 1 (it's balanced-ish) and everything rare hits the cap.
The cap is there because the true ratio for radial would be over 200, and a weight that size makes training unstable, a handful of hard rare positives would dominate every batch.

The focal loss experiment uses the standard Lin et al. formulation with alpha 0.25 and gamma 2, through the built-in Keras BinaryFocalCrossentropy.
The idea is different from per-class weighting: instead of weighting classes by frequency, focal down-weights whatever examples the model already finds easy, whoever they are.

What actually happened is more interesting than either loss just winning.
On the fine-tuned CNN, focal matched weighted BCE on mAP (0.565 vs 0.561) but was clearly better on macro F1 (0.507 vs 0.474), with the gains concentrated in rare classes like radial and none.
But on the frozen DINOv2 head the exact same focal setup was clearly worse than weighted BCE (0.467 vs 0.513 mAP), and it specifically lost the rare classes it's supposed to help.
My interpretation is that focal's benefit comes from redirecting representation learning toward hard examples, and when the backbone is frozen there's no representation learning to redirect, so all you're left with is a weighting scheme that's less well matched to the class frequencies than the explicit per-class weights.
I didn't find this obvious in advance and it's probably the most useful single takeaway from the loss experiments.

## The DINOv2 comparison

The point of this experiment was to compare two kinds of pretrained representations on the same task: supervised ImageNet CNN features that get fine-tuned, versus self-supervised ViT features used frozen.

The setup is linear probing, more or less.
I run every image through frozen DINOv2 ViT-S/14 once and save the embeddings (the CLS token concatenated with the mean of the patch tokens, 768 dims total).
Then I train only a small head on those embeddings: either literally one dense layer (the linear probe) or one hidden layer of 512 with dropout (the MLP).
Because the embeddings are precomputed, a full head training run takes under a minute, which made it cheap to try the variants.
Everything else (splits, weighted BCE, threshold selection, metrics) is identical to the CNN experiments on purpose, otherwise the comparison wouldn't mean much.

Results in short:

- Frozen DINOv2 (0.513 mAP) beats frozen EfficientNet (0.474) comfortably, so the self-supervised features really are better out of the box.
- It still loses to the fine-tuned CNN (0.561), so on this dataset being able to adapt the backbone beats having better frozen features.
- The linear probe (0.509) is nearly as good as the MLP (0.513) on mAP, so the DINOv2 embedding is close to linearly separable for these labels, the MLP mostly helps macro F1.
- DINOv2 wins specific classes that depend on whole-frame geometry: radial (AP 0.671 vs 0.468), vanishing point, symmetric.
  That pattern fits the intuition that ViT attention sees global structure that a CNN has to build up through many layers.
- Scaling the frozen backbone to ViT-B/14 did not help (0.500 vs 0.513).
  I expected at least a small gain here and didn't get one.
  My best guess is that with only ~7,600 training images the head is the bottleneck, not the representation, so extra embedding width just adds parameters to overfit with.

If I pushed this further the next step would be actually fine-tuning DINOv2 rather than probing it, but that's a much heavier training job and I wanted to keep the comparison clean (frozen vs fine-tuned is the interesting axis).

## Grad-CAM

Grad-CAM answers "which part of the image made the model say rule of thirds".
The implementation takes the gradient of one class's score with respect to the last convolutional feature map (top_activation in EfficientNetB0, a 7x7x1280 tensor), averages the gradients per channel to get channel importances, then takes the weighted sum of the channels and a ReLU.
That gives a 7x7 heat map which gets upsampled onto the original image.

Two implementation notes.
First, the augmentation layers sit in front of the backbone in my model, but they're identity at inference, so the Grad-CAM graph can run the backbone directly on the resized image and the probabilities come out the same as the normal predict path.
Second, upsampling the 7x7 map with cubic interpolation overshoots the 0 to 1 range slightly, and if you cast that straight to uint8 for the colormap it wraps around and the hottest pixels come out as the coldest color.
That one produced a genuinely confusing image before I found it, the fix is just clipping to 0..1 after the resize.

The sanity checks look right: for a seascape the horizontal explanation lights up exactly along the horizon, which is what you'd hope the model is using.

## Things I would try next

- Higher input resolution.
  Curved compositions have the worst recall of the common classes and I suspect thin leading lines just vanish at 224x224.
- Something about golden ratio.
  Every model fails on it (AP around 0.09) and it's heavily confused with rule of thirds.
  Honestly the labels themselves may not be separable, the two rules place subjects in nearly the same spots, so this might be a label problem rather than a model problem.
- Fine-tuning DINOv2 end to end for a fair "best ViT vs best CNN" comparison.
- Using the element geometry annotations (the lines and boxes CADB provides) as auxiliary supervision instead of throwing them away.
- Calibrating the probabilities, right now the thresholds absorb the calibration problem but the raw sigmoid outputs run hot for common classes.
