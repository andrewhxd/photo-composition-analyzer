"""Grad-CAM explanations for the EfficientNet composition model.

Answers "which image regions drove this composition prediction?" by
back-propagating a class score to the last convolutional feature map
(EfficientNetB0's top_activation) and weighting its channels by the
pooled gradients.

Usage:
    python -m src.gradcam photo.jpg rule_of_thirds -o heatmap.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .data import LABELS


class GradCAM:
    def __init__(self, model=None, model_dir: str = "artifacts", image_size: int = 224):
        import keras

        if model is None:
            model = keras.models.load_model(
                os.path.join(model_dir, "model.keras"), compile=False
            )
        self.image_size = image_size
        # Augmentation layers are identity at inference, so running the
        # backbone directly on the resized image reproduces the model output.
        backbone = model.get_layer("efficientnetb0")
        conv_output = backbone.get_layer("top_activation").output
        self.grad_model = keras.Model(backbone.inputs, [conv_output, backbone.output])
        self.head = model.get_layer("composition_probs")

    def heatmap(self, rgb: np.ndarray, label: str) -> np.ndarray:
        """Compute a [0, 1] Grad-CAM map at the input image's resolution.

        rgb: HxWx3 uint8 array. label: one of LABELS.
        """
        import cv2
        import tensorflow as tf

        class_idx = LABELS.index(label)
        resized = cv2.resize(
            rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        batch = tf.constant(resized.astype(np.float32)[None, ...])

        with tf.GradientTape() as tape:
            conv_out, pooled = self.grad_model(batch, training=False)
            probs = self.head(pooled)
            score = probs[:, class_idx]
        grads = tape.gradient(score, conv_out)

        channel_weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.einsum("bhwc,bc->bhw", conv_out, channel_weights)[0]
        cam = tf.nn.relu(cam).numpy()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        # Cubic interpolation overshoots outside [0, 1], which would wrap
        # around when converted to uint8 for the colormap.
        return np.clip(cam, 0.0, 1.0)

    def overlay(self, rgb: np.ndarray, label: str, alpha: float = 0.45) -> np.ndarray:
        """Blend the heatmap over the image; returns an RGB uint8 array."""
        import cv2

        cam = self.heatmap(rgb, label)
        colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        # Fade the colormap in with heat so calm regions keep the photo visible.
        weight = (alpha * cam)[..., None]
        blended = rgb.astype(np.float32) * (1 - weight) + colored.astype(np.float32) * weight
        return blended.astype(np.uint8)


def main(argv=None):
    import cv2

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="Image file to explain")
    p.add_argument("label", choices=LABELS, help="Composition class to explain")
    p.add_argument("-o", "--output", default=None,
                   help="Output path (default: <image>_gradcam_<label>.png)")
    p.add_argument("--model-dir", default="artifacts")
    args = p.parse_args(argv)

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    overlay = GradCAM(model_dir=args.model_dir).overlay(rgb, args.label)
    out = args.output or (
        os.path.splitext(args.image)[0] + f"_gradcam_{args.label}.png"
    )
    cv2.imwrite(out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM for '{args.label}' -> {out}")


if __name__ == "__main__":
    main()
