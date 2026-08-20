"""EfficientNetB0 backbone with a multi-label composition head."""

from __future__ import annotations

import keras
import tensorflow as tf

from .data import NUM_LABELS


def build_model(
    num_labels: int = NUM_LABELS,
    image_size: int = 224,
    dropout: float = 0.3,
    augment: bool = True,
) -> keras.Model:
    """EfficientNetB0 (ImageNet weights, frozen) + sigmoid multi-label head.

    Expects inputs in [0, 255]; EfficientNet's internal rescaling handles
    normalization. Augmentation layers are active only during training, and
    deliberately avoid crops that could change the composition itself.
    """
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs
    if augment:
        x = keras.layers.RandomFlip("horizontal", name="aug_flip")(x)
        x = keras.layers.RandomRotation(0.01, fill_mode="nearest", name="aug_rotation")(x)
        x = keras.layers.RandomBrightness(0.1, value_range=(0.0, 255.0), name="aug_brightness")(x)
        x = keras.layers.RandomContrast(0.1, name="aug_contrast")(x)

    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", pooling="avg"
    )
    backbone.trainable = False
    x = backbone(x)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_labels, activation="sigmoid", name="composition_probs")(x)
    return keras.Model(inputs, outputs, name="composition_analyzer")


def get_backbone(model: keras.Model) -> keras.Model:
    for layer in model.layers:
        if layer.name.startswith("efficientnet"):
            return layer
    raise ValueError("EfficientNet backbone not found in model")


def unfreeze_top_blocks(model: keras.Model, num_blocks: int = 2) -> int:
    """Unfreeze the last `num_blocks` EfficientNet blocks (plus the top conv).

    BatchNormalization layers stay frozen: updating their statistics on a small
    dataset destabilizes fine-tuning.
    """
    backbone = get_backbone(model)
    backbone.trainable = True

    # EfficientNetB0 blocks are named block1a..block7a; unfreeze from the end.
    all_block_ids = [f"block{i}" for i in range(1, 8)]
    unfrozen_ids = all_block_ids[len(all_block_ids) - num_blocks:]
    prefixes = tuple(unfrozen_ids) + ("top_",)

    trainable_count = 0
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        elif layer.name.startswith(prefixes):
            layer.trainable = True
            trainable_count += 1
        else:
            layer.trainable = False
    return trainable_count


class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Binary cross-entropy with a per-class weight on positive examples.

    Rare composition classes (radial, none, pattern) are outnumbered ~100:1 by
    negatives; without weighting the model can minimize loss by never
    predicting them.
    """

    def __init__(self, pos_weight, name="weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = tf.constant(pos_weight, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = keras.config.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss = -(
            self.pos_weight * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config["pos_weight"] = self.pos_weight.numpy().tolist()
        return config


def compile_model(
    model: keras.Model,
    learning_rate: float,
    pos_weight=None,
    loss_name: str = "weighted_bce",
) -> None:
    if loss_name == "focal":
        # alpha/gamma follow Lin et al. 2017; class balancing is global here,
        # unlike the per-class weights of weighted BCE.
        loss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True, alpha=0.25, gamma=2.0
        )
    elif pos_weight is not None:
        loss = WeightedBinaryCrossentropy(pos_weight)
    else:
        loss = keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
        metrics=[
            keras.metrics.AUC(name="auc_pr", curve="PR", multi_label=True),
            keras.metrics.BinaryAccuracy(name="bin_acc"),
        ],
    )
