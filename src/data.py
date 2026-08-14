"""CADB annotation parsing and tf.data input pipeline.

The CADB dataset (https://github.com/bcmi/Image-Composition-Assessment-Dataset-CADB)
provides per-image composition class annotations. Each image is annotated with one
or more of 13 composition classes, or "none" when no rule is clearly present.
This module turns those annotations into multi-hot vectors and builds tf.data
pipelines for training and evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical label order used everywhere: annotation parsing, the model head,
# metrics, thresholds, and the API response.
LABELS = [
    "center",
    "rule_of_thirds",
    "golden_ratio",
    "triangle",
    "horizontal",
    "vertical",
    "diagonal",
    "symmetric",
    "curved",
    "radial",
    "vanishing_point",
    "pattern",
    "fill_the_frame",
    "none",
]

NUM_LABELS = len(LABELS)

# Raw annotation names vary in spacing/case between files; normalize to LABELS.
_CANONICAL = {label.replace("_", ""): label for label in LABELS}


def canonical_label(raw: str) -> str | None:
    key = raw.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return _CANONICAL.get(key)


def _labels_from_annotation(value) -> set[str]:
    """Extract composition class names from one image's annotation entry.

    Handles the formats found in CADB: a list of class names, a dict keyed by
    class name (values are element geometry), or a single string.
    """
    if isinstance(value, dict):
        raw_names = list(value.keys())
    elif isinstance(value, (list, tuple)):
        raw_names = [v for v in value if isinstance(v, str)]
        # Lists of dicts, e.g. [{"class": "...", "elements": [...]}].
        for v in value:
            if isinstance(v, dict):
                for k in ("class", "category", "label", "type"):
                    if k in v and isinstance(v[k], str):
                        raw_names.append(v[k])
    elif isinstance(value, str):
        raw_names = [value]
    else:
        raw_names = []

    found = set()
    for raw in raw_names:
        label = canonical_label(raw)
        if label is not None:
            found.add(label)
    return found


def load_annotations(data_root: str, annotation_file: str = "composition_elements.json") -> pd.DataFrame:
    """Parse CADB composition class annotations into a DataFrame.

    Returns a DataFrame with columns: image_id, image_path, and one 0/1 column
    per label in LABELS.
    """
    ann_path = os.path.join(data_root, annotation_file)
    with open(ann_path) as f:
        annotations = json.load(f)

    images_dir = os.path.join(data_root, "images")
    rows = []
    skipped_missing_image = 0
    skipped_no_labels = 0
    for image_name, value in annotations.items():
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            skipped_missing_image += 1
            continue
        labels = _labels_from_annotation(value)
        if not labels:
            skipped_no_labels += 1
            continue
        row = {"image_id": os.path.splitext(image_name)[0], "image_path": image_path}
        for label in LABELS:
            row[label] = int(label in labels)
        rows.append(row)

    if skipped_missing_image or skipped_no_labels:
        print(
            f"load_annotations: skipped {skipped_missing_image} entries with missing "
            f"image files and {skipped_no_labels} with no recognized labels"
        )
    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No usable annotations parsed from {ann_path}")
    return df


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _stable_bucket(image_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def make_splits(
    df: pd.DataFrame,
    data_root: str,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 17,
) -> Splits:
    """Deterministic image-level train/val/test split.

    Uses the official CADB split.json (train/test) when present so results are
    comparable with published work, carving a validation set out of train with
    a stable hash. Falls back to a fully hash-based split otherwise.
    """
    split_path = os.path.join(data_root, "split.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            official = json.load(f)
        by_split = {
            name: {os.path.splitext(n)[0] for n in names}
            for name, names in official.items()
        }
        test_ids = by_split.get("test", set())
        test = df[df.image_id.isin(test_ids)]
        trainval = df[~df.image_id.isin(test_ids)]
        buckets = trainval.image_id.map(lambda i: _stable_bucket(i, seed))
        val = trainval[buckets < val_fraction]
        train = trainval[buckets >= val_fraction]
    else:
        buckets = df.image_id.map(lambda i: _stable_bucket(i, seed))
        test = df[buckets < test_fraction]
        val = df[(buckets >= test_fraction) & (buckets < test_fraction + val_fraction)]
        train = df[buckets >= test_fraction + val_fraction]

    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def label_frequencies(df: pd.DataFrame) -> pd.Series:
    return df[LABELS].sum().astype(int)


def positive_class_weights(df: pd.DataFrame, max_weight: float = 10.0) -> np.ndarray:
    """Per-class positive weights (neg/pos ratio, capped) for weighted BCE."""
    pos = df[LABELS].sum().to_numpy(dtype=np.float64)
    neg = len(df) - pos
    weights = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(weights, 1.0, max_weight).astype(np.float32)


def build_dataset(
    df: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = False,
    repeat: bool = False,
    seed: int = 17,
):
    """tf.data pipeline: decode -> resize -> batch.

    Pixel values stay in [0, 255]; Keras EfficientNet includes input
    normalization, and augmentation layers live inside the training model.
    """
    import tensorflow as tf

    paths = df.image_path.to_numpy()
    targets = df[LABELS].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()

    def _load(path, target):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32)
        return image, target

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
