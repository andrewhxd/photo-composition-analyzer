"""Two-stage transfer-learning training for the composition classifier.

Stage 1 trains only the new multi-label head with the EfficientNet backbone
frozen. Stage 2 unfreezes the last backbone blocks and fine-tunes at a lower
learning rate. The best checkpoint (by validation PR-AUC) is kept.

Usage:
    python -m src.train --data-root data/CADB_Dataset --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import os

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import (
    LABELS,
    build_dataset,
    label_frequencies,
    load_annotations,
    make_splits,
    positive_class_weights,
)
from .model import build_model, compile_model, unfreeze_top_blocks


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/CADB_Dataset")
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=8)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--unfreeze-blocks", type=int, default=2,
                   help="EfficientNet blocks to unfreeze in stage 2 (0 skips fine-tuning)")
    p.add_argument("--loss", choices=["weighted_bce", "focal"], default="weighted_bce",
                   help="Training loss: per-class weighted BCE or binary focal loss")
    p.add_argument("--no-class-weighting", action="store_true",
                   help="Use plain BCE instead of positive-class-weighted BCE")
    p.add_argument("--max-pos-weight", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=0,
                   help="Truncate each split to N images (smoke testing only)")
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args(argv)


def plot_history(histories: list[dict], output_dir: str) -> None:
    merged: dict[str, list] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "auc_pr", "PR-AUC (mAP proxy)"),
    ]:
        ax.plot(merged.get(metric, []), label="train")
        ax.plot(merged.get(f"val_{metric}", []), label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_annotations(args.data_root)
    splits = make_splits(df, args.data_root, seed=args.seed)
    if args.limit:
        splits.train = splits.train.head(args.limit)
        splits.val = splits.val.head(max(args.limit // 4, 8))
        splits.test = splits.test.head(max(args.limit // 4, 8))

    print(f"train={len(splits.train)}  val={len(splits.val)}  test={len(splits.test)}")
    print("train label frequencies:")
    print(label_frequencies(splits.train).to_string())

    pos_weight = None
    if not args.no_class_weighting and args.loss == "weighted_bce":
        pos_weight = positive_class_weights(splits.train, args.max_pos_weight)
        print("positive class weights:",
              {label: round(float(w), 2) for label, w in zip(LABELS, pos_weight)})

    train_ds = build_dataset(splits.train, args.image_size, args.batch_size,
                             shuffle=True, seed=args.seed)
    val_ds = build_dataset(splits.val, args.image_size, args.batch_size)

    model = build_model(image_size=args.image_size)
    model.summary(line_length=100)

    checkpoint_path = os.path.join(args.output_dir, "model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_auc_pr", mode="max",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc_pr", mode="max", patience=4,
            restore_best_weights=True, verbose=1,
        ),
    ]

    histories = []

    print(f"\n=== Stage 1: train head with frozen backbone (loss={args.loss}) ===")
    compile_model(model, args.head_lr, pos_weight, args.loss)
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.head_epochs, callbacks=callbacks)
    histories.append(history.history)

    if args.finetune_epochs > 0 and args.unfreeze_blocks > 0:
        print(f"\n=== Stage 2: fine-tune last {args.unfreeze_blocks} blocks ===")
        unfrozen = unfreeze_top_blocks(model, args.unfreeze_blocks)
        print(f"unfroze {unfrozen} layers")
        compile_model(model, args.finetune_lr, pos_weight, args.loss)
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=args.finetune_epochs, callbacks=callbacks)
        histories.append(history.history)

    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump(LABELS, f, indent=2)
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(histories, f)
    plot_history(histories, args.output_dir)
    print(f"\nBest checkpoint saved to {checkpoint_path}")
    print("Next: python -m src.evaluate --split val --select-thresholds, then --split test")


if __name__ == "__main__":
    main()
