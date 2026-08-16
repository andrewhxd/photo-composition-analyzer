"""Evaluation: multi-label metrics, per-class thresholds, and reports.

Typical flow:
    # pick per-class thresholds on the validation set and store them
    python -m src.evaluate --split val --select-thresholds

    # final, one-time evaluation on the held-out test set
    python -m src.evaluate --split test

Reports (JSON + markdown table + plots) are written to the artifacts dir.
"""

from __future__ import annotations

import argparse
import json
import os

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from .data import LABELS, build_dataset, load_annotations, make_splits


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/CADB_Dataset")
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--select-thresholds", action="store_true",
                   help="Choose one threshold per class maximizing F1 and save them")
    p.add_argument("--limit", type=int, default=0, help="Truncate split (smoke testing)")
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args(argv)


def load_thresholds(model_dir: str) -> np.ndarray:
    path = os.path.join(model_dir, "thresholds.json")
    if os.path.exists(path):
        with open(path) as f:
            stored = json.load(f)
        return np.array([stored[label] for label in LABELS], dtype=np.float32)
    return np.full(len(LABELS), 0.5, dtype=np.float32)


def select_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-class threshold maximizing F1, swept over a fixed grid."""
    grid = np.arange(0.05, 0.96, 0.05)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        scores = [f1_score(y_true[:, i], y_prob[:, i] >= t, zero_division=0) for t in grid]
        thresholds[i] = grid[int(np.argmax(scores))]
    return thresholds


def compute_report(y_true, y_prob, thresholds) -> dict:
    y_pred = (y_prob >= thresholds).astype(int)
    per_class = {}
    ap_values = []
    for i, label in enumerate(LABELS):
        support = int(y_true[:, i].sum())
        ap = average_precision_score(y_true[:, i], y_prob[:, i]) if support else float("nan")
        if support:
            ap_values.append(ap)
        per_class[label] = {
            "support": support,
            "ap": round(float(ap), 4) if support else None,
            "precision": round(float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
            "recall": round(float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
            "f1": round(float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
            "threshold": round(float(thresholds[i]), 2),
        }
    return {
        "num_images": int(len(y_true)),
        "mAP": round(float(np.mean(ap_values)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "per_class": per_class,
    }


def report_markdown(report: dict, split: str) -> str:
    lines = [
        f"## Evaluation on `{split}` ({report['num_images']} images)",
        "",
        f"- mAP: **{report['mAP']:.4f}**",
        f"- Macro F1: **{report['macro_f1']:.4f}**",
        f"- Micro F1: **{report['micro_f1']:.4f}**",
        "",
        "| Class | Support | AP | Precision | Recall | F1 | Threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, m in report["per_class"].items():
        ap = f"{m['ap']:.3f}" if m["ap"] is not None else "-"
        lines.append(
            f"| {label} | {m['support']} | {ap} | {m['precision']:.3f} "
            f"| {m['recall']:.3f} | {m['f1']:.3f} | {m['threshold']:.2f} |"
        )
    return "\n".join(lines) + "\n"


# Single-hue sequential ramp (light to dark blue) for magnitude encodings.
SEQUENTIAL_BLUES = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"
]
BAR_BLUE = "#2a78d6"
GRID_GRAY = "#e1e0d9"
MUTED_INK = "#898781"


def plot_cooccurrence(y_true: np.ndarray, out_path: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    co = y_true.T @ y_true
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUES)
    im = ax.imshow(np.log1p(co), cmap=cmap)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=90)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_title("Label co-occurrence (log scale)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_ap(report: dict, out_path: str) -> None:
    items = [(label, m["ap"]) for label, m in report["per_class"].items() if m["ap"] is not None]
    labels, aps = zip(*items)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, aps, color=BAR_BLUE, width=0.62)
    ax.set_ylabel("Average precision", color=MUTED_INK)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=90, colors=MUTED_INK)
    ax.tick_params(axis="y", colors=MUTED_INK)
    ax.grid(axis="y", color=GRID_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    df = load_annotations(args.data_root)
    splits = make_splits(df, args.data_root, seed=args.seed)
    split_df = getattr(splits, args.split)
    if args.limit:
        split_df = split_df.head(args.limit)

    model = keras.models.load_model(
        os.path.join(args.model_dir, "model.keras"), compile=False
    )
    ds = build_dataset(split_df, args.image_size, args.batch_size)
    y_prob = model.predict(ds, verbose=1)
    y_true = split_df[LABELS].to_numpy(dtype=int)

    if args.select_thresholds:
        if args.split == "test":
            raise SystemExit("Refusing to tune thresholds on the test set")
        thresholds = select_thresholds(y_true, y_prob)
        with open(os.path.join(args.model_dir, "thresholds.json"), "w") as f:
            json.dump({l: float(t) for l, t in zip(LABELS, thresholds)}, f, indent=2)
        print("Saved per-class thresholds to thresholds.json")
    else:
        thresholds = load_thresholds(args.model_dir)

    report = compute_report(y_true, y_prob, thresholds)
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))

    with open(os.path.join(args.model_dir, f"report_{args.split}.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.model_dir, f"report_{args.split}.md"), "w") as f:
        f.write(report_markdown(report, args.split))
    plot_cooccurrence(y_true, os.path.join(args.model_dir, f"cooccurrence_{args.split}.png"))
    plot_per_class_ap(report, os.path.join(args.model_dir, f"per_class_ap_{args.split}.png"))
    print(f"Reports written to {args.model_dir}/report_{args.split}.{{json,md}}")


if __name__ == "__main__":
    main()
