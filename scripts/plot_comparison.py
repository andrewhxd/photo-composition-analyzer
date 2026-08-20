"""Per-class AP comparison chart across experiments (for the README).

Reads report_test.json from each experiment's artifacts dir and renders a
grouped bar chart into docs/comparison_ap.png.

Run: .venv/bin/python scripts/plot_comparison.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import LABELS  # noqa: E402

# Validated 3-slot categorical palette; series order is fixed, not cycled.
EXPERIMENTS = [
    ("EfficientNet fine-tuned (weighted BCE)", "artifacts", "#2a78d6"),
    ("EfficientNet fine-tuned (focal)", "artifacts_focal", "#eb6834"),
    ("DINOv2 frozen + MLP head", "artifacts_dinov2", "#1baf7a"),
]
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"


def main():
    series = []
    for name, model_dir, color in EXPERIMENTS:
        with open(os.path.join(model_dir, "report_test.json")) as f:
            report = json.load(f)
        aps = [report["per_class"][label]["ap"] for label in LABELS]
        series.append((name, [np.nan if a is None else a for a in aps], color))

    x = np.arange(len(LABELS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (name, aps, color) in enumerate(series):
        ax.bar(x + (i - 1) * width, aps, width - 0.03, label=name, color=color)

    ax.set_ylabel("Test average precision", color=MUTED)
    ax.set_ylim(0, 1)
    ax.set_xticks(x, [l.replace("_", " ") for l in LABELS], rotation=45,
                  ha="right", color=INK)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = os.path.join("docs", "comparison_ap.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
