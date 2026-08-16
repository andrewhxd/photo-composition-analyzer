"""Single-image composition prediction.

Usage:
    python -m src.inference photo.jpg [more.jpg ...] [--overlay out_dir]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import LABELS


class CompositionPredictor:
    """Loads the trained model plus stored per-class thresholds."""

    def __init__(self, model_dir: str = "artifacts", image_size: int = 224):
        import keras

        self.model = keras.models.load_model(
            os.path.join(model_dir, "model.keras"), compile=False
        )
        self.image_size = image_size
        thresholds_path = os.path.join(model_dir, "thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path) as f:
                stored = json.load(f)
            self.thresholds = np.array([stored[l] for l in LABELS], dtype=np.float32)
        else:
            self.thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)

    def predict_array(self, rgb: np.ndarray) -> list[dict]:
        """rgb: HxWx3 uint8 array. Returns per-label results sorted by probability."""
        import cv2

        resized = cv2.resize(
            rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        batch = resized.astype(np.float32)[None, ...]
        probs = self.model.predict(batch, verbose=0)[0]
        results = [
            {
                "label": label,
                "probability": round(float(p), 4),
                "threshold": round(float(t), 2),
                "predicted": bool(p >= t),
            }
            for label, p, t in zip(LABELS, probs, self.thresholds)
        ]
        results.sort(key=lambda r: r["probability"], reverse=True)
        return results

    def predict_file(self, image_path: str) -> list[dict]:
        import cv2

        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.predict_array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def draw_overlay(image_path: str, results: list[dict], out_path: str) -> None:
    """Save a copy of the image with a rule-of-thirds grid and predicted labels."""
    import cv2

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    color, thickness = (255, 255, 255), max(1, w // 640)
    for fx in (1 / 3, 2 / 3):
        cv2.line(bgr, (int(w * fx), 0), (int(w * fx), h), color, thickness)
        cv2.line(bgr, (0, int(h * fx)), (w, int(h * fx)), color, thickness)
    predicted = [r for r in results if r["predicted"]]
    text = ", ".join(f"{r['label']} {r['probability']:.2f}" for r in predicted) or "none above threshold"
    scale = max(0.5, w / 1280)
    cv2.putText(bgr, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), int(3 * scale) + 1, cv2.LINE_AA)
    cv2.putText(bgr, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), int(scale) + 1, cv2.LINE_AA)
    cv2.imwrite(out_path, bgr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+", help="Image file(s) to analyze")
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--overlay", metavar="OUT_DIR",
                   help="Also save copies with a rule-of-thirds grid + labels")
    args = p.parse_args(argv)

    predictor = CompositionPredictor(args.model_dir)
    for image_path in args.images:
        results = predictor.predict_file(image_path)
        print(f"\n{image_path}")
        for r in results:
            marker = "*" if r["predicted"] else " "
            bar = "#" * int(r["probability"] * 30)
            print(f" {marker} {r['label']:16s} {r['probability']:.3f} {bar}")
        if args.overlay:
            os.makedirs(args.overlay, exist_ok=True)
            out_path = os.path.join(args.overlay, os.path.basename(image_path))
            draw_overlay(image_path, results, out_path)
            print(f"   overlay -> {out_path}")


if __name__ == "__main__":
    main()
