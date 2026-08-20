"""DINOv2 linear-probe baseline (Experiment D).

Compares self-supervised ViT representations against the fine-tuned CNN:
a frozen DINOv2 ViT-S/14 backbone produces one embedding per image
(CLS token concatenated with the mean patch token), and only a small MLP
head is trained on the CADB composition labels. Splits, labels, metrics,
and threshold selection are identical to the EfficientNet experiments.

Usage:
    python -m src.dinov2 extract                 # one-time embedding pass (PyTorch)
    python -m src.dinov2 train                   # train MLP head (Keras)
    python -m src.dinov2 evaluate --split val --select-thresholds
    python -m src.dinov2 evaluate --split test
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import LABELS, load_annotations, make_splits, positive_class_weights

DEFAULT_DIR = "artifacts_dinov2"
EMBED_FILE = "embeddings.npz"


def extract(args):
    import torch
    from PIL import Image

    df = load_annotations(args.data_root)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.arch)
    model.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def load_image(path):
        img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return (x - mean) / std

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            paths = df.image_path.iloc[start:start + args.batch_size]
            batch = torch.stack([load_image(p) for p in paths]).to(device)
            out = model.forward_features(batch)
            emb = torch.cat(
                [out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)], dim=1
            )
            embeddings.append(emb.cpu().numpy())
            if (start // args.batch_size) % 20 == 0:
                print(f"embedded {start + len(paths)}/{len(df)}")

    embeddings = np.concatenate(embeddings).astype(np.float32)
    os.makedirs(args.model_dir, exist_ok=True)
    np.savez(
        os.path.join(args.model_dir, EMBED_FILE),
        image_id=df.image_id.to_numpy(),
        embedding=embeddings,
    )
    print(f"Saved {embeddings.shape} embeddings to {args.model_dir}/{EMBED_FILE}")


def _load_embeddings(model_dir: str) -> dict[str, np.ndarray]:
    data = np.load(os.path.join(model_dir, EMBED_FILE), allow_pickle=True)
    return dict(zip(data["image_id"], data["embedding"]))


def _split_arrays(split_df, emb_by_id):
    x = np.stack([emb_by_id[i] for i in split_df.image_id])
    y = split_df[LABELS].to_numpy(np.float32)
    return x, y


def train(args):
    import keras

    from .model import WeightedBinaryCrossentropy

    keras.utils.set_random_seed(args.seed)
    df = load_annotations(args.data_root)
    splits = make_splits(df, args.data_root, seed=args.seed)
    emb_by_id = _load_embeddings(args.embeddings_from or args.model_dir)
    x_train, y_train = _split_arrays(splits.train, emb_by_id)
    x_val, y_val = _split_arrays(splits.val, emb_by_id)

    if args.head == "linear":
        hidden = []
    else:
        hidden = [keras.layers.Dense(512, activation="gelu"), keras.layers.Dropout(0.3)]
    head = keras.Sequential(
        [keras.Input(shape=(x_train.shape[1],))]
        + hidden
        + [keras.layers.Dense(len(LABELS), activation="sigmoid")],
        name=f"dinov2_{args.head}_head",
    )
    if args.loss == "focal":
        loss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True, alpha=0.25, gamma=2.0
        )
    else:
        loss = WeightedBinaryCrossentropy(positive_class_weights(splits.train))
    head.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=[keras.metrics.AUC(name="auc_pr", curve="PR", multi_label=True)],
    )
    os.makedirs(args.model_dir, exist_ok=True)
    head.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=256,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc_pr", mode="max", patience=8,
                restore_best_weights=True, verbose=1,
            )
        ],
        verbose=2,
    )
    head.save(os.path.join(args.model_dir, "head.keras"))
    with open(os.path.join(args.model_dir, "labels.json"), "w") as f:
        json.dump(LABELS, f, indent=2)
    print(f"Saved head to {args.model_dir}/head.keras")


def evaluate(args):
    import keras

    from .evaluate import compute_report, report_markdown, select_thresholds

    df = load_annotations(args.data_root)
    splits = make_splits(df, args.data_root, seed=args.seed)
    split_df = getattr(splits, args.split)
    emb_by_id = _load_embeddings(args.embeddings_from or args.model_dir)
    x, y_true = _split_arrays(split_df, emb_by_id)
    y_true = y_true.astype(int)

    head = keras.models.load_model(os.path.join(args.model_dir, "head.keras"), compile=False)
    y_prob = head.predict(x, batch_size=256, verbose=0)

    thresholds_path = os.path.join(args.model_dir, "thresholds.json")
    if args.select_thresholds:
        if args.split == "test":
            raise SystemExit("Refusing to tune thresholds on the test set")
        thresholds = select_thresholds(y_true, y_prob)
        with open(thresholds_path, "w") as f:
            json.dump({l: float(t) for l, t in zip(LABELS, thresholds)}, f, indent=2)
        print("Saved per-class thresholds")
    elif os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            stored = json.load(f)
        thresholds = np.array([stored[l] for l in LABELS], dtype=np.float32)
    else:
        thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)

    report = compute_report(y_true, y_prob, thresholds)
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))
    with open(os.path.join(args.model_dir, f"report_{args.split}.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.model_dir, f"report_{args.split}.md"), "w") as f:
        f.write(report_markdown(report, args.split))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["extract", "train", "evaluate"])
    p.add_argument("--data-root", default="data/CADB_Dataset")
    p.add_argument("--model-dir", default=DEFAULT_DIR)
    p.add_argument("--arch", default="dinov2_vits14")
    p.add_argument("--head", choices=["mlp", "linear"], default="mlp")
    p.add_argument("--loss", choices=["weighted_bce", "focal"], default="weighted_bce")
    p.add_argument("--embeddings-from", default=None,
                   help="Reuse embeddings.npz from another model dir")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--select-thresholds", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    {"extract": extract, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
