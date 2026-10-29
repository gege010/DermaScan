"""Frozen ImageNet embeddings + linear classifier; explicit patient/lesion-grouped splits.

Usage: python -m scripts.train_compact --csv data/curated.csv --domain clinical
CSV: image_path,label,group_id,split (train/val/test). Supply true patient/lesion groups.
This trains a research model, never a clinically validated diagnostic system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def validate_manifest(rows: list[dict], base: Path) -> list[dict]:
    if not rows:
        raise ValueError("Manifest is empty")
    groups: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for row in rows:
        if any(
            not str(row.get(key, "")).strip()
            for key in ("image_path", "label", "group_id", "split")
        ):
            raise ValueError("Every row needs image_path,label,group_id,split")
        split, group = row["split"], row["group_id"]
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        if group in groups and groups[group] != split:
            raise ValueError(f"Group leakage across splits: {group}")
        groups[group] = split
        path = (base / row["image_path"]).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate image content in manifest; deduplicate before training")
        hashes[digest] = split
        row["image_path"] = str(path)
    label_sets = [{r["label"] for r in rows if r["split"] == s} for s in ("train", "val", "test")]
    if len(label_sets[0]) < 2 or any(s != label_sets[0] for s in label_sets):
        raise ValueError("Every split must contain the same two or more classes")
    return rows


def main():
    import csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--domain", choices=["clinical", "dermoscopy"], required=True)
    parser.add_argument("--output", type=Path, default=Path("models/compact"))
    parser.add_argument(
        "--max-per-class",
        type=int,
        help="Optional train-only sample cap, never changes validation/test",
    )
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as file:
        rows = validate_manifest(list(csv.DictReader(file)), args.csv.parent)

    import joblib
    import numpy as np
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.dermascan.imaging import prepare_image

    tf.keras.utils.set_random_seed(42)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3)
    )
    backbone.trainable = False
    labels = sorted({row["label"] for row in rows})
    rng = np.random.default_rng(42)
    data = {}
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        if split == "train" and args.max_per_class:
            if args.max_per_class < 2:
                raise ValueError("Use at least two training images per class")
            subset = [
                r
                for label in labels
                for r in rng.permutation([r for r in subset if r["label"] == label])[
                    : args.max_per_class
                ]
            ]
        vectors = []
        for start in range(0, len(subset), 16):
            batch = np.concatenate(
                [
                    prepare_image(Path(r["image_path"]).read_bytes()).tensor((224, 224))
                    for r in subset[start : start + 16]
                ]
            )
            vectors.extend(backbone.predict(batch, verbose=0))
        data[split] = (np.asarray(vectors), np.asarray([r["label"] for r in subset]))
    best, best_score = None, -1.0
    for c in (0.01, 0.1, 1.0, 10.0):
        candidate = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42),
        )
        candidate.fit(*data["train"])
        score = f1_score(data["val"][1], candidate.predict(data["val"][0]), average="macro")
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    x_test, y_test = data["test"]
    probs = best.predict_proba(x_test)
    predicted = best.classes_[probs.argmax(axis=1)]
    report = {
        "validation_f1_macro": best_score,
        "test": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=best.classes_).tolist(),
        "log_loss_uncalibrated": log_loss(y_test, probs, labels=best.classes_),
        "sample_counts": {s: len(data[s][1]) for s in data},
        "clinical_validation": False,
        "limitations": [
            "ImageNet pretraining, not dermatology-specialized",
            "Uncalibrated scores",
            "No validated out-of-distribution detector",
            "External validation and clinician review required",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    backbone.save(args.output / "embedding_backbone.keras")
    joblib.dump(best, args.output / "classifier.joblib")
    manifest = {
        "schema_version": 1,
        "classes": best.classes_.tolist(),
        "input_size": [224, 224],
        "domain": args.domain,
        "preprocessing": "RGB float32 [0,255], EfficientNet internal rescaling",
        "clinical_validation": False,
        "manifest_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
    }
    (args.output / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Research model saved to {args.output}. Test macro F1: {report['test']['macro avg']['f1-score']:.3f}"
    )


if __name__ == "__main__":
    main()
