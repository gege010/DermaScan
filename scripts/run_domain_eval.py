"""
scripts/run_domain_eval.py
──────────────────────────
Jalankan HANYA evaluate_per_domain() tanpa training ulang.
Berguna setelah training selesai dan model sudah tersimpan.

Usage (dari root repo):
    python scripts/run_domain_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR  = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Model di-save dengan Keras 2 (HDF5 format).
# Set env var SEBELUM import tensorflow agar tf.keras masih pakai Keras 2 API.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from train import (
    BATCH_SIZE_P2,
    DATA_CSV,
    IMG_SIZE,
    SAVE_DIR,
    evaluate_model,
    evaluate_per_domain,
    focal_loss,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_model(model_path: Path) -> tf.keras.Model:
    """
    Load model yang di-save oleh Keras 2 (format HDF5, magic bytes 89 48 44 46).

    TF 2.18 bundled keras 3 yang menolak HDF5. Solusi:
    1. Coba tf_keras (package Keras 2 yang masih dimaintain untuk TF 2.x)
    2. Fallback ke tf.keras dengan compile=False
    """
    custom_objs = {"focal_loss": focal_loss(gamma=2.0, alpha=0.25)}

    # Coba tf_keras dulu (install: pip install tf_keras)
    try:
        import tf_keras  # noqa: F401
        model = tf_keras.models.load_model(
            str(model_path),
            custom_objects=custom_objs,
            compile=False,
        )
        logger.info(f"Loaded via tf_keras from {model_path.name}")
        return model
    except ImportError:
        logger.warning("tf_keras not installed. Trying tf.keras with legacy flag...")

    # Fallback: tf.keras langsung (mungkin bisa jika TF_USE_LEGACY_KERAS=1 aktif)
    try:
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objs,
            compile=False,
        )
        logger.info(f"Loaded via tf.keras from {model_path.name}")
        return model
    except Exception as e:
        logger.error(
            f"Cannot load model: {e}\n"
            "Install tf_keras dengan: pip install tf_keras"
        )
        raise


def main():
    # ── Cari model terbaik ────────────────────────────────────────────────────
    candidates = [
        SAVE_DIR / "skin_model_best.keras",
        SAVE_DIR / "skin_model_phase2.keras",
        SAVE_DIR / "skin_model_phase1.keras",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        logger.error(f"Tidak ada model ditemukan di {SAVE_DIR}. Jalankan src/train.py dulu.")
        sys.exit(1)

    logger.info(f"Menggunakan model: {model_path.name}")
    model = _load_model(model_path)

    # ── Load class names ──────────────────────────────────────────────────────
    class_names_path = SAVE_DIR / "class_names.json"
    if not class_names_path.exists():
        logger.error(f"class_names.json tidak ditemukan di {SAVE_DIR}")
        sys.exit(1)

    with open(class_names_path) as f:
        class_names = json.load(f)
    logger.info(f"Classes ({len(class_names)}): {class_names}")

    # ── Rebuild test split (harus identik dengan train.py: seed=42, 80/10/10) ─
    df = pd.read_csv(DATA_CSV)
    _, temp_df = train_test_split(df, test_size=0.2, stratify=df["dx"], random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["dx"], random_state=42)
    logger.info(f"Test set: {len(test_df)} samples")

    val_aug = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    )
    test_gen = val_aug.flow_from_dataframe(
        test_df,
        x_col="image_path",
        y_col="dx",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE_P2,
        shuffle=False,
        seed=42,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running predictions on test set...")
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.asarray(test_gen.classes, dtype=np.int64)

    # ── Full evaluation (per-class metrics) ───────────────────────────────────
    logger.info("Running full evaluation (per-class metrics)...")
    overall_metrics, per_class_metrics = evaluate_model(model, test_gen, class_names)

    with open(SAVE_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)
    with open(SAVE_DIR / "per_class_metrics.json", "w") as f:
        json.dump(per_class_metrics, f, indent=2)
    logger.info("Saved → evaluation_metrics.json + per_class_metrics.json")

    # ── Domain evaluation ─────────────────────────────────────────────────────
    # Re-run predictions dengan fresh generator untuk domain eval
    test_gen.reset()
    y_pred_probs2 = model.predict(test_gen, verbose=0)
    y_pred2 = np.argmax(y_pred_probs2, axis=1)
    y_true2 = np.asarray(test_gen.classes, dtype=np.int64)

    logger.info("Running per-domain evaluation (HAM10000 vs SD-198)...")
    domain_metrics = evaluate_per_domain(y_true2, y_pred2, class_names, per_class_metrics)

    # ── Save & print ──────────────────────────────────────────────────────────
    out_path = SAVE_DIR / "domain_metrics.json"
    with open(out_path, "w") as f:
        json.dump(domain_metrics, f, indent=2)
    logger.info(f"Saved → {out_path}")

    logger.info("\n" + "=" * 55)
    logger.info("PER-DOMAIN SUMMARY")
    logger.info("=" * 55)
    for domain, dm in domain_metrics.items():
        logger.info(
            f"  {domain:<10} | Acc: {dm['accuracy']:.3f} "
            f"| F1: {dm['f1_weighted']:.3f} "
            f"| Cross-err: {dm['cross_domain_error_pct']:.1f}%"
            f"  (n={dm['n_samples']})"
        )
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
