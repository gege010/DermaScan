"""
model/train.py
──────────────
DermaScan — 2-Phase EfficientNetB0 Training Script (Ultra-Lean Version)

Usage:
    python model/train.py

Outputs:
    model/saved_model/skin_model_best.keras
    model/saved_model/class_names.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

IMG_SIZE = (224, 224)
BATCH_SIZE_P1 = 32
BATCH_SIZE_P2 = 16
EPOCHS_P1 = 20
EPOCHS_P2 = 30
UNFREEZE_TOP_N = 20

DATA_CSV = Path("data/processed/combined_clean.csv")
SAVE_DIR = Path("model/saved_model")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_splits(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples, {df['dx'].nunique()} classes.")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["dx"], random_state=42
    )
    return train_df, val_df, test_df

def make_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
) -> tuple:
    """Build Keras ImageDataGenerators with augmentation for training."""
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    common_kwargs = dict(
        x_col="image_path",
        y_col="dx",           
        target_size=IMG_SIZE,
        class_mode="categorical",
        seed=42,
    )

    train_gen = train_aug.flow_from_dataframe(
        train_df, batch_size=batch_size, shuffle=True, **common_kwargs
    )
    val_gen = val_aug.flow_from_dataframe(
        val_df, batch_size=batch_size, shuffle=False, **common_kwargs
    )
    test_gen = val_aug.flow_from_dataframe(
        test_df, batch_size=batch_size, shuffle=False, **common_kwargs
    )
    return train_gen, val_gen, test_gen


# ─── Model Architecture ───────────────────────────────────────────────────────

def build_model(num_classes: int, freeze_base: bool = True) -> Model:
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = not freeze_base

    if not freeze_base:
        for layer in base.layers[:-UNFREEZE_TOP_N]:
            layer.trainable = False
        logger.info(f"Unfrozen top {UNFREEZE_TOP_N} layers of EfficientNetB0.")

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    output = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=base.input, outputs=output)


# ─── Training ─────────────────────────────────────────────────────────────────

def get_class_weights(train_gen) -> dict:
    labels = train_gen.classes
    unique_classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    return dict(zip(unique_classes.tolist(), weights.tolist()))


def get_callbacks(phase: int) -> list:
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5 if phase == 1 else 7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(SAVE_DIR / f"best_phase{phase}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return callbacks


def evaluate_model(model: Model, test_gen) -> dict[str, float]:
    """Compute accuracy, F1, precision, recall on test set."""
    from sklearn.metrics import classification_report

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, output_dict=True)
    weighted = report["weighted avg"]

    metrics = {
        "test_accuracy": float(np.mean(y_pred == y_true)),
        "test_f1_weighted": weighted["f1-score"],
        "test_precision_weighted": weighted["precision"],
        "test_recall_weighted": weighted["recall"],
    }
    logger.info(f"Test Results → {metrics}")
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Data ──────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_splits(DATA_CSV)

    train_gen_p1, val_gen, test_gen = make_generators(
        train_df, val_df, test_df, BATCH_SIZE_P1
    )
    num_classes = len(train_gen_p1.class_indices)
    class_names = list(train_gen_p1.class_indices.keys())

    with open(SAVE_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info(f"Saved {num_classes} class names to {SAVE_DIR}/class_names.json")

    class_weights = get_class_weights(train_gen_p1)

    # ── Phase 1: Train Classification Head ───────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 — Training classification head (frozen base)")
    logger.info("=" * 60)

    model = build_model(num_classes, freeze_base=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    model.fit(
        train_gen_p1,
        validation_data=val_gen,
        epochs=EPOCHS_P1,
        class_weight=class_weights,
        callbacks=get_callbacks(phase=1),
    )

    # ── Phase 2: Fine-tune Top Layers ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"PHASE 2 — Fine-tuning top {UNFREEZE_TOP_N} layers")
    logger.info("=" * 60)

    train_gen_p2, val_gen_p2, test_gen_p2 = make_generators(
        train_df, val_df, test_df, BATCH_SIZE_P2
    )

    model.trainable = True
    for layer in model.layers[:-UNFREEZE_TOP_N]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen_p2,
        validation_data=val_gen_p2,
        epochs=EPOCHS_P2,
        class_weight=class_weights,
        callbacks=get_callbacks(phase=2),
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    test_metrics = evaluate_model(model, test_gen_p2)
    
    # Simpan metrik agar bisa dibaca oleh Streamlit
    with open(SAVE_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Save Final Model ──────────────────────────────────────────────────
    # Nama file disinkronkan agar langsung terbaca oleh api/main.py
    final_path = str(SAVE_DIR / "skin_model_best.keras")
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()