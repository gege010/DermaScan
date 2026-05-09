"""
src/train.py
────────────
DermaScan — 2-Phase EfficientNetB3 Training Script

Perbaikan dari revisi reviewer:
  - Upgrade backbone: EfficientNetB0 → EfficientNetB3 (lebih dalam, akurasi lebih tinggi)
  - Focal Loss menggantikan standard CrossEntropy untuk mengatasi class imbalance
  - Augmentasi lebih agresif: rotation 180°, color jitter, shear, cutout-style cropping
  - Per-class precision, recall, F1 dilaporkan dan disimpan ke JSON
  - ROC-AUC per kelas (macro & weighted) dihitung dan disimpan
  - Confusion matrix disimpan sebagai PNG untuk dashboard Streamlit
  - EfficientNet preprocessing bawaan digunakan (tidak manual rescale)
  - Model input size dinaikkan: 224 → 300 (optimal untuk B3)
  - Domain-aware evaluation: HAM10000 vs SD-198 dianalisis secara TERPISAH
    untuk transparansi dataset mismatch (lihat DESIGN_DECISIONS.md ADR-001)

Usage (dari root repo):
    python src/train.py

Outputs:
    models/skin_model_best.keras
    models/class_names.json
    models/evaluation_metrics.json      ← aggregate metrics (accuracy, F1, AUC)
    models/per_class_metrics.json       ← per-class F1, precision, recall, AUC
    models/domain_metrics.json          ← HAM10000 vs SD-198 metrics TERPISAH
    models/confusion_matrix.png         ← normalized confusion matrix
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Tambahkan direktori src ke sys.path agar utils bisa diimport
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

# EfficientNetB3 optimal input size (vs 224 untuk B0)
IMG_SIZE = (300, 300)
BATCH_SIZE_P1 = 32
BATCH_SIZE_P2 = 16
EPOCHS_P1 = 25
EPOCHS_P2 = 40

# Jumlah layer top yang di-unfreeze di fase 2 (B3 punya lebih banyak layer)
UNFREEZE_TOP_N = 30

# Path relatif terhadap root repo (satu level di atas src/)
ROOT_DIR = SRC_DIR.parent
DATA_CSV = ROOT_DIR / "data" / "processed" / "combined_clean.csv"
SAVE_DIR = ROOT_DIR / "models"
LOG_DIR  = ROOT_DIR / "logs" / "tensorboard"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Domain Mapping ───────────────────────────────────────────────────────────
# Digunakan untuk evaluasi per-domain (HAM10000 vs SD-198) secara terpisah.
# Ini penting untuk transparansi dataset mismatch (lihat DESIGN_DECISIONS.md ADR-001).
#
# HAM10000 menggunakan label kode standar (akiec, bcc, bkl, df, mel, nv, vasc)
# SD-198 menggunakan label deskriptif (Acne, Vitiligo, dst)
HAM10000_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
SD198_CLASSES = {
    "Acne", "SJS-TEN", "Vitiligo", "Nail_psoriasis",
    "Hyperpigmentation", "Dry", "Oily", "Combination", "Normal",
}


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss untuk mengatasi class imbalance secara eksplisit di loss function.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    - gamma > 0 mengurangi dampak loss dari contoh mudah (well-classified).
    - alpha menyeimbangkan bobot antar kelas.
    - Lebih kuat dari class_weight saat imbalance ratio sangat tinggi.

    Reference: Lin et al. (2017) https://arxiv.org/abs/1708.02002
    """
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred  = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Cross entropy component
        ce = -y_true * tf.math.log(y_pred)

        # Focal weight: (1 - p_t)^gamma
        p_t  = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        weight = tf.pow(1.0 - p_t, gamma)

        fl = alpha * weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_splits(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples, {df['dx'].nunique()} classes.")

    # Stratified split: 80% train / 10% val / 10% test
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["dx"], random_state=42
    )
    logger.info(
        f"Split → train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}"
    )

    # Log distribusi kelas untuk verifikasi imbalance
    logger.info("Class distribution (train):")
    for cls, cnt in train_df["dx"].value_counts().items():
        pct = cnt / len(train_df) * 100
        logger.info(f"  {cls:<30} {cnt:>5} ({pct:.1f}%)")

    return train_df, val_df, test_df


def make_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tuple:
    """
    Build Keras ImageDataGenerators.

    Augmentasi training diperketat sesuai rekomendasi medical imaging:
    - rotation_range=180°: lesi kulit tidak punya orientasi tetap
    - color jitter (brightness + channel_shift): variasi pencahayaan kamera
    - horizontal + vertical flip: simetri lesi
    - zoom + shear: variasi skala dan distorsi kamera

    EfficientNetB3 menggunakan preprocessing internal tf.keras.applications,
    sehingga TIDAK perlu rescale=1/255 manual. Input harus [0, 255].
    """
    train_aug = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=180,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.20,
        shear_range=0.10,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.75, 1.25],
        channel_shift_range=20.0,
        fill_mode="reflect",
    )
    val_aug = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    )

    common_kwargs = dict(
        x_col="image_path",
        y_col="dx",
        target_size=img_size,
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
    """
    EfficientNetB3 classification model.

    Mengapa B3 bukan B0?
    - B3: ~12M params vs B0 ~5.3M params — lebih dalam dan ekspresif.
    - Pada benchmark HAM10000, B3/B4 secara konsisten mencapai 85-92% accuracy.
    - B0 terlalu dangkal untuk 16-class imbalanced medical problem.
    - Trade-off: training lebih lambat, tapi akurasi signifikan lebih tinggi.
    """
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = not freeze_base

    if not freeze_base:
        # Freeze semua kecuali top N layers + biarkan BatchNorm tetap frozen
        # (mem-freeze BN saat fine-tuning adalah best practice untuk TF)
        for layer in base.layers[:-UNFREEZE_TOP_N]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        trainable_count = sum(1 for l in base.layers if l.trainable)
        logger.info(f"Unfrozen {trainable_count} layers of EfficientNetB3.")

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(512, activation="relu", name="head_dense_1")(x)
    x = layers.Dropout(0.4, name="head_dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_2")(x)
    x = layers.Dropout(0.3, name="head_dropout_2")(x)
    output = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=base.input, outputs=output)


# ─── Training Utilities ───────────────────────────────────────────────────────

def get_class_weights(train_gen) -> dict:
    """Hitung class weights yang seimbang untuk kompensasi imbalance."""
    labels = train_gen.classes
    unique_classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    cw = dict(zip(unique_classes.tolist(), weights.tolist()))
    logger.info(f"Class weights range: min={min(cw.values()):.3f}, max={max(cw.values()):.3f}")
    return cw


def get_callbacks(phase: int, model_name: str = "skin_model") -> list:
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7 if phase == 1 else 10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(SAVE_DIR / f"{model_name}_phase{phase}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(
            log_dir=str(LOG_DIR / f"phase{phase}"),
            histogram_freq=1,
        ),
    ]
    return callbacks


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    model: Model,
    test_gen,
    class_names: list[str],
) -> tuple[dict, dict]:
    """
    Evaluasi komprehensif:
    - Overall: accuracy, weighted F1/precision/recall
    - Per-class: F1, precision, recall
    - ROC-AUC: per-class, macro, weighted
    - Confusion matrix PNG
    """
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # ── Classification report ──────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    logger.info("\n" + classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    weighted = report["weighted avg"]
    macro    = report["macro avg"]

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    n_classes  = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    try:
        roc_auc_macro    = roc_auc_score(y_true_bin, y_pred_probs, average="macro",    multi_class="ovr")
        roc_auc_weighted = roc_auc_score(y_true_bin, y_pred_probs, average="weighted", multi_class="ovr")
        per_class_auc    = {}
        for i, name in enumerate(class_names):
            try:
                per_class_auc[name] = float(
                    roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
                )
            except Exception:
                per_class_auc[name] = None
    except Exception as e:
        logger.warning(f"ROC-AUC computation failed: {e}")
        roc_auc_macro = roc_auc_weighted = 0.0
        per_class_auc = {n: None for n in class_names}

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    overall_metrics = {
        "test_accuracy":          float(np.mean(y_pred == y_true)),
        "test_f1_weighted":       float(weighted["f1-score"]),
        "test_f1_macro":          float(macro["f1-score"]),
        "test_precision_weighted":float(weighted["precision"]),
        "test_recall_weighted":   float(weighted["recall"]),
        "roc_auc_macro":          float(roc_auc_macro),
        "roc_auc_weighted":       float(roc_auc_weighted),
    }

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_class_metrics = {}
    for name in class_names:
        cls_report = report.get(name, {})
        per_class_metrics[name] = {
            "precision": round(float(cls_report.get("precision", 0)), 4),
            "recall":    round(float(cls_report.get("recall",    0)), 4),
            "f1_score":  round(float(cls_report.get("f1-score",  0)), 4),
            "support":   int(cls_report.get("support", 0)),
            "roc_auc":   round(per_class_auc.get(name, 0) or 0, 4),
        }

    logger.info(f"\nOverall Test Results:")
    for k, v in overall_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Confusion Matrix ───────────────────────────────────────────────────────
    _save_confusion_matrix(y_true, y_pred, class_names)

    return overall_metrics, per_class_metrics


def evaluate_per_domain(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    per_class_metrics: dict,
) -> dict:
    """
    Evaluasi per-domain: hitung metrik terpisah untuk HAM10000 dan SD-198.

    Ini menjawab pertanyaan kritis: apakah model perform baik di kedua domain,
    atau hanya bagus di salah satu dan buruk di lainnya?

    Mengapa ini penting:
    - HAM10000 (dermoscopy) dan SD-198 (foto klinis) punya distribusi gambar
      yang sangat berbeda (image domain gap).
    - Jika model bagus secara keseluruhan tapi buruk di HAM10000 (kondisi klinis
      serius), itu adalah risk yang harus dilaporkan secara transparan.
    - Per-domain accuracy membuktikan bahwa mixed dataset training memberikan
      benefit di kedua domain, atau mengidentifikasi mana yang perlu ditingkatkan.

    See: DESIGN_DECISIONS.md ADR-001
    """
    # Pastikan input adalah numpy array — boolean indexing tidak bekerja di Python list
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # Build label → index mapping (case-sensitive)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    # Case-insensitive fallback: handle ketidaksesuaian kapital (Acne vs acne, Dry vs dry, dll)
    class_to_idx_lower = {name.lower(): i for i, name in enumerate(class_names)}

    def _resolve_class(c: str) -> int | None:
        if c in class_to_idx:
            return class_to_idx[c]
        if c.lower() in class_to_idx_lower:
            return class_to_idx_lower[c.lower()]
        return None

    domain_results = {}
    for domain_name, domain_classes in [
        ("HAM10000", HAM10000_CLASSES),
        ("SD198", SD198_CLASSES),
    ]:
        # Filter indices yang termasuk domain ini (case-insensitive)
        domain_indices = [
            idx for c in domain_classes
            if (idx := _resolve_class(c)) is not None
        ]

        if not domain_indices:
            logger.warning(f"No classes found for domain {domain_name}. Check class names.")
            continue

        # Mask samples yang true label-nya termasuk domain ini
        mask = np.isin(y_true, domain_indices)
        if mask.sum() == 0:
            logger.warning(f"No test samples found for domain {domain_name}.")
            continue

        y_true_dom = y_true[mask]
        y_pred_dom = y_pred[mask]

        # Map ke local indices (0..N) untuk classification_report
        # Gunakan nama kelas aktual (dari dataset), bukan dari domain_classes set
        local_names = [
            class_names[idx] for c in domain_classes
            if (idx := _resolve_class(c)) is not None
        ]
        local_idx   = [_resolve_class(c) for c in domain_classes if _resolve_class(c) is not None]
        idx_remap   = {orig: new for new, orig in enumerate(local_idx)}

        y_true_loc = np.array([idx_remap[i] for i in y_true_dom])
        y_pred_loc = np.array([idx_remap.get(i, -1) for i in y_pred_dom])

        # Filter pred yang valid (model mungkin prediksi kelas dari domain lain)
        valid_mask  = y_pred_loc >= 0
        cross_domain_errors = int((~valid_mask).sum())

        domain_accuracy = float(np.mean(y_true_dom == y_pred_dom))

        # F1 hanya dari samples yang prediksinya in-domain
        dom_report = classification_report(
            y_true_loc[valid_mask],
            y_pred_loc[valid_mask],
            target_names=local_names,
            output_dict=True,
            zero_division=0,
        ) if valid_mask.sum() > 0 else {}

        weighted_avg = dom_report.get("weighted avg", {})

        domain_results[domain_name] = {
            "n_samples":              int(mask.sum()),
            "n_classes":              len(local_names),
            "class_names":            local_names,
            "accuracy":               round(domain_accuracy, 4),
            "f1_weighted":            round(float(weighted_avg.get("f1-score",  0)), 4),
            "precision_weighted":     round(float(weighted_avg.get("precision", 0)), 4),
            "recall_weighted":        round(float(weighted_avg.get("recall",    0)), 4),
            "cross_domain_errors":    cross_domain_errors,
            "cross_domain_error_pct": round(cross_domain_errors / mask.sum() * 100, 2) if mask.sum() > 0 else 0,
            "per_class_f1": {
                name: round(float(dom_report.get(name, {}).get("f1-score", 0)), 4)
                for name in local_names
            },
        }

        logger.info(
            f"\nDomain: {domain_name}"
            f"\n  Samples:              {mask.sum()}"
            f"\n  Accuracy:             {domain_accuracy:.4f}"
            f"\n  F1 (Weighted):        {weighted_avg.get('f1-score', 0):.4f}"
            f"\n  Cross-domain errors:  {cross_domain_errors} ({domain_results[domain_name]['cross_domain_error_pct']:.1f}%)"
            f"\n  Per-class F1:"
        )
        for name, f1 in domain_results[domain_name]["per_class_f1"].items():
            bar = "█" * int(f1 * 20)
            logger.info(f"    {name:<25} {f1:.3f}  {bar}")

    return domain_results


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    """Simpan confusion matrix sebagai PNG resolusi tinggi."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 8},
        linewidths=0.3,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(
        "Normalized Confusion Matrix — DermaScan EfficientNetB3",
        fontsize=13, pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()

    out_path = SAVE_DIR / "confusion_matrix.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Seed ──────────────────────────────────────────────────────────────────
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_splits(DATA_CSV)

    train_gen_p1, val_gen, test_gen = make_generators(
        train_df, val_df, test_df, BATCH_SIZE_P1
    )
    num_classes  = len(train_gen_p1.class_indices)
    class_names  = list(train_gen_p1.class_indices.keys())

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")

    with open(SAVE_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info(f"Saved {num_classes} class names to {SAVE_DIR}/class_names.json")

    class_weights = get_class_weights(train_gen_p1)

    # ── Phase 1: Train Classification Head ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 — Training classification head (EfficientNetB3, frozen base)")
    logger.info("=" * 60)

    model = build_model(num_classes, freeze_base=True)

    # Phase 1: Focal Loss dengan LR lebih tinggi
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.25),
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

    # ── Phase 2: Fine-tune Top Layers ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"PHASE 2 — Fine-tuning top {UNFREEZE_TOP_N} layers (EfficientNetB3)")
    logger.info("=" * 60)

    train_gen_p2, val_gen_p2, test_gen_p2 = make_generators(
        train_df, val_df, test_df, BATCH_SIZE_P2
    )

    # Unfreeze top N layers (BatchNorm tetap frozen — best practice)
    model.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    for layer in model.layers[:-UNFREEZE_TOP_N]:
        layer.trainable = False

    # Phase 2: LR sangat kecil + focal loss untuk fine-tuning stabil
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )

    model.fit(
        train_gen_p2,
        validation_data=val_gen_p2,
        epochs=EPOCHS_P2,
        class_weight=class_weights,
        callbacks=get_callbacks(phase=2),
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    overall_metrics, per_class_metrics = evaluate_model(model, test_gen_p2, class_names)

    # ── Simpan model & metrik LEBIH DULU sebelum domain eval ─────────────────
    # (domain eval bisa gagal, tapi model dan metrik utama harus tersimpan)
    final_path = str(SAVE_DIR / "skin_model_best.keras")
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    with open(SAVE_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    with open(SAVE_DIR / "per_class_metrics.json", "w") as f:
        json.dump(per_class_metrics, f, indent=2)

    logger.info("Core metrics saved.")

    # ── Per-domain evaluation (HAM10000 vs SD-198) ────────────────────────────
    # Dijalankan SETELAH save agar crash di sini tidak menghilangkan model
    try:
        test_gen_p2.reset()
        y_pred_probs_eval = model.predict(test_gen_p2, verbose=0)
        y_pred_eval = np.argmax(y_pred_probs_eval, axis=1)
        y_true_eval = test_gen_p2.classes

        domain_metrics = evaluate_per_domain(y_true_eval, y_pred_eval, class_names, per_class_metrics)

        with open(SAVE_DIR / "domain_metrics.json", "w") as f:
            json.dump(domain_metrics, f, indent=2)
        logger.info("Domain metrics saved.")
    except Exception as e:
        logger.warning(f"Per-domain evaluation failed (non-fatal): {e}")
        domain_metrics = {}

    logger.info("Training complete.")

    # ── Summary Report ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Backbone:     EfficientNetB3 (300x300)")
    logger.info(f"Num Classes:  {num_classes}")
    logger.info(f"Test Acc:     {overall_metrics['test_accuracy']:.4f}")
    logger.info(f"F1 Weighted:  {overall_metrics['test_f1_weighted']:.4f}")
    logger.info(f"ROC-AUC Mac:  {overall_metrics['roc_auc_macro']:.4f}")
    logger.info(f"ROC-AUC Wgt:  {overall_metrics['roc_auc_weighted']:.4f}")

    logger.info("\nPer-class F1:")
    for name, m in sorted(per_class_metrics.items(), key=lambda x: x[1]["f1_score"]):
        bar = "█" * int(m["f1_score"] * 20)
        logger.info(f"  {name:<30} {m['f1_score']:.3f}  {bar}")

    logger.info("\nPer-domain Summary (Dataset Transparency):")
    for domain, dm in domain_metrics.items():
        logger.info(
            f"  {domain:<12} → Acc: {dm['accuracy']:.3f} | "
            f"F1: {dm['f1_weighted']:.3f} | "
            f"Cross-domain err: {dm['cross_domain_error_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()