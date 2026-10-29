"""Optional inference; no model download or external photo transfer at request time."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from .clinical import DISPLAY_NAMES
from .imaging import PreparedImage

logger = logging.getLogger(__name__)
HAM_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


class ModelService:
    def __init__(self, directory: Path, enable_legacy: bool = False):
        self.directory, self.enable_legacy = directory, enable_legacy
        self.model = self.classifier = None
        self.classes: list[str] = []
        self.kind, self.domain, self.size = "unavailable", "clinical", (300, 300)
        self.lock = threading.Lock()

    def load(self) -> None:
        manifest = self.directory / "embedding_manifest.json"
        legacy = self.directory / "skin_model_best.keras"
        if not manifest.exists() and not (self.enable_legacy and legacy.exists()):
            return
        try:
            import tensorflow as tf

            if manifest.exists():
                import joblib

                meta = json.loads(manifest.read_text())
                if meta.get("schema_version") != 1 or meta.get("domain") not in {
                    "clinical",
                    "dermoscopy",
                }:
                    raise ValueError("Unsupported model manifest")
                self.classes, self.size, self.domain = (
                    meta["classes"],
                    tuple(meta["input_size"]),
                    meta["domain"],
                )
                if not self.classes or len(self.classes) != len(set(self.classes)):
                    raise ValueError("Invalid class metadata")
                self.model = tf.keras.models.load_model(
                    self.directory / "embedding_backbone.keras", compile=False
                )
                # Only trusted operator-created artifacts; never load uploaded joblib files.
                self.classifier = joblib.load(self.directory / "classifier.joblib")
                if list(self.classifier.classes_) != self.classes:
                    raise ValueError("Classifier label order mismatch")
                self.kind = "embedding-research"
            else:
                try:
                    self.model = tf.keras.models.load_model(legacy, compile=False)
                except (ValueError, TypeError):
                    import tf_keras

                    self.model = tf_keras.models.load_model(str(legacy), compile=False)
                self.classes = json.loads((self.directory / "class_names.json").read_text())
                if not isinstance(self.classes, list) or self.model.output_shape[-1] != len(
                    self.classes
                ):
                    raise ValueError("Model label order/shape mismatch")
                self.size = tuple(self.model.input_shape[1:3])
                self.kind = "legacy-research"
        except Exception:
            logger.exception("Local model could not be initialized")
            self.model = self.classifier = None
            self.kind = "unavailable"

    def describe(self) -> dict:
        return {
            "available": self.model is not None,
            "mode": self.kind,
            "clinical_validation": False,
            "class_count": len(self.classes),
            "message": "Skor model adalah hasil eksperimen, bukan probabilitas diagnosis.",
        }

    def predict(self, image: PreparedImage, image_kind: str) -> list[dict]:
        if self.model is None or (self.kind == "embedding-research" and image_kind != self.domain):
            return []
        with self.lock:
            output = np.asarray(self.model.predict(image.tensor(self.size), verbose=0))[0]
            if self.classifier is not None:
                output = self.classifier.predict_proba(output[None, :])[0]
        if (
            len(output) != len(self.classes)
            or not np.isfinite(output).all()
            or (output < 0).any()
            or not np.isclose(output.sum(), 1, atol=0.01)
        ):
            raise ValueError("Invalid class distribution")
        indices = np.argsort(output)[::-1][:3]
        if self.kind == "legacy-research":
            if (self.classes[int(indices[0])] in HAM_CLASSES) != (image_kind == "dermoscopy"):
                return []
        return [
            {
                "code": self.classes[int(i)],
                "name": DISPLAY_NAMES.get(self.classes[int(i)], self.classes[int(i)]),
                "score": round(float(output[i]), 4),
            }
            for i in indices
        ]
