"""Verify local legacy loading and tensor compatibility, not diagnostic accuracy.

Run in an optional ML environment: python -m scripts.smoke_legacy_model
Only a synthetic checker pattern is used. No patient photos or new training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.dermascan.imaging import PreparedImage
from src.dermascan.inference import ModelService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    service = ModelService(args.model_dir, enable_legacy=True)
    service.load()
    if service.model is None:
        raise RuntimeError(
            "Legacy model unavailable. Install the documented ML runtime and inspect startup errors."
        )
    y, x = np.indices((320, 320))
    gray = ((x // 16 + y // 16) % 2 * 100 + 70).astype(np.uint8)
    image = PreparedImage(np.stack([gray, gray, gray], axis=-1), 320, 320, [])
    output = np.asarray(service.model.predict(image.tensor(service.size), verbose=0))
    assert output.shape == (1, len(service.classes))
    assert np.isfinite(output).all() and (output >= 0).all()
    assert np.isclose(output[0].sum(), 1, atol=0.01)
    report = {
        "loaded": True,
        "mode": service.kind,
        "input_shape": list(service.model.input_shape),
        "output_shape": list(output.shape),
        "class_count": len(service.classes),
        "finite_distribution": True,
        "pixel_range": [
            int(image.tensor(service.size).min()),
            int(image.tensor(service.size).max()),
        ],
        "clinical_accuracy_evaluated": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
