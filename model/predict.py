from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from utils.gradcam import GradCAM
from utils.logger import get_logger

logger = get_logger(__name__)

# Jadikan default constant, tapi bisa di-override via argparse
DEFAULT_MODEL_PATH = Path("model/saved_model/skin_model_best.keras") 
DEFAULT_CLASS_NAMES_PATH = Path("model/saved_model/class_names.json") # Diubah ke class_names.json
IMG_SIZE = (224, 224)


def load_artifacts(model_path: Path, class_names_path: Path) -> tuple[tf.keras.Model, list[str]]:
    """Load model and class names from disk once."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python model/train.py` first."
        )
    model = tf.keras.models.load_model(str(model_path))
    
    with open(class_names_path) as f:
        class_names = json.load(f)
        
    logger.info(f"Loaded model from {model_path} with {len(class_names)} class names.")
    return model, class_names


def preprocess(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess image."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.uint8)
    model_input = np.expand_dims(arr.astype(np.float32) / 255.0, axis=0)
    return model_input, arr


def predict(
    image_path: str | Path,
    model: tf.keras.Model,
    class_names: list[str],
    top_k: int = 3,
    generate_gradcam: bool = False,
) -> dict:
    """Run inference on a single image using a pre-loaded model."""
    model_input, original_arr = preprocess(image_path)

    probs = model.predict(model_input, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    result = {
        "predicted_class": class_names[top_indices[0]],
        "confidence": float(probs[top_indices[0]]),
        "top_predictions": [
            {"class": class_names[i], "probability": float(probs[i])}
            for i in top_indices
        ],
        "gradcam_heatmap_base64": None,
    }

    if generate_gradcam:
        try:
            cam = GradCAM(model)
            heatmap = cam.generate(model_input, class_idx=int(top_indices[0]))
            overlay = cam.overlay_heatmap(original_arr, heatmap)
            result["gradcam_heatmap_base64"] = cam.to_base64(overlay)
            logger.info("Grad-CAM heatmap generated.")
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="DermaScan Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to the .keras model file")
    parser.add_argument("--classes", type=str, default=str(DEFAULT_CLASS_NAMES_PATH), help="Path to class_names.json")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmap")
    args = parser.parse_args()

    # Load artifacts HANYA SATU KALI di sini
    model, class_names = load_artifacts(Path(args.model), Path(args.classes))

    # Oper model dan class_names ke fungsi predict
    result = predict(
        image_path=args.image, 
        model=model, 
        class_names=class_names, 
        top_k=args.top_k, 
        generate_gradcam=args.gradcam
    )

    print(f"\n{'='*50}")
    print(f"  PREDICTION: {result['predicted_class']}")
    print(f"  CONFIDENCE: {result['confidence']:.2%}")
    print(f"{'='*50}")
    print("\nTop predictions:")
    for i, p in enumerate(result["top_predictions"], 1):
        bar = "█" * int(p["probability"] * 30)
        print(f"  {i}. {p['class']:<30} {p['probability']:.2%}  {bar}")

    if result["gradcam_heatmap_base64"]:
        out_path = Path(args.image).stem + "_gradcam.png"
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(result["gradcam_heatmap_base64"]))
        print(f"\nGrad-CAM saved to: {out_path}")


if __name__ == "__main__":
    main()