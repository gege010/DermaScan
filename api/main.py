"""
api/main.py
───────────
DermaScan FastAPI Backend

Endpoints:
  GET  /health    — Service health check
  POST /predict   — Skin condition classification + AI analysis
  GET  /docs      — Swagger UI (auto-generated)
"""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from utils.gradcam import GradCAM
from utils.groq_analyzer import GroqAnalyzer
from utils.logger import get_logger
from utils.tavily_search import TavilySearch

from dotenv import load_dotenv
load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────

logger = get_logger(__name__)

MODEL_PATH = Path("model/saved_model/skin_model_best.keras")
CLASS_NAMES_PATH = Path("model/saved_model/class_names.json") 
IMG_SIZE = (224, 224)
TOP_K = 3

# Global model state (loaded once at startup)
model: tf.keras.Model | None = None
CLASS_NAMES: list[str] = []
gradcam: GradCAM | None = None
groq_analyzer = GroqAnalyzer()
tavily_search = TavilySearch()

# Urgency mapping
URGENCY_MAP = {
    "Melanoma": "SEGERA",
    "Basal Cell Carcinoma": "SEGERA",
    "Actinic Keratosis": "SEGERA",
    "SJS/TEN": "SEGERA",
    "Benign Keratosis": "PERHATIAN",
    "Vascular Lesion": "PERHATIAN",
    "Psoriasis Nail": "PERHATIAN",
    "Vitiligo": "PERHATIAN",
    "Acne": "PERHATIAN",
}

# ─── Lifespan (model loading) ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, CLASS_NAMES, gradcam

    logger.info("Loading DermaScan model...")
    try:
        if not MODEL_PATH.exists():
            logger.warning(f"Model file not found at {MODEL_PATH}.")
        else:
            model = tf.keras.models.load_model(str(MODEL_PATH))
            gradcam = GradCAM(model)
            logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

    try:
        if CLASS_NAMES_PATH.exists():
            with open(CLASS_NAMES_PATH) as f:
                raw_data = json.load(f)
                
                if isinstance(raw_data, dict) and "class_names" in raw_data:
                    codes = raw_data["class_names"]
                    display_dict = raw_data.get("class_display_names", {})
                    CLASS_NAMES = [display_dict.get(c, c) for c in codes]
                elif isinstance(raw_data, dict):
                    CLASS_NAMES = list(raw_data.values())
                else:
                    CLASS_NAMES = raw_data
                    
            logger.info(f"Loaded {len(CLASS_NAMES)} class names.")
        else:
            logger.warning("Class names metadata not found.")
    except Exception as e:
        logger.error(f"Failed to load class names: {e}")

    yield  # Menandakan startup selesai, aplikasi siap menerima request

    logger.info("Shutting down DermaScan API.")
    
    if model is not None:
        del model
        tf.keras.backend.clear_session()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DermaScan API",
    description="AI-powered skin condition classifier using EfficientNetB0 + Groq LLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─── Response Schemas ─────────────────────────────────────────────────────────

class ClassProbability(BaseModel):
    class_name: str = Field(..., alias="class")
    probability: float

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3_predictions: list[ClassProbability]
    urgency_level: str
    ai_analysis: dict[str, Any]
    references: Any
    gradcam_heatmap_base64: str | None = None
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    version: str = "1.0.0"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.uint8)
    # Pembagian / 255.0 agar model membaca rentang [0, 1]
    model_input = np.expand_dims(arr.astype(np.float32) / 255.0, axis=0)
    return model_input, arr

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Returns service health status and model load state."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        num_classes=len(CLASS_NAMES),
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(file: UploadFile = File(..., description="Skin image (JPEG or PNG)")):
    """
    Analyze a skin image and return predictions, Grad-CAM, Groq analysis, and Tavily references.
    """
    if model is None or not CLASS_NAMES:
        raise HTTPException(status_code=503, detail="Model not loaded. Please retry later.")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    t0 = time.perf_counter()

    # Read and preprocess image (Synchronous read untuk menghindari error Threadpool)
    try:
        image_bytes = file.file.read()
        model_input, original_arr = preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    finally:
        file.file.close()

    # CNN Prediction
    predictions = model.predict(model_input, verbose=0)[0]
    top_k_indices = np.argsort(predictions)[::-1][:TOP_K]

    predicted_idx = int(top_k_indices[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])

    top_k = [
        ClassProbability(**{"class": CLASS_NAMES[i], "probability": float(predictions[i])})
        for i in top_k_indices
    ]

    # Grad-CAM
    heatmap_b64 = None
    if gradcam is not None:
        try:
            heatmap = gradcam.generate(model_input, class_idx=predicted_idx)
            overlay = gradcam.overlay_heatmap(original_arr, heatmap)
            heatmap_b64 = gradcam.to_base64(overlay)
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed (non-fatal): {e}")

    # AI Analysis (Groq)
    try:
        ai_result = groq_analyzer.analyze(predicted_class, confidence)
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        ai_result = {"error": "AI analysis unavailable"}

    # Web Search (Tavily) - Dipisah dari Groq agar jika salah satu gagal, yang lain tetap jalan
    try:
        references = tavily_search.search(f"{predicted_class} skin condition treatment")
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        references = []

    inference_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Prediction: {predicted_class} ({confidence:.2%}) | "
        f"Urgency: {URGENCY_MAP.get(predicted_class, 'NORMAL')} | "
        f"Time: {inference_ms:.0f}ms"
    )

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        top_3_predictions=top_k,
        urgency_level=URGENCY_MAP.get(predicted_class, "NORMAL"),
        ai_analysis=ai_result,
        references=references,
        gradcam_heatmap_base64=heatmap_b64,
        inference_time_ms=round(inference_ms, 2),
    )