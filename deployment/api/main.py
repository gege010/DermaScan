"""DermaScan v3: bounded uploads, symptom-led escalation, optional research model."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.dermascan.clinical import (
    REVIEW_CLASSES,
    SOURCES,
    SYMPTOMS,
    education_for,
    initial_assessment,
)
from src.dermascan.imaging import MAX_BYTES, prepare_image
from src.dermascan.inference import ModelService
from src.dermascan.schemas import AnalysisResponse, ModelInfo

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
logger = logging.getLogger("dermascan.api")
VERSION = "3.0.0"


def create_app(service: ModelService | None = None) -> FastAPI:
    model_service = service or ModelService(
        Path(os.getenv("MODEL_DIR", str(ROOT / "models"))),
        os.getenv("ENABLE_LEGACY_MODEL", "false").lower() == "true",
    )
    requests_by_client: OrderedDict[str, deque] = OrderedDict()
    max_requests = int(os.getenv("ANALYSIS_RATE_LIMIT", "12"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(model_service.load)
        app.state.analysis_slots = asyncio.Semaphore(2)
        yield

    app = FastAPI(
        title="DermaScan",
        version=VERSION,
        lifespan=lifespan,
        description="Skin education and research image assessment. Not a diagnostic medical device.",
    )
    app.state.model_service = model_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv(
            "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def security(request: Request, call_next):
        request_id = uuid4().hex
        if request.url.path in {"/api/v1/analyze", "/predict"}:
            # Do not trust client-provided forwarding headers.
            client = request.client.host if request.client else "unknown"
            now = time.monotonic()
            bucket = requests_by_client.setdefault(client, deque())
            requests_by_client.move_to_end(client)
            while bucket and bucket[0] < now - 60:
                bucket.popleft()
            if len(bucket) >= max_requests:
                return JSONResponse(
                    {"detail": "Terlalu banyak analisis. Coba lagi dalam satu menit."},
                    status_code=429,
                    headers={"Retry-After": "60", "Cache-Control": "no-store"},
                )
            bucket.append(now)
            while len(requests_by_client) > 2048:
                requests_by_client.popitem(last=False)
            try:
                if int(request.headers.get("content-length", "0")) > MAX_BYTES + 64 * 1024:
                    return JSONResponse({"detail": "Foto maksimal 10 MB."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Ukuran permintaan tidak valid."}, status_code=400)
        response = await call_next(request)
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
                "X-Request-ID": request_id,
            }
        )
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "version": VERSION, "model": model_service.describe()}

    @app.get("/ready")
    def ready():
        if os.getenv("REQUIRE_MODEL", "false").lower() == "true" and model_service.model is None:
            return JSONResponse({"status": "model_unavailable"}, status_code=503)
        return {"status": "ready", "mode": model_service.kind}

    @app.get("/api/v1/knowledge")
    def knowledge():
        return json.loads((ROOT / "content" / "education.json").read_text(encoding="utf-8"))

    @app.get("/api/v1/model", response_model=ModelInfo)
    def model_info():
        return model_service.describe()

    @app.post("/predict", include_in_schema=False)
    def legacy_predict():
        raise HTTPException(
            410,
            "Endpoint lama dihentikan. Gunakan /api/v1/analyze dengan persetujuan dan konteks foto.",
        )

    @app.post("/api/v1/analyze", response_model=AnalysisResponse)
    async def analyze(
        request: Request,
        consent: Annotated[bool, Form()],
        image_kind: Annotated[Literal["clinical", "dermoscopy"], Form()] = "clinical",
        symptoms: Annotated[str, Form(max_length=1000)] = "[]",
        duration: Annotated[Literal["days", "weeks", "months", "unknown"], Form()] = "unknown",
        area: Annotated[Literal["face", "body", "hands", "scalp", "other"], Form()] = "face",
        file: Annotated[UploadFile | None, File()] = None,
    ):
        if not consent:
            raise HTTPException(422, "Persetujuan pemrosesan foto diperlukan.")
        try:
            selected = json.loads(symptoms)
            if (
                not isinstance(selected, list)
                or len(selected) > len(SYMPTOMS)
                or any(not isinstance(s, str) or s not in SYMPTOMS for s in selected)
            ):
                raise ValueError
            selected = list(dict.fromkeys(selected))
        except (ValueError, TypeError):
            raise HTTPException(422, "Pilihan gejala tidak valid.")
        assessment = initial_assessment(selected)
        quality = None
        candidates: list[dict] = []
        warnings: list[str] = []
        started = time.perf_counter()
        if assessment["status"] not in {"urgent", "review"}:
            if file is None:
                raise HTTPException(422, "Pilih satu foto kulit terlebih dahulu.")
            try:
                contents = await file.read(MAX_BYTES + 1)
            finally:
                await file.close()
            if len(contents) > MAX_BYTES:
                raise HTTPException(413, "Foto maksimal 10 MB.")
            try:
                image = await run_in_threadpool(prepare_image, contents)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            quality = {
                "width": image.width,
                "height": image.height,
                "issues": image.issues,
                "scope": "Pemeriksaan teknis; tidak membuktikan foto berisi kulit.",
            }
            if image.issues:
                assessment = {
                    "status": "retake",
                    "title": "Mari ambil foto yang lebih jelas",
                    "summary": "Detail foto belum cukup untuk analisis yang berguna.",
                    "next_steps": image.issues,
                }
            else:
                try:
                    async with asyncio.timeout(30):
                        async with request.app.state.analysis_slots:
                            candidates = await run_in_threadpool(
                                model_service.predict, image, image_kind
                            )
                except TimeoutError:
                    raise HTTPException(503, "Layanan analisis sedang penuh. Coba lagi sebentar.")
                except Exception:
                    logger.exception("Research inference failed")
                    warnings.append(
                        "Model lokal gagal memproses foto. Tidak ada hasil model yang digunakan."
                    )
                if candidates:
                    assessment = {
                        "status": "research",
                        "title": "Pola visual tersedia untuk ditinjau",
                        "summary": "Model menemukan kemiripan dengan label dataset. Hasil ini belum membuktikan kondisi kulit Anda dan tidak boleh digunakan untuk memilih obat.",
                        "next_steps": initial_assessment([])["next_steps"],
                    }
                    if candidates[0]["code"] in REVIEW_CLASSES:
                        assessment = initial_assessment(["changing_mole"])
                        assessment["summary"] = (
                            "Hasil eksperimen menyentuh kategori yang perlu evaluasi langsung. Ini bukan konfirmasi penyakit; diskusikan keluhan dengan dokter kulit."
                        )
                    warnings.append(
                        "Skor belum dikalibrasi secara klinis. Skor tinggi tidak menjamin prediksi benar, termasuk pada foto di luar cakupan."
                    )
                else:
                    warnings.append(
                        "Tidak ada model tervalidasi yang dapat menyimpulkan kondisi dari foto ini. Tidak ada diagnosis yang dibuat."
                    )
        if file is not None and not file.file.closed:
            await file.close()
        return {
            "id": uuid4().hex,
            "assessment": assessment,
            "context": {
                "symptoms": [SYMPTOMS[s] for s in selected],
                "duration": duration,
                "area": area,
                "image_kind": image_kind,
            },
            "quality": quality,
            "candidates": candidates,
            "education": [] if assessment["status"] == "urgent" else education_for(selected),
            "references": SOURCES,
            "warnings": warnings,
            "model": model_service.describe(),
            "privacy": "Foto tidak disimpan oleh aplikasi dan tidak dikirim ke penyedia AI eksternal.",
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    return app


app = create_app()
