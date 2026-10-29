import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from deployment.api.main import create_app
from src.dermascan.imaging import prepare_image
from src.dermascan.inference import ModelService


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANALYSIS_RATE_LIMIT", "100")
    with TestClient(create_app(ModelService(Path("not-a-model-directory")))) as app:
        yield app


def photo(size=(320, 320), dark=False):
    pixels = (
        np.zeros((*size, 3), dtype=np.uint8)
        if dark
        else np.random.default_rng(42).integers(45, 210, (*size, 3), dtype=np.uint8)
    )
    file = io.BytesIO()
    Image.fromarray(pixels).save(file, format="PNG")
    return file.getvalue()


def submit(client, contents=None, **fields):
    return client.post(
        "/api/v1/analyze",
        data={"consent": "true", **fields},
        files={"file": ("photo.png", contents or photo(), "image/png")},
    )


def test_api_works_without_tensorflow_and_never_fabricates_diagnosis(client):
    result = submit(client).json()
    assert result["assessment"]["status"] == "unsupported"
    assert result["candidates"] == []
    assert result["model"]["clinical_validation"] is False
    assert len(result["references"]) == 3


def test_consent_required(client):
    assert submit(client, consent="false").status_code == 422


@pytest.mark.parametrize("symptom", ["breathing", "blisters_fever"])
def test_emergency_overrides_photo_and_model(client, symptom):
    response = client.post(
        "/api/v1/analyze", data={"consent": "true", "symptoms": json.dumps([symptom])}
    )
    assert response.status_code == 200
    assert response.json()["assessment"]["status"] == "urgent"
    assert response.json()["education"] == []
    assert response.json()["candidates"] == []


def test_changing_lesion_cannot_be_reassured_by_model(client):
    result = submit(client, symptoms='["changing_mole"]').json()
    assert result["assessment"]["status"] == "review"
    assert not result["candidates"]


@pytest.mark.parametrize("symptoms", ["{}", '["invalid"]', "[{}]", "invalid"])
def test_untrusted_symptoms_rejected(client, symptoms):
    assert submit(client, symptoms=symptoms).status_code == 422


def test_image_signature_not_just_mime(client):
    assert submit(client, b"this is not an image").status_code == 422


def test_image_quality_retakes(client):
    assert submit(client, photo(dark=True)).json()["assessment"]["status"] == "retake"


def test_missing_file_and_invalid_enum(client):
    assert client.post("/api/v1/analyze", data={"consent": "true"}).status_code == 422
    assert submit(client, area="invalid").status_code == 422


def test_image_range_contract():
    image = prepare_image(photo())
    array = image.tensor((300, 300))
    assert array.shape == (1, 300, 300, 3)
    assert array.dtype == np.float32
    assert array.max() > 1


def test_health_is_explicit_and_readiness_configurable(client, monkeypatch):
    assert client.get("/health").json()["model"]["available"] is False
    assert client.get("/ready").status_code == 200
    monkeypatch.setenv("REQUIRE_MODEL", "true")
    assert client.get("/ready").status_code == 503
    assert client.get("/health").headers["cache-control"] == "no-store"


def test_model_exception_degrades_without_exposing_error(client):
    def broken(*args):
        raise RuntimeError("sensitive internal detail")

    client.app.state.model_service.predict = broken
    response = submit(client)
    assert response.status_code == 200
    assert "sensitive internal detail" not in response.text
    assert response.json()["assessment"]["status"] == "unsupported"


def test_raw_label_mapping_does_not_fall_back_to_normal(client):
    client.app.state.model_service.predict = lambda *args: [
        {"code": "mel", "name": "Melanoma", "score": 0.99}
    ]
    result = submit(client).json()
    assert result["assessment"]["status"] == "review"
    assert result["model"]["clinical_validation"] is False


def test_file_size_limit(client):
    response = submit(client, b"0" * (10 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("ANALYSIS_RATE_LIMIT", "1")
    with TestClient(create_app(ModelService(Path("missing")))) as client:
        assert submit(client).status_code == 200
        assert submit(client).status_code == 429


def test_content_and_legacy_migration(client):
    assert len(client.get("/api/v1/knowledge").json()) >= 6
    assert client.post("/predict").status_code == 410
