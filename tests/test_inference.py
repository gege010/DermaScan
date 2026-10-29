from pathlib import Path

import numpy as np
import pytest

from src.dermascan.imaging import PreparedImage
from src.dermascan.inference import ModelService


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.input = None

    def predict(self, tensor, verbose=0):
        self.input = tensor
        return np.array([self.result])


def service(scores):
    result = ModelService(Path("missing"))
    result.model = FakeModel(scores)
    result.classes = ["acne", "mel"]
    result.kind = "legacy-research"
    return result


def image():
    return PreparedImage(np.full((320, 320, 3), 160, dtype=np.uint8), 320, 320, [])


def test_modality_conflict_does_not_renormalize():
    model = service([0.1, 0.9])
    assert model.predict(image(), "clinical") == []
    assert model.predict(image(), "dermoscopy")[0]["code"] == "mel"
    assert model.model.input.shape == (1, 300, 300, 3)
    assert model.model.input.max() == 160


def test_compact_model_never_runs_on_wrong_modality():
    model = service([0.7, 0.3])
    model.kind = "embedding-research"
    model.domain = "dermoscopy"
    assert model.predict(image(), "clinical") == []
    assert model.model.input is None


@pytest.mark.parametrize("scores", [[float("nan"), 0.0], [-0.2, 1.2], [0.1, 0.1]])
def test_bad_model_output_is_rejected(scores):
    with pytest.raises(ValueError, match="distribution"):
        service(scores).predict(image(), "clinical")
