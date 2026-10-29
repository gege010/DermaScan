import pytest

from scripts.train_compact import validate_manifest


def test_group_leakage_is_rejected(tmp_path):
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    rows = [
        {"image_path": "a", "label": "a", "group_id": "patient-1", "split": "train"},
        {"image_path": "b", "label": "a", "group_id": "patient-1", "split": "test"},
    ]
    with pytest.raises(ValueError, match="Group leakage"):
        validate_manifest(rows, tmp_path)


def test_duplicate_content_is_rejected(tmp_path):
    (tmp_path / "a").write_bytes(b"same image")
    (tmp_path / "b").write_bytes(b"same image")
    with pytest.raises(ValueError, match="Duplicate image"):
        validate_manifest(
            [
                {"image_path": p, "label": "a", "group_id": p, "split": s}
                for p, s in [("a", "train"), ("b", "test")]
            ],
            tmp_path,
        )
