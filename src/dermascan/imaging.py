"""One image contract for API, CLI and embedding training. Pixels stay in [0,255]."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass
class PreparedImage:
    pixels: np.ndarray
    width: int
    height: int
    issues: list[str]

    def tensor(self, size: tuple[int, int]) -> np.ndarray:
        resized = Image.fromarray(self.pixels).resize(size, Image.Resampling.NEAREST)
        return np.asarray(resized, dtype=np.float32)[None, ...]


def prepare_image(contents: bytes) -> PreparedImage:
    if not contents or len(contents) > MAX_BYTES:
        raise ValueError("Gunakan foto JPG, PNG, atau WebP berukuran maksimal 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as source:
                if source.format not in ALLOWED_FORMATS:
                    raise ValueError("Format foto harus JPG, PNG, atau WebP.")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Resolusi foto terlalu besar. Gunakan maksimal 20 megapiksel.")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Gunakan foto diam, bukan gambar animasi.")
                source.verify()
            with Image.open(io.BytesIO(contents)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                pixels = np.asarray(image).copy()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Foto tidak dapat dibaca. Pilih file gambar lain.") from exc
    issues: list[str] = []
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    if min(width, height) < 224:
        issues.append(
            "Resolusi terlalu rendah. Gunakan foto dengan sisi terpendek minimal 224 piksel."
        )
    if float(gray.mean()) < 35:
        issues.append("Foto terlalu gelap. Ambil ulang di dekat cahaya alami.")
    if float(gray.mean()) > 235:
        issues.append("Foto terlalu terang. Hindari flash dan cahaya langsung.")
    if float(gray.std()) < 8:
        issues.append(
            "Detail foto tidak cukup terlihat. Pastikan area kulit tajam dan memenuhi bingkai."
        )
    return PreparedImage(pixels, width, height, issues)
