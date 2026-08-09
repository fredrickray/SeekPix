"""Image loading with HEIC/HEIF support (iPhone photos)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ModuleNotFoundError:
    pass


def load_image(path: str | Path) -> Image.Image:
    """Open any supported image as RGB, honouring EXIF orientation."""
    image = Image.open(Path(path))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load as a BGR array for OpenCV-based models such as InsightFace."""
    rgb = np.asarray(load_image(path))
    return rgb[:, :, ::-1].copy()
