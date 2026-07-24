"""Generate and cache image thumbnails on disk."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def generate_thumbnail(
    source: str | Path,
    dest_dir: str | Path,
    *,
    size: int = 256,
    photo_id: int | None = None,
) -> Path:
    """Create a JPEG thumbnail; returns the output path."""
    source = Path(source)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{photo_id}_{source.stem}" if photo_id is not None else source.stem
    dest = dest_dir / f"{stem}.jpg"

    with Image.open(source) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        img.save(dest, format="JPEG", quality=85)

    return dest
