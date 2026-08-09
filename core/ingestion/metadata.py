"""EXIF extraction — date taken and GPS when present."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import exifread

from core.ingestion.image_io import load_image


@dataclass
class ImageMetadata:
    width: Optional[int] = None
    height: Optional[int] = None
    taken_at: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


def _rationally_to_float(values) -> Optional[float]:
    try:
        deg, minutes, seconds = values
        return float(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _gps_coord(tags: dict, coord_key: str, ref_key: str) -> Optional[float]:
    if coord_key not in tags or ref_key not in tags:
        return None
    value = _rationally_to_float(tags[coord_key].values)
    if value is None:
        return None
    ref = str(tags[ref_key].values)
    if ref in ("S", "W"):
        value = -value
    return value


def extract_metadata(path: str | Path) -> ImageMetadata:
    path = Path(path)
    meta = ImageMetadata()

    try:
        meta.width, meta.height = load_image(path).size
    except OSError:
        pass

    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except OSError:
        return meta

    date_tag = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    if date_tag:
        raw = str(date_tag.values if hasattr(date_tag, "values") else date_tag)
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                meta.taken_at = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue

    meta.latitude = _gps_coord(tags, "GPS GPSLatitude", "GPS GPSLatitudeRef")
    meta.longitude = _gps_coord(tags, "GPS GPSLongitude", "GPS GPSLongitudeRef")
    return meta
