"""Deduplicate / find prior appearances of a person."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.face_pipeline.matcher import FaceMatcher
from core.storage.models import FaceMatch, Photo


def find_prior_appearances(
    matcher: FaceMatcher,
    path: str | Path,
    *,
    top_k: int = 20,
    threshold: Optional[float] = None,
) -> list[FaceMatch]:
    """Return indexed faces that match the probe image above threshold."""
    return matcher.find_matches(path, top_k=top_k, threshold=threshold)


def unique_photos(matches: list[FaceMatch]) -> list[Photo]:
    """Collapse face matches to unique photos, keeping best score order."""
    seen: set[int] = set()
    photos: list[Photo] = []
    for match in matches:
        if match.photo.id in seen:
            continue
        seen.add(match.photo.id)
        photos.append(match.photo)
    return photos
