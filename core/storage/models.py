"""Domain dataclasses shared across pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Photo:
    id: int
    filepath: str
    filename: str
    thumbnail_path: Optional[str] = None
    taken_at: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    clip_vector_id: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class Face:
    id: int
    photo_id: int
    face_vector_id: int
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    det_score: Optional[float] = None
    created_at: Optional[datetime] = None


@dataclass
class SearchResult:
    photo: Photo
    score: float


@dataclass
class FaceMatch:
    face: Face
    photo: Photo
    score: float


@dataclass
class IndexResult:
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
