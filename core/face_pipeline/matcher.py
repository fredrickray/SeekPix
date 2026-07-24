"""Face matching against the face FAISS index."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from core.config import Settings, get_settings
from core.face_pipeline.embedder import (
    FACE_DIM,
    DetectedFace,
    FaceEmbedder,
    get_face_embedder,
)
from core.storage.db import MetadataDB
from core.storage.models import FaceMatch
from core.storage.vector_store import VectorStore


class FaceMatcher:
    def __init__(
        self,
        db: MetadataDB,
        face_store: VectorStore,
        embedder: Optional[FaceEmbedder] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.db = db
        self.face_store = face_store
        self.embedder = embedder or get_face_embedder()
        self.settings = settings or get_settings()

    def find_matches(
        self,
        path: str | Path,
        *,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> list[FaceMatch]:
        """Detect faces in probe image and find similar faces in the index."""
        threshold = (
            self.settings.face_match_threshold if threshold is None else threshold
        )
        detected = self.embedder.detect_and_embed(path)
        if not detected:
            return []

        # Use the highest-confidence face as the probe
        probe = max(detected, key=lambda f: f.det_score)
        return self.match_embedding(probe.embedding, top_k=top_k, threshold=threshold)

    def match_embedding(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> list[FaceMatch]:
        threshold = (
            self.settings.face_match_threshold if threshold is None else threshold
        )
        hits = self.face_store.search(embedding, top_k=top_k)
        matches: list[FaceMatch] = []
        for vector_id, score in hits:
            if score < threshold:
                continue
            face = self.db.get_face_by_vector_id(vector_id)
            if face is None:
                continue
            photo = self.db.get_photo(face.photo_id)
            if photo is None:
                continue
            matches.append(FaceMatch(face=face, photo=photo, score=score))
        return matches

    def verify_pair(
        self,
        path_a: str | Path,
        path_b: str | Path,
    ) -> Optional[float]:
        """Return cosine similarity between primary faces, or None if no face."""
        faces_a = self.embedder.detect_and_embed(path_a)
        faces_b = self.embedder.detect_and_embed(path_b)
        if not faces_a or not faces_b:
            return None
        a = max(faces_a, key=lambda f: f.det_score).embedding
        b = max(faces_b, key=lambda f: f.det_score).embedding
        return float(np.dot(a, b))


def build_face_store(settings: Optional[Settings] = None) -> VectorStore:
    settings = settings or get_settings()
    return VectorStore(dim=FACE_DIM, index_path=settings.face_index_path)
