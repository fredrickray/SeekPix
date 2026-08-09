"""InsightFace detector + ArcFace embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import Settings, get_settings

# buffalo_l ArcFace embedding dim
FACE_DIM = 512


@dataclass
class DetectedFace:
    embedding: np.ndarray
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    det_score: float


class FaceEmbedder:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        # Lazy import — insightface is heavy and may download models on first use
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.settings.face_device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(
            ctx_id=0 if self.settings.face_device == "cuda" else -1,
            det_size=(self.settings.face_det_size, self.settings.face_det_size),
        )
        self.dim = FACE_DIM

    def detect_and_embed(self, path: str | Path) -> list[DetectedFace]:
        from core.ingestion.image_io import load_image_bgr

        image = load_image_bgr(path)

        faces = self.app.get(image)
        results: list[DetectedFace] = []
        for face in faces:
            bbox = face.bbox.astype(float)
            emb = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            # L2-normalize for cosine via IP
            emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
            results.append(
                DetectedFace(
                    embedding=emb,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    det_score=float(getattr(face, "det_score", 0.0)),
                )
            )
        return results


@lru_cache(maxsize=1)
def get_face_embedder() -> FaceEmbedder:
    return FaceEmbedder()
