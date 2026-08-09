"""Vector index — exact cosine search over L2-normalized embeddings.

Backed by a single NumPy matrix persisted as .npy. At prototype scale
(thousands of photos) a brute-force matrix product is exact and fast, and it
keeps native ANN libraries out of the process. Swapping in FAISS or pgvector
later only requires reimplementing this class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class VectorStore:
    """Append-only vector index. IDs are row positions starting at 0."""

    def __init__(self, dim: int, index_path: Path) -> None:
        self.dim = dim
        self.index_path = Path(index_path).with_suffix(".npy")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._vectors: Optional[np.ndarray] = None
        self._pending: list[np.ndarray] = []

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = self._load_or_create()
        if self._pending:
            self._vectors = np.vstack([self._vectors, *self._pending])
            self._pending.clear()
        return self._vectors

    def _load_or_create(self) -> np.ndarray:
        if self.index_path.exists():
            arr = np.load(self.index_path).astype(np.float32, copy=False)
            if arr.ndim != 2 or arr.shape[1] != self.dim:
                raise ValueError(
                    f"Index shape {arr.shape} incompatible with dim {self.dim} "
                    f"({self.index_path})"
                )
            return arr
        return np.zeros((0, self.dim), dtype=np.float32)

    def count(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> list[int]:
        """Add one or more vectors. Returns the assigned integer IDs."""
        arr = self._prepare(vectors)
        start = self.count()
        self._pending.append(arr)
        return list(range(start, start + arr.shape[0]))

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return (vector_id, cosine score) pairs, best first."""
        matrix = self.vectors
        if matrix.shape[0] == 0:
            return []
        q = self._prepare(query)[0]
        scores = matrix @ q
        k = min(top_k, scores.shape[0])
        # argpartition avoids a full sort when the index grows large
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def save(self) -> None:
        np.save(self.index_path, self.vectors)

    def reset(self) -> None:
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._pending.clear()
        self.index_path.unlink(missing_ok=True)

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {arr.shape[1]}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)
