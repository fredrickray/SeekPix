"""FAISS vector index wrapper — add, search, persist."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class VectorStore:
    """Flat IP (inner-product) index over L2-normalized vectors.

    With normalized embeddings, inner product == cosine similarity.
    Vector IDs are sequential integers starting at 0, matching FAISS positions.
    """

    def __init__(self, dim: int, index_path: Path) -> None:
        self.dim = dim
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index: Optional[faiss.IndexFlatIP] = None

    @property
    def index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            self._index = self._load_or_create()
        return self._index

    def _load_or_create(self) -> faiss.IndexFlatIP:
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.d != self.dim:
                raise ValueError(
                    f"Index dim {index.d} != expected {self.dim} "
                    f"({self.index_path})"
                )
            return index
        return faiss.IndexFlatIP(self.dim)

    def count(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray) -> list[int]:
        """Add one or more vectors. Returns assigned integer IDs."""
        arr = self._prepare(vectors)
        start = self.count()
        self.index.add(arr)
        return list(range(start, start + arr.shape[0]))

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return list of (vector_id, score) sorted by descending score."""
        if self.count() == 0:
            return []
        q = self._prepare(query)
        k = min(top_k, self.count())
        scores, ids = self.index.search(q, k)
        results: list[tuple[int, float]] = []
        for score, vid in zip(scores[0], ids[0]):
            if vid < 0:
                continue
            results.append((int(vid), float(score)))
        return results

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))

    def reset(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        if self.index_path.exists():
            self.index_path.unlink()

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"Expected dim {self.dim}, got {arr.shape[1]}"
            )
        # L2-normalize so IP == cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms
