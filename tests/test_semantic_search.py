"""Semantic search unit tests — no CLIP download required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.semantic_search.search import SemanticSearch
from core.services.context import AppContext
from core.storage.vector_store import VectorStore


class FakeClipEmbedder:
    """Deterministic embedder: text and image vectors live in a fixed space."""

    dim = 512

    def __init__(self) -> None:
        self._image_vectors: dict[str, np.ndarray] = {}

    def register_image(self, path: str | Path, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        self._image_vectors[str(path)] = vec
        return vec

    def embed_image(self, path: str | Path) -> np.ndarray:
        key = str(path)
        if key not in self._image_vectors:
            raise KeyError(f"Unregistered image: {path}")
        return self._image_vectors[key]

    def embed_text(self, text: str) -> np.ndarray:
        # Map a few queries onto registered image seeds so ranking is predictable.
        table = {
            "red car": 1,
            "blue sky": 2,
            "birthday cake": 3,
        }
        seed = table.get(text.strip().lower(), 99)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)


@pytest.fixture()
def search_env(tmp_path: Path, tmp_settings, db):
    photos_dir = tmp_path / "imgs"
    photos_dir.mkdir()
    clip_store = VectorStore(dim=512, index_path=tmp_settings.clip_index_path)
    face_store = VectorStore(dim=512, index_path=tmp_settings.face_index_path)
    ctx = AppContext(
        settings=tmp_settings,
        db=db,
        clip_store=clip_store,
        face_store=face_store,
    )
    embedder = FakeClipEmbedder()

    names = [("red_car.jpg", 1), ("blue_sky.jpg", 2), ("cake.jpg", 3)]
    for name, seed in names:
        path = photos_dir / name
        Image.new("RGB", (32, 32), color=(seed * 40, 10, 10)).save(path)
        vec = embedder.register_image(path, seed)
        photo = db.insert_photo(filepath=str(path), filename=name)
        ids = clip_store.add(vec)
        db.update_photo_clip_id(photo.id, ids[0])

    engine = SemanticSearch(db, clip_store, embedder=embedder)
    return engine, ctx


def test_search_ranks_matching_photo_first(search_env) -> None:
    engine, _ = search_env
    results = engine.search("red car", top_k=3)
    assert results
    assert results[0].photo.filename == "red_car.jpg"
    assert results[0].score > results[-1].score


def test_search_empty_query_returns_nothing(search_env) -> None:
    engine, _ = search_env
    assert engine.search("   ") == []


def test_search_unknown_query_still_returns_top_k(search_env) -> None:
    engine, _ = search_env
    results = engine.search("something obscure", top_k=2)
    assert len(results) == 2
