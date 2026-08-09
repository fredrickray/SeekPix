"""Shared pytest fixtures — isolated temp data dirs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.config import Settings
from core.services.context import AppContext, reset_context
from core.storage.db import MetadataDB
from core.storage.vector_store import VectorStore


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    photos = data / "photos"
    thumbs = data / "thumbnails"
    indexes = data / "indexes"
    for p in (photos, thumbs, indexes):
        p.mkdir(parents=True)
    return Settings(
        data_dir=data,
        photos_dir=photos,
        thumbnails_dir=thumbs,
        indexes_dir=indexes,
        db_path=data / "metadata.db",
        clip_model="ViT-B-32",
        clip_pretrained="openai",
        clip_device="cpu",
        clip_index_path=indexes / "clip.npy",
        face_det_size=640,
        face_match_threshold=0.40,
        face_device="cpu",
        face_index_path=indexes / "faces.npy",
        thumb_size=128,
        api_host="127.0.0.1",
        api_port=8000,
    )


@pytest.fixture()
def db(tmp_settings: Settings) -> MetadataDB:
    return MetadataDB(tmp_settings.db_path)


@pytest.fixture()
def clip_store(tmp_settings: Settings) -> VectorStore:
    return VectorStore(dim=512, index_path=tmp_settings.clip_index_path)


@pytest.fixture()
def face_store(tmp_settings: Settings) -> VectorStore:
    return VectorStore(dim=512, index_path=tmp_settings.face_index_path)


@pytest.fixture()
def ctx(
    tmp_settings: Settings,
    db: MetadataDB,
    clip_store: VectorStore,
    face_store: VectorStore,
) -> AppContext:
    reset_context()
    return AppContext(
        settings=tmp_settings,
        db=db,
        clip_store=clip_store,
        face_store=face_store,
    )


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), color=(200, 40, 40)).save(path, format="JPEG")
    return path


@pytest.fixture()
def random_vec() -> np.ndarray:
    rng = np.random.default_rng(42)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v
