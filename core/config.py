"""Application configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of core/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    photos_dir: Path
    thumbnails_dir: Path
    indexes_dir: Path
    db_path: Path

    clip_model: str
    clip_pretrained: str
    clip_device: str
    clip_index_path: Path

    face_det_size: int
    face_match_threshold: float
    face_device: str
    face_index_path: Path

    thumb_size: int

    api_host: str
    api_port: int

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.photos_dir,
            self.thumbnails_dir,
            self.indexes_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    indexes_dir = _resolve_path(_env("SEEKPIX_INDEXES_DIR", "data/indexes"))
    return Settings(
        data_dir=_resolve_path(_env("SEEKPIX_DATA_DIR", "data")),
        photos_dir=_resolve_path(_env("SEEKPIX_PHOTOS_DIR", "data/photos")),
        thumbnails_dir=_resolve_path(
            _env("SEEKPIX_THUMBNAILS_DIR", "data/thumbnails")
        ),
        indexes_dir=indexes_dir,
        db_path=_resolve_path(_env("SEEKPIX_DB_PATH", "data/metadata.db")),
        clip_model=_env("SEEKPIX_CLIP_MODEL", "ViT-B-32-quickgelu"),
        clip_pretrained=_env("SEEKPIX_CLIP_PRETRAINED", "openai"),
        clip_device=_env("SEEKPIX_CLIP_DEVICE", "cpu"),
        clip_index_path=indexes_dir / "clip.npy",
        face_det_size=_env_int("SEEKPIX_FACE_DET_SIZE", 640),
        face_match_threshold=_env_float("SEEKPIX_FACE_MATCH_THRESHOLD", 0.40),
        face_device=_env("SEEKPIX_FACE_DEVICE", "cpu"),
        face_index_path=indexes_dir / "faces.npy",
        thumb_size=_env_int("SEEKPIX_THUMB_SIZE", 256),
        api_host=_env("SEEKPIX_API_HOST", "0.0.0.0"),
        api_port=_env_int("SEEKPIX_API_PORT", 8000),
    )
