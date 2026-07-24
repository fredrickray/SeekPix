"""Shared runtime wiring for DB + vector stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.config import Settings, get_settings
from core.storage.db import MetadataDB
from core.storage.vector_store import VectorStore

# Embedding dims — keep context free of heavy model imports
CLIP_DIM = 512
FACE_DIM = 512


@dataclass
class AppContext:
    settings: Settings
    db: MetadataDB
    clip_store: VectorStore
    face_store: VectorStore

    def save_indexes(self) -> None:
        self.clip_store.save()
        self.face_store.save()


_ctx: Optional[AppContext] = None


def get_context(settings: Optional[Settings] = None) -> AppContext:
    global _ctx
    if _ctx is None:
        settings = settings or get_settings()
        settings.ensure_dirs()
        _ctx = AppContext(
            settings=settings,
            db=MetadataDB(settings.db_path),
            clip_store=VectorStore(dim=CLIP_DIM, index_path=settings.clip_index_path),
            face_store=VectorStore(dim=FACE_DIM, index_path=settings.face_index_path),
        )
    return _ctx


def reset_context() -> None:
    """Test helper — drop cached context."""
    global _ctx
    _ctx = None
