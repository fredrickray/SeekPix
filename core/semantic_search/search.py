"""Semantic search: embed query text, look up the CLIP vector index."""

from __future__ import annotations

from typing import Optional

from core.config import Settings, get_settings
from core.semantic_search.clip_embedder import CLIP_DIM, ClipEmbedder, get_clip_embedder
from core.storage.db import MetadataDB
from core.storage.models import SearchResult
from core.storage.vector_store import VectorStore


class SemanticSearch:
    def __init__(
        self,
        db: MetadataDB,
        clip_store: VectorStore,
        embedder: Optional[ClipEmbedder] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.db = db
        self.clip_store = clip_store
        self.embedder = embedder or get_clip_embedder()
        self.settings = settings or get_settings()

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if not query.strip():
            return []
        vector = self.embedder.embed_text(query)
        hits = self.clip_store.search(vector, top_k=top_k)
        results: list[SearchResult] = []
        for vector_id, score in hits:
            photo = self.db.get_photo_by_clip_id(vector_id)
            if photo is None:
                continue
            results.append(SearchResult(photo=photo, score=score))
        return results


def build_clip_store(settings: Optional[Settings] = None) -> VectorStore:
    settings = settings or get_settings()
    return VectorStore(dim=CLIP_DIM, index_path=settings.clip_index_path)
