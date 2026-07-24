"""Search and face-verification use-cases."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.face_pipeline.dedupe import find_prior_appearances, unique_photos
from core.face_pipeline.matcher import FaceMatcher
from core.semantic_search.search import SemanticSearch
from core.services.context import AppContext, get_context
from core.storage.models import FaceMatch, Photo, SearchResult


def search_photos(
    query: str,
    *,
    top_k: int = 10,
    ctx: Optional[AppContext] = None,
) -> list[SearchResult]:
    ctx = ctx or get_context()
    engine = SemanticSearch(ctx.db, ctx.clip_store)
    return engine.search(query, top_k=top_k)


def list_library(
    *,
    limit: int = 100,
    offset: int = 0,
    ctx: Optional[AppContext] = None,
) -> list[Photo]:
    ctx = ctx or get_context()
    return ctx.db.list_photos(limit=limit, offset=offset)


def library_stats(ctx: Optional[AppContext] = None) -> dict:
    ctx = ctx or get_context()
    return {
        "photos": ctx.db.count_photos(),
        "faces": ctx.db.count_faces(),
        "clip_vectors": ctx.clip_store.count(),
        "face_vectors": ctx.face_store.count(),
    }


def find_same_person(
    path: str | Path,
    *,
    top_k: int = 20,
    threshold: Optional[float] = None,
    ctx: Optional[AppContext] = None,
) -> list[FaceMatch]:
    ctx = ctx or get_context()
    matcher = FaceMatcher(ctx.db, ctx.face_store)
    return find_prior_appearances(
        matcher, path, top_k=top_k, threshold=threshold
    )


def find_same_person_photos(
    path: str | Path,
    *,
    top_k: int = 20,
    threshold: Optional[float] = None,
    ctx: Optional[AppContext] = None,
) -> list[Photo]:
    matches = find_same_person(
        path, top_k=top_k, threshold=threshold, ctx=ctx
    )
    return unique_photos(matches)


def verify_faces(
    path_a: str | Path,
    path_b: str | Path,
    *,
    ctx: Optional[AppContext] = None,
) -> Optional[float]:
    ctx = ctx or get_context()
    matcher = FaceMatcher(ctx.db, ctx.face_store)
    return matcher.verify_pair(path_a, path_b)
