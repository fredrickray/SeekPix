"""Library hygiene — delete photos and compact orphaned vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.services.context import AppContext, get_context


@dataclass
class DeleteResult:
    photo_id: int
    filename: str
    deleted_file: bool
    deleted_thumbnail: bool
    removed_clip_vectors: int
    removed_face_vectors: int


@dataclass
class CompactResult:
    clip_before: int
    clip_after: int
    faces_before: int
    faces_after: int


def delete_photo(
    photo_id: int,
    *,
    ctx: Optional[AppContext] = None,
    delete_library_file: bool = True,
) -> DeleteResult:
    """Remove a photo from the library and compact vector indexes.

    Source files under ``data/photos/`` (uploads) are deleted. Paths outside
    the library (e.g. ``--no-copy`` indexed demo folders) are left on disk.
    """
    ctx = ctx or get_context()
    photo = ctx.db.get_photo(photo_id)
    if photo is None:
        raise LookupError(f"No photo with id {photo_id}")

    faces = ctx.db.list_faces_for_photo(photo_id)
    clip_remove = (
        {photo.clip_vector_id} if photo.clip_vector_id is not None else set()
    )
    face_remove = {f.face_vector_id for f in faces}

    thumb_path = Path(photo.thumbnail_path) if photo.thumbnail_path else None
    file_path = Path(photo.filepath)

    if not ctx.db.delete_photo(photo_id):
        raise LookupError(f"No photo with id {photo_id}")

    deleted_thumb = False
    if thumb_path is not None:
        deleted_thumb = _unlink(thumb_path)

    deleted_file = False
    if delete_library_file and _is_under(file_path, ctx.settings.photos_dir):
        deleted_file = _unlink(file_path)

    if clip_remove:
        mapping = ctx.clip_store.compact(clip_remove)
        ctx.db.remap_clip_vector_ids(mapping)
        ctx.clip_store.save()

    if face_remove:
        mapping = ctx.face_store.compact(face_remove)
        ctx.db.remap_face_vector_ids(mapping)
        ctx.face_store.save()

    return DeleteResult(
        photo_id=photo_id,
        filename=photo.filename,
        deleted_file=deleted_file,
        deleted_thumbnail=deleted_thumb,
        removed_clip_vectors=len(clip_remove),
        removed_face_vectors=len(face_remove),
    )


def compact_orphans(ctx: Optional[AppContext] = None) -> CompactResult:
    """Drop vector rows that no longer map to a photo or face."""
    ctx = ctx or get_context()

    live_clip = {
        p.clip_vector_id
        for p in ctx.db.list_all_photos()
        if p.clip_vector_id is not None
    }
    live_faces = {f.face_vector_id for f in ctx.db.list_all_faces()}

    clip_before = ctx.clip_store.count()
    face_before = ctx.face_store.count()

    clip_orphans = {i for i in range(clip_before) if i not in live_clip}
    face_orphans = {i for i in range(face_before) if i not in live_faces}

    if clip_orphans:
        mapping = ctx.clip_store.compact(clip_orphans)
        ctx.db.remap_clip_vector_ids(mapping)
        ctx.clip_store.save()

    if face_orphans:
        mapping = ctx.face_store.compact(face_orphans)
        ctx.db.remap_face_vector_ids(mapping)
        ctx.face_store.save()

    return CompactResult(
        clip_before=clip_before,
        clip_after=ctx.clip_store.count(),
        faces_before=face_before,
        faces_after=ctx.face_store.count(),
    )


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _unlink(path: Path) -> bool:
    if not path.is_file():
        return False
    path.unlink()
    return True
