"""Index photos: ingest metadata + CLIP + faces."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional, Sequence

from core.config import Settings
from core.face_pipeline.embedder import FaceEmbedder, get_face_embedder
from core.ingestion.metadata import extract_metadata
from core.ingestion.scanner import scan_folder
from core.ingestion.thumbnails import generate_thumbnail
from core.semantic_search.clip_embedder import ClipEmbedder, get_clip_embedder
from core.services.context import AppContext, get_context
from core.storage.models import IndexResult, Photo


ProgressCallback = Callable[[int, int, str], None]


def index_folder(
    folder: str | Path,
    *,
    ctx: Optional[AppContext] = None,
    recursive: bool = True,
    copy_into_library: bool = True,
    run_faces: bool = True,
    clip_embedder: Optional[ClipEmbedder] = None,
    face_embedder: Optional[FaceEmbedder] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> IndexResult:
    """Scan folder and index every new image into metadata + vector stores."""
    paths = scan_folder(folder, recursive=recursive)
    return index_paths(
        paths,
        ctx=ctx,
        copy_into_library=copy_into_library,
        run_faces=run_faces,
        clip_embedder=clip_embedder,
        face_embedder=face_embedder,
        on_progress=on_progress,
    )


def index_paths(
    paths: Sequence[str | Path],
    *,
    ctx: Optional[AppContext] = None,
    copy_into_library: bool = True,
    run_faces: bool = True,
    clip_embedder: Optional[ClipEmbedder] = None,
    face_embedder: Optional[FaceEmbedder] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> IndexResult:
    """Index an explicit list of image paths."""
    ctx = ctx or get_context()
    settings = ctx.settings
    paths = [Path(p) for p in paths]
    result = IndexResult()
    total = len(paths)

    clip = clip_embedder or get_clip_embedder()
    faces = face_embedder or (get_face_embedder() if run_faces else None)

    for i, src in enumerate(paths, start=1):
        try:
            photo = _index_one(
                src,
                ctx=ctx,
                settings=settings,
                clip=clip,
                faces=faces,
                copy_into_library=copy_into_library,
            )
            if photo is None:
                result.skipped += 1
            else:
                result.indexed += 1
        except Exception as exc:  # noqa: BLE001 — collect per-file failures
            result.failed += 1
            result.errors.append(f"{src}: {exc}")
        # Reported after the file is done so counts never overstate progress
        if on_progress:
            on_progress(i, total, src.name)

    ctx.save_indexes()
    return result


def reserve_library_path(
    filename: str,
    *,
    ctx: Optional[AppContext] = None,
) -> Path:
    """Pick an unused path inside the photo library for an uploaded file.

    Only the base name of the upload is used, so a client cannot steer writes
    outside the library with a crafted filename.
    """
    ctx = ctx or get_context()
    safe_name = Path(filename).name or "upload.jpg"
    dest = ctx.settings.photos_dir / safe_name
    if not dest.exists():
        return dest

    counter = 2
    while True:
        candidate = ctx.settings.photos_dir / f"{dest.stem}_{counter}{dest.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _index_one(
    src: Path,
    *,
    ctx: AppContext,
    settings: Settings,
    clip: ClipEmbedder,
    faces: Optional[FaceEmbedder],
    copy_into_library: bool,
) -> Optional[Photo]:
    # Destination path inside library (or absolute source if not copying)
    if copy_into_library:
        dest = settings.photos_dir / src.name
        # Avoid collisions by prefixing stem if name exists and differs
        if dest.exists() and dest.resolve() != src.resolve():
            dest = settings.photos_dir / f"{src.stem}_{src.stat().st_size}{src.suffix}"
        if not dest.exists():
            shutil.copy2(src, dest)
        stored_path = dest
    else:
        stored_path = src.resolve()

    existing = ctx.db.get_photo_by_path(str(stored_path))
    if existing is not None:
        return None

    meta = extract_metadata(stored_path)
    photo = ctx.db.insert_photo(
        filepath=str(stored_path),
        filename=stored_path.name,
        taken_at=meta.taken_at,
        latitude=meta.latitude,
        longitude=meta.longitude,
        width=meta.width,
        height=meta.height,
    )

    thumb = generate_thumbnail(
        stored_path,
        settings.thumbnails_dir,
        size=settings.thumb_size,
        photo_id=photo.id,
    )
    with ctx.db.connect() as conn:
        conn.execute(
            "UPDATE photos SET thumbnail_path = ? WHERE id = ?",
            (str(thumb), photo.id),
        )

    # CLIP
    clip_vec = clip.embed_image(stored_path)
    clip_ids = ctx.clip_store.add(clip_vec)
    ctx.db.update_photo_clip_id(photo.id, clip_ids[0])

    # Faces
    if faces is not None:
        detected = faces.detect_and_embed(stored_path)
        for face in detected:
            face_ids = ctx.face_store.add(face.embedding)
            ctx.db.insert_face(
                photo_id=photo.id,
                face_vector_id=face_ids[0],
                bbox_x1=face.bbox[0],
                bbox_y1=face.bbox[1],
                bbox_x2=face.bbox[2],
                bbox_y2=face.bbox[3],
                det_score=face.det_score,
            )

    return ctx.db.get_photo(photo.id)
