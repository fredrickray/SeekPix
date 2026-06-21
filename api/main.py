"""SeekPix HTTP API — consumed by SeekPix UI."""

from __future__ import annotations

import io
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from core.config import get_settings
from core.ingestion.image_io import load_image
from core.services.context import get_context
from core.services.indexing import index_folder
from core.services.search import (
    find_same_person,
    library_stats,
    list_library,
    search_photos,
    verify_faces,
)
from core.storage.models import Photo

# Formats every browser can render directly; anything else is transcoded to JPEG
# on the way out, which matters because iPhone libraries are mostly HEIC.
WEB_SAFE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
WEB_SAFE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_dirs()
    get_context(settings)
    yield


app = FastAPI(
    title="SeekPix API",
    version="0.1.0",
    description="Local photo search and face verification backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)


class IndexRequest(BaseModel):
    folder: str
    recursive: bool = True
    run_faces: bool = True


class PhotoOut(BaseModel):
    id: int
    filename: str
    thumbnail_url: str
    image_url: str
    score: Optional[float] = None

    @classmethod
    def from_photo(cls, photo: Photo, score: Optional[float] = None) -> "PhotoOut":
        return cls(
            id=photo.id,
            filename=photo.filename,
            thumbnail_url=f"/photos/{photo.id}/thumbnail",
            image_url=f"/photos/{photo.id}/file",
            score=score,
        )


class FaceMatchOut(BaseModel):
    photo: PhotoOut
    face_id: int
    score: float


class StatsOut(BaseModel):
    photos: int
    faces: int
    clip_vectors: int
    face_vectors: int


class IndexOut(BaseModel):
    indexed: int
    skipped: int
    failed: int
    errors: list[str]


class VerifyOut(BaseModel):
    score: Optional[float]
    matched: Optional[bool] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats", response_model=StatsOut)
def stats() -> StatsOut:
    return StatsOut(**library_stats())


@app.get("/photos", response_model=list[PhotoOut])
def photos(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[PhotoOut]:
    items = list_library(limit=limit, offset=offset)
    return [PhotoOut.from_photo(p) for p in items]


@app.get("/photos/{photo_id}/thumbnail")
def photo_thumbnail(photo_id: int) -> FileResponse:
    photo = _get_photo_or_404(photo_id)
    if not photo.thumbnail_path:
        raise HTTPException(
            status_code=404, detail=f"No thumbnail for photo {photo_id}"
        )
    path = Path(photo.thumbnail_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Thumbnail missing: {path.name}")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/photos/{photo_id}/file")
def photo_file(photo_id: int) -> Response:
    photo = _get_photo_or_404(photo_id)
    path = Path(photo.filepath)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Image missing: {path.name}")

    suffix = path.suffix.lower()
    if suffix in WEB_SAFE_SUFFIXES:
        return FileResponse(path, media_type=WEB_SAFE_MEDIA_TYPES[suffix])

    buffer = io.BytesIO()
    load_image(path).save(buffer, format="JPEG", quality=90)
    return Response(content=buffer.getvalue(), media_type="image/jpeg")


@app.post("/search", response_model=list[PhotoOut])
def search(body: SearchRequest) -> list[PhotoOut]:
    results = search_photos(body.query, top_k=body.top_k)
    return [PhotoOut.from_photo(r.photo, score=r.score) for r in results]


@app.post("/index", response_model=IndexOut)
def index(body: IndexRequest) -> IndexOut:
    folder = Path(body.folder)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Invalid folder: {folder}")
    result = index_folder(
        folder,
        recursive=body.recursive,
        run_faces=body.run_faces,
    )
    return IndexOut(
        indexed=result.indexed,
        skipped=result.skipped,
        failed=result.failed,
        errors=result.errors,
    )


@app.post("/faces/find", response_model=list[FaceMatchOut])
async def faces_find(
    file: UploadFile = File(...),
    top_k: int = Query(20, ge=1, le=100),
    threshold: Optional[float] = Query(None),
) -> list[FaceMatchOut]:
    tmp = _save_upload(file)
    try:
        matches = find_same_person(tmp, top_k=top_k, threshold=threshold)
        return [
            FaceMatchOut(
                photo=PhotoOut.from_photo(m.photo, score=m.score),
                face_id=m.face.id,
                score=m.score,
            )
            for m in matches
        ]
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/faces/verify", response_model=VerifyOut)
async def faces_verify(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
) -> VerifyOut:
    settings = get_settings()
    path_a = _save_upload(file_a)
    path_b = _save_upload(file_b)
    try:
        score = verify_faces(path_a, path_b)
        matched = None if score is None else score >= settings.face_match_threshold
        return VerifyOut(score=score, matched=matched)
    finally:
        path_a.unlink(missing_ok=True)
        path_b.unlink(missing_ok=True)


def _get_photo_or_404(photo_id: int) -> Photo:
    photo = get_context().db.get_photo(photo_id)
    if photo is None:
        raise HTTPException(status_code=404, detail=f"No photo with id {photo_id}")
    return photo


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"
    fd, name = tempfile.mkstemp(suffix=suffix)
    path = Path(name)
    with open(fd, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return path
