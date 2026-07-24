"""SeekPix HTTP API — consumed by SeekPix UI."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import get_settings
from core.services.context import get_context
from core.services.indexing import index_folder
from core.services.search import (
    find_same_person,
    library_stats,
    list_library,
    search_photos,
    verify_faces,
)

app = FastAPI(
    title="SeekPix API",
    version="0.1.0",
    description="Local photo search and face verification backend",
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
    filepath: str
    filename: str
    thumbnail_path: Optional[str] = None
    score: Optional[float] = None


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


@app.on_event("startup")
def _startup() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    get_context(settings)


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
    return [
        PhotoOut(
            id=p.id,
            filepath=p.filepath,
            filename=p.filename,
            thumbnail_path=p.thumbnail_path,
        )
        for p in items
    ]


@app.post("/search", response_model=list[PhotoOut])
def search(body: SearchRequest) -> list[PhotoOut]:
    results = search_photos(body.query, top_k=body.top_k)
    return [
        PhotoOut(
            id=r.photo.id,
            filepath=r.photo.filepath,
            filename=r.photo.filename,
            thumbnail_path=r.photo.thumbnail_path,
            score=r.score,
        )
        for r in results
    ]


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
                photo=PhotoOut(
                    id=m.photo.id,
                    filepath=m.photo.filepath,
                    filename=m.photo.filename,
                    thumbnail_path=m.photo.thumbnail_path,
                    score=m.score,
                ),
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


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"
    fd, name = tempfile.mkstemp(suffix=suffix)
    path = Path(name)
    with open(fd, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return path
