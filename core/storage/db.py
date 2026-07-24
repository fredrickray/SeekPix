"""SQLite metadata store — source of truth for photos and faces."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from core.storage.models import Face, Photo

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath        TEXT NOT NULL UNIQUE,
    filename        TEXT NOT NULL,
    thumbnail_path  TEXT,
    taken_at        TEXT,
    latitude        REAL,
    longitude       REAL,
    clip_vector_id  INTEGER,
    width           INTEGER,
    height          INTEGER,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id        INTEGER NOT NULL,
    face_vector_id  INTEGER NOT NULL,
    bbox_x1         REAL NOT NULL,
    bbox_y1         REAL NOT NULL,
    bbox_x2         REAL NOT NULL,
    bbox_y2         REAL NOT NULL,
    det_score       REAL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_photos_clip_vector_id ON photos(clip_vector_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_face_vector_id ON faces(face_vector_id);
"""


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _row_to_photo(row: sqlite3.Row) -> Photo:
    return Photo(
        id=row["id"],
        filepath=row["filepath"],
        filename=row["filename"],
        thumbnail_path=row["thumbnail_path"],
        taken_at=_parse_dt(row["taken_at"]),
        latitude=row["latitude"],
        longitude=row["longitude"],
        clip_vector_id=row["clip_vector_id"],
        width=row["width"],
        height=row["height"],
        created_at=_parse_dt(row["created_at"]),
    )


def _row_to_face(row: sqlite3.Row) -> Face:
    return Face(
        id=row["id"],
        photo_id=row["photo_id"],
        face_vector_id=row["face_vector_id"],
        bbox_x1=row["bbox_x1"],
        bbox_y1=row["bbox_y1"],
        bbox_x2=row["bbox_x2"],
        bbox_y2=row["bbox_y2"],
        det_score=row["det_score"],
        created_at=_parse_dt(row["created_at"]),
    )


class MetadataDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def get_photo_by_path(self, filepath: str) -> Optional[Photo]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM photos WHERE filepath = ?",
                (filepath,),
            ).fetchone()
        return _row_to_photo(row) if row else None

    def get_photo(self, photo_id: int) -> Optional[Photo]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM photos WHERE id = ?",
                (photo_id,),
            ).fetchone()
        return _row_to_photo(row) if row else None

    def get_photo_by_clip_id(self, clip_vector_id: int) -> Optional[Photo]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM photos WHERE clip_vector_id = ?",
                (clip_vector_id,),
            ).fetchone()
        return _row_to_photo(row) if row else None

    def list_photos(self, limit: int = 100, offset: int = 0) -> list[Photo]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM photos ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_photo(r) for r in rows]

    def count_photos(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM photos").fetchone()
        return int(row["c"])

    def insert_photo(
        self,
        *,
        filepath: str,
        filename: str,
        thumbnail_path: Optional[str] = None,
        taken_at: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        clip_vector_id: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Photo:
        now = datetime.utcnow().isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO photos (
                    filepath, filename, thumbnail_path, taken_at,
                    latitude, longitude, clip_vector_id, width, height, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filepath,
                    filename,
                    thumbnail_path,
                    taken_at.isoformat() if taken_at else None,
                    latitude,
                    longitude,
                    clip_vector_id,
                    width,
                    height,
                    now,
                ),
            )
            photo_id = int(cur.lastrowid)
        photo = self.get_photo(photo_id)
        assert photo is not None
        return photo

    def update_photo_clip_id(self, photo_id: int, clip_vector_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE photos SET clip_vector_id = ? WHERE id = ?",
                (clip_vector_id, photo_id),
            )

    def insert_face(
        self,
        *,
        photo_id: int,
        face_vector_id: int,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        det_score: Optional[float] = None,
    ) -> Face:
        now = datetime.utcnow().isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO faces (
                    photo_id, face_vector_id, bbox_x1, bbox_y1,
                    bbox_x2, bbox_y2, det_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    photo_id,
                    face_vector_id,
                    bbox_x1,
                    bbox_y1,
                    bbox_x2,
                    bbox_y2,
                    det_score,
                    now,
                ),
            )
            face_id = int(cur.lastrowid)
        face = self.get_face(face_id)
        assert face is not None
        return face

    def get_face(self, face_id: int) -> Optional[Face]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM faces WHERE id = ?",
                (face_id,),
            ).fetchone()
        return _row_to_face(row) if row else None

    def get_face_by_vector_id(self, face_vector_id: int) -> Optional[Face]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM faces WHERE face_vector_id = ?",
                (face_vector_id,),
            ).fetchone()
        return _row_to_face(row) if row else None

    def list_faces_for_photo(self, photo_id: int) -> list[Face]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM faces WHERE photo_id = ? ORDER BY id",
                (photo_id,),
            ).fetchall()
        return [_row_to_face(r) for r in rows]

    def count_faces(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM faces").fetchone()
        return int(row["c"])
