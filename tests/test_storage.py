"""Phase 1 — storage foundation tests (no ML models required)."""

from __future__ import annotations

import numpy as np

from core.storage.db import MetadataDB
from core.storage.vector_store import VectorStore


def test_insert_and_get_photo(db: MetadataDB) -> None:
    photo = db.insert_photo(
        filepath="/tmp/a.jpg",
        filename="a.jpg",
        width=100,
        height=80,
    )
    assert photo.id >= 1
    loaded = db.get_photo(photo.id)
    assert loaded is not None
    assert loaded.filename == "a.jpg"
    assert db.get_photo_by_path("/tmp/a.jpg") is not None


def test_clip_vector_id_roundtrip(db: MetadataDB) -> None:
    photo = db.insert_photo(filepath="/tmp/b.jpg", filename="b.jpg")
    db.update_photo_clip_id(photo.id, 7)
    by_clip = db.get_photo_by_clip_id(7)
    assert by_clip is not None
    assert by_clip.id == photo.id


def test_face_insert_and_lookup(db: MetadataDB) -> None:
    photo = db.insert_photo(filepath="/tmp/c.jpg", filename="c.jpg")
    face = db.insert_face(
        photo_id=photo.id,
        face_vector_id=3,
        bbox_x1=1,
        bbox_y1=2,
        bbox_x2=10,
        bbox_y2=20,
        det_score=0.99,
    )
    assert face.id >= 1
    assert db.get_face_by_vector_id(3) is not None
    assert len(db.list_faces_for_photo(photo.id)) == 1


def test_vector_store_add_search_persist(
    clip_store: VectorStore,
    random_vec: np.ndarray,
    tmp_settings,
) -> None:
    ids = clip_store.add(random_vec)
    assert ids == [0]
    hits = clip_store.search(random_vec, top_k=1)
    assert len(hits) == 1
    assert hits[0][0] == 0
    assert hits[0][1] > 0.99

    clip_store.save()
    reloaded = VectorStore(dim=512, index_path=tmp_settings.clip_index_path)
    hits2 = reloaded.search(random_vec, top_k=1)
    assert hits2[0][0] == 0
    assert reloaded.count() == 1


def test_scanner_finds_images(sample_image, tmp_path) -> None:
    from core.ingestion.scanner import scan_folder

    other = tmp_path / "notes.txt"
    other.write_text("nope")
    found = scan_folder(tmp_path)
    assert sample_image.resolve() in [p.resolve() for p in found]


def test_thumbnail_generation(sample_image, tmp_settings) -> None:
    from core.ingestion.thumbnails import generate_thumbnail

    dest = generate_thumbnail(
        sample_image,
        tmp_settings.thumbnails_dir,
        size=32,
        photo_id=1,
    )
    assert dest.exists()
    assert dest.suffix == ".jpg"
