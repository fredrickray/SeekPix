"""Delete + vector compact hygiene tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from core.services.context import AppContext
from core.services.library import compact_orphans, delete_photo
from core.storage.vector_store import VectorStore


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_vector_store_compact_renumbers(tmp_path: Path) -> None:
    store = VectorStore(dim=512, index_path=tmp_path / "clip.npy")
    store.add(_vec(1))
    store.add(_vec(2))
    store.add(_vec(3))
    mapping = store.compact({1})
    assert store.count() == 2
    assert mapping == {0: 0, 2: 1}
    hits = store.search(_vec(3), top_k=1)
    assert hits[0][0] == 1


def test_delete_photo_compacts_indexes_and_files(
    tmp_settings, db, clip_store, face_store, tmp_path
) -> None:
    photos_dir = tmp_settings.photos_dir
    thumbs = tmp_settings.thumbnails_dir

    path_a = photos_dir / "a.jpg"
    path_b = photos_dir / "b.jpg"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path_a)
    Image.new("RGB", (8, 8), color=(200, 20, 30)).save(path_b)
    thumb_a = thumbs / "1_a.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(thumb_a)

    photo_a = db.insert_photo(
        filepath=str(path_a), filename="a.jpg", thumbnail_path=str(thumb_a)
    )
    clip_ids = clip_store.add(_vec(1))
    db.update_photo_clip_id(photo_a.id, clip_ids[0])
    face_ids = face_store.add(_vec(11))
    db.insert_face(
        photo_id=photo_a.id,
        face_vector_id=face_ids[0],
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=1,
        bbox_y2=1,
    )

    photo_b = db.insert_photo(filepath=str(path_b), filename="b.jpg")
    clip_ids_b = clip_store.add(_vec(2))
    db.update_photo_clip_id(photo_b.id, clip_ids_b[0])
    face_ids_b = face_store.add(_vec(22))
    db.insert_face(
        photo_id=photo_b.id,
        face_vector_id=face_ids_b[0],
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=1,
        bbox_y2=1,
    )
    clip_store.save()
    face_store.save()

    ctx = AppContext(
        settings=tmp_settings,
        db=db,
        clip_store=clip_store,
        face_store=face_store,
    )

    result = delete_photo(photo_a.id, ctx=ctx)
    assert result.filename == "a.jpg"
    assert result.deleted_file is True
    assert result.deleted_thumbnail is True
    assert not path_a.exists()
    assert not thumb_a.exists()
    assert path_b.exists()

    assert db.get_photo(photo_a.id) is None
    assert db.count_photos() == 1
    assert db.count_faces() == 1
    assert clip_store.count() == 1
    assert face_store.count() == 1

    remaining = db.get_photo(photo_b.id)
    assert remaining is not None
    assert remaining.clip_vector_id == 0
    faces = db.list_faces_for_photo(photo_b.id)
    assert faces[0].face_vector_id == 0


def test_compact_orphans_drops_unreferenced_vectors(
    tmp_settings, db, clip_store, face_store
) -> None:
    photo = db.insert_photo(filepath="/tmp/only.jpg", filename="only.jpg")
    clip_store.add(_vec(1))
    clip_store.add(_vec(2))  # orphan
    db.update_photo_clip_id(photo.id, 0)
    face_store.add(_vec(3))
    face_store.add(_vec(4))  # orphan
    db.insert_face(
        photo_id=photo.id,
        face_vector_id=0,
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=1,
        bbox_y2=1,
    )

    ctx = AppContext(
        settings=tmp_settings,
        db=db,
        clip_store=clip_store,
        face_store=face_store,
    )
    result = compact_orphans(ctx)
    assert result.clip_before == 2
    assert result.clip_after == 1
    assert result.faces_before == 2
    assert result.faces_after == 1
    assert db.get_photo(photo.id).clip_vector_id == 0
    assert db.list_faces_for_photo(photo.id)[0].face_vector_id == 0
