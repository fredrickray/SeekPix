"""Face pipeline tests.

Labeled fixture pairs are optional. When tests/fixtures/pairs.json is missing,
only the always-on identity checks run (self-match via FakeFaceEmbedder).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.face_pipeline.dedupe import unique_photos
from core.face_pipeline.embedder import DetectedFace
from core.face_pipeline.matcher import FaceMatcher
from core.services.context import AppContext
from core.storage.models import FaceMatch, Photo
from core.storage.vector_store import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"
PAIRS_JSON = FIXTURES / "pairs.json"


class FakeFaceEmbedder:
    dim = 512

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def detect_and_embed(self, path: str | Path) -> list[DetectedFace]:
        key = Path(path).name
        if key not in self._vectors:
            return []
        return [
            DetectedFace(
                embedding=self._vectors[key],
                bbox=(0, 0, 10, 10),
                det_score=0.99,
            )
        ]


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_matcher_finds_same_identity_above_threshold(
    tmp_settings, db, face_store
) -> None:
    alice = _unit(7)
    bob = _unit(99)
    vectors = {"alice.jpg": alice, "bob.jpg": bob}
    embedder = FakeFaceEmbedder(vectors)

    photo_a = db.insert_photo(filepath="/tmp/alice.jpg", filename="alice.jpg")
    ids = face_store.add(alice)
    db.insert_face(
        photo_id=photo_a.id,
        face_vector_id=ids[0],
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=1,
        bbox_y2=1,
    )

    photo_b = db.insert_photo(filepath="/tmp/bob.jpg", filename="bob.jpg")
    ids_b = face_store.add(bob)
    db.insert_face(
        photo_id=photo_b.id,
        face_vector_id=ids_b[0],
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=1,
        bbox_y2=1,
    )

    matcher = FaceMatcher(
        db, face_store, embedder=embedder, settings=tmp_settings
    )
    # tmp_settings.face_match_threshold is 0.40; alice vs alice is ~1.0
    matches = matcher.find_matches("/tmp/alice.jpg", top_k=5, threshold=0.8)
    assert matches
    assert matches[0].photo.filename == "alice.jpg"
    assert matches[0].score > 0.99

    # Bob must not appear as a high-confidence alice match
    high = matcher.find_matches("/tmp/alice.jpg", top_k=5, threshold=0.85)
    assert all(m.photo.filename != "bob.jpg" for m in high)


def test_verify_pair_scores_identical_faces(
    tmp_settings, db, face_store
) -> None:
    vec = _unit(3)
    embedder = FakeFaceEmbedder({"a.jpg": vec, "b.jpg": vec})
    matcher = FaceMatcher(
        db, face_store, embedder=embedder, settings=tmp_settings
    )
    score = matcher.verify_pair("/tmp/a.jpg", "/tmp/b.jpg")
    assert score is not None
    assert score > 0.99


def test_verify_pair_returns_none_without_faces(
    tmp_settings, db, face_store
) -> None:
    embedder = FakeFaceEmbedder({})
    matcher = FaceMatcher(
        db, face_store, embedder=embedder, settings=tmp_settings
    )
    assert matcher.verify_pair("/tmp/missing.jpg", "/tmp/also.jpg") is None


def test_unique_photos_keeps_best_first() -> None:
    from core.storage.models import Face

    face = Face(
        id=1,
        photo_id=1,
        face_vector_id=0,
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=1,
        bbox_y2=1,
    )
    p1 = Photo(id=1, filepath="/a", filename="a.jpg")
    p2 = Photo(id=2, filepath="/b", filename="b.jpg")
    matches = [
        FaceMatch(face=face, photo=p1, score=0.9),
        FaceMatch(face=face, photo=p1, score=0.5),
        FaceMatch(face=face, photo=p2, score=0.8),
    ]
    photos = unique_photos(matches)
    assert [p.id for p in photos] == [1, 2]


@pytest.mark.skipif(
    not PAIRS_JSON.exists(),
    reason="tests/fixtures/pairs.json not present — add labeled pairs to enable",
)
def test_labeled_pairs_respect_threshold() -> None:
    """Integration check against real InsightFace + labeled fixtures."""
    from core.config import get_settings
    from core.services.search import verify_faces

    data = json.loads(PAIRS_JSON.read_text())
    base = PAIRS_JSON.parent
    threshold = get_settings().face_match_threshold

    for a, b in data.get("same_person", []):
        score = verify_faces(base / a, base / b)
        assert score is not None, f"No face in {a} or {b}"
        assert score >= threshold, (
            f"Same-person pair scored {score:.3f} < threshold {threshold:.2f}: "
            f"{a} vs {b}"
        )

    for a, b in data.get("different_people", []):
        score = verify_faces(base / a, base / b)
        assert score is not None, f"No face in {a} or {b}"
        assert score < threshold, (
            f"Different-people pair scored {score:.3f} >= threshold "
            f"{threshold:.2f}: {a} vs {b}"
        )
