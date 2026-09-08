#!/usr/bin/env python3
"""Suggest a face-match threshold from labeled pairs or a library probe.

Labeled mode (preferred):
  python scripts/calibrate_face_threshold.py --pairs tests/fixtures/pairs.json

Library probe mode (exploratory — not ground truth):
  python scripts/calibrate_face_threshold.py --probe face.jpg --against-library
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.face_pipeline.embedder import get_face_embedder
from core.face_pipeline.matcher import FaceMatcher
from core.services.context import get_context


def _primary_embedding(path: Path):
    faces = get_face_embedder().detect_and_embed(path)
    if not faces:
        raise ValueError(f"No face detected in {path}")
    return max(faces, key=lambda f: f.det_score).embedding


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def score_pairs(
    pairs: list[tuple[Path, Path]],
) -> list[tuple[str, str, float | None, str | None]]:
    rows: list[tuple[str, str, float | None, str | None]] = []
    for left, right in pairs:
        try:
            score = _cosine(_primary_embedding(left), _primary_embedding(right))
            rows.append((left.name, right.name, score, None))
        except Exception as exc:  # noqa: BLE001 — report per pair
            rows.append((left.name, right.name, None, str(exc)))
    return rows


def suggest_threshold(
    same_scores: list[float],
    different_scores: list[float],
) -> float | None:
    if not same_scores or not different_scores:
        return None
    # Sit halfway between the weakest same-person and strongest different-person
    # score, clamped to a sensible ArcFace cosine band.
    mid = (min(same_scores) + max(different_scores)) / 2.0
    return float(np.clip(mid, 0.45, 0.70))


def run_labeled(pairs_path: Path) -> int:
    data = json.loads(pairs_path.read_text())
    base = pairs_path.parent

    same_paths = [
        (base / a, base / b) for a, b in data.get("same_person", [])
    ]
    diff_paths = [
        (base / a, base / b) for a, b in data.get("different_people", [])
    ]

    print("=== Same person ===")
    same_rows = score_pairs(same_paths)
    same_scores = [s for *_, s, err in same_rows if s is not None and not err]
    for a, b, score, err in same_rows:
        if err:
            print(f"  FAIL  {a} vs {b}: {err}")
        else:
            print(f"  {score:.3f}  {a} vs {b}")

    print("\n=== Different people ===")
    diff_rows = score_pairs(diff_paths)
    diff_scores = [s for *_, s, err in diff_rows if s is not None and not err]
    for a, b, score, err in diff_rows:
        if err:
            print(f"  FAIL  {a} vs {b}: {err}")
        else:
            print(f"  {score:.3f}  {a} vs {b}")

    if same_scores:
        print(
            f"\nSame-person range:      "
            f"{min(same_scores):.3f} … {max(same_scores):.3f}"
        )
    if diff_scores:
        print(
            f"Different-people range: "
            f"{min(diff_scores):.3f} … {max(diff_scores):.3f}"
        )

    suggested = suggest_threshold(same_scores, diff_scores)
    current = get_context().settings.face_match_threshold
    print(f"\nCurrent threshold:   {current:.2f}")
    if suggested is None:
        print(
            "Suggested threshold: (need at least one successful same-person "
            "and one different-people pair)"
        )
        return 1

    print(f"Suggested threshold: {suggested:.2f}")
    print(
        "\nTo apply, set SEEKPIX_FACE_MATCH_THRESHOLD in .env and restart "
        "the API."
    )
    if min(same_scores) <= max(diff_scores):
        print(
            "WARNING: score ranges overlap — more / better labeled pairs "
            "are needed before trusting a single cut-off."
        )
        return 2
    return 0


def run_probe(probe: Path, top_k: int) -> int:
    ctx = get_context()
    matcher = FaceMatcher(ctx.db, ctx.face_store)
    matches = matcher.find_matches(probe, top_k=top_k, threshold=0.0)
    print(f"Probe: {probe}")
    print(f"Library faces: {ctx.face_store.count()}")
    print(f"Top {top_k} matches (threshold ignored for inspection):\n")
    if not matches:
        print("  (no faces in library or no face in probe)")
        return 1
    for m in matches:
        print(f"  {m.score:.3f}  {m.photo.filename}  (face_id={m.face.id})")
    print(
        f"\nCurrent threshold: {ctx.settings.face_match_threshold:.2f} "
        f"— matches at or above this would be returned by /faces/find"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        help="Path to pairs.json with labeled same/different pairs",
    )
    parser.add_argument("--probe", type=Path, help="Probe image for library mode")
    parser.add_argument(
        "--against-library",
        action="store_true",
        help="Score probe against the indexed face library",
    )
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    if args.pairs:
        raise SystemExit(run_labeled(args.pairs.resolve()))
    if args.probe and args.against_library:
        raise SystemExit(run_probe(args.probe.resolve(), args.top_k))

    parser.error("Provide --pairs PATH, or --probe PATH --against-library")


if __name__ == "__main__":
    main()
