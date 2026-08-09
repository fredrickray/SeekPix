#!/usr/bin/env python3
"""CLI: rebuild vector indexes from scratch (after model change)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.face_pipeline.embedder import get_face_embedder
from core.semantic_search.clip_embedder import get_clip_embedder
from core.services.context import get_context, reset_context
from core.storage.db import MetadataDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild SeekPix vector indexes")
    parser.add_argument(
        "--faces-only",
        action="store_true",
        help="Only rebuild face index",
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
        help="Only rebuild CLIP index",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    reset_context()
    ctx = get_context(settings)
    db: MetadataDB = ctx.db

    do_clip = not args.faces_only
    do_faces = not args.clip_only

    if do_clip:
        print("Rebuilding CLIP index...")
        ctx.clip_store.reset()
        clip = get_clip_embedder()
        photos = db.list_photos(limit=1_000_000)
        for photo in photos:
            vec = clip.embed_image(photo.filepath)
            ids = ctx.clip_store.add(vec)
            db.update_photo_clip_id(photo.id, ids[0])
            print(f"  clip #{photo.id} {photo.filename}")
        ctx.clip_store.save()

    if do_faces:
        print("Rebuilding face index...")
        ctx.face_store.reset()
        # Clear face rows — will re-insert
        with db.connect() as conn:
            conn.execute("DELETE FROM faces")
        faces = get_face_embedder()
        photos = db.list_photos(limit=1_000_000)
        for photo in photos:
            detected = faces.detect_and_embed(photo.filepath)
            for face in detected:
                face_ids = ctx.face_store.add(face.embedding)
                db.insert_face(
                    photo_id=photo.id,
                    face_vector_id=face_ids[0],
                    bbox_x1=face.bbox[0],
                    bbox_y1=face.bbox[1],
                    bbox_x2=face.bbox[2],
                    bbox_y2=face.bbox[3],
                    det_score=face.det_score,
                )
            print(f"  faces #{photo.id} {photo.filename} ({len(detected)})")
        ctx.face_store.save()

    print("Rebuild complete.")


if __name__ == "__main__":
    main()
