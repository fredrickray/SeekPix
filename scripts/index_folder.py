#!/usr/bin/env python3
"""CLI: bulk-index a photo folder into SeekPix storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/index_folder.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.services.context import get_context
from core.services.indexing import index_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Index photos into SeekPix")
    parser.add_argument("--folder", required=True, help="Folder of images to index")
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Index in place (do not copy into data/photos)",
    )
    parser.add_argument(
        "--no-faces",
        action="store_true",
        help="Skip face detection/embedding (CLIP only)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subfolders",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    ctx = get_context(settings)

    def progress(i: int, total: int, name: str) -> None:
        print(f"[{i}/{total}] {name}")

    result = index_folder(
        args.folder,
        ctx=ctx,
        recursive=not args.no_recursive,
        copy_into_library=not args.no_copy,
        run_faces=not args.no_faces,
        on_progress=progress,
    )

    print(
        f"Done. indexed={result.indexed} skipped={result.skipped} "
        f"failed={result.failed}"
    )
    for err in result.errors:
        print(f"  ERROR: {err}", file=sys.stderr)
    if result.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
