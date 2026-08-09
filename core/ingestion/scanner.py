"""Walk a folder and discover image files."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


def scan_folder(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Return sorted list of image paths under folder."""
    root = Path(folder).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pattern = "**/*" if recursive else "*"
    paths = [
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)
