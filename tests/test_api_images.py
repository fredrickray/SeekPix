"""API image-serving tests — no ML models required."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from core.config import get_settings


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Boot the API against a throwaway library containing one photo."""
    data = tmp_path / "data"
    for name in ("photos", "thumbnails", "indexes"):
        (data / name).mkdir(parents=True)

    monkeypatch.setenv("SEEKPIX_DATA_DIR", str(data))
    monkeypatch.setenv("SEEKPIX_PHOTOS_DIR", str(data / "photos"))
    monkeypatch.setenv("SEEKPIX_THUMBNAILS_DIR", str(data / "thumbnails"))
    monkeypatch.setenv("SEEKPIX_INDEXES_DIR", str(data / "indexes"))
    monkeypatch.setenv("SEEKPIX_DB_PATH", str(data / "metadata.db"))
    get_settings.cache_clear()

    from core.services import context as context_module

    context_module.reset_context()

    import api.main as api_main

    importlib.reload(api_main)

    source = data / "photos" / "cat.png"
    Image.new("RGB", (40, 30), color=(10, 120, 200)).save(source)
    thumb = data / "thumbnails" / "1_cat.jpg"
    Image.new("RGB", (20, 15), color=(10, 120, 200)).save(thumb, format="JPEG")

    ctx = context_module.get_context()
    photo = ctx.db.insert_photo(
        filepath=str(source),
        filename=source.name,
        thumbnail_path=str(thumb),
    )

    with TestClient(api_main.app) as client:
        yield client, photo.id

    get_settings.cache_clear()
    context_module.reset_context()


def test_photos_expose_urls_not_filesystem_paths(api_client) -> None:
    client, photo_id = api_client
    body = client.get("/photos").json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["thumbnail_url"] == f"/photos/{photo_id}/thumbnail"
    assert item["image_url"] == f"/photos/{photo_id}/file"
    assert "filepath" not in item
    assert "thumbnail_path" not in item


def test_thumbnail_is_served_as_jpeg(api_client) -> None:
    client, photo_id = api_client
    resp = client.get(f"/photos/{photo_id}/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_web_safe_original_keeps_its_format(api_client) -> None:
    client, photo_id = api_client
    resp = client.get(f"/photos/{photo_id}/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_unknown_photo_returns_404(api_client) -> None:
    client, _ = api_client
    assert client.get("/photos/4242/thumbnail").status_code == 404
    assert client.get("/photos/4242/file").status_code == 404


def test_missing_thumbnail_file_returns_404(api_client) -> None:
    client, photo_id = api_client
    from core.services.context import get_context

    photo = get_context().db.get_photo(photo_id)
    Path(photo.thumbnail_path).unlink()
    assert client.get(f"/photos/{photo_id}/thumbnail").status_code == 404
