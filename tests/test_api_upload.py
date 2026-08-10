"""Upload, job-tracking, and pagination tests.

Indexing is stubbed out so these run without downloading CLIP or InsightFace;
what's under test is the HTTP contract and the job lifecycle, not the models.
"""

from __future__ import annotations

import importlib
import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from core.config import get_settings
from core.storage.models import IndexResult


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    from core.services import jobs as jobs_module

    context_module.reset_context()
    jobs_module.reset_job_registry()

    import api.main as api_main

    importlib.reload(api_main)

    # Replace real indexing with a recorder: no models, no embeddings.
    indexed: list[list[Path]] = []

    def fake_index_paths(paths, *, on_progress=None, **kwargs):
        paths = [Path(p) for p in paths]
        indexed.append(paths)
        for i, path in enumerate(paths, start=1):
            if on_progress:
                on_progress(i, len(paths), path.name)
        return IndexResult(indexed=len(paths))

    monkeypatch.setattr(api_main, "index_paths", fake_index_paths)

    with TestClient(api_main.app) as client:
        yield client, data, indexed

    get_settings.cache_clear()
    context_module.reset_context()
    jobs_module.reset_job_registry()


def _png_bytes(color=(20, 90, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _await_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    """Poll until the job leaves a non-terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_upload_stores_files_and_indexes_them(api_client) -> None:
    client, data, indexed = api_client
    resp = client.post(
        "/photos/upload",
        files=[
            ("files", ("a.png", _png_bytes(), "image/png")),
            ("files", ("b.png", _png_bytes((200, 30, 30)), "image/png")),
        ],
    )
    assert resp.status_code == 202
    job = resp.json()
    assert job["kind"] == "upload"
    assert job["total"] == 2

    finished = _await_job(client, job["id"])
    assert finished["status"] == "completed"
    assert finished["indexed"] == 2
    assert finished["processed"] == 2

    assert (data / "photos" / "a.png").is_file()
    assert (data / "photos" / "b.png").is_file()
    # Files are already inside the library, so indexing must not re-copy them
    assert indexed and {p.name for p in indexed[0]} == {"a.png", "b.png"}


def test_upload_renames_instead_of_overwriting(api_client) -> None:
    client, data, _ = api_client
    for _ in range(2):
        resp = client.post(
            "/photos/upload",
            files=[("files", ("same.png", _png_bytes(), "image/png"))],
        )
        _await_job(client, resp.json()["id"])

    assert (data / "photos" / "same.png").is_file()
    assert (data / "photos" / "same_2.png").is_file()


def test_upload_rejects_unsupported_files(api_client) -> None:
    client, _, _ = api_client
    resp = client.post(
        "/photos/upload",
        files=[("files", ("clip.mp4", b"not an image", "video/mp4"))],
    )
    assert resp.status_code == 400
    assert "clip.mp4" in resp.json()["detail"]


def test_upload_reports_partial_rejections_on_the_job(api_client) -> None:
    client, _, _ = api_client
    resp = client.post(
        "/photos/upload",
        files=[
            ("files", ("good.png", _png_bytes(), "image/png")),
            ("files", ("clip.mov", b"nope", "video/quicktime")),
        ],
    )
    assert resp.status_code == 202
    finished = _await_job(client, resp.json()["id"])
    assert finished["indexed"] == 1
    assert finished["failed"] == 1
    assert any("clip.mov" in e for e in finished["errors"])


def test_uploaded_filename_cannot_escape_the_library(api_client) -> None:
    client, data, _ = api_client
    resp = client.post(
        "/photos/upload",
        files=[("files", ("../../escaped.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 202
    _await_job(client, resp.json()["id"])

    assert (data / "photos" / "escaped.png").is_file()
    assert not (data.parent.parent / "escaped.png").exists()


def test_index_rejects_bad_folder(api_client) -> None:
    client, _, _ = api_client
    resp = client.post("/index", json={"folder": "/definitely/not/here"})
    assert resp.status_code == 400


def test_unknown_job_returns_404(api_client) -> None:
    client, _, _ = api_client
    assert client.get("/jobs/nope").status_code == 404


def test_jobs_listing_includes_submitted_job(api_client) -> None:
    client, _, _ = api_client
    resp = client.post(
        "/photos/upload",
        files=[("files", ("x.png", _png_bytes(), "image/png"))],
    )
    job_id = resp.json()["id"]
    _await_job(client, job_id)
    assert job_id in [j["id"] for j in client.get("/jobs").json()]


def test_photos_pagination_reports_total(api_client) -> None:
    client, _, _ = api_client
    files = [
        ("files", (f"p{i}.png", _png_bytes((i * 20 % 255, 40, 90)), "image/png"))
        for i in range(3)
    ]
    resp = client.post("/photos/upload", files=files)
    _await_job(client, resp.json()["id"])

    # Rows come from the stubbed indexer's caller, so insert them directly
    from core.services.context import get_context

    ctx = get_context()
    for i in range(3):
        ctx.db.insert_photo(filepath=f"/tmp/p{i}.png", filename=f"p{i}.png")

    page = client.get("/photos?limit=2&offset=0").json()
    assert page["total"] == 3
    assert page["limit"] == 2
    assert len(page["items"]) == 2
