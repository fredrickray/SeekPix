# SeekPix (Backend)

Local-first photo search and face verification API for **SeekPix UI**.

Search runs only against photos you ingest — no external photo library.

## Architecture

```text
SeekPix UI  ──HTTP──►  api/ (FastAPI)
                            │
                       services/   (use-cases)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ingestion    semantic_search   face_pipeline
              └─────────────┬─────────────┘
                            ▼
                         storage
                SQLite + 2 vector indexes + files
```

`core/` never imports FastAPI. The API and CLI both call the same service functions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Bulk-index a folder
python scripts/index_folder.py --folder /path/to/photos

# Run API for SeekPix UI
uvicorn api.main:app --reload --port 8000
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /stats` | Photo / face / vector counts |
| `GET /photos?limit=&offset=` | Browse the library; returns `total` for pagination |
| `GET /photos/{id}/thumbnail` | Thumbnail bytes (always JPEG) |
| `GET /photos/{id}/file` | Full image; non-web formats such as HEIC are transcoded to JPEG |
| `POST /search` | `{"query": "red car", "top_k": 10}` |
| `POST /photos/upload` | Multipart image upload; returns a job to poll |
| `POST /index` | `{"folder": "/path/to/photos"}`; returns a job to poll |
| `GET /jobs/{id}` | Progress of an upload or indexing run |
| `GET /jobs` | Recent jobs |
| `POST /faces/find` | Upload a probe face, get photos containing that person |
| `POST /faces/verify` | Upload two photos, get a similarity score |

### Uploading and indexing

Indexing runs far longer than an HTTP request should, so `POST /photos/upload`
and `POST /index` return `202` with a job the client polls:

```json
{ "id": "f446ebcf…", "status": "running", "processed": 1, "total": 2,
  "indexed": 1, "failed": 1, "errors": ["clip.mov: unsupported file type"] }
```

Uploads are stored in `data/photos/`, renamed rather than overwritten on name
collisions, and unsupported files (video, for example) are rejected per file
instead of failing the whole batch. Jobs run one at a time, since the vector
indexes are held in memory and saved as a unit.

Photo responses carry `thumbnail_url` and `image_url` rather than server
filesystem paths, so the frontend can render them directly:

```json
{
  "id": 16,
  "filename": "IMG_6337.JPEG",
  "thumbnail_url": "/photos/16/thumbnail",
  "image_url": "/photos/16/file",
  "score": 0.251
}
```

## Build phases

| Phase | Scope |
|-------|--------|
| 0 | Skeleton — config, layout, deps |
| 1 | Storage — SQLite + vector index wrappers |
| 2 | Ingestion — scan, EXIF, thumbnails |
| 3 | Semantic search — CLIP + search + CLI |
| 4 | Face pipeline — detect, embed, match, dedupe |
| 5 | Services + API — use-cases + FastAPI for UI |

## Face match threshold

Default `SEEKPIX_FACE_MATCH_THRESHOLD` is **0.70** (cosine similarity).

That value was chosen from a probe against the demo library: strong candidates
scored ≥ 0.77, ambiguous mid-scores sat around 0.60–0.63, and clear non-matches
were ≤ 0.14. Identity verification prefers rejecting a borderline same-person
pair over accepting a different person, so the cut sits above the ambiguous
band.

Refine it with labeled fixtures:

```bash
cp tests/fixtures/pairs.example.json tests/fixtures/pairs.json
# edit paths, drop images under tests/fixtures/
python scripts/calibrate_face_threshold.py --pairs tests/fixtures/pairs.json
```

Then set `SEEKPIX_FACE_MATCH_THRESHOLD` in `.env` and restart the API.

