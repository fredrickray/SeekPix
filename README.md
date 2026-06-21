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
| `GET /photos?limit=&offset=` | Browse the library |
| `GET /photos/{id}/thumbnail` | Thumbnail bytes (always JPEG) |
| `GET /photos/{id}/file` | Full image; non-web formats such as HEIC are transcoded to JPEG |
| `POST /search` | `{"query": "red car", "top_k": 10}` |
| `POST /index` | `{"folder": "/path/to/photos"}` |
| `POST /faces/find` | Upload a probe face, get photos containing that person |
| `POST /faces/verify` | Upload two photos, get a similarity score |

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

## Data

All runtime data lives under `data/` (gitignored). Whatever you ingest is the entire searchable library for that instance.
