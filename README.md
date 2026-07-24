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
                   SQLite + FAISS×2 + files
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

## Build phases

| Phase | Scope |
|-------|--------|
| 0 | Skeleton — config, layout, deps |
| 1 | Storage — SQLite + FAISS wrappers |
| 2 | Ingestion — scan, EXIF, thumbnails |
| 3 | Semantic search — CLIP + search + CLI |
| 4 | Face pipeline — detect, embed, match, dedupe |
| 5 | Services + API — use-cases + FastAPI for UI |

## Data

All runtime data lives under `data/` (gitignored). Whatever you ingest is the entire searchable library for that instance.
