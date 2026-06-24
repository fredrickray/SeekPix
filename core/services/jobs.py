"""Background job tracking for long-running indexing runs.

Indexing a large folder takes far longer than an HTTP request should, so the
API starts a job and lets the client poll for progress. Jobs are serialized:
the vector stores are held in memory and saved as a unit, so two concurrent
indexing runs would race on the same objects.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from core.storage.models import IndexResult


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    total: int = 0
    processed: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    current_file: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "total": self.total,
            "processed": self.processed,
            "indexed": self.indexed,
            "skipped": self.skipped,
            "failed": self.failed,
            "current_file": self.current_file,
            "errors": list(self.errors),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


JobWork = Callable[[Job], IndexResult]


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    def create(self, kind: str, *, total: int = 0) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, total=total)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            )
        return jobs[:limit]

    def submit(self, job: Job, work: JobWork) -> Job:
        thread = threading.Thread(
            target=self._run, args=(job, work), name=f"seekpix-{job.kind}", daemon=True
        )
        thread.start()
        return job

    def _run(self, job: Job, work: JobWork) -> None:
        # Serialized so concurrent runs cannot corrupt the shared vector stores.
        with self._run_lock:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            try:
                result = work(job)
                job.indexed = result.indexed
                job.skipped = result.skipped
                # Accumulated, because callers may have recorded failures
                # (such as rejected uploads) before the work started.
                job.failed += result.failed
                job.errors.extend(result.errors)
                job.status = JobStatus.COMPLETED
            except Exception as exc:  # noqa: BLE001 — surfaced via job status
                job.error = str(exc)
                job.status = JobStatus.FAILED
            finally:
                job.current_file = None
                job.finished_at = datetime.utcnow()


_registry: Optional[JobRegistry] = None


def get_job_registry() -> JobRegistry:
    global _registry
    if _registry is None:
        _registry = JobRegistry()
    return _registry


def reset_job_registry() -> None:
    """Test helper — drop all tracked jobs."""
    global _registry
    _registry = None


def make_progress_callback(job: Job) -> Callable[[int, int, str], None]:
    def on_progress(processed: int, total: int, filename: str) -> None:
        job.total = total
        job.processed = processed
        job.current_file = filename

    return on_progress
