"""In-process job queue for long-running evaluations.

Deliberately not Celery. This project's deployment target is a single free-tier
container, where a broker plus a worker pool would be more moving parts than the
workload justifies. What is needed is: submit a job, poll its status, stream
progress, and never block the event loop.

Jobs run in a thread pool because model fitting is CPU-bound and would otherwise
stall every other request. If throughput ever outgrows this, the interface is
small enough to swap for a real broker without touching the routers.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from forecaster.logging import get_logger

log = get_logger(__name__)

#: Retention window for finished jobs. Long enough for a client to poll, short
#: enough that a long-lived process does not leak memory.
JOB_TTL_SECONDS = 3600
MAX_WORKERS = 2


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at

    @property
    def is_terminal(self) -> bool:
        return self.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            **self.metadata,
        }


class JobQueue:
    def __init__(self, max_workers: int = MAX_WORKERS) -> None:
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="forecaster-job"
        )
        self._events: dict[str, asyncio.Event] = {}

    def submit(
        self,
        kind: str,
        fn: Callable[..., Any],
        *args: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Job:
        """Queue a callable. Returns immediately with a QUEUED job."""
        self._evict_expired()

        job = Job(id=uuid.uuid4().hex[:16], kind=kind, metadata=metadata or {})
        self._jobs[job.id] = job
        self._events[job.id] = asyncio.Event()

        loop = asyncio.get_running_loop()

        def runner() -> None:
            job.state = JobState.RUNNING
            job.started_at = time.time()
            self._notify(loop, job.id)
            try:
                # The callable receives a progress reporter it can call freely.
                job.result = fn(*args, progress=_Reporter(job, loop, self), **kwargs)
                job.state = JobState.SUCCEEDED
                job.progress = 1.0
            except Exception as exc:  # noqa: BLE001 -- surfaced via job.error
                job.state = JobState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                log.warning("job_failed", job_id=job.id, kind=kind, error=job.error)
            finally:
                job.finished_at = time.time()
                self._notify(loop, job.id)

        self._executor.submit(runner)
        log.info("job_submitted", job_id=job.id, kind=kind, **(metadata or {}))
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self, *, kind: str | None = None, limit: int = 50) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: -j.created_at)
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        return jobs[:limit]

    def cancel(self, job_id: str) -> bool:
        """Mark a queued job cancelled.

        Threads already running are not interrupted -- Python offers no safe
        way to kill one mid-fit. A running job is left to finish.
        """
        job = self._jobs.get(job_id)
        if job is None or job.is_terminal:
            return False
        if job.state is JobState.QUEUED:
            job.state = JobState.CANCELLED
            job.finished_at = time.time()
            return True
        return False

    async def wait_for_update(self, job_id: str, timeout: float = 30.0) -> Job | None:
        """Block until the job changes state, for WebSocket streaming."""
        event = self._events.get(job_id)
        if event is None:
            return None
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError:
            pass
        event.clear()
        return self._jobs.get(job_id)

    def _notify(self, loop: asyncio.AbstractEventLoop, job_id: str) -> None:
        event = self._events.get(job_id)
        if event is not None:
            loop.call_soon_threadsafe(event.set)

    def _evict_expired(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        stale = [
            jid for jid, job in self._jobs.items()
            if job.is_terminal and (job.finished_at or job.created_at) < cutoff
        ]
        for jid in stale:
            self._jobs.pop(jid, None)
            self._events.pop(jid, None)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class _Reporter:
    """Progress handle handed to job callables.

    Matches the ``advance``/``update`` interface the harness and ingestion
    service already expect, so the same code works under the CLI's Rich status
    display and under the API's WebSocket.
    """

    def __init__(self, job: Job, loop: asyncio.AbstractEventLoop, queue: JobQueue) -> None:
        self._job = job
        self._loop = loop
        self._queue = queue
        self._done = 0
        self._total = 0

    def set_total(self, total: int) -> None:
        self._total = max(1, total)

    def advance(self, n: int = 1) -> None:
        self._done += n
        if self._total:
            self._job.progress = min(0.99, self._done / self._total)
        self._queue._notify(self._loop, self._job.id)

    def update(self, message: str, progress: float | None = None) -> None:
        self._job.message = message
        if progress is not None:
            self._job.progress = min(0.99, max(0.0, progress))
        self._queue._notify(self._loop, self._job.id)


_queue: JobQueue | None = None


def get_queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue
