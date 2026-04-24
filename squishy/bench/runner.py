"""Generic async batch runner for benchmarks.
 
- Concurrency cap (asyncio.Semaphore)
- Per-task timeout
- Best-effort error capture — one task failing does not stop the batch
- Streaming JSONL writer with thread-safe flush (one line per result)
"""
 
from __future__ import annotations
 
import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
 
log = logging.getLogger("squishy.bench.runner")
 
 
class BenchTask(Protocol):
    """Minimum shape a task must expose."""
 
    id: str
 
 
@dataclass
class BenchResult:
    task_id: str
    success: bool
    prediction: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_s: float = 0.0
 
 
class PredictionWriter:
    """Append-only JSONL writer, flushed per result. Safe for one writer."""
 
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._fh = self.path.open("a", encoding="utf-8")
 
    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            # fsync after flush so long bench runs survive a kernel panic or
            # SIGKILL. Python flush() only pushes user-space buffers; the OS
            # page cache can still drop data.
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                # Some file systems (tmpfs, overlays in certain containers)
                # reject fsync. Log and keep going — durability becomes
                # best-effort rather than a crash.
                log.debug("fsync unavailable on %s", self.path)
 
    def close(self) -> None:
        self._fh.close()
 
    async def __aenter__(self) -> PredictionWriter:
        return self
 
    async def __aexit__(self, *_: Any) -> None:
        self.close()
 
 
async def run_batch(
    tasks: list[Any],
    runner: Callable[[Any], Awaitable[BenchResult]],
    *,
    concurrency: int = 4,
    per_task_timeout: float | None = None,
    writer: PredictionWriter | None = None,
    on_progress: Callable[[BenchResult, int, int], None] | None = None,
) -> list[BenchResult]:
    """Run ``runner`` over ``tasks`` with bounded concurrency.
 
    ``runner`` receives one task and returns a BenchResult. Timeouts and
    exceptions are caught and recorded; one task's failure does not stop
    the batch.
    """
    sem = asyncio.Semaphore(concurrency)
    total = len(tasks)
    completed = 0
    lock = asyncio.Lock()
 
    async def _one(task: Any) -> BenchResult:
        nonlocal completed
        async with sem:
            t0 = time.monotonic()
            task_id = getattr(task, "id", None) or str(getattr(task, "get", lambda _k: None)("id") or "?")
            try:
                if per_task_timeout is not None:
                    async with asyncio.timeout(per_task_timeout):
                        result = await runner(task)
                else:
                    result = await runner(task)
            except TimeoutError:
                result = BenchResult(
                    task_id=task_id,
                    success=False,
                    error=f"per-task timeout ({per_task_timeout}s)",
                    elapsed_s=time.monotonic() - t0,
                )
            except Exception as e:  # noqa: BLE001
                log.exception("task %s raised", task_id)
                result = BenchResult(
                    task_id=task_id,
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                    elapsed_s=time.monotonic() - t0,
                )
 
            if writer is not None:
                record = {"task_id": result.task_id, **result.prediction}
                if result.artifacts:
                    record["artifacts"] = result.artifacts
                if not result.success:
                    record["error"] = result.error
                await writer.write(record)
 
            async with lock:
                completed += 1
                if on_progress is not None:
                    on_progress(result, completed, total)
 
            return result
 
    results = await asyncio.gather(*(_one(t) for t in tasks))
    return results
