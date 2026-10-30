"""
Background task runner for long-running effGen agent executions.

Uses threading.Thread (not multiprocessing) so tasks share the in-memory
agent / model and avoid serialization overhead. A simple priority queue
schedules pending tasks; running tasks expose status, progress, result,
and basic cancel/pause/resume controls.

Worker threads target a module-level loop bound to an internal state object
(never the runner itself), so the runner can be garbage-collected even while
idle workers wait; a ``weakref.finalize`` then stops the threads cleanly. This
avoids the "thread left running / GC'd without close" leak that a bound-method
worker target would cause.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


def _redact(text: str) -> str:
    """Scrub secrets from an error string before it is surfaced/logged."""
    try:
        from ..observability.redact import get_redactor
        return get_redactor().scrub(text)
    except Exception:  # pragma: no cover - redaction is best-effort
        return text


class TaskStatus(StrEnum):
    """Lifecycle state of a background task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(order=True)
class _PrioritizedItem:
    priority: int
    seq: int
    task_id: str = field(compare=False)


@dataclass
class BackgroundTask:
    """A queued or running task with its status, progress and result."""

    task_id: str
    task_text: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5
    progress: float = 0.0
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _RunnerState:
    """Mutable shared state for a runner's worker threads.

    Held by the worker threads (so they reference *this*, not the public
    ``BackgroundTaskRunner``). When the runner is collected, its finalizer flips
    ``shutdown`` and notifies, letting idle workers exit.
    """

    def __init__(self, agent: Any):
        self.agent = agent
        self.tasks: dict[str, BackgroundTask] = {}
        self.queue: list[_PrioritizedItem] = []
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.pause_events: dict[str, threading.Event] = {}
        self.cancel_flags: dict[str, threading.Event] = {}
        self.progress_callbacks: dict[str, Callable[[BackgroundTask], None]] = {}
        self.seq = 0
        self.shutdown = False
        self.workers: list[threading.Thread] = []


def _request_shutdown(state: _RunnerState) -> None:
    """Signal all workers to stop. Safe to call from a finalizer / atexit."""
    with state.cv:
        state.shutdown = True
        state.cv.notify_all()


def _worker_loop(state: _RunnerState) -> None:
    while True:
        with state.cv:
            while not state.queue and not state.shutdown:
                state.cv.wait()
            if state.shutdown and not state.queue:
                return
            item = heapq.heappop(state.queue)
            task = state.tasks.get(item.task_id)
            if task is None or task.status == TaskStatus.CANCELLED:
                continue
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

        _execute(state, task)


def _execute(state: _RunnerState, task: BackgroundTask) -> None:
    try:
        # Honor pause / cancel before starting heavy work
        state.pause_events[task.task_id].wait()
        if state.cancel_flags[task.task_id].is_set():
            with state.lock:
                task.status = TaskStatus.CANCELLED
                task.finished_at = time.time()
            return

        run_kwargs = task.metadata.get("run_kwargs", {}) or {}
        result = state.agent.run(task.task_text, **run_kwargs)

        with state.lock:
            if state.cancel_flags[task.task_id].is_set():
                task.status = TaskStatus.CANCELLED
            elif getattr(result, "success", True) is False:
                # A reported failure from the agent: the run
                # came back unsuccessful. Surface it as FAILED with the typed,
                # redacted reason rather than reporting a silent COMPLETED.
                task.status = TaskStatus.FAILED
                task.result = result
                reason = ""
                meta = getattr(result, "metadata", None)
                if isinstance(meta, dict):
                    reason = meta.get("error") or meta.get("reason") or ""
                task.error = _redact(reason or getattr(result, "output", "") or "Task failed")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.progress = 1.0
            task.finished_at = time.time()
    except Exception as e:
        with state.lock:
            task.status = TaskStatus.FAILED
            task.error = _redact(f"{type(e).__name__}: {e}")
            task.finished_at = time.time()
    finally:
        cb = state.progress_callbacks.get(task.task_id)
        if cb:
            try:
                cb(task)
            except Exception:
                logger.debug(
                    "Progress callback for task %s raised", task.task_id, exc_info=True
                )


class BackgroundTaskRunner:
    """
    Execute agent tasks in background worker threads.

    Usage:
        runner = BackgroundTaskRunner(agent, max_workers=2)
        task_id = runner.submit("Long task...", priority=1)
        status = runner.get_status(task_id)
        result = runner.get_result(task_id, wait=True)
        runner.shutdown()                 # or use as a context manager

    The runner stops its worker threads automatically when it is garbage
    collected and at interpreter exit, so a forgotten ``shutdown()`` never
    leaves threads running. Prefer the explicit ``shutdown()`` / context
    manager for prompt cleanup.
    """

    def __init__(self, agent: Any, max_workers: int = 1) -> None:
        self.agent = agent
        self.max_workers = max_workers
        self._state = _RunnerState(agent)

        for i in range(max_workers):
            t = threading.Thread(
                target=_worker_loop,
                args=(self._state,),
                name=f"effgen-bg-worker-{i}",
                daemon=True,
            )
            t.start()
            self._state.workers.append(t)

        # Stop the worker threads when this runner is collected, and as a
        # backstop at interpreter shutdown (weakref.finalize runs at exit too).
        # The finalizer closes over `state` only (never `self`), so it does not
        # keep the runner alive.
        self._finalizer = weakref.finalize(self, _request_shutdown, self._state)

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        task_text: str,
        priority: int = 5,
        progress_callback: Callable[[BackgroundTask], None] | None = None,
        **run_kwargs: Any,
    ) -> str:
        """Enqueue a task and return its id. Lower priority value = higher priority.

        Args:
            task_text: The task to run.
            priority: Queue priority; a lower number runs sooner.
            progress_callback: Called with the task each time its state changes.
            **run_kwargs: Extra keyword arguments forwarded to the agent's run.
        """
        s = self._state
        task_id = str(uuid.uuid4())
        task = BackgroundTask(
            task_id=task_id,
            task_text=task_text,
            priority=priority,
            metadata={"run_kwargs": run_kwargs},
        )
        with s.cv:
            s.tasks[task_id] = task
            s.pause_events[task_id] = threading.Event()
            s.pause_events[task_id].set()  # not paused
            s.cancel_flags[task_id] = threading.Event()
            if progress_callback:
                s.progress_callbacks[task_id] = progress_callback
            s.seq += 1
            heapq.heappush(s.queue, _PrioritizedItem(priority, s.seq, task_id))
            s.cv.notify()
        return task_id

    # ------------------------------------------------------------------ status / result
    def _require_task(self, task_id: str) -> BackgroundTask:
        """Look up a task, raising a clear error for an unknown id.

        The caller holds ``self._state.lock``. A bare ``dict[task_id]`` would
        raise an opaque ``KeyError('<uuid>')``; surface the actual problem
        instead so a script can act on it.
        """
        task = self._state.tasks.get(task_id)
        if task is None:
            raise KeyError(
                f"No background task with id '{task_id}'. It may have already "
                "been cleared, or the id is wrong — use list_tasks() to see "
                "active task ids."
            )
        return task

    def get_status(self, task_id: str) -> TaskStatus:
        """Return the current status of the task *task_id*."""
        with self._state.lock:
            return self._require_task(task_id).status

    def get_task(self, task_id: str) -> BackgroundTask:
        """Return the task record for *task_id* (raises ``KeyError`` if unknown)."""
        with self._state.lock:
            return self._require_task(task_id)

    def get_result(self, task_id: str, wait: bool = False, timeout: float | None = None) -> Any:
        """Return the task's result, optionally blocking until it finishes.

        Args:
            task_id: The id :meth:`submit` returned.
            wait: Block until the task finishes instead of returning at once.
            timeout: Seconds to block for when *wait* is set.
        """
        s = self._state
        if wait:
            deadline = None if timeout is None else time.time() + timeout
            while True:
                with s.lock:
                    task = self._require_task(task_id)
                    if task.status in (
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                    ):
                        return task.result
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError(f"Task {task_id} did not finish in time")
                time.sleep(0.05)
        with s.lock:
            return self._require_task(task_id).result

    def list_tasks(self) -> list[BackgroundTask]:
        """Return every known task, whatever its status."""
        with self._state.lock:
            return list(self._state.tasks.values())

    # ------------------------------------------------------------------ control
    def cancel(self, task_id: str) -> bool:
        """Cancel a pending or running task; returns whether a cancel was issued."""
        s = self._state
        with s.lock:
            task = s.tasks.get(task_id)
            if task is None:
                return False
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.finished_at = time.time()
                return True
            if task.status in (TaskStatus.RUNNING, TaskStatus.PAUSED):
                s.cancel_flags[task_id].set()
                # Unblock any pause wait so the worker can observe the cancel
                s.pause_events[task_id].set()
                return True
            return False

    def pause(self, task_id: str) -> bool:
        """Pause a running task at its next checkpoint; returns whether it paused."""
        s = self._state
        with s.lock:
            task = s.tasks.get(task_id)
            if task is None or task.status != TaskStatus.RUNNING:
                return False
            s.pause_events[task_id].clear()
            task.status = TaskStatus.PAUSED
            return True

    def resume(self, task_id: str) -> bool:
        """Resume a paused task; returns whether it was resumed."""
        s = self._state
        with s.lock:
            task = s.tasks.get(task_id)
            if task is None or task.status != TaskStatus.PAUSED:
                return False
            s.pause_events[task_id].set()
            task.status = TaskStatus.RUNNING
            return True

    def shutdown(self, wait: bool = False) -> None:
        """Stop the workers (optionally joining them); repeat calls are no-ops."""
        # `finalizer()` runs `_request_shutdown(state)` exactly once and
        # de-registers the atexit hook.
        self._finalizer()
        if wait:
            for t in self._state.workers:
                t.join(timeout=5)

    # Allow `with BackgroundTaskRunner(...) as runner:` for prompt cleanup.
    close = shutdown

    def __enter__(self) -> "BackgroundTaskRunner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown(wait=True)
