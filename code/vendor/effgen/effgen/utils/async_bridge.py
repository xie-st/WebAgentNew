"""Synchronous/asynchronous bridging helpers.

A single, correct way to drive a coroutine to completion from synchronous code,
regardless of whether an event loop is already running in the calling thread.

Why this exists
---------------
Synchronous entry points (e.g. ``Model.generate``) sometimes need to await an
async primitive such as a rate-limit coordinator.  The naive pattern

    loop = asyncio.get_event_loop()
    if loop.is_running():
        ...  # skip
    else:
        loop.run_until_complete(coro)

is wrong in two ways: ``get_event_loop()`` is deprecated when no loop is running,
and *skipping* the work under a running loop (Jupyter, FastAPI, any async caller)
silently disables it — exactly where it matters most.  :func:`run_coroutine_sync`
instead always runs the coroutine to completion: directly when the calling thread
has no running loop, or on a dedicated worker thread (blocking the caller) when a
loop is already running.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import TypeVar

__all__ = ["run_coroutine_sync", "is_loop_blocked"]

T = TypeVar("T")

# Event loops whose owning thread is currently inside the worker-thread branch
# of run_coroutine_sync() below — i.e. synchronously blocked on done.wait() and
# unable to service any callback scheduled on it (run_coroutine_threadsafe,
# call_soon_threadsafe, ...) until that call returns. A loop-bound async
# resource (an MCP stdio session is the known case) that tries to marshal work
# back onto one of these loops can detect the guaranteed deadlock immediately
# via is_loop_blocked() instead of waiting out a timeout that can never
# resolve in its favor.
_blocked_loops: set[int] = set()
_blocked_loops_lock = threading.Lock()


def is_loop_blocked(loop: asyncio.AbstractEventLoop | None) -> bool:
    """True when *loop* cannot service callbacks because its thread is bridge-blocked.

    The owning thread is synchronously blocked inside a
    :func:`run_coroutine_sync` bridge.

    A caller about to marshal work onto *loop* via
    ``asyncio.run_coroutine_threadsafe`` should check this first: if it is
    ``True``, that marshal can never complete (the thread that would run it is
    the same thread waiting on this bridge) and the caller should fail fast
    instead of waiting out a timeout.
    """
    if loop is None:
        return False
    with _blocked_loops_lock:
        return id(loop) in _blocked_loops


def run_coroutine_sync(coro: Awaitable[T], timeout: float = 120.0) -> T:
    """Run an awaitable from synchronous code and return its result.

    Behaviour:

    * **No running loop in the calling thread** — drive the coroutine directly
      with :func:`asyncio.run`.
    * **A loop is already running in the calling thread** (Jupyter, FastAPI, or
      any async caller that reached synchronous code) — run the coroutine on a
      dedicated worker thread with its own event loop and *block* the calling
      thread until it finishes.  The work is never silently skipped.

    Args:
        coro: The coroutine / awaitable to run.
        timeout: Maximum seconds to wait when bridging via a worker thread.

    Returns:
        The value produced by the coroutine.

    Raises:
        Any exception raised by the coroutine, plus :class:`TimeoutError` (with
        an actionable message) if the worker-thread path exceeds ``timeout`` —
        e.g. when a *loop-bound* async resource (such as an MCP stdio session)
        is driven from synchronous code under a running event loop and can never
        complete on the worker loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread — safe to drive it directly. We do not
        # catch RuntimeError from ``asyncio.run`` here so that a RuntimeError
        # raised *inside* the coroutine is not mistaken for "loop already
        # running" and re-run on an already-consumed coroutine.
        return asyncio.run(coro)  # type: ignore[arg-type]

    # A loop is already running in this thread; we cannot use asyncio.run or
    # block the live loop. Hand the coroutine to a fresh loop on a *daemon*
    # worker thread and wait synchronously, bounded by ``timeout``.
    #
    # We deliberately avoid ``ThreadPoolExecutor`` here. Its context manager
    # joins the worker on exit, and even ``shutdown(wait=False)`` leaves a
    # non-daemon thread that the ``concurrent.futures`` atexit hook joins at
    # interpreter shutdown. If the coroutine awaits a primitive bound to the
    # *calling* loop (now blocked on us) it deadlocks — and either join would
    # then wait forever, making ``timeout`` ineffective and turning the call
    # into an indefinite hang (the classic symptom of driving an MCP stdio tool
    # from a synchronous ``Agent.run()`` under a running loop). A daemon thread
    # lets us surface a bounded, actionable error and never blocks the caller
    # (or process exit) on a stuck worker.
    box: dict[str, object] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            box["value"] = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001 — propagated to the caller
            box["error"] = exc
        finally:
            done.set()

    outer_loop = asyncio.get_running_loop()
    with _blocked_loops_lock:
        _blocked_loops.add(id(outer_loop))
    try:
        threading.Thread(
            target=_worker, name="effgen-async-bridge", daemon=True
        ).start()

        if not done.wait(timeout):
            raise TimeoutError(
                f"Async work did not complete within {timeout:.0f}s while bridging "
                "from synchronous code under a running event loop. This usually "
                "means a loop-bound async resource (e.g. an MCP stdio tool session) "
                "is being driven from a synchronous call such as Agent.run(); the "
                "resource is bound to the calling loop, which is blocked waiting on "
                "this bridge. Use the async entry point instead — e.g. "
                "`await agent.run_async(...)` — so the tool runs on the calling loop."
            )
    finally:
        with _blocked_loops_lock:
            _blocked_loops.discard(id(outer_loop))
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]
