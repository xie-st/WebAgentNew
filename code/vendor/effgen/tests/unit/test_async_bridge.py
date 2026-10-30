"""Unit tests for ``effgen.utils.run_coroutine_sync``.

The bridge must drive a coroutine to completion whether or not an event loop is
already running in the calling thread — and must never silently skip the work.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from effgen.utils import run_coroutine_sync
from effgen.utils.async_bridge import is_loop_blocked
from effgen.utils.async_bridge import run_coroutine_sync as direct_import


def test_exported_from_utils_package():
    """The helper is importable from the package facade and the submodule."""
    assert run_coroutine_sync is direct_import


def test_runs_with_no_running_loop():
    async def work():
        await asyncio.sleep(0)
        return 41 + 1

    assert run_coroutine_sync(work()) == 42


def test_runs_from_inside_a_running_loop():
    """When a loop is already running in the thread, the coroutine still runs
    (on a worker thread) and the result is returned to the caller."""

    async def outer():
        async def inner():
            await asyncio.sleep(0.01)
            return "done"

        # Calling run_coroutine_sync from inside a running loop must not raise
        # "asyncio.run() cannot be called from a running event loop" and must
        # actually execute inner().
        return run_coroutine_sync(inner())

    assert asyncio.run(outer()) == "done"


def test_actually_executes_under_running_loop_not_skipped():
    """Regression: the work is performed, not no-op'd, under a loop."""
    ran = threading.Event()

    async def side_effect():
        await asyncio.sleep(0)
        ran.set()
        return True

    async def outer():
        return run_coroutine_sync(side_effect())

    result = asyncio.run(outer())
    assert result is True
    assert ran.is_set(), "coroutine body must have executed"


def test_propagates_exceptions_no_loop():
    async def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_coroutine_sync(boom())


def test_propagates_runtimeerror_from_coroutine_no_loop():
    """A RuntimeError raised *inside* the coroutine must surface as-is and not
    be mistaken for "loop already running" (which would try to re-run an
    already-consumed coroutine)."""

    async def boom():
        raise RuntimeError("inner runtime error")

    with pytest.raises(RuntimeError, match="inner runtime error"):
        run_coroutine_sync(boom())


def test_propagates_exceptions_under_running_loop():
    async def boom():
        raise ValueError("nested kaboom")

    async def outer():
        return run_coroutine_sync(boom())

    with pytest.raises(ValueError, match="nested kaboom"):
        asyncio.run(outer())


def test_blocks_until_complete_under_running_loop():
    """The caller blocks for the duration of the bridged coroutine."""

    async def slow():
        await asyncio.sleep(0.2)
        return "ok"

    async def outer():
        t0 = time.monotonic()
        res = run_coroutine_sync(slow())
        return res, time.monotonic() - t0

    res, elapsed = asyncio.run(outer())
    assert res == "ok"
    assert elapsed >= 0.18, f"should have blocked ~0.2s, got {elapsed:.3f}s"


def test_loop_bound_coroutine_times_out_instead_of_hanging():
    """A coroutine awaiting a primitive bound to the *calling* loop can never
    complete on the worker loop — historically this deadlocked forever (the
    worker-thread join hung even after the result timeout). It must now raise a
    bounded, actionable ``TimeoutError`` instead, and the caller must regain
    control promptly (this is exactly the MCP-stdio-tool-in-``Agent.run()``
    trap)."""

    async def outer():
        # An Event bound to THIS loop; the worker loop can never see it set.
        ev = asyncio.Event()

        async def loop_bound():
            await ev.wait()
            return "never"

        t0 = time.monotonic()
        try:
            run_coroutine_sync(loop_bound(), timeout=1.0)
        except TimeoutError as exc:
            return time.monotonic() - t0, str(exc)
        raise AssertionError("expected a TimeoutError, none raised")

    elapsed, msg = asyncio.run(outer())
    # Bounded by ~timeout, not an indefinite hang.
    assert 1.0 <= elapsed < 5.0, f"should surface ~1s, got {elapsed:.2f}s"
    # The message must teach the async-native escape hatch.
    assert "run_async" in msg


def test_timeout_worker_is_daemon_so_it_never_blocks_exit():
    """The leaked worker on the timeout path is a daemon thread, so it never
    keeps the interpreter (or a clean shutdown) hanging on a deadlocked loop."""

    async def outer():
        ev = asyncio.Event()

        async def loop_bound():
            await ev.wait()

        with pytest.raises(TimeoutError):
            run_coroutine_sync(loop_bound(), timeout=0.3)

    asyncio.run(outer())
    lingering = [
        th for th in threading.enumerate()
        if th.name == "effgen-async-bridge" and not th.daemon
    ]
    assert not lingering, "bridge worker must be a daemon thread"


def test_is_loop_blocked_true_only_during_the_bridge_window():
    """A loop-bound async resource (an MCP session is the known case) can ask
    is_loop_blocked() to detect a guaranteed deadlock immediately instead of
    waiting out a timeout."""

    async def outer():
        this_loop = asyncio.get_running_loop()
        assert is_loop_blocked(this_loop) is False

        seen_blocked = {}

        async def probe():
            # Runs on the worker thread while `this_loop` is stuck in
            # done.wait() back on the outer thread.
            seen_blocked["value"] = is_loop_blocked(this_loop)
            return "ok"

        assert run_coroutine_sync(probe()) == "ok"
        assert seen_blocked["value"] is True
        # Cleared once the bridge returns.
        assert is_loop_blocked(this_loop) is False

    asyncio.run(outer())


def test_is_loop_blocked_false_for_none_and_unrelated_loop():
    assert is_loop_blocked(None) is False

    unrelated_loop = asyncio.new_event_loop()
    try:
        assert is_loop_blocked(unrelated_loop) is False
    finally:
        unrelated_loop.close()
