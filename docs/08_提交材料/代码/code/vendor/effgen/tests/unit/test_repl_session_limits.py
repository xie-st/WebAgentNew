"""Resource bounds on the Python REPL's session worker subprocesses.

Uses real worker subprocesses (no mocks of the behaviour under test) to cover:

* the live-session cap — many distinct ``session_id`` values keep at most
  ``max_sessions`` workers alive, evicting the least recently used one;
* eviction order — a session that is still being used survives, an idle older
  one is stopped;
* the cap under concurrency — every call still returns a correct result;
* worker lifetime — a session's variables survive the thread that made the
  call, and workers are started from the shared spawner thread;
* dropping the tool without awaiting ``cleanup()`` does not leave workers
  running.
"""

from __future__ import annotations

import asyncio
import gc
import os
import subprocess
import threading
import time

import pytest

from effgen.tools.builtin.python_repl import PythonREPL


def _child_pids() -> set[int]:
    """PIDs of this process's direct children (Linux ``/proc``)."""
    me = os.getpid()
    pids: set[int] = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                # The comm field can contain spaces/parens; fields follow ") ".
                ppid = int(fh.read().rsplit(") ", 1)[-1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == me:
            pids.add(int(entry))
    return pids


pytestmark = pytest.mark.skipif(
    not os.path.isdir("/proc"), reason="child-process accounting needs /proc"
)


class TestReplSessionLimit:
    async def test_distinct_sessions_stay_bounded(self):
        """Many distinct session ids keep at most ``max_sessions`` workers."""
        repl = PythonREPL(max_sessions=3)
        baseline = _child_pids()
        try:
            for i in range(12):
                result = await repl.execute(code=f"v = {i}", session_id=f"s{i}", timeout=15)
                assert result.success is True, result.error
                live = _child_pids() - baseline
                assert len(live) <= 3, f"{len(live)} workers alive, limit is 3"
        finally:
            await repl.cleanup()
        await asyncio.sleep(0.2)
        assert _child_pids() - baseline == set()

    async def test_evicted_session_starts_empty_and_kept_session_persists(self):
        """The oldest session loses its variables; a reused one keeps them."""
        repl = PythonREPL(max_sessions=2)
        try:
            assert (await repl.execute(code="a = 1", session_id="old", timeout=15)).success
            assert (await repl.execute(code="b = 2", session_id="keep", timeout=15)).success
            # Touch "keep" so "old" is the least recently used, then push past
            # the limit with a third session.
            assert (await repl.execute(code="b", session_id="keep", timeout=15)).success
            assert (await repl.execute(code="c = 3", session_id="new", timeout=15)).success

            kept = await repl.execute(code="b", session_id="keep", timeout=15)
            assert kept.success is True
            assert kept.output["result"] == 2

            evicted = await repl.execute(code="a", session_id="old", timeout=15)
            assert evicted.success is False
            assert "a" in (evicted.error or "")
        finally:
            await repl.cleanup()

    async def test_limit_holds_under_concurrency(self):
        """Concurrent calls over more sessions than the limit all succeed.

        A session running a call is never stopped, so the count can exceed the
        limit while the burst is in flight — but only up to the number of
        distinct sessions, never one worker per call, and the pool is trimmed
        back to the limit as the calls finish.
        """
        repl = PythonREPL(max_sessions=4)
        baseline = _child_pids()
        peak = 0
        try:

            async def one(i: int):
                nonlocal peak
                result = await repl.execute(
                    code=f"x = {i}\nx * 3", session_id=f"c{i % 10}", timeout=30
                )
                peak = max(peak, len(_child_pids() - baseline))
                return i, result

            results = await asyncio.gather(*(one(i) for i in range(40)))
            for i, result in results:
                assert result.success is True, result.error
                assert result.output["result"] == i * 3
            assert peak <= 10, f"{peak} workers alive for 10 sessions"

            await asyncio.sleep(0.3)
            settled = len(_child_pids() - baseline)
            assert settled <= 4, f"{settled} workers still alive, limit is 4"
        finally:
            await repl.cleanup()

    def test_limit_is_at_least_one(self):
        """A zero/negative limit is raised to one rather than blocking use."""
        assert PythonREPL(max_sessions=0)._max_sessions == 1
        assert PythonREPL(max_sessions=-5)._max_sessions == 1

    def test_limit_read_from_environment(self, monkeypatch):
        """``EFFGEN_REPL_MAX_SESSIONS`` sets the default; junk falls back."""
        monkeypatch.setenv("EFFGEN_REPL_MAX_SESSIONS", "3")
        assert PythonREPL()._max_sessions == 3
        monkeypatch.setenv("EFFGEN_REPL_MAX_SESSIONS", "not-a-number")
        assert PythonREPL()._max_sessions == PythonREPL.DEFAULT_MAX_SESSIONS
        # An explicit argument still wins over the environment.
        monkeypatch.setenv("EFFGEN_REPL_MAX_SESSIONS", "3")
        assert PythonREPL(max_sessions=6)._max_sessions == 6


class TestReplWorkerLifetime:
    def test_session_state_survives_the_calling_thread(self):
        """Variables persist when successive calls run on different threads.

        The worker asks the kernel to kill it if its parent dies, and Linux
        delivers that when the creating *thread* exits. A worker started on an
        event loop's default executor would be killed as soon as that thread was
        recycled, and the session's variables would vanish with it.
        """
        repl = PythonREPL()
        try:
            first = asyncio.run(repl.execute(code="kept = 41", session_id="s", timeout=15))
            assert first.success is True, first.error
            # A separate asyncio.run means a separate executor thread pool.
            second = asyncio.run(repl.execute(code="kept + 1", session_id="s", timeout=15))
            assert second.success is True, second.error
            assert second.output["result"] == 42
        finally:
            asyncio.run(repl.cleanup())

    def test_worker_is_not_started_on_the_calling_thread(self, monkeypatch):
        """Workers come from the shared spawner thread, not the caller's."""
        from effgen.tools.builtin import python_repl as repl_mod

        seen: list[str] = []
        real_popen = subprocess.Popen

        def recording_popen(*args, **kwargs):
            seen.append(threading.current_thread().name)
            return real_popen(*args, **kwargs)

        # Restored by monkeypatch even if the assertions below fail, so no other
        # test ever sees the patched stdlib.
        monkeypatch.setattr(repl_mod.subprocess, "Popen", recording_popen)

        repl = PythonREPL()
        try:
            result = asyncio.run(repl.execute(code="1 + 1", session_id="s", timeout=15))
            assert result.success is True, result.error
        finally:
            asyncio.run(repl.cleanup())
        assert seen == ["effgen-repl-spawner"], seen


class TestReplDroppedWithoutCleanup:
    async def test_dropping_the_tool_stops_its_workers(self):
        """A REPL that is garbage-collected leaves no worker running."""
        baseline = _child_pids()
        repl = PythonREPL()
        assert (await repl.execute(code="y = 1", session_id="default", timeout=15)).success
        assert _child_pids() - baseline, "expected a live worker while in use"

        del repl
        gc.collect()
        await asyncio.sleep(0.3)
        assert _child_pids() - baseline == set(), "worker outlived the dropped tool"

    async def test_cleanup_is_still_safe_after_the_finalizer_ran(self):
        """Calling cleanup() on a REPL whose workers are gone does not raise."""
        repl = PythonREPL()
        assert (await repl.execute(code="1 + 1", session_id="default", timeout=15)).success
        repl._finalizer()
        await repl.cleanup()
        # Usable again afterwards: a fresh worker is started on demand.
        result = await repl.execute(code="2 + 2", session_id="default", timeout=15)
        assert result.success is True
        assert result.output["result"] == 4
        await repl.cleanup()


class TestPerCallTimeoutIsDeclared:
    """``timeout`` is a real per-call parameter, so it is declared as one.

    The tool honours a per-call ``timeout``, but it was missing from the
    declared parameter list — so every caller that passed one was told its
    parameter was being ignored (it was not), and a model reading the schema
    could not set one at all.
    """

    def test_timeout_is_in_the_declared_parameters(self):
        names = [p.name for p in PythonREPL().metadata.parameters]
        assert "timeout" in names, f"declared parameters: {names}"

    def test_timeout_is_not_offered_to_the_model(self):
        """The value raises the cap as well as lowers it, so the model never
        sees it: a model that could set its own would be able to lift the
        limit that stops a runaway program."""
        meta = PythonREPL().metadata
        model_names = [p.name for p in meta.model_facing_parameters]
        assert "timeout" not in model_names, model_names
        assert "timeout" not in meta.to_json_schema()["parameters"]["properties"]

    def test_passing_a_timeout_logs_no_unknown_parameter_warning(self, caplog):
        repl = PythonREPL()
        with caplog.at_level("WARNING"):
            valid, error = repl.validate_parameters(
                code="1 + 1", session_id="default", timeout=15
            )
        assert valid is True, error
        unknown = [r for r in caplog.records if "unknown parameter" in r.getMessage()]
        assert not unknown, [r.getMessage() for r in unknown]

    async def test_a_per_call_timeout_still_bounds_the_run(self):
        """The declared parameter is the one the implementation acts on."""
        repl = PythonREPL()
        try:
            started = time.monotonic()
            result = await repl.execute(
                code="n = 0\nwhile True:\n    n += 1", session_id="spin", timeout=3
            )
            elapsed = time.monotonic() - started
            assert result.success is False
            assert "timed out" in (result.error or "").lower(), result.error
            assert elapsed < 15, f"per-call timeout ignored: ran {elapsed:.1f}s"
        finally:
            await repl.cleanup()

    async def test_a_fractional_timeout_is_accepted(self):
        """``execute`` takes ``float | int``, so the declaration must too.

        Declaring a narrower type turns a call that used to work into a
        parameter-validation failure.
        """
        repl = PythonREPL()
        try:
            valid, error = repl.validate_parameters(code="1 + 1", timeout=2.5)
            assert valid is True, error
            started = time.monotonic()
            result = await repl.execute(
                code="n = 0\nwhile True:\n    n += 1", session_id="frac", timeout=2.5
            )
            elapsed = time.monotonic() - started
            assert result.success is False
            assert "timed out" in (result.error or "").lower(), result.error
            assert elapsed < 15, f"fractional timeout ignored: ran {elapsed:.1f}s"
        finally:
            await repl.cleanup()

    def test_the_declared_timeout_matches_the_one_the_tool_was_built_with(self):
        """A tool built with a different cap advertises that cap, not the default."""
        repl = PythonREPL(timeout=7)
        assert repl.metadata.timeout_seconds == 7
        declared = next(p for p in repl.metadata.parameters if p.name == "timeout")
        assert "7s" in declared.description, declared.description
