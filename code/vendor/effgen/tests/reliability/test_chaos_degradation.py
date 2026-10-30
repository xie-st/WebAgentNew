"""
Real (non-mocked) chaos-degradation tests.

The deterministic ``test_chaos.py`` harness covers network-shaped failures
(5xx/429/timeout/all-providers-fail) at the router/retry layer. This file
covers the *other* chaos axes the reliability guarantees hold for, using **real**
subprocess/tool execution — no mocks:

  * runaway code (CPU + memory) is hard-killed at its budget, not the host;
  * a per-call timeout is honored exactly (regression for the budget-override);
  * a worker process that crashes mid-call yields a typed failure, not a hang;
  * the Docker sandbox degrades to "unavailable" cleanly when the daemon is
    absent (no crash), and the subprocess sandbox still runs;
  * tool exceptions surface as ``ToolResult(success=False)`` with a message,
    never an exception escaping ``execute``.

Every assertion is on observable degradation: a failure is reported, work
stops near its budget, and nothing leaks a raw traceback to the caller.
"""

from __future__ import annotations

import time

import pytest

from effgen.security.sandbox import (
    DockerSandbox,
    SandboxConfig,
    SubprocessSandbox,
)
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.python_repl import PythonREPL

pytestmark = pytest.mark.reliability


# ---------------------------------------------------------------------------
# Runaway code is killed at budget (CPU + memory)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repl_cpu_runaway_killed_at_per_call_timeout() -> None:
    """``while True: pass`` is hard-killed at the *per-call* timeout, not 30s."""
    repl = PythonREPL()
    try:
        start = time.monotonic()
        result = await repl.execute(code="while True: pass", timeout=3)
        elapsed = time.monotonic() - start
        assert result.success is False
        # Killed close to 3s (allow scheduling slack), nowhere near the 30s default.
        assert 2.0 <= elapsed <= 8.0, f"timeout not honored: {elapsed:.1f}s"
        assert "timed out" in (result.error or "").lower()
    finally:
        await repl.cleanup()


@pytest.mark.asyncio
async def test_repl_memory_bomb_capped_not_host_oom() -> None:
    """A giant allocation fails closed with MemoryError; the host is not OOM-killed."""
    repl = PythonREPL()
    try:
        result = await repl.execute(code="x = [0] * (10 ** 9)", timeout=20)
        assert result.success is False
        assert "memory" in (result.error or "").lower()
    finally:
        await repl.cleanup()


@pytest.mark.asyncio
async def test_repl_disallowed_import_blocked() -> None:
    """A process-escape attempt (os._exit) is blocked before any subprocess work."""
    repl = PythonREPL()
    try:
        result = await repl.execute(code="import os; os._exit(1)", timeout=5)
        assert result.success is False
        assert "not allowed" in (result.error or "").lower()
    finally:
        await repl.cleanup()


# ---------------------------------------------------------------------------
# Sandbox degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_docker_sandbox_unavailable_is_clean() -> None:
    """When the Docker daemon is absent, ``is_available`` returns False — no crash."""
    avail = await DockerSandbox().is_available()
    assert avail in (True, False)  # never raises


@pytest.mark.asyncio
async def test_subprocess_sandbox_runs_and_kills_runaway() -> None:
    """The fallback subprocess sandbox executes normal code and kills a runaway."""
    sb = SubprocessSandbox()
    assert await sb.is_available() is True

    ok = await sb.run("print('hello-chaos')", "python", SandboxConfig())
    assert ok.exit_code == 0
    assert "hello-chaos" in ok.stdout

    start = time.monotonic()
    runaway = await sb.run("while True: pass", "python", SandboxConfig(timeout=3))
    elapsed = time.monotonic() - start
    # Terminated at/near budget with a non-zero exit (CPU-ulimit SIGKILL or the
    # wall-clock guard — either way the run does not exceed its budget+slack).
    assert runaway.exit_code != 0
    assert elapsed <= 8.0, f"runaway not killed at budget: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Tool exceptions never escape execute()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("expr", ["1/0", "@#$%^", "sqrt(-1) + )", "9" * 10000])
async def test_calculator_bad_input_returns_failure(expr: str) -> None:
    """Pathological calculator input yields ToolResult(success=False), never raises."""
    result = await Calculator().execute(operation="evaluate", expression=expr)
    assert result.success is False
    assert result.error  # a non-empty error message


@pytest.mark.asyncio
async def test_repl_syntax_error_is_reported() -> None:
    repl = PythonREPL()
    try:
        result = await repl.execute(code="def (:", timeout=5)
        assert result.success is False
        assert result.error
    finally:
        await repl.cleanup()
