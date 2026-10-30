"""
Bounded concurrency / volume stress tests (no live API, no GPU).

These exercise the hot in-process paths under concurrency and high call volume
to flush out shared-state races, unbounded growth, and resource leaks. They are
deliberately bounded (seconds, not hours) and marked ``expensive`` + ``slow`` so
CI can deselect them; the unbounded soak/GPU stress lives in the dedicated
driver scripts, not in the test suite.

Covered:
  * concurrent async tool execution (calculator + REPL) — no shared-state
    corruption, every call returns a well-formed ToolResult;
  * REPL worker lifecycle under churn — many sessions created/reset without
    leaking processes, and a high volume of distinct session ids under
    concurrency staying within the live-session limit;
  * high-volume pure-function parsing (finish-reason, JSON extraction, config
    merge) — stable, no slow degradation or memory blowup.
"""

from __future__ import annotations

import asyncio
import gc

import pytest

from effgen.config.loader import Config
from effgen.core.structured_output import extract_json_from_text, validate_json_schema
from effgen.models._adapter_utils import normalize_finish_reason
from effgen.tools.builtin.calculator import Calculator

pytestmark = [pytest.mark.expensive, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Concurrent tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculator_concurrent_volume() -> None:
    """500 concurrent calculator calls all return correct, well-formed results."""
    calc = Calculator()

    async def one(i: int):
        r = await calc.execute(operation="evaluate", expression=f"{i} * {i}")
        return i, r

    results = await asyncio.gather(*(one(i) for i in range(500)))
    assert len(results) == 500
    for i, r in results:
        assert r.success is True
        assert r.output["result"] == i * i


@pytest.mark.asyncio
async def test_repl_session_churn_no_process_leak() -> None:
    """Creating + resetting many REPL sessions does not leak worker processes."""
    import os

    from effgen.tools.builtin.python_repl import PythonREPL

    def _child_count() -> int:
        # Count direct child PIDs of this process via /proc (Linux).
        try:
            me = os.getpid()
            n = 0
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/stat") as fh:
                        # The comm field can hold spaces/parens; fields follow ") ".
                        ppid = int(fh.read().rsplit(") ", 1)[-1].split()[1])
                except (OSError, IndexError, ValueError):
                    continue
                if ppid == me:
                    n += 1
            return n
        except OSError:
            return -1

    repl = PythonREPL()
    try:
        baseline = _child_count()
        for i in range(25):
            r = await repl.execute(code=f"v = {i} * 2", session_id=f"s{i % 5}", timeout=10)
            assert r.success is True
            if i % 5 == 0:
                repl.reset_session(f"s{i % 5}")
        # At most one live worker per the 5 distinct sessions, plus slack.
        await repl.cleanup()
        gc.collect()
        await asyncio.sleep(0.5)
        after = _child_count()
        if baseline >= 0 and after >= 0:
            assert after <= baseline + 1, f"worker leak: {baseline} -> {after}"
    finally:
        await repl.cleanup()


@pytest.mark.asyncio
async def test_repl_high_session_volume_stays_within_limit() -> None:
    """400 concurrent calls over 100 session ids stay within the session limit.

    ``session_id`` is a model-supplied parameter, so a run that varies it must
    not turn into one live interpreter per session.
    """
    import os

    from effgen.tools.builtin.python_repl import PythonREPL

    def _children() -> set[int]:
        me = os.getpid()
        pids: set[int] = set()
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/stat") as fh:
                    ppid = int(fh.read().rsplit(") ", 1)[-1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            if ppid == me:
                pids.add(int(pid))
        return pids

    limit = 6
    in_flight = 32
    repl = PythonREPL(max_sessions=limit)
    baseline = _children()
    gate = asyncio.Semaphore(in_flight)
    peak = 0
    try:

        async def one(i: int):
            nonlocal peak
            async with gate:
                result = await repl.execute(
                    code=f"n = {i}\nn + 1", session_id=f"v{i % 100}", timeout=60
                )
            peak = max(peak, len(_children() - baseline))
            return i, result

        results = await asyncio.gather(*(one(i) for i in range(400)))
        for i, result in results:
            assert result.success is True, result.error
            assert result.output["result"] == i + 1
        # A session running a call is never stopped, so a burst can exceed the
        # limit — but only by the number of calls in flight, never by the number
        # of distinct session ids.
        assert peak <= limit + in_flight, f"{peak} workers alive for 100 session ids"

        await asyncio.sleep(0.5)
        settled = len(_children() - baseline)
        assert settled <= limit, f"{settled} workers still alive, limit is {limit}"
    finally:
        await repl.cleanup()
        await asyncio.sleep(0.3)
    assert _children() - baseline == set()


# ---------------------------------------------------------------------------
# High-volume pure-function parsing
# ---------------------------------------------------------------------------


def test_finish_reason_high_volume_stable() -> None:
    """50k finish-reason normalizations never raise and always return a str."""
    samples = [
        "stop", "STOP", "length", "max_tokens", "end_turn", "tool_use",
        "tool_calls", "FinishReason.STOP", "FinishReason.MAX_TOKENS",
        None, 1, 2, 9, ".", "weird-thing", "", "safety", "recitation",
    ]
    for i in range(50_000):
        out = normalize_finish_reason(samples[i % len(samples)])
        assert isinstance(out, str) and out


def test_json_extraction_high_volume_stable() -> None:
    """10k JSON extractions over messy text never crash."""
    blobs = [
        'prose {"a": 1} more',
        "```json\n[{\"x\": 2}]\n```",
        "no json here",
        "[1, 2, 3]",
        '{"nested": {"deep": [1, {"k": "v"}]}}',
        "{",  # unbalanced
        "prose [{}] more",  # array of objects
    ]
    for i in range(10_000):
        out = extract_json_from_text(blobs[i % len(blobs)])
        assert out is None or isinstance(out, str)


def test_config_merge_high_volume_no_growth() -> None:
    """Repeated deep-merges into one Config converge (no unbounded key growth)."""
    cfg = Config(data={})
    for i in range(5_000):
        cfg.update({"models": {"m": {"ctx": i, "name": f"m{i % 10}"}}, "n": i})
    d = cfg.to_dict()
    # Keys are overwritten, not accumulated — bounded shape.
    assert set(d.keys()) == {"models", "n"}
    assert set(d["models"]["m"].keys()) == {"ctx", "name"}


def test_short_term_memory_bounded_no_growth() -> None:
    """Short-term memory enforces its cap under heavy append load (no unbounded growth)."""
    from effgen.memory.short_term import ShortTermMemory

    stm = ShortTermMemory(max_messages=50)
    for i in range(2_000):
        stm.add_user_message(f"message number {i} with some padding " * 4)
    msgs = stm.get_messages()
    assert len(msgs) <= 50, f"memory not bounded: {len(msgs)} messages"
    # Token accounting stays finite/positive, not runaway.
    tokens = stm.get_token_count()
    assert 0 < tokens < 1_000_000


def test_rag_chunking_high_volume_stable() -> None:
    """Chunking large documents repeatedly is stable, bounded, and never crashes."""
    from effgen.rag.chunking import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=200, overlap=20)
    big = "The quick brown fox jumps over the lazy dog. " * 1_000
    for i in range(200):
        chunks = chunker.chunk(big, doc_id=f"doc{i}")
        assert chunks
        # Chunk sizes respect the configured bound (plus separator slack).
        assert all(len(c.content) <= 400 for c in chunks)


def test_validate_high_volume_stable() -> None:
    """5k schema validations return (bool, str|None), never raise."""
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    for i in range(5_000):
        data = {"name": f"u{i}", "age": i} if i % 2 else {"age": i}
        valid, err = validate_json_schema(data, schema)
        assert isinstance(valid, bool)
        assert err is None or isinstance(err, str)
