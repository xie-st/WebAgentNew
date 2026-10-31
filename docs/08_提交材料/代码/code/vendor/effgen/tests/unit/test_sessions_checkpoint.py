"""Durability tests for sessions, checkpoints, and background tasks.

Covers the round-trip formats, corruption reporting (a clear typed error that
names the file rather than a raw stack trace), model-aware resume, and the
background runner's thread lifecycle (no leaked threads after GC / shutdown).

No live model calls — these exercise the persistence/runtime plumbing only.
The live multi-turn continuity + resume proofs live in
tests/integration/test_resume_live.py.
"""

from __future__ import annotations

import gc
import threading
import time

import pytest

from effgen.core.background import BackgroundTask, BackgroundTaskRunner, TaskStatus
from effgen.core.checkpoint import Checkpoint, CheckpointManager
from effgen.core.session import Session, SessionManager, _default_session_dir
from effgen.errors import CorruptStateError

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- sessions
def test_session_round_trip(tmp_path):
    s = Session(session_id="abc", agent_name="tester")
    s.add_user_message("hello")
    s.add_assistant_message("hi there")
    path = s.save(str(tmp_path))
    assert path.endswith("abc.json")

    loaded = Session.load("abc", str(tmp_path))
    assert loaded.session_id == "abc"
    assert [m["role"] for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[0]["content"] == "hello"


def test_session_manager_list_delete_export(tmp_path):
    mgr = SessionManager(str(tmp_path))
    s = Session(session_id="s1", agent_name="a")
    s.add_user_message("q")
    s.add_assistant_message("a")
    s.save(str(tmp_path))

    listing = mgr.list_sessions()
    assert len(listing) == 1
    assert listing[0]["session_id"] == "s1"
    assert listing[0]["messages"] == 2

    text = mgr.export("s1", format="text")
    assert "[user] q" in text and "[assistant] a" in text
    js = mgr.export("s1", format="json")
    assert '"session_id": "s1"' in js

    assert mgr.delete("s1") is True
    assert mgr.delete("s1") is False
    assert mgr.list_sessions() == []


def test_session_export_missing_raises_filenotfound(tmp_path):
    mgr = SessionManager(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        mgr.export("nope")


def test_corrupt_session_raises_typed_error_naming_file(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text('{"session_id": "broken", "messages": [')  # truncated
    with pytest.raises(CorruptStateError) as ei:
        Session.load("broken", str(tmp_path))
    assert "broken.json" in str(ei.value)
    assert ei.value.kind == "session"


def test_corrupt_session_skipped_in_listing(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    good = Session(session_id="ok", agent_name="a")
    good.save(str(tmp_path))
    listing = SessionManager(str(tmp_path)).list_sessions()
    assert [e["session_id"] for e in listing] == ["ok"]


def test_scan_names_unreadable_session_files(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    Session(session_id="ok", agent_name="a").save(str(tmp_path))

    sessions, unreadable = SessionManager(str(tmp_path)).scan()

    assert [e["session_id"] for e in sessions] == ["ok"]
    assert [u["file"] for u in unreadable] == ["broken.json"]
    assert unreadable[0]["reason"]


def test_listing_reports_per_session_model_and_cost(tmp_path):
    s = Session(session_id="priced", agent_name="a")
    s.add_message("user", "q", model="gpt-5-nano", run_id="r1", cost_usd=0.002)
    s.add_message("assistant", "a", model="gpt-5-nano", run_id="r1", cost_usd=0.002)
    s.add_message("assistant", "a2", model="gpt-5-nano", run_id="r2", cost_usd=0.003)
    s.metadata["model"] = "gpt-5-nano"
    s.save(str(tmp_path))

    entry = SessionManager(str(tmp_path)).list_sessions()[0]

    assert entry["model"] == "gpt-5-nano"
    # One cost per turn: the pair sharing run r1 counts once.
    assert entry["cost_usd"] == pytest.approx(0.005)


def test_listing_cost_is_none_when_no_turn_was_priced(tmp_path):
    s = Session(session_id="unpriced", agent_name="a")
    s.add_message("user", "q")
    s.add_message("assistant", "a")
    s.save(str(tmp_path))

    assert SessionManager(str(tmp_path)).list_sessions()[0]["cost_usd"] is None


def test_session_atomic_save_no_partial_file(tmp_path):
    s = Session(session_id="atomic")
    s.save(str(tmp_path))
    # No leftover .tmp file after a successful save.
    assert not list(tmp_path.glob("*.tmp"))


def test_session_dir_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom_sessions"
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(target))
    assert _default_session_dir() == str(target)
    mgr = SessionManager()
    assert mgr.sessions_dir == str(target)


def test_session_home_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("EFFGEN_SESSIONS_DIR", raising=False)
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    assert _default_session_dir().endswith("sessions")
    assert str(tmp_path / "home") in _default_session_dir()


def test_session_cleanup_retention(tmp_path):
    mgr = SessionManager(str(tmp_path))
    old = Session(session_id="old")
    old.save(str(tmp_path))
    # Backdate updated_at far into the past.
    old.updated_at = "2000-01-01T00:00:00"
    import json
    (tmp_path / "old.json").write_text(json.dumps(old.to_dict()))

    fresh = Session(session_id="fresh")
    fresh.save(str(tmp_path))

    removed = mgr.cleanup(older_than_days=30)
    assert removed == 1
    assert [e["session_id"] for e in mgr.list_sessions()] == ["fresh"]


# --------------------------------------------------------------------------- checkpoints
def test_checkpoint_round_trip_records_model(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    cp = Checkpoint(
        checkpoint_id="cp1", agent_name="a", task="do x",
        iteration=2, model="Qwen/Qwen2.5-1.5B-Instruct", scratchpad="step1",
    )
    cid = mgr.save(cp)
    loaded = mgr.load(cid)
    assert loaded.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert loaded.scratchpad == "step1"
    assert mgr.load_latest().checkpoint_id == "cp1"


def test_checkpoint_missing_raises_filenotfound(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        mgr.load("does-not-exist")


def test_corrupt_checkpoint_raises_typed_error_naming_file(tmp_path):
    bad = tmp_path / "cp.json"
    bad.write_text('{"checkpoint_id": "cp", "iteration":')  # truncated
    mgr = CheckpointManager(str(tmp_path))
    with pytest.raises(CorruptStateError) as ei:
        mgr.load("cp")
    assert "cp.json" in str(ei.value)
    assert ei.value.kind == "checkpoint"


def test_corrupt_checkpoint_by_path_raises_typed(tmp_path):
    bad = tmp_path / "explicit.json"
    bad.write_text("garbage{")
    mgr = CheckpointManager(str(tmp_path))
    with pytest.raises(CorruptStateError):
        mgr.load(str(bad))


def test_checkpoint_sqlite_round_trip(tmp_path):
    mgr = CheckpointManager(str(tmp_path), backend="sqlite")
    cp = Checkpoint(checkpoint_id="s1", agent_name="a", task="t", iteration=1, model="m")
    mgr.save(cp)
    assert mgr.load("s1").model == "m"
    assert mgr.load_latest().checkpoint_id == "s1"
    assert any(c["checkpoint_id"] == "s1" for c in mgr.list_checkpoints())
    assert mgr.delete("s1") is True


def test_old_checkpoint_without_model_still_loads(tmp_path):
    # Simulate a 0.2.x checkpoint written before the `model` field existed.
    import json
    payload = {
        "checkpoint_id": "legacy", "agent_name": "a", "task": "t",
        "iteration": 1, "scratchpad": "", "tool_calls": 0, "tokens_used": 0,
        "memory": {}, "tool_states": {}, "metadata": {},
        "created_at": "2025-01-01T00:00:00",
    }
    (tmp_path / "legacy.json").write_text(json.dumps(payload))
    cp = CheckpointManager(str(tmp_path)).load("legacy")
    assert cp.model == ""  # default, no crash


# --------------------------------------------------------------------------- background
class _FakeAgent:
    name = "fake"

    def __init__(self, fail=False, sleep=0.02):
        self.fail = fail
        self.sleep = sleep

    def run(self, task, **kw):
        time.sleep(self.sleep)
        if self.fail:
            # An OpenAI-shaped key that the redactor must scrub from the error.
            raise RuntimeError("boom sk-abcdefABCDEF0123456789abcdefABCDEF0123 leaked?")
        return type("R", (), {"output": f"done:{task}", "success": True})()


def test_background_basic_lifecycle():
    runner = BackgroundTaskRunner(_FakeAgent(), max_workers=2)
    try:
        tid = runner.submit("hello")
        res = runner.get_result(tid, wait=True, timeout=5)
        assert res.output == "done:hello"
        assert runner.get_status(tid) == TaskStatus.COMPLETED
    finally:
        runner.shutdown(wait=True)


def test_background_unknown_id_clear_error():
    """An unknown task id raises a clear, actionable message (not a bare
    KeyError('<uuid>') a caller can't interpret)."""
    runner = BackgroundTaskRunner(_FakeAgent(), max_workers=1)
    try:
        for accessor in (runner.get_status, runner.get_task, runner.get_result):
            with pytest.raises(KeyError) as ei:
                accessor("no-such-id")
            msg = str(ei.value)
            assert "no-such-id" in msg
            assert "list_tasks()" in msg
    finally:
        runner.shutdown(wait=True)


def test_background_failed_task_redacted_error():
    runner = BackgroundTaskRunner(_FakeAgent(fail=True), max_workers=1)
    try:
        tid = runner.submit("x")
        runner.get_result(tid, wait=True, timeout=5)
        task = runner.get_task(tid)
        assert task.status == TaskStatus.FAILED
        assert task.error and "RuntimeError" in task.error
        # The secret-looking token must be redacted out of the surfaced error.
        assert "sk-abcdefABCDEF0123456789abcdefABCDEF0123" not in task.error
        assert "REDACTED" in task.error
    finally:
        runner.shutdown(wait=True)


def test_background_unsuccessful_result_marked_failed():
    class UnsuccessfulAgent:
        name = "u"

        def run(self, task, **kw):
            return type("R", (), {
                "output": "could not finish",
                "success": False,
                "metadata": {"error": "ModelNotFoundError: bad-model"},
            })()

    runner = BackgroundTaskRunner(UnsuccessfulAgent(), max_workers=1)
    try:
        tid = runner.submit("x")
        runner.get_result(tid, wait=True, timeout=5)
        task = runner.get_task(tid)
        assert task.status == TaskStatus.FAILED
        assert "ModelNotFoundError" in (task.error or "")
    finally:
        runner.shutdown(wait=True)


def test_background_concurrent_tasks():
    runner = BackgroundTaskRunner(_FakeAgent(sleep=0.05), max_workers=4)
    try:
        ids = [runner.submit(f"t{i}") for i in range(12)]
        results = {i: runner.get_result(i, wait=True, timeout=10) for i in ids}
        assert all(r.success for r in results.values())
        assert {runner.get_status(i) for i in ids} == {TaskStatus.COMPLETED}
    finally:
        runner.shutdown(wait=True)


def test_background_cancel_pending():
    # One worker, first task occupies it; the second stays PENDING and is cancelled.
    runner = BackgroundTaskRunner(_FakeAgent(sleep=0.3), max_workers=1)
    try:
        t1 = runner.submit("first")
        t2 = runner.submit("second", priority=9)
        assert runner.cancel(t2) is True
        runner.get_result(t1, wait=True, timeout=5)
        assert runner.get_status(t2) == TaskStatus.CANCELLED
    finally:
        runner.shutdown(wait=True)


def test_background_no_thread_leak_after_gc():
    before = threading.active_count()
    runner = BackgroundTaskRunner(_FakeAgent(), max_workers=3)
    tid = runner.submit("x")
    runner.get_result(tid, wait=True, timeout=5)
    del runner
    gc.collect()
    # Give the worker threads a moment to observe the finalizer's shutdown.
    deadline = time.time() + 3
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.05)
    assert threading.active_count() == before


def test_background_context_manager_cleans_up():
    before = threading.active_count()
    with BackgroundTaskRunner(_FakeAgent(), max_workers=2) as runner:
        tid = runner.submit("x")
        runner.get_result(tid, wait=True, timeout=5)
    deadline = time.time() + 3
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.05)
    assert threading.active_count() == before


def test_background_dataclass_defaults():
    t = BackgroundTask(task_id="t", task_text="x")
    assert t.status == TaskStatus.PENDING
    assert t.progress == 0.0
