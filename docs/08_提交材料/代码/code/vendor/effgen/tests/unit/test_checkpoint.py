"""Unit tests for effgen.core.checkpoint."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from effgen.core.checkpoint import Checkpoint, CheckpointManager


def _make_cp(agent_name: str = "agent", iteration: int = 1) -> Checkpoint:
    return Checkpoint(
        checkpoint_id="",
        agent_name=agent_name,
        task="solve x",
        iteration=iteration,
        scratchpad="thinking…",
        partial_output="42",
        tool_calls=2,
        tokens_used=100,
        memory={"short_term": {"messages": [{"role": "user", "content": "hi"}]}},
        tool_states={"calc": {"name": "calc", "class": "Calculator", "module": "x"}},
        metadata={"source": "unit-test"},
    )


class TestCheckpointDataclass:
    def test_to_dict_roundtrip(self):
        cp = _make_cp()
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.task == cp.task
        assert cp2.tool_calls == cp.tool_calls
        assert cp2.memory == cp.memory


class TestFilesystemBackend:
    def test_unsupported_backend_raises(self, tmp_path):
        with pytest.raises(ValueError):
            CheckpointManager(checkpoint_dir=str(tmp_path), backend="redis")

    def test_save_assigns_id(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        cp = _make_cp()
        cp_id = mgr.save(cp)
        assert cp_id
        assert (tmp_path / f"{cp_id}.json").exists()
        assert (tmp_path / "latest.json").exists()

    def test_save_then_load_by_id(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        cp_id = mgr.save(_make_cp())
        loaded = mgr.load(cp_id)
        assert loaded.task == "solve x"
        assert loaded.tool_calls == 2

    def test_load_by_path(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        cp_id = mgr.save(_make_cp())
        path = tmp_path / f"{cp_id}.json"
        loaded = mgr.load(str(path))
        assert loaded.task == "solve x"

    def test_load_missing_id_raises(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            mgr.load("does-not-exist")

    def test_load_latest_uses_pointer_file(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        mgr.save(_make_cp(agent_name="a1", iteration=1))
        mgr.save(_make_cp(agent_name="a2", iteration=99))
        latest = mgr.load_latest()
        assert latest.iteration == 99

    def test_load_latest_with_no_checkpoints_raises(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            mgr.load_latest()

    def test_list_checkpoints_filters_latest_pointer(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        mgr.save(_make_cp(agent_name="a", iteration=1))
        mgr.save(_make_cp(agent_name="a", iteration=2))
        listed = mgr.list_checkpoints()
        assert len(listed) == 2
        assert all(c["agent_name"] == "a" for c in listed)
        # latest.json must not appear
        assert not any(c["checkpoint_id"] == "latest" for c in listed)

    def test_delete_removes_file(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        cp_id = mgr.save(_make_cp())
        assert mgr.delete(cp_id) is True
        assert mgr.delete(cp_id) is False  # already gone

    def test_list_checkpoints_skips_invalid_json(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        mgr.save(_make_cp())
        # Write a junk file alongside
        (tmp_path / "junk.json").write_text("{not valid json")
        listed = mgr.list_checkpoints()
        # The one valid checkpoint remains
        assert len(listed) == 1


class TestSqliteBackend:
    def test_save_and_load_by_id(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), backend="sqlite")
        cp_id = mgr.save(_make_cp())
        loaded = mgr.load(cp_id)
        assert loaded.task == "solve x"

    def test_load_missing_raises(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), backend="sqlite")
        with pytest.raises(FileNotFoundError):
            mgr.load("nope")

    def test_load_latest_returns_most_recent(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), backend="sqlite")
        mgr.save(_make_cp(iteration=1))
        # Distinct created_at by pre-setting field via Checkpoint.from_dict
        from datetime import datetime, timedelta
        cp = _make_cp(iteration=42)
        cp.created_at = (datetime.now() + timedelta(hours=1)).isoformat()
        mgr.save(cp)
        latest = mgr.load_latest()
        assert latest.iteration == 42

    def test_load_latest_empty_raises(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), backend="sqlite")
        with pytest.raises(FileNotFoundError):
            mgr.load_latest()

    def test_list_and_delete(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), backend="sqlite")
        cp_id1 = mgr.save(_make_cp(iteration=1))
        mgr.save(_make_cp(iteration=2))
        assert len(mgr.list_checkpoints()) == 2
        assert mgr.delete(cp_id1) is True
        assert len(mgr.list_checkpoints()) == 1
        assert mgr.delete(cp_id1) is False  # already gone


class TestSnapshotAgent:
    def test_snapshot_collects_tool_metadata(self):
        class _StubTool:
            pass

        agent = SimpleNamespace(name="bot", tools={"calc": _StubTool()})
        cp = CheckpointManager.snapshot_agent(agent, task="t", iteration=3)
        assert cp.agent_name == "bot"
        assert cp.iteration == 3
        assert "calc" in cp.tool_states
        assert cp.tool_states["calc"]["class"] == "_StubTool"

    def test_snapshot_handles_short_term_memory(self):
        class _STM:
            def to_dict(self):
                return {"messages": []}

        agent = SimpleNamespace(name="a", tools={}, short_term_memory=_STM())
        cp = CheckpointManager.snapshot_agent(agent, task="x", iteration=0)
        assert "short_term" in cp.memory
        assert cp.memory["short_term"] == {"messages": []}

    def test_snapshot_swallows_memory_errors(self):
        class _BadSTM:
            def to_dict(self):
                raise RuntimeError("boom")

        agent = SimpleNamespace(name="a", tools={}, short_term_memory=_BadSTM())
        cp = CheckpointManager.snapshot_agent(agent, task="x", iteration=0)
        # Should not raise; memory dict simply lacks short_term
        assert "short_term" not in cp.memory

    def test_new_id_includes_agent_name(self):
        cp_id = CheckpointManager._new_id("solver")
        assert cp_id.startswith("solver-")
        assert len(cp_id.split("-")[-1]) == 8
