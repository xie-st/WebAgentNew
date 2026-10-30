"""Resuming a workflow run that did not finish.

The point of a checkpoint is that work already paid for is not paid for twice,
so most of these tests assert on how many times an agent was actually called
rather than on the answer it produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from effgen.core.workflow import WorkflowDAG, WorkflowNode
from effgen.core.workflow_checkpoint import (
    FileCheckpointStore,
    InMemoryCheckpointStore,
    WorkflowCheckpoint,
)


@dataclass
class _Response:
    output: str
    success: bool = True
    tokens_used: int = 3
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CountingAgent:
    """Records how many times it ran, which is what resuming has to change."""

    def __init__(self, name: str, *, succeed: bool = True) -> None:
        self.name = name
        self.succeed = succeed
        self.runs = 0

    def run(self, task: str, mode: Any = None, context: Any = None, **kw) -> _Response:
        self.runs += 1
        if not self.succeed:
            return _Response(
                output="failed",
                success=False,
                metadata={"error": {"type": "ModelError", "message": "no"}},
            )
        return _Response(output=f"{self.name}-out")

    async def run_async(self, task: str, mode: Any = None, context: Any = None, **kw):
        return self.run(task, mode=mode, context=context, **kw)

    def close(self) -> None:
        pass


def _chain(*agents: CountingAgent) -> WorkflowDAG:
    """A -> B -> C, one node per agent, in the order given."""
    dag = WorkflowDAG("chain")
    for agent in agents:
        dag.add_node(WorkflowNode(id=agent.name, agent=agent))
    for left, right in zip(agents, agents[1:]):
        dag.connect(left.name, right.name)
    return dag


class TestWithoutACheckpoint:
    """Nothing changes for a run that does not ask for one."""

    def test_a_plain_run_still_works(self):
        a, b = CountingAgent("a"), CountingAgent("b")
        result = _chain(a, b).run("go")
        assert result.success
        assert (a.runs, b.runs) == (1, 1)

    def test_a_run_id_without_a_store_is_refused(self):
        with pytest.raises(ValueError, match="checkpoint store"):
            _chain(CountingAgent("a")).run("go", run_id="x")

    def test_a_store_without_a_run_id_is_refused(self):
        with pytest.raises(ValueError, match="run_id"):
            _chain(CountingAgent("a")).run("go", checkpoint=InMemoryCheckpointStore())

    def test_an_empty_workflow_still_reports_a_bad_argument_pair(self):
        """The empty-workflow result used to be returned before the check ran.

        A caller who mistyped the pair then got a plausible-looking failed
        result instead of being told the call was wrong.
        """
        with pytest.raises(ValueError, match="checkpoint store"):
            WorkflowDAG("empty").run("go", run_id="x")
        with pytest.raises(ValueError, match="run_id"):
            WorkflowDAG("empty").run("go", checkpoint=InMemoryCheckpointStore())


class TestSavingProgress:
    """What ends up in the store."""

    def test_a_finished_run_records_every_node(self):
        store = InMemoryCheckpointStore()
        a, b = CountingAgent("a"), CountingAgent("b")
        _chain(a, b).run("go", checkpoint=store, run_id="r1")

        saved = store.load("r1")
        assert saved is not None
        assert set(saved.completed) == {"a", "b"}
        assert saved.metadata["complete"] is True

    def test_a_failed_run_records_what_did_finish(self):
        store = InMemoryCheckpointStore()
        a = CountingAgent("a")
        b = CountingAgent("b", succeed=False)
        result = _chain(a, b).run("go", checkpoint=store, run_id="r2")

        assert not result.success
        saved = store.load("r2")
        assert set(saved.completed) == {"a"}
        assert "b" in saved.failed
        assert saved.metadata["complete"] is False


class TestResuming:
    """The behaviour the feature exists for."""

    def test_a_completed_node_is_not_run_again(self):
        store = InMemoryCheckpointStore()
        a = CountingAgent("a")
        b = CountingAgent("b", succeed=False)
        _chain(a, b).run("go", checkpoint=store, run_id="r3")
        assert (a.runs, b.runs) == (1, 1)

        # The failure is addressed and the same run continues.
        a2, b2 = CountingAgent("a"), CountingAgent("b")
        result = _chain(a2, b2).run("go", checkpoint=store, run_id="r3")

        assert result.success
        assert a2.runs == 0, "the completed node was run a second time"
        assert b2.runs == 1, "the failed node was not retried"

    def test_a_resumed_node_output_still_reaches_downstream(self):
        store = InMemoryCheckpointStore()
        a = CountingAgent("a")
        b = CountingAgent("b", succeed=False)
        _chain(a, b).run("go", checkpoint=store, run_id="r4")

        a2, b2 = CountingAgent("a"), CountingAgent("b")
        dag = _chain(a2, b2)
        result = dag.run("go", checkpoint=store, run_id="r4")

        assert result.outputs["a"] == "a-out"
        assert result.outputs["b"] == "b-out"

    def test_a_finished_run_replays_without_calling_a_model(self):
        store = InMemoryCheckpointStore()
        _chain(CountingAgent("a"), CountingAgent("b")).run(
            "go", checkpoint=store, run_id="r5",
        )

        a2, b2 = CountingAgent("a"), CountingAgent("b")
        result = _chain(a2, b2).run("go", checkpoint=store, run_id="r5")

        assert result.success
        assert (a2.runs, b2.runs) == (0, 0)

    def test_deleting_the_checkpoint_starts_the_run_over(self):
        store = InMemoryCheckpointStore()
        _chain(CountingAgent("a")).run("go", checkpoint=store, run_id="r6")
        store.delete("r6")

        a2 = CountingAgent("a")
        _chain(a2).run("go", checkpoint=store, run_id="r6")
        assert a2.runs == 1

    def test_an_unknown_run_id_simply_starts_from_the_beginning(self):
        store = InMemoryCheckpointStore()
        a = CountingAgent("a")
        result = _chain(a).run("go", checkpoint=store, run_id="never-seen")
        assert result.success and a.runs == 1


class TestGraphMismatch:
    """Resuming into a different graph would mix two workflows' outputs."""

    def test_a_changed_node_set_is_refused(self):
        store = InMemoryCheckpointStore()
        _chain(CountingAgent("a"), CountingAgent("b")).run(
            "go", checkpoint=store, run_id="r7",
        )

        wider = _chain(CountingAgent("a"), CountingAgent("b"), CountingAgent("c"))
        with pytest.raises(ValueError, match="different graph"):
            wider.run("go", checkpoint=store, run_id="r7")

    def test_the_refusal_names_what_moved(self):
        store = InMemoryCheckpointStore()
        _chain(CountingAgent("a")).run("go", checkpoint=store, run_id="r8")

        with pytest.raises(ValueError) as excinfo:
            _chain(CountingAgent("a"), CountingAgent("z")).run(
                "go", checkpoint=store, run_id="r8",
            )
        assert "'z'" in str(excinfo.value)


class TestFileStore:
    """Progress that outlives the process."""

    def test_a_run_survives_a_new_store_object(self, tmp_path):
        a = CountingAgent("a")
        b = CountingAgent("b", succeed=False)
        _chain(a, b).run(
            "go", checkpoint=FileCheckpointStore(tmp_path), run_id="disk-1",
        )

        # A different store instance is the closest a test gets to a new process.
        a2, b2 = CountingAgent("a"), CountingAgent("b")
        result = _chain(a2, b2).run(
            "go", checkpoint=FileCheckpointStore(tmp_path), run_id="disk-1",
        )
        assert result.success
        assert a2.runs == 0 and b2.runs == 1

    def test_a_run_id_cannot_escape_the_directory(self, tmp_path):
        store = FileCheckpointStore(tmp_path / "runs")
        store.save(WorkflowCheckpoint(run_id="../../escape"))
        written = list((tmp_path / "runs").glob("*.json"))
        assert len(written) == 1
        assert written[0].parent == tmp_path / "runs"

    def test_a_corrupt_file_reads_as_no_checkpoint(self, tmp_path):
        store = FileCheckpointStore(tmp_path)
        store.save(WorkflowCheckpoint(run_id="bad"))
        next(tmp_path.glob("bad.json")).write_text("{not json", encoding="utf-8")
        # Losing saved progress is survivable; refusing to run at all is not.
        assert store.load("bad") is None

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        store = FileCheckpointStore(tmp_path)
        store.save(WorkflowCheckpoint(run_id="tidy"))
        assert [p.name for p in tmp_path.iterdir()] == ["tidy.json"]

    def test_listing_returns_the_run_ids(self, tmp_path):
        store = FileCheckpointStore(tmp_path)
        store.save(WorkflowCheckpoint(run_id="one"))
        store.save(WorkflowCheckpoint(run_id="two"))
        assert store.list_runs() == ["one", "two"]


class TestCheckpointRecord:
    """The stored form."""

    def test_it_round_trips_through_json(self):
        original = WorkflowCheckpoint(
            run_id="r", workflow="w", node_ids=["a"],
            completed={"a": "out"}, outputs={"a": "out"},
        )
        restored = WorkflowCheckpoint.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )
        assert restored.run_id == "r"
        assert restored.completed == {"a": "out"}

    def test_an_unserializable_output_is_stored_as_text(self):
        checkpoint = WorkflowCheckpoint(run_id="r", completed={"a": object()})
        # One awkward value must not cost the run its whole saved progress.
        assert isinstance(checkpoint.to_dict()["completed"]["a"], str)

    def test_a_checkpoint_from_an_older_version_still_loads(self):
        restored = WorkflowCheckpoint.from_dict({"run_id": "r"})
        assert restored.run_id == "r"
        assert restored.completed == {}

    def test_a_failed_node_is_not_treated_as_done(self):
        checkpoint = WorkflowCheckpoint(run_id="r", failed={"a": "boom"})
        assert not checkpoint.is_done("a")


class TestStoreFailure:
    """A store that cannot be written to is not a reason to lose the run."""

    def test_a_saving_error_does_not_fail_the_run(self, caplog):
        class Broken:
            def save(self, checkpoint):
                raise OSError("disk full")

            def load(self, run_id):
                return None

            def delete(self, run_id):
                pass

        a = CountingAgent("a")
        result = _chain(a).run("go", checkpoint=Broken(), run_id="r9")
        assert result.success and a.runs == 1
