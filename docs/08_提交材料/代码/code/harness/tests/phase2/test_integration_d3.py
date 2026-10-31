"""D3: local-site L01 through the state ledger, effects, events and judge."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from d3_helpers import SimulatedKill, _action, run_l01

from webagent.clock import FakeClock, SystemClock
from webagent.effgen_bridge import DriverStatus, create_driver
from webagent.events import REQUIRED_FIELDS, STATE_FIELDS, EventLogger, read_events
from webagent.models.fake import ScriptedModel
from webagent.protocol import ToolRequest
from webagent.session import Session, SessionState
from webagent.state import (
    STATE_TOOL_NAMES,
    EffectClass,
    EffectPolicy,
    EffectStatus,
    GoalStatus,
    StatePersistenceError,
    StateStore,
    StatefulLoopRunner,
    create_stateful_driver,
    observed_page_fact_id,
)
from webagent.testing import FakeBackend, FakeElement, FakePage, FakeSite
from webagent.tools import build_web_action


def _single_button_site() -> FakeSite:
    return FakeSite(
        [
            FakePage(
                url="http://fixture.local/form",
                title="Fixture form",
                ax_text="A form with one save control.\nELEMENTS:\n[1] button Save",
                elements=(FakeElement(role="button", name="Save"),),
            )
        ]
    )


def test_effect_policy_classifies_only_declared_writes(make_observation) -> None:
    observation = make_observation(
        elements={
            "1": ("button", "Save draft"),
            "2": ("button", "Cancel draft"),
            "3": ("button", "Send comment"),
            "4": ("link", "Next page"),
            "5": ("textbox", "Query"),
        }
    )
    policy = EffectPolicy()

    def classify(tool: str, arguments: dict) -> object:
        action = build_web_action(ToolRequest(tool, arguments), observation)
        return policy.classify(action, observation)

    assert classify("web_click", {"target": "1"}).effect_class is EffectClass.DRAFT_SAVE
    assert classify("web_click", {"target": "2"}).effect_class is EffectClass.DRAFT_CLEAR
    assert classify("web_click", {"target": "3"}).effect_class is EffectClass.COMMENT
    assert classify("web_click", {"target": "4"}) is None
    assert classify("web_click", {"target": "5"}) is None
    assert classify("web_scroll", {"direction": "down"}) is None
    assert classify("web_type", {"target": "5", "text": "plain"}) is None
    assert (
        classify("web_type", {"target": "5", "text": "submitted", "submit": True}).effect_class
        is EffectClass.SUBMIT
    )


def test_events_state_fields_are_optional_and_backward_compatible(tmp_path: Path) -> None:
    legacy = EventLogger("d3-events-legacy", tmp_path / "legacy.jsonl", clock=FakeClock())
    legacy.emit("run_start", result={"note": "legacy"})
    legacy.close()
    event = read_events(tmp_path / "legacy.jsonl")[0]
    assert set(REQUIRED_FIELDS) <= set(event)
    for field in STATE_FIELDS:
        assert field in event and event[field] is None

    contextual = EventLogger("d3-events-ctx", tmp_path / "ctx.jsonl", clock=FakeClock())
    contextual.set_state_context(
        lambda: {
            "state_revision": 7,
            "branch_id": "root",
            "plan_version": 3,
        },
        goal_id="g1",
    )
    contextual.emit("run_start", result={"note": "ledger"})
    contextual.close()
    event = read_events(tmp_path / "ctx.jsonl")[0]
    assert event["state_revision"] == 7
    assert event["goal_id"] == "g1"
    assert event["branch_id"] == "root"
    assert event["plan_version"] == 3
    assert set(REQUIRED_FIELDS) <= set(event)


def test_take_action_hook_runs_before_consume_and_can_block_dispatch(
    tmp_path: Path,
) -> None:
    site = _single_button_site()
    backend = FakeBackend("d3-hook", site, start_url="http://fixture.local/form")
    clock = FakeClock()
    session = Session("d3-hook", runtime_dir=tmp_path / "runtime", clock=clock)
    driver = None
    try:
        session.start_task("click save")
        session.ingest_observation(backend.current_observation)
        driver = create_driver(
            session,
            ScriptedModel([_action("web_click", {"target": "1"}), "Final Answer: done"]),
        )
        step = driver.start()
        assert step.status is DriverStatus.ACTION
        assert step.action is not None
        seen: list[str] = []

        def refuse(action) -> None:
            seen.append(action.action_id)
            raise RuntimeError("ledger refused this dispatch")

        with pytest.raises(RuntimeError, match="refused"):
            driver.take_action(before_dispatch=refuse)
        assert seen == [step.action.action_id]
        assert session.proposed_action is not None
        assert session.in_flight_action is None
        assert backend.executed_actions == []
        assert not [event for event in read_events(session.evidence.events_path)
                    if event["event_type"] == "dispatched"]

        dispatched = driver.take_action(before_dispatch=lambda action: seen.append("ok"))
        assert dispatched.action_id == step.action.action_id
        assert seen[-1] == "ok"
        result = backend.execute(dispatched)
        step = driver.accept_observation(result.observation)
        assert step.status is DriverStatus.COMPLETED
        assert len(backend.executed_actions) == 1
    finally:
        if driver is not None:
            driver.close()
        session.close()


def test_effect_save_failure_stops_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _single_button_site()
    backend = FakeBackend("d3-fail", site, start_url="http://fixture.local/form")
    clock = FakeClock()
    session = Session("d3-fail", runtime_dir=tmp_path / "runtime", clock=clock)
    store = StateStore(tmp_path / "runtime", "d3-fail")
    store.initialize("click save")
    driver = None
    try:
        session.start_task("click save")
        session.ingest_observation(backend.current_observation)
        driver = create_stateful_driver(
            session,
            ScriptedModel([_action("web_click", {"target": "1"}), "Final Answer: done"]),
            store,
        )
        runner = StatefulLoopRunner(driver, backend, store)

        def fail_write(_payload: bytes) -> None:
            raise StatePersistenceError("simulated disk failure")

        monkeypatch.setattr(store, "_atomic_write", fail_write)
        step = runner.run()
        assert step.status is DriverStatus.CANCELLED
        assert isinstance(runner.last_error, StatePersistenceError)
        assert runner.effects_prepared == []
        assert runner.effects_confirmed == []
        assert backend.executed_actions == []
        assert store.state.revision == 0
        assert store.state.pending_effects == {}
        assert session.state is SessionState.CANCELLED
        assert not [
            event
            for event in read_events(session.evidence.events_path)
            if event["event_type"] == "dispatched"
        ]
    finally:
        if driver is not None:
            driver.close()
        session.close()


def test_l01_end_to_end_ledger_effects_judge_and_reload(tmp_path: Path) -> None:
    run = run_l01(tmp_path / "runtime", seed=101)

    assert run.killed is False
    assert run.step is not None and run.step.status is DriverStatus.COMPLETED
    assert run.session.finish_result is not None
    assert run.session.finish_result.reason.value == "completed"
    assert run.prepared.site.submission_count() == 0

    # The task text and plan are in the ledger, and the state tools were built
    # through the driver's extra_tools path.
    assert run.state_after["task"] == run.task
    assert run.state_after["plan_version"] == 1
    assert {"n1"} == set(run.state_after["plan"])
    assert STATE_TOOL_NAMES <= set(run.runner.driver.agent.tools)

    # The observed fact is backed by real, hashed screenshot files. Before the
    # judge it is observed progress, not success: the goal is not verified.
    fact = run.state_before_judge["facts"][observed_page_fact_id(run.target_id)]
    assert fact["kind"] == "observed"
    assert fact["object"] == run.target_id
    assert fact["evidence"]
    for ref in fact["evidence"]:
        path = run.run_dir / ref["reference"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]
    assert run.view_before_judge["verified_goal_ids"] == []
    assert run.state_before_judge["goals"]["g1"]["status"] == GoalStatus.RUNNING.value
    assert run.view_before_judge["completion_status"] == "incomplete"

    # The same object was observed more than once; only distinct objects count.
    assert len(run.detail_observations) >= 2
    assert set(run.detail_observations) == {run.target_id}
    progress = run.object_progress_before_judge
    assert progress["objects_total"] == 1
    assert progress["objects_completed"] == 1
    assert progress["objects_remaining"] == 0
    assert progress["completed_object_ids"] == [run.target_id]
    assert fact["revision"] == len(run.detail_observations)

    # One declared write (Cancel draft): saved pending before dispatch, then
    # confirmed from the observed feedback.
    assert len(run.runner.effects_prepared) == 1
    assert len(run.runner.effects_confirmed) == 1
    prepared = run.runner.effects_prepared[0]
    confirmed = run.runner.effects_confirmed[0]
    assert prepared.action_id == confirmed.action_id
    assert prepared.object_id == run.target_id
    assert prepared.target.isdigit()
    assert "draft_clear" in prepared.note
    assert prepared.status is EffectStatus.PENDING
    assert confirmed.status is EffectStatus.CONFIRMED
    assert confirmed.created_revision == prepared.created_revision
    assert confirmed.updated_revision is not None
    assert confirmed.updated_revision > prepared.created_revision
    live = run.state_after["pending_effects"][prepared.action_id]
    assert live["status"] == EffectStatus.CONFIRMED.value
    assert prepared.action_id in run.executed_action_ids

    # The judge ran externally and its pass is the evidence-backed transition.
    assert run.judge_result is not None and run.judge_result.passed
    assert run.verified is True
    assert run.state_after["goals"]["g1"]["status"] == GoalStatus.VERIFIED.value
    assert run.view_after["verified_goal_ids"] == ["g1"]
    assert run.view_after["completion_status"] == "complete"
    assert run.state_after["goals"]["g1"]["evidence"]
    assert run.state_after["revision"] > run.state_before_judge["revision"]

    # Events carry the ledger revision/goal/branch/plan fields at the right
    # stages: proposed before prepare, dispatched after prepare, feedback after
    # confirmation.
    action_events = [
        event for event in run.events if event.get("action_id") == prepared.action_id
    ]
    proposed = next(event for event in action_events if event["event_type"] == "proposed")
    dispatched = next(event for event in action_events if event["event_type"] == "dispatched")
    feedback = next(event for event in action_events if event["event_type"] == "feedback")
    for event in (proposed, dispatched, feedback):
        assert event["goal_id"] == "g1"
        assert event["branch_id"] == "root"
        assert event["plan_version"] == 1
    assert proposed["state_revision"] < prepared.created_revision
    assert dispatched["state_revision"] == prepared.created_revision
    assert feedback["state_revision"] == confirmed.updated_revision
    assert proposed["result"] is None and dispatched["result"] is None
    assert feedback["result"] == "observed"
    for event in run.events:
        assert set(REQUIRED_FIELDS) <= set(event)

    # A fresh StateStore has no memory of the run and loads the same snapshot.
    assert run.reload_equal, "reloaded state.json differs from the in-memory state"
    assert run.reloaded_state["revision"] == run.state_after["revision"]
    assert run.reloaded_state["pending_effects"] == run.state_after["pending_effects"]
    assert run.reloaded_state["facts"] == run.state_after["facts"]


def test_failed_judge_does_not_write_verified(tmp_path: Path) -> None:
    run = run_l01(tmp_path / "runtime", seed=303, judge_evidence="none")

    assert run.step is not None and run.step.status is DriverStatus.COMPLETED
    assert run.judge_result is not None and not run.judge_result.passed
    assert any(
        check.name == "screenshot" and not check.passed
        for check in run.judge_result.checks
    )
    # The observed Fact and its evidence exist, but a failed judge is not
    # promoted into verified: observed progress is not success.
    fact = run.state_after["facts"][observed_page_fact_id(run.target_id)]
    assert fact["kind"] == "observed" and fact["evidence"]
    assert run.verified is False
    assert run.state_after["goals"]["g1"]["status"] == GoalStatus.RUNNING.value
    assert run.view_after["verified_goal_ids"] == []
    assert run.view_after["completion_status"] == "incomplete"
    assert run.reload_equal


def test_midrun_kill_leaves_pending_effect_and_reload_preserves_it(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    run = run_l01(runtime, seed=202, kill_before_cancel=True)

    assert run.killed is True
    assert isinstance(run.kill_error, SimulatedKill)
    assert run.step is None
    assert len(run.runner.effects_prepared) == 1
    effect = run.runner.effects_prepared[0]
    assert effect.status is EffectStatus.PENDING
    assert effect.action_id not in run.executed_action_ids
    assert run.state_after["pending_effects"][effect.action_id]["status"] == (
        EffectStatus.PENDING.value
    )
    assert run.reload_equal

    fresh = StateStore(runtime, run.run_id)
    fresh.load()
    assert fresh.state.revision == run.state_after["revision"]
    assert len(fresh.state.unconfirmed_effects) == 1
    loaded_effect = fresh.state.unconfirmed_effects[0]
    assert loaded_effect.action_id == effect.action_id
    assert loaded_effect.status is EffectStatus.PENDING
    # Loading does not send or clear the residue, and emits no event of its own.
    before = len(read_events(run.session.evidence.events_path))
    assert len(read_events(run.session.evidence.events_path)) == before
    assert fresh.state.object_progress()["objects_completed"] == 1
