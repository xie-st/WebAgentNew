"""E08 control part: restart from the complete snapshot and re-observe.

Only public StateStore/EvidenceStore entry points are used. No generator stack
is restored and no pending action is replayed.
"""

from __future__ import annotations

from pathlib import Path

from webagent.clock import FakeClock
from webagent.control import read_control_events, restart_from_snapshot
from webagent.events import read_events
from webagent.protocol import ToolRequest
from webagent.session import Session
from webagent.state import EffectStatus, StateStore
from webagent.testing import FakeBackend, FakeElement, FakePage, FakeSite
from webagent.tools import build_web_action


def _site() -> FakeSite:
    return FakeSite(
        [
            FakePage(
                url="http://fixture.local/product/P-1",
                title="Fixture product",
                ax_text="A fixture product page.",
                elements=(FakeElement(role="button", name="Cancel draft"),),
            )
        ]
    )


def _pending_effect_snapshot(runtime: Path) -> tuple[str, str, str, int]:
    old_run = "e08-old"
    site = _site()
    backend = FakeBackend(old_run, site, start_url="http://fixture.local/product/P-1")
    clock = FakeClock()
    session = Session(old_run, runtime_dir=runtime, clock=clock)
    store = StateStore(runtime, old_run, clock=clock)
    action_id = ""
    executed = 0
    try:
        session.start_task("cancel the draft once")
        observation = backend.observe()
        session.ingest_observation(observation)
        store.initialize("cancel the draft once")
        action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
        action_id = action.action_id
        effect = store.prepare_effect(
            action_id=action.action_id,
            target="1",
            object_id="P-1",
            branch_id="root",
            observation_id=observation.observation_id,
            note="draft_clear: simulated write that took effect before interruption",
        )
        assert effect.status is EffectStatus.PENDING
        session.propose(action)
        session.consume(action.action_id)
        backend.execute(action)
        executed = len(backend.executed_actions)
        # Simulate the process interruption immediately after the write took
        # effect and before any feedback was ingested or confirmed.
    finally:
        backend.close()
        session.close()
    return old_run, str(store.state_path), action_id, executed


def test_restart_loads_snapshot_registers_branch_and_reobserves(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    old_run, state_path, executed_action_id, old_executed = _pending_effect_snapshot(runtime)
    assert Path(state_path).is_file()
    assert old_executed == 1

    new_run = "e08-new"
    new_backend = FakeBackend(new_run, _site(), start_url="http://fixture.local/product/P-1")
    try:
        result = restart_from_snapshot(
            runtime,
            old_run,
            branch_id="restart-1",
            new_run_id=new_run,
            observer=lambda: new_backend.observe(),
        )
    finally:
        new_backend.close()

    assert result.generator_restored is False
    assert result.replayed_actions == ()
    assert result.branches == {"root": old_run, "restart-1": new_run}
    assert result.loaded_revision >= 1
    assert len(result.pending_effects) == 1
    effect = result.pending_effects[0]
    assert effect.action_id == executed_action_id
    assert effect.status is EffectStatus.PENDING
    assert result.observation.run_id == new_run
    assert result.session_state == "running"
    assert new_backend.executed_actions == []

    old_events = read_events(runtime / "runs" / old_run / "events.jsonl")
    assert any(
        event["event_type"] == "dispatched"
        and event["action_id"] == executed_action_id
        for event in old_events
    )
    assert not any(
        event["event_type"] == "feedback"
        and event["action_id"] == executed_action_id
        for event in old_events
    )

    # The restarted run's event/control files exist and contain no replay.
    run_dir = runtime / "runs" / new_run
    assert (run_dir / "control_events.jsonl").is_file()
    control_events = read_control_events(run_dir / "control_events.jsonl")
    assert [event["event_type"] for event in control_events] == [
        "restart_loaded",
        "reobserved",
    ]
    assert control_events[0]["details"]["generator_restored"] is False
    assert control_events[1]["observation_id"] == result.observation.observation_id

    session_events = read_events(run_dir / "events.jsonl")
    assert not any(
        event["event_type"] == "dispatched" for event in session_events
    )
    # The loaded snapshot still has the residue; loading does not clear it.
    reloaded = StateStore(runtime, old_run)
    reloaded.load()
    assert len(reloaded.pending_effects) == 1
    assert reloaded.pending_effects[0].status is EffectStatus.PENDING
