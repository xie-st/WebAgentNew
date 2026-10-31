"""E05/E06: model results that return during a pause.

The fake clock makes the pause window exact. The scripted model is a real
EffGen model path; the fake backend is only the environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from webagent.budget import Budget, Deadline
from webagent.clock import FakeClock
from webagent.control import (
    ControlAwareModel,
    ControlCoordinator,
    ControlPaused,
    ControlState,
    ControlledLoopRunner,
    MemoryControlSource,
    read_control_events,
)
from webagent.effgen_bridge import DriverStatus, create_driver
from webagent.events import read_events
from webagent.models.fake import ScriptedModel
from webagent.protocol import ExecutionResult, Observation, ToolRequest
from webagent.session import Session, SessionState
from webagent.testing import FakeBackend, FakeElement, FakePage, FakeSite
from webagent.tools import build_web_action


def _form_site() -> FakeSite:
    return FakeSite(
        [
            FakePage(
                url="http://fixture.local/form",
                title="Fixture form",
                ax_text="A draft form with one save control.",
                elements=(
                    FakeElement(role="textbox", name="Review draft"),
                    FakeElement(role="button", name="Save draft"),
                ),
            )
        ]
    )


def _action(tool: str, arguments: dict) -> str:
    return (
        "Thought: next step.\n"
        f"Action: {tool}\nAction Input: {json.dumps(arguments, ensure_ascii=False)}"
    )


class PausingModel(ScriptedModel):
    """Requests a pause inside the first model call, then returns a proposal."""

    def __init__(self, coordinator: ControlCoordinator, clock: FakeClock) -> None:
        super().__init__([_action("web_click", {"target": "2"}), "Final Answer: done"])
        self.coordinator = coordinator
        self.clock = clock

    def generate(self, prompt: str, config=None, **kwargs):  # type: ignore[override]
        if self.call_count == 0:
            self.coordinator.request_pause("user_pause", source="model")
            self.clock.advance(2.5)
        return super().generate(prompt, config, **kwargs)


class PauseThenResumeSource(MemoryControlSource):
    """Queue one pause, then perform the user edit and return explicit resume."""

    def __init__(self, clock: FakeClock, *, wait_seconds: float = 30.0) -> None:
        super().__init__()
        self.clock = clock
        self.wait_seconds = wait_seconds
        self.coordinator: ControlCoordinator | None = None
        self.probe = None
        self.resumed = False
        self.probe_at_handoff: dict = {}
        self.probe_after_handoff: dict = {}
        self.send("pause")

    def poll(self) -> str | None:
        command = super().poll()
        if command is not None:
            return command
        coordinator = self.coordinator
        if (
            coordinator is not None
            and coordinator.state is ControlState.USER_CONTROL
            and not self.resumed
        ):
            self.resumed = True
            if self.probe is not None:
                self.probe_at_handoff = dict(self.probe())
            self.clock.advance(self.wait_seconds)
            if self.probe is not None:
                self.probe_after_handoff = dict(self.probe())
            return "resume"
        return None


class AdvancingResumeSource(MemoryControlSource):
    def __init__(self, clock: FakeClock, seconds: float) -> None:
        super().__init__()
        self.clock = clock
        self.seconds = seconds
        self.coordinator: ControlCoordinator | None = None
        self.probe = None
        self.sent = False
        self.probe_at_handoff: dict = {}
        self.probe_after_handoff: dict = {}

    def poll(self) -> str | None:
        command = super().poll()
        if command is not None:
            return command
        coordinator = self.coordinator
        if (
            coordinator is not None
            and coordinator.state is ControlState.USER_CONTROL
            and not self.sent
        ):
            self.sent = True
            if self.probe is not None:
                self.probe_at_handoff = dict(self.probe())
            self.clock.advance(self.seconds)
            if self.probe is not None:
                self.probe_after_handoff = dict(self.probe())
            return "resume"
        return None


class CancelDuringHandoffSource(MemoryControlSource):
    def __init__(self) -> None:
        super().__init__()
        self.coordinator: ControlCoordinator | None = None

    def poll(self) -> str | None:
        command = super().poll()
        if command is not None:
            return command
        if (
            self.coordinator is not None
            and self.coordinator.state is ControlState.USER_CONTROL
        ):
            return "cancel"
        return None


class FailingModel(ScriptedModel):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__([])
        self.clock = clock

    def generate(self, prompt: str, config=None, **kwargs):  # type: ignore[override]
        self.clock.advance(0.75)
        raise RuntimeError("simulated model failure")


def _session(clock: FakeClock, tmp_path: Path, run_id: str) -> Session:
    session = Session(
        run_id,
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        budget=Budget(deadline=Deadline(clock.now() + 300, clock=clock), clock=clock),
    )
    session.start_task("controlled fake task")
    return session


def test_late_model_result_is_counted_and_its_proposal_is_discarded(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    backend = FakeBackend("e05-late", _form_site(), start_url="http://fixture.local/form")
    session = _session(clock, tmp_path, "e05-late")
    session.ingest_observation(backend.current_observation)
    source = AdvancingResumeSource(clock, 30)
    coordinator = ControlCoordinator(
        "e05-late",
        clock=clock,
        source=source,
        platform_deadline_epoch=clock.now() + 300,
        wait_timeout_seconds=1.0,
    )
    source.coordinator = coordinator
    model = ControlAwareModel(
        PausingModel(coordinator, clock),
        session=session,
        coordinator=coordinator,
        clock=clock,
    )
    source.probe = lambda: {
        "model_calls": len(model.calls),
        "backend_actions": len(backend.executed_actions),
        "event_count": len(read_events(session.evidence.events_path)),
    }
    driver = None
    try:
        driver = create_driver(session, model)
        runner = ControlledLoopRunner(driver, backend, coordinator)
        step = runner.run()
    finally:
        if driver is not None:
            driver.close()
        session.close()
        coordinator.close()

    assert step.status is DriverStatus.COMPLETED
    assert backend.executed_actions == []
    counts = session.model_call_counts()
    assert counts == {"platform_calls": 2, "local_attempts": 2, "failures": 0}
    accounting = coordinator.accounting.snapshot()
    assert accounting["local_attempts"] == 2
    assert accounting["latency_ms_total"] == pytest.approx(2500.0)
    assert accounting["paused_at_return"] == 1
    assert accounting["discarded_proposals"] == 1

    events = read_events(session.evidence.events_path)
    proposed = [event for event in events if event["event_type"] == "proposed"]
    dispatched = [event for event in events if event["event_type"] == "dispatched"]
    feedback = [event for event in events if event["event_type"] == "feedback"]
    assert len(proposed) == 1
    assert dispatched == []
    assert feedback == []
    dropped = coordinator.discarded_proposals[0]
    assert dropped["action_id"] == proposed[0]["action_id"]
    assert dropped["executed"] is False and dropped["consumed"] is False

    # Zero new model call / page action / event during the explicit wait.
    assert source.probe_at_handoff == source.probe_after_handoff
    assert source.probe_at_handoff["model_calls"] == 1
    assert source.probe_at_handoff["backend_actions"] == 0

    control_events = read_control_events(
        tmp_path / "runtime" / "runs" / "e05-late" / "control_events.jsonl"
    )
    assert any(
        event["event_type"] == "model_result_during_pause" for event in control_events
    )
    snapshot = coordinator.snapshot().to_dict()
    assert snapshot["total_seconds"] == pytest.approx(32.5)
    assert snapshot["human_wait_seconds"] == pytest.approx(30.0)
    assert snapshot["execution_seconds"] == pytest.approx(2.5)
    assert snapshot["deadline_extended"] is False


def test_user_pause_before_dispatch_drops_proposal_and_resumes_on_new_observation(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    backend = FakeBackend("e06-pause", _form_site(), start_url="http://fixture.local/form")
    session = _session(clock, tmp_path, "e06-pause")
    session.ingest_observation(backend.current_observation)
    source = PauseThenResumeSource(clock, wait_seconds=12)
    coordinator = ControlCoordinator(
        "e06-pause",
        clock=clock,
        source=source,
        platform_deadline_epoch=clock.now() + 300,
        wait_timeout_seconds=1.0,
    )
    source.coordinator = coordinator
    model = ControlAwareModel(
        ScriptedModel(
            [
                _action("web_click", {"target": "2"}),
                _action("web_click", {"target": "2"}),
                "Final Answer: done",
            ]
        ),
        session=session,
        coordinator=coordinator,
        clock=clock,
    )
    source.probe = lambda: {
        "model_calls": len(model.calls),
        "backend_actions": len(backend.executed_actions),
        "event_count": len(read_events(session.evidence.events_path)),
    }
    driver = None
    old_action = None
    try:
        driver = create_driver(session, model)
        runner = ControlledLoopRunner(driver, backend, coordinator)
        step = runner.run()
    finally:
        if driver is not None:
            old_action = driver.pending_action
            driver.close()
        session.close()
        coordinator.close()

    assert step.status is DriverStatus.COMPLETED
    assert len(backend.executed_actions) == 1
    executed = backend.executed_actions[0]
    dropped = coordinator.discarded_proposals[0]
    assert dropped["executed"] is False
    assert executed.action_id != dropped["action_id"]
    assert executed.observation_id != dropped["observation_id"]

    events = read_events(session.evidence.events_path)
    feedback = [event for event in events if event["event_type"] == "feedback"]
    assert [event["action_id"] for event in feedback] == [executed.action_id]
    assert dropped["action_id"] not in {event["action_id"] for event in feedback}
    assert source.probe_at_handoff == source.probe_after_handoff

    control_events = read_control_events(
        tmp_path / "runtime" / "runs" / "e06-pause" / "control_events.jsonl"
    )
    resumed = next(event for event in control_events if event["event_type"] == "resumed")
    assert resumed["page_epoch"] == 1
    assert resumed["observation_id"] == executed.observation_id
    assert dropped["observation_id"] != executed.observation_id
    assert coordinator.snapshot().human_wait_seconds == pytest.approx(12.0)


def test_session_consume_gate_blocks_dispatch_while_paused(tmp_path: Path) -> None:
    clock = FakeClock()
    backend = FakeBackend("e05-gate", _form_site(), start_url="http://fixture.local/form")
    session = _session(clock, tmp_path, "e05-gate")
    session.ingest_observation(backend.current_observation)
    coordinator = ControlCoordinator(
        "e05-gate", clock=clock, platform_deadline_epoch=clock.now() + 300
    )
    coordinator.start()
    session.set_action_gate(coordinator.gate)
    action = build_web_action(
        ToolRequest("web_click", {"target": "2"}), session.observation
    )
    session.propose(action)
    assert session.proposed_action is not None

    coordinator.request_pause("user_pause", source="user")
    with pytest.raises(ControlPaused):
        session.consume(action.action_id)
    assert session.state is SessionState.RUNNING
    assert session.proposed_action is not None
    assert session.in_flight_action is None
    assert backend.executed_actions == []
    assert not [
        event
        for event in read_events(session.evidence.events_path)
        if event["event_type"] == "dispatched"
    ]


def test_initial_visible_overlay_pauses_before_the_loop_opens(tmp_path: Path) -> None:
    clock = FakeClock()
    auth_site = FakeSite(
        [
            FakePage(
                url="http://fixture.local/login",
                title="Sign in required",
                ax_text=(
                    "HEADINGS:\n- h1: Sign in required\nTEXT:\n"
                    "- Your session has expired. Sign in in this browser to restore the page."
                ),
                elements=(FakeElement(role="button", name="Sign in"),),
            )
        ]
    )
    backend = FakeBackend("e05-initial", auth_site, start_url="http://fixture.local/login")
    session = _session(clock, tmp_path, "e05-initial")
    session.ingest_observation(backend.current_observation)
    source = AdvancingResumeSource(clock, 5)
    coordinator = ControlCoordinator(
        "e05-initial",
        clock=clock,
        source=source,
        platform_deadline_epoch=clock.now() + 300,
        wait_timeout_seconds=1.0,
    )
    source.coordinator = coordinator
    model = ControlAwareModel(
        ScriptedModel(["Final Answer: user completed the sign-in step"]),
        session=session,
        coordinator=coordinator,
        clock=clock,
    )
    source.probe = lambda: {
        "model_calls": len(model.calls),
        "backend_actions": len(backend.executed_actions),
    }
    driver = None
    try:
        driver = create_driver(session, model)
        runner = ControlledLoopRunner(driver, backend, coordinator)
        step = runner.run()
    finally:
        if driver is not None:
            driver.close()
        session.close()
        coordinator.close()

    assert step.status is DriverStatus.COMPLETED
    assert backend.executed_actions == []
    assert source.probe_at_handoff == {"model_calls": 0, "backend_actions": 0}
    takeover = next(
        event for event in read_control_events(
            tmp_path / "runtime" / "runs" / "e05-initial" / "control_events.jsonl"
        )
        if event["event_type"] == "takeover_detected"
    )
    assert takeover["details"]["kind"] == "login"
    assert coordinator.snapshot().page_epoch == 1


def test_cancel_during_handoff_keeps_the_executed_action_feedback(tmp_path: Path) -> None:
    clock = FakeClock()
    auth_site = FakeSite(
        [
            FakePage(
                url="http://fixture.local/form",
                title="Fixture form",
                ax_text="A draft form with one save control.",
                elements=(
                    FakeElement(role="textbox", name="Review draft"),
                    FakeElement(
                        role="button",
                        name="Save draft",
                        href="http://fixture.local/login",
                    ),
                ),
            ),
            FakePage(
                url="http://fixture.local/login",
                title="Sign in required",
                ax_text=(
                    "HEADINGS:\n- h1: Sign in required\nTEXT:\n"
                    "- Your session has expired. Sign in in this browser to restore the page."
                ),
                elements=(FakeElement(role="button", name="Sign in"),),
            ),
        ]
    )
    backend = FakeBackend("e06-cancel-feedback", auth_site, start_url="http://fixture.local/form")
    session = _session(clock, tmp_path, "e06-cancel-feedback")
    session.ingest_observation(backend.current_observation)
    source = CancelDuringHandoffSource()
    coordinator = ControlCoordinator(
        "e06-cancel-feedback",
        clock=clock,
        source=source,
        platform_deadline_epoch=clock.now() + 300,
        wait_timeout_seconds=1.0,
    )
    source.coordinator = coordinator
    model = ControlAwareModel(
        ScriptedModel([_action("web_click", {"target": "2"})]),
        session=session,
        coordinator=coordinator,
        clock=clock,
    )
    driver = None
    final_session_state = None
    try:
        driver = create_driver(session, model)
        runner = ControlledLoopRunner(driver, backend, coordinator)
        step = runner.run()
    finally:
        if driver is not None:
            driver.close()
        final_session_state = session.state
        session.close()
        coordinator.close()

    assert step.status is DriverStatus.CANCELLED
    assert len(backend.executed_actions) == 1
    executed = backend.executed_actions[0].action_id
    events = read_events(session.evidence.events_path)
    feedback = [
        event
        for event in events
        if event["event_type"] == "feedback" and event["action_id"] == executed
    ]
    assert len(feedback) == 1
    assert feedback[0]["result"] == "observed"
    assert coordinator.snapshot().state == ControlState.CANCELLED.value
    assert final_session_state is SessionState.CANCELLED


def test_failed_model_call_still_counts_in_all_three_buckets(tmp_path: Path) -> None:
    clock = FakeClock()
    session = _session(clock, tmp_path, "e06-failure")
    coordinator = ControlCoordinator(
        "e06-failure", clock=clock, platform_deadline_epoch=clock.now() + 300
    )
    coordinator.start()
    model = ControlAwareModel(
        FailingModel(clock), session=session, coordinator=coordinator, clock=clock
    )
    with pytest.raises(RuntimeError, match="simulated model failure"):
        model.generate("prompt")

    assert session.model_call_counts() == {
        "platform_calls": 0,
        "local_attempts": 1,
        "failures": 1,
    }
    accounting = coordinator.accounting.snapshot()
    assert accounting["platform_calls"] == 0
    assert accounting["local_attempts"] == 1
    assert accounting["failures"] == 1
    assert accounting["latency_ms_total"] == pytest.approx(750.0)


def test_local_pause_does_not_extend_the_platform_deadline(tmp_path: Path) -> None:
    clock = FakeClock()
    deadline_epoch = clock.now() + 60
    backend = FakeBackend("e07-runner-deadline", _form_site(), start_url="http://fixture.local/form")
    session = Session(
        "e07-runner-deadline",
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        budget=Budget(deadline=Deadline(deadline_epoch, clock=clock), clock=clock),
    )
    session.start_task("pause past the platform deadline")
    session.ingest_observation(backend.current_observation)
    source = PauseThenResumeSource(clock, wait_seconds=70)
    coordinator = ControlCoordinator(
        "e07-runner-deadline",
        clock=clock,
        source=source,
        platform_deadline_epoch=deadline_epoch,
        wait_timeout_seconds=1.0,
    )
    source.coordinator = coordinator
    model = ControlAwareModel(
        ScriptedModel([_action("web_click", {"target": "2"})]),
        session=session,
        coordinator=coordinator,
        clock=clock,
    )
    driver = None
    try:
        driver = create_driver(session, model)
        runner = ControlledLoopRunner(driver, backend, coordinator)
        step = runner.run()
    finally:
        if driver is not None:
            driver.close()
        session.close()
        coordinator.close()

    assert step.status is DriverStatus.CANCELLED
    assert backend.executed_actions == []
    assert session.deadline.epoch_seconds == pytest.approx(deadline_epoch)
    snapshot = coordinator.snapshot().to_dict()
    assert snapshot["state"] == ControlState.BLOCKED.value
    assert snapshot["deadline_extended"] is False
    assert snapshot["platform_deadline_epoch"] == pytest.approx(deadline_epoch)
    assert snapshot["official_remaining_seconds"] == pytest.approx(-10.0)
    control_events = read_control_events(
        tmp_path / "runtime" / "runs" / "e07-runner-deadline" / "control_events.jsonl"
    )
    assert any(event["event_type"] == "deadline_blocked" for event in control_events)
