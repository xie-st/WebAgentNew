from __future__ import annotations

from pathlib import Path

import pytest

from webagent.clock import FakeClock
from webagent.errors import ActionRejected, RejectReason
from webagent.events import read_events
from webagent.protocol import Feedback, FeedbackStatus, Finish, ToolRequest, WebAction
from webagent.session import Session
from webagent.testing import FakeBackend
from webagent.tools import build_web_action, guard_requests


def start_session(session: Session, backend: FakeBackend) -> None:
    session.start_task("offline browser task")
    session.ingest_observation(backend.current_observation)


def test_legal_action_executes_and_returns_bound_feedback(
    backend: FakeBackend, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    start_session(session, backend)
    observation = session.observation
    action = guard_requests([ToolRequest("web_click", {"target": "2"})], observation).action
    assert action is not None
    session.propose(action)
    session.consume()
    result = backend.execute(action)
    assert result.feedback.status is FeedbackStatus.OBSERVED
    assert result.observation.previous_action_id == action.action_id
    assert result.observation.feedback is not None
    assert result.observation.feedback.action_id == action.action_id
    session.ingest_observation(result.observation)
    assert session.observation.url == "http://fixture.local/results"
    session.finish(Finish(claim="searched", message="clicked search", reason="completed"))
    session.close()
    events = read_events(session.evidence.events_path)
    feedback = [event for event in events if event["event_type"] == "feedback"]
    assert feedback and feedback[-1]["result"] == "observed"
    assert feedback[-1]["action_id"] == action.action_id


def test_multiple_actions_never_reach_session(
    backend: FakeBackend, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    start_session(session, backend)
    requests = [
        ToolRequest("web_click", {"target": "2"}),
        ToolRequest("web_scroll", {"direction": "down"}),
    ]
    with pytest.raises(ActionRejected) as excinfo:
        guard_requests(requests, session.observation)
    assert excinfo.value.reason is RejectReason.MULTIPLE_ACTIONS
    assert session.proposed_action is None
    assert backend.executed_actions == []
    events = read_events(session.evidence.events_path)
    assert not any(event["event_type"] == "dispatched" for event in events)
    session.close()


def test_stale_action_after_new_observation(
    backend: FakeBackend, runtime: Path, clock: FakeClock, make_observation
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    start_session(session, backend)
    old_observation = session.observation
    old_action = build_web_action(ToolRequest("web_click", {"target": "2"}), old_observation)
    session.ingest_observation(
        make_observation(observation_id="obs-manual-2", url="http://fixture.local/results")
    )
    with pytest.raises(ActionRejected) as excinfo:
        session.propose(old_action)
    assert excinfo.value.reason is RejectReason.STALE_OBSERVATION
    assert "obs-manual-2" in excinfo.value.correction
    session.close()


def test_same_number_different_element_is_resolved_from_full_reference(
    make_observation, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("task")
    first = make_observation(observation_id="obs-1", elements={"1": ("textbox", "Query")})
    second = make_observation(observation_id="obs-2", elements={"1": ("link", "Result")})
    session.ingest_observation(first)
    session.ingest_observation(second)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), second)
    session.propose(action)
    assert session.proposed_action.observation_id == "obs-2"
    assert session.resolve_target(action).role == "link"
    session.close()


def test_identical_page_content_is_still_a_new_observation(
    make_observation, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("task")
    first = make_observation(observation_id="obs-1", ax_text="same", url="http://fixture.local/same")
    second = make_observation(observation_id="obs-2", ax_text="same", url="http://fixture.local/same")
    session.ingest_observation(first)
    session.ingest_observation(second)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), second)
    session.propose(action)
    observations = [
        event
        for event in read_events(session.evidence.events_path)
        if event["event_type"] == "observation"
    ]
    assert len(observations) == 2
    session.close()


def test_duplicate_consumption_does_not_execute_twice(
    backend: FakeBackend, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    start_session(session, backend)
    action = build_web_action(ToolRequest("web_click", {"target": "2"}), session.observation)
    session.propose(action)
    session.consume()
    with pytest.raises(ActionRejected) as excinfo:
        session.consume(action.action_id)
    assert excinfo.value.reason is RejectReason.DUPLICATE_CONSUMPTION
    backend.execute(action)
    assert len(backend.executed_actions) == 1
    session.close()


def test_two_scrolls_with_same_parameters_are_two_bound_actions(
    backend: FakeBackend, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    start_session(session, backend)
    first = guard_requests(
        [ToolRequest("web_scroll", {"direction": "down"})], session.observation
    ).action
    session.propose(first)
    session.consume()
    first_result = backend.execute(first)
    session.ingest_observation(first_result.observation)
    second = guard_requests(
        [ToolRequest("web_scroll", {"direction": "down"})], session.observation
    ).action
    assert second.observation_id != first.observation_id
    assert second.args == first.args
    session.propose(second)
    session.consume()
    second_result = backend.execute(second)
    session.ingest_observation(second_result.observation)
    assert len(backend.executed_actions) == 2
    assert "SCROLL_POSITION: 2" in second_result.observation.ax_text
    events = read_events(session.evidence.events_path)
    assert len([event for event in events if event["event_type"] == "dispatched"]) == 2
    assert len([event for event in events if event["event_type"] == "feedback"]) == 2
    session.close()


def test_pause_resume_reobserves_and_rejects_old_action(
    backend: FakeBackend, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    start_session(session, backend)
    old_observation = session.observation
    old_action = build_web_action(ToolRequest("web_click", {"target": "2"}), old_observation)
    session.pause()
    fresh = backend.observe()
    assert fresh.previous_action_id is None
    session.resume(fresh)
    with pytest.raises(ActionRejected) as excinfo:
        session.propose(old_action)
    assert excinfo.value.reason is RejectReason.STALE_OBSERVATION
    new_action = build_web_action(ToolRequest("web_click", {"target": "2"}), fresh)
    session.propose(new_action)
    session.consume()
    result = backend.execute(new_action)
    session.ingest_observation(result.observation)
    session.close()


def test_wrong_page_action_is_rejected(backend: FakeBackend, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    start_session(session, backend)
    observation = session.observation
    wrong = WebAction(
        action_id="act-wrong",
        run_id=observation.run_id,
        page_id="page-other",
        observation_id=observation.observation_id,
        op="click",
        args={"target": "1"},
    )
    with pytest.raises(ActionRejected) as excinfo:
        session.propose(wrong)
    assert excinfo.value.reason is RejectReason.WRONG_PAGE
    assert session.proposed_action is None
    session.close()
