from __future__ import annotations

from pathlib import Path

import pytest

from webagent.budget import Budget, BudgetLimits, Deadline
from webagent.clock import FakeClock
from webagent.errors import ActionRejected, DeadlineExceeded, RejectReason, SessionStateError
from webagent.events import read_events
from webagent.protocol import Feedback, FeedbackStatus, Finish, ToolRequest
from webagent.session import Session, SessionState
from webagent.tools import build_web_action


def feedback_observation(make_observation, action_id: str, observation_id: str = "obs-2", **kwargs):
    feedback = Feedback(
        status=FeedbackStatus.OBSERVED,
        action_id=action_id,
        observation_id=observation_id,
        message="page changed",
    )
    return make_observation(
        observation_id=observation_id,
        previous_action_id=action_id,
        feedback=feedback,
        **kwargs,
    )


def test_reset_without_task_then_start(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.reset()
    assert session.state is SessionState.READY
    with pytest.raises(SessionStateError):
        session.start_task("   ")
    with pytest.raises(SessionStateError):
        session.ingest_observation(make_observation())
    session.start_task("find the sample result")
    assert session.state is SessionState.RUNNING
    session.close()


def test_full_lifecycle_and_event_distinction(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click search")
    observation = make_observation()
    session.ingest_observation(observation)
    action = build_web_action(
        ToolRequest("web_click", {"target": "2"}), observation, action_id="act-1"
    )
    session.propose(action)
    assert session.proposed_action is action
    session.consume()
    assert session.in_flight_action is action
    session.ingest_observation(feedback_observation(make_observation, "act-1"))
    session.finish(Finish(claim="done", message="clicked search", reason="completed"))
    assert session.state is SessionState.FINISHED
    session.close()
    assert session.state is SessionState.CLOSED
    events = read_events(session.evidence.events_path)
    event_types = [event["event_type"] for event in events]
    assert event_types.index("proposed") < event_types.index("dispatched") < event_types.index("feedback")
    proposed = next(event for event in events if event["event_type"] == "proposed")
    dispatched = next(event for event in events if event["event_type"] == "dispatched")
    feedback = next(event for event in events if event["event_type"] == "feedback")
    claim = next(event for event in events if event["event_type"] == "agent_claim")
    assert proposed["result"] is None
    assert dispatched["result"] is None
    assert feedback["result"] == "observed"
    assert claim["result"]["reason"] == "completed"
    assert "score" not in claim["result"]
    result = session.evidence.result_path.read_text(encoding="utf-8")
    assert "completed" in result


def test_pause_requires_no_in_flight(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click")
    observation = make_observation()
    session.ingest_observation(observation)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
    session.propose(action)
    session.consume()
    with pytest.raises(SessionStateError):
        session.pause()
    session.ingest_observation(feedback_observation(make_observation, action.action_id))
    session.pause()
    assert session.state is SessionState.PAUSED
    with pytest.raises(ActionRejected) as excinfo:
        session.propose(build_web_action(ToolRequest("web_click", {"target": "1"}), session.observation))
    assert excinfo.value.reason is RejectReason.PAUSED
    session.close()
    assert session.state is SessionState.CLOSED
    event_count = len(read_events(session.evidence.events_path))
    session.close()
    assert len(read_events(session.evidence.events_path)) == event_count


def test_pause_drops_unconsumed_proposal(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click")
    observation = make_observation()
    session.ingest_observation(observation)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
    session.propose(action)
    session.pause()
    assert session.proposed_action is None
    cancelled = [
        event for event in read_events(session.evidence.events_path) if event["event_type"] == "cancelled"
    ]
    assert cancelled and cancelled[-1]["action_id"] == action.action_id
    session.close()


def test_resume_requires_fresh_observation_and_invalidates_old_action(
    make_observation, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click")
    first = make_observation()
    session.ingest_observation(first)
    old_action = build_web_action(
        ToolRequest("web_click", {"target": "1"}), first, action_id="act-old"
    )
    session.pause()
    with pytest.raises(SessionStateError):
        session.resume(first)
    second = make_observation(observation_id="obs-2", elements={"1": ("link", "New element")})
    session.resume(second)
    assert session.state is SessionState.RUNNING
    with pytest.raises(ActionRejected) as excinfo:
        session.propose(old_action)
    assert excinfo.value.reason is RejectReason.STALE_OBSERVATION
    new_action = build_web_action(ToolRequest("web_click", {"target": "1"}), second)
    session.propose(new_action)
    assert session.proposed_action is new_action
    session.close()


def test_duplicate_consumption_is_rejected(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click")
    observation = make_observation()
    session.ingest_observation(observation)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
    session.propose(action)
    session.consume()
    with pytest.raises(ActionRejected) as excinfo:
        session.consume(action.action_id)
    assert excinfo.value.reason is RejectReason.DUPLICATE_CONSUMPTION
    session.ingest_observation(feedback_observation(make_observation, action.action_id))
    with pytest.raises(ActionRejected) as excinfo:
        session.consume(action.action_id)
    assert excinfo.value.reason is RejectReason.DUPLICATE_CONSUMPTION
    dispatched = [
        event for event in read_events(session.evidence.events_path) if event["event_type"] == "dispatched"
    ]
    assert len(dispatched) == 1
    session.close()


def test_dispatched_action_requires_bound_feedback(
    make_observation, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click")
    observation = make_observation()
    session.ingest_observation(observation)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
    session.propose(action)
    session.consume()
    with pytest.raises(SessionStateError):
        session.ingest_observation(make_observation(observation_id="obs-2"))
    with pytest.raises(SessionStateError):
        session.ingest_observation(
            make_observation(
                observation_id="obs-3",
                previous_action_id="act-other",
                feedback=Feedback(
                    status=FeedbackStatus.OBSERVED,
                    action_id="act-other",
                    observation_id="obs-3",
                    message="wrong action",
                ),
            )
        )
    session.ingest_observation(feedback_observation(make_observation, action.action_id, "obs-4"))
    assert session.in_flight_action is None
    session.close()


def test_feedback_without_dispatch_is_rejected(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click")
    session.ingest_observation(make_observation())
    foreign = feedback_observation(make_observation, "act-foreign", "obs-2")
    with pytest.raises(SessionStateError):
        session.ingest_observation(foreign)
    session.close()


def test_reset_clears_previous_task(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("first")
    first = make_observation()
    session.ingest_observation(first)
    old_action = build_web_action(ToolRequest("web_click", {"target": "1"}), first)
    session.reset()
    assert session.state is SessionState.READY
    assert session.observation is None
    session.start_task("second")
    second = make_observation(observation_id="obs-2")
    session.ingest_observation(second)
    with pytest.raises(ActionRejected) as excinfo:
        session.propose(old_action)
    assert excinfo.value.reason is RejectReason.STALE_OBSERVATION
    new_action = build_web_action(ToolRequest("web_click", {"target": "1"}), second)
    session.propose(new_action)
    session.reset()
    session.start_task("third")
    page_two = make_observation(page_id="page-2", observation_id="obs-3")
    session.ingest_observation(page_two)
    assert session.page_id == "page-2"
    session.close()


def test_duplicate_observation_id_and_wrong_page(
    make_observation, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("observe")
    observation = make_observation()
    session.ingest_observation(observation)
    with pytest.raises(SessionStateError):
        session.ingest_observation(observation)
    with pytest.raises(SessionStateError):
        session.ingest_observation(make_observation(page_id="page-2", observation_id="obs-3"))
    session.close()


def test_finish_with_pending_action_rejected(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click")
    observation = make_observation()
    session.ingest_observation(observation)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
    session.propose(action)
    with pytest.raises(SessionStateError):
        session.finish(Finish(claim="done", message="x", reason="completed"))
    session.close()


def test_cancel_writes_terminal_result_without_fake_feedback(
    make_observation, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("click")
    observation = make_observation()
    session.ingest_observation(observation)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
    session.propose(action)
    session.cancel("user cancelled")
    assert session.state is SessionState.CANCELLED
    events = read_events(session.evidence.events_path)
    assert not any(event["event_type"] == "feedback" for event in events)
    assert any(event["event_type"] == "cancelled" for event in events)
    result = session.evidence.result_path.read_text(encoding="utf-8")
    assert "cancelled" in result
    session.close()
    assert session.state is SessionState.CLOSED


def test_model_call_events_and_count(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("call model")
    session.ingest_observation(make_observation())
    session.record_model_call(
        request_id="req-1",
        input_tokens=100,
        output_tokens=7,
        token_count_source="estimate:chars_div_4",
        latency_ms=12.5,
    )
    session.record_model_call(
        request_id=None,
        input_tokens=None,
        output_tokens=None,
        token_count_source="unknown",
        latency_ms=None,
        failure_type="transport",
    )
    assert session.model_calls_total == 2
    events = read_events(session.evidence.events_path)
    calls = [event for event in events if event["event_type"] == "model_call"]
    assert [event["model_calls_total"] for event in calls] == [1, 2]
    assert calls[0]["token_count_source"] == "estimate:chars_div_4"
    assert calls[1]["model_request_id"] is None
    session.close()


def test_task_notes_are_task_local(make_observation, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("notes")
    session.ingest_observation(make_observation())
    session.add_task_note("the result title is Sample")
    assert session.task_notes == ("the result title is Sample",)
    session.reset()
    assert session.task_notes == ()
    session.close()


def test_two_runs_do_not_share_state(
    make_observation, runtime: Path, clock: FakeClock
) -> None:
    first_session = Session("run-a", runtime_dir=runtime, clock=clock)
    first_session.start_task("first run")
    first_observation = make_observation(run_id="run-a", observation_id="obs-a-1")
    first_session.ingest_observation(first_observation)
    old_action = build_web_action(
        ToolRequest("web_click", {"target": "1"}), first_observation
    )
    first_session.cancel("simulate a new run")
    first_session.close()

    second_session = Session("run-b", runtime_dir=runtime, clock=clock)
    second_session.start_task("second run")
    second_session.ingest_observation(
        make_observation(run_id="run-b", observation_id="obs-b-1")
    )
    with pytest.raises(ActionRejected) as excinfo:
        second_session.propose(old_action)
    assert excinfo.value.reason is RejectReason.WRONG_RUN
    assert second_session.snapshot().observation_id == "obs-b-1"
    second_session.close()
    assert first_session.evidence.events_path != second_session.evidence.events_path


def test_deadline_blocks_actions_and_is_logged(make_observation, runtime: Path, clock: FakeClock) -> None:
    deadline = Deadline(clock.now() + 10, clock=clock)
    budget = Budget(BudgetLimits(), deadline=deadline, clock=clock)
    session = Session("run-a", runtime_dir=runtime, clock=clock, budget=budget)
    session.start_task("click")
    observation = make_observation()
    session.ingest_observation(observation)
    clock.advance(11)
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
    with pytest.raises(DeadlineExceeded):
        session.propose(action)
    assert session.proposed_action is None
    events = read_events(session.evidence.events_path)
    assert any(event["event_type"] == "deadline" for event in events)
    assert not any(event["event_type"] == "proposed" for event in events)
    session.close()
