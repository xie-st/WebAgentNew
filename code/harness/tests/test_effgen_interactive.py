"""A2 acceptance: the vendored interactive loop driven by the harness session.

Every test here is offline: a scripted EffGen model, the A1 fake site and backend,
a fake clock. No real model, browser or platform is contacted.
"""

from __future__ import annotations

import contextvars
import dataclasses
import json
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest
from effgen.core import agent as effgen_agent
from effgen.core.agent_config import AgentMode
from effgen.models.base import GenerationResult
from effgen.prompts.tool_contract import ToolUsePolicy

from webagent.budget import Budget, Deadline
from webagent.clock import FakeClock
from webagent.config import AppConfig
from webagent.effgen_bridge import (
    BridgeError,
    DriverStatus,
    EffGenLoopDriver,
    HarnessToolRuntime,
    LoopSettings,
    build_effgen_agent,
    build_effgen_config,
    build_harness_tools,
    create_driver,
)
from webagent.errors import ActionRejected
from webagent.events import read_events
from webagent.models.fake import ScriptedModel
from webagent.protocol import ActionOp, ArtifactSet, TerminationReason, ToolRequest
from webagent.session import Session, SessionState
from webagent.testing import FakeBackend, FakeSite
from webagent.tools import TOOL_SPECS, build_web_action

RUN = "run-a"
TASK = "find the sample result and open it"


class BatchThenTextModel(ScriptedModel):
    """One native tool-call batch, then the scripted text turns."""

    def __init__(self, calls: list[dict[str, Any]], responses: list[str]) -> None:
        super().__init__(responses)
        self._batch_calls = calls
        self._batch_sent = False

    def generate(self, prompt: str, config: Any = None, **kwargs: Any) -> GenerationResult:
        if not self._batch_sent:
            self._batch_sent = True
            self._record(prompt, config, None, None, None)
            return GenerationResult(
                text="",
                tokens_used=5,
                finish_reason="tool_calls",
                model_name=self.model_name,
                metadata={"tool_calls": self._batch_calls},
            )
        return super().generate(prompt, config, **kwargs)


class DeadlineAdvancingModel(ScriptedModel):
    """Each model turn spends ``seconds`` of the task's deadline."""

    def __init__(self, responses: list[str], clock: FakeClock, seconds: float) -> None:
        super().__init__(responses)
        self.clock = clock
        self.seconds = seconds

    def generate(self, prompt: str, config: Any = None, **kwargs: Any) -> GenerationResult:
        self.clock.advance(self.seconds)
        return super().generate(prompt, config, **kwargs)


class ScopeCheckingModel(ScriptedModel):
    """Records whether each model call ran inside an installed run scope."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.scoped: list[bool] = []

    def generate(self, prompt: str, config: Any = None, **kwargs: Any) -> GenerationResult:
        self.scoped.append(
            effgen_agent._run_scope_var.get() is not None
            and effgen_agent._call_state_var.get() is not None
        )
        return super().generate(prompt, config, **kwargs)


def _action(tool: str, arguments: Mapping[str, Any]) -> str:
    return (
        "Thought: next page step.\n"
        f"Action: {tool}\nAction Input: {json.dumps(dict(arguments))}"
    )


def _call(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"c-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(dict(arguments))},
    }


@pytest.fixture
def site(fixtures_dir: Path) -> FakeSite:
    return FakeSite.from_json(fixtures_dir / "fake_site.json")


@pytest.fixture
def backend(site: FakeSite) -> FakeBackend:
    return FakeBackend(RUN, site, start_url=site.default_url)


def _session(
    tmp_path: Path,
    clock: FakeClock,
    backend: FakeBackend,
    *,
    deadline_seconds: float | None = None,
) -> Session:
    budget = None
    if deadline_seconds is not None:
        budget = Budget(
            deadline=Deadline(clock.now() + deadline_seconds, clock=clock),
            clock=clock,
        )
    session = Session(
        RUN,
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        budget=budget,
    )
    session.start_task(TASK)
    session.ingest_observation(backend.current_observation)
    return session


def _driver(
    session: Session,
    model: ScriptedModel,
    *,
    settings: LoopSettings | None = None,
) -> EffGenLoopDriver:
    return create_driver(session, model, settings=settings)


def _events(session: Session) -> list[dict[str, Any]]:
    return read_events(session.evidence.events_path)


def _types(session: Session) -> list[str]:
    return [str(event["event_type"]) for event in _events(session)]


# --------------------------------------------------------------------- A03


def test_pause_resume_keeps_one_loop_and_records_only_after_feedback(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([
        _action("task_note", {"note": "wanted: the sample result"}),
        _action("web_type", {"target": "1", "text": "sample"}),
        _action("web_click", {"target": "2"}),
        _action("web_observe", {}),
        "Thought: found it.\nFinal Answer: opened the sample result",
    ])
    driver = _driver(session, model)
    agent = driver.agent

    # The internal note runs inside the same advance; the page call pauses it.
    step = driver.start()
    assert step.status is DriverStatus.ACTION
    assert step.request is not None and step.request.tool == "web_type"
    assert session.task_notes == ("wanted: the sample result",)
    assert backend.executed_actions == []
    assert session.observation is not None
    assert session.observation.observation_id == "obs-0"
    assert "feedback" not in _types(session)
    assert len([event for event in _events(session) if event["event_type"] == "dispatched"]) == 0
    # The action is proposed but not consumed: the environment owes it.
    assert session.proposed_action is not None
    assert session.in_flight_action is None

    action = driver.take_action()
    assert action.op is ActionOp.TYPE
    assert session.in_flight_action is not None
    assert backend.executed_actions == [], "take_action only hands the action over"
    dispatched = [event for event in _events(session) if event["event_type"] == "dispatched"]
    assert len(dispatched) == 1 and dispatched[0]["result"] is None

    result = backend.execute(action)
    step = driver.accept_observation(result.observation)
    assert step.status is DriverStatus.ACTION
    assert step.request is not None and step.request.tool == "web_click"
    assert driver.agent is agent, "the same agent and loop survive every send"
    assert session.task == TASK
    assert session.state is SessionState.RUNNING
    assert len(backend.executed_actions) == 1
    # The resumed prompt is the real observation the environment returned: the
    # typed value and the feedback for the action the model just made.
    assert result.observation.observation_id in model.prompts[2]
    assert "value='sample'" in model.prompts[2]
    assert "typed into element 1" in model.prompts[2]
    assert len([event for event in _events(session) if event["event_type"] == "feedback"]) == 1

    second = driver.take_action()
    second_result = backend.execute(second)
    step = driver.accept_observation(second_result.observation)

    assert step.status is DriverStatus.COMPLETED
    assert step.response is not None and step.response.success is True
    assert driver.agent is agent
    assert len(model.calls) == 5
    assert "opened the sample result" in step.message
    # The observation from the second send reached the last model input, and the
    # note's result is still in the same conversation.
    assert second_result.observation.observation_id in model.prompts[3]
    assert "Results for the submitted query." in model.prompts[3]
    assert any("Saved task note" in prompt for prompt in model.prompts[1:])

    thread = step.response.metadata["thread"]
    kinds = [type(item).__name__ for item in thread.steps]
    assert kinds.count("TaskStep") == 1
    actions = [item.tool for item in thread.steps if type(item).__name__ == "ActionStep"]
    assert actions == ["task_note", "web_type", "web_click", "web_observe"]
    observations = [
        item.text
        for item in thread.steps
        if type(item).__name__ == "ObservationStep"
    ]
    assert any(result.observation.observation_id in text for text in observations)
    assert step.response.metadata["interactive_requests"] == 2

    assert session.finish_result is not None
    assert session.finish_result.reason is TerminationReason.COMPLETED
    assert _types(session).count("agent_claim") == 1
    session.close()
    driver.close()
    driver.close()
    assert json.loads(session.evidence.result_path.read_text(encoding="utf-8"))[
        "termination"
    ] == "completed"


def test_pause_drops_an_unconsumed_proposal_without_a_feedback(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([_action("web_click", {"target": "2"})])
    driver = _driver(session, model)
    step = driver.start()
    assert step.status is DriverStatus.ACTION
    session.pause()
    assert session.proposed_action is None
    assert session.state is SessionState.PAUSED
    assert backend.executed_actions == []
    fresh = backend.observe()
    session.resume(fresh)
    assert session.state is SessionState.RUNNING
    assert session.observation is not None and session.observation.observation_id == fresh.observation_id
    driver.close()
    session.close()


# --------------------------------------------------------------------- A04


def test_loop_path_rejects_an_unknown_number_and_sends_the_correction(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([
        _action("web_click", {"target": "99"}),
        _action("web_click", {"target": "2"}),
        "Thought: opened.\nFinal Answer: done",
    ])
    driver = _driver(session, model)
    step = driver.start()

    assert step.status is DriverStatus.ACTION
    assert step.corrections == 1
    assert backend.executed_actions == [], "a refused action is never executed"
    assert len(model.calls) == 2, "the correction earned one more model turn"
    correction_text = model.prompts[1]
    assert "unknown_element" in correction_text
    assert "observation_id=obs-0" in correction_text
    assert "1, 2" in correction_text

    events = [event for event in _events(session) if event["event_type"] == "format_correction"]
    assert len(events) == 1
    assert events[0]["result"]["reason"] == "unknown_element"
    assert events[0]["result"]["request_id"] == "ireq-1"
    assert step.request is not None and step.request.request_id != "ireq-1"

    action = driver.take_action()
    result = backend.execute(action)
    step = driver.accept_observation(result.observation)
    assert step.status is DriverStatus.COMPLETED
    assert [item.args["target"] for item in backend.executed_actions] == ["2"]
    assert len([event for event in _events(session) if event["event_type"] == "dispatched"]) == 1
    session.close()


def test_two_corrections_then_the_run_stops(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([
        _action("web_click", {"target": "90"}),
        _action("web_click", {"target": "91"}),
        _action("web_click", {"target": "92"}),
        _action("web_click", {"target": "93"}),
    ])
    driver = _driver(session, model, settings=LoopSettings(max_format_corrections=2))
    step = driver.start()

    assert step.status is DriverStatus.FORMAT_ERROR
    assert step.corrections == 3
    assert len(model.calls) == 3, "at most two correction calls are appended"
    assert backend.executed_actions == []
    assert session.finish_result is not None
    assert session.finish_result.reason is TerminationReason.ERROR
    assert _types(session).count("format_correction") == 3
    session.close()


def test_a_batch_of_page_actions_is_refused_before_dispatch(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = BatchThenTextModel(
        [_call("web_click", {"target": "1"}), _call("web_click", {"target": "2"})],
        ["Thought: retry.\nFinal Answer: the batch was refused"],
    )
    driver = _driver(session, model)
    step = driver.start()

    assert step.status is DriverStatus.COMPLETED
    assert backend.executed_actions == []
    assert "one model output contained 2 web actions" in model.prompts[1]
    assert "observation_id=obs-0" in model.prompts[1]
    batch_events = [
        event
        for event in _events(session)
        if event["event_type"] == "format_correction" and event["result"]["scope"] == "batch"
    ]
    assert len(batch_events) == 1
    assert batch_events[0]["result"]["reason"] == "multiple_actions"
    assert batch_events[0]["result"]["tools"] == ["web_click", "web_click"]
    session.close()


def test_the_same_number_is_a_new_action_in_a_new_observation(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([
        _action("web_click", {"target": "1"}),
        _action("web_click", {"target": "2"}),
        _action("web_click", {"target": "1"}),
        "Thought: opened.\nFinal Answer: opened the detail page",
    ])
    driver = _driver(session, model)
    step = driver.start()
    while step.status is DriverStatus.ACTION:
        action = driver.take_action()
        assert action.run_id == RUN
        assert session.observation is not None
        assert action.observation_id == session.observation.observation_id
        result = backend.execute(action)
        step = driver.accept_observation(result.observation)

    assert step.status is DriverStatus.COMPLETED
    assert [action.args["target"] for action in backend.executed_actions] == ["1", "2", "1"]
    observation_ids = {action.observation_id for action in backend.executed_actions}
    assert len(observation_ids) == 3, "each action bound to its own observation"
    assert backend.current_url == "http://fixture.local/detail"
    session.close()


def test_a_consumed_action_cannot_be_taken_twice(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([_action("web_click", {"target": "2"})])
    driver = _driver(session, model)
    driver.start()
    driver.take_action()
    with pytest.raises(BridgeError):
        driver.take_action()
    assert backend.executed_actions == []
    driver.close()
    session.close()


def test_two_presses_on_an_unchanged_page_are_two_steps(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    """Identical page content is still a new observation, so the step is taken."""

    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([
        _action("web_press", {"keys": "Enter"}),
        _action("web_press", {"keys": "Enter"}),
        "Thought: both presses went through.\nFinal Answer: pressed twice",
    ])
    driver = _driver(session, model)
    step = driver.start()
    first = backend.execute(driver.take_action())
    step = driver.accept_observation(first.observation)
    second = backend.execute(driver.take_action())
    step = driver.accept_observation(second.observation)

    assert step.status is DriverStatus.COMPLETED
    assert first.observation.ax_text == second.observation.ax_text
    assert first.observation.observation_id != second.observation.observation_id
    assert len(backend.executed_actions) == 2
    assert len([event for event in _events(session) if event["event_type"] == "dispatched"]) == 2
    assert len([event for event in _events(session) if event["event_type"] == "feedback"]) == 2
    session.close()


# --------------------------------------------------------------------- A05


def test_bridge_configuration_is_the_phase_one_configuration() -> None:
    model = ScriptedModel([])
    config = build_effgen_config(model, [], app_config=AppConfig(profile="fake", backend="fake"))
    assert config.mode is AgentMode.SINGLE
    assert config.tool_calling_mode == "react"
    assert config.capability_probe is False
    assert config.enable_sub_agents is False
    assert config.enable_fallback is False
    assert config.speculative_execution is False
    assert config.enable_memory is False
    assert config.models is None
    assert config.tool_use is ToolUsePolicy.AUTO


def test_harness_tools_follow_the_one_schema() -> None:
    runtime = HarnessToolRuntime()
    tools = build_harness_tools(runtime)
    assert [tool.name for tool in tools] == [spec.name for spec in TOOL_SPECS]
    note = next(tool for tool in tools if tool.name == "task_note")
    # Internal tools answer synchronously: no event loop and no worker thread.
    unbound = note.execute(note="offline")
    assert unbound.success is False
    page = next(tool for tool in tools if tool.name == "web_click")
    refused = page.execute(target="1")
    assert refused.success is False
    assert "executes in the environment" in (refused.error or "")


def test_ordinary_run_executes_internal_tools_but_refuses_page_actions(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    runtime = HarnessToolRuntime(session)
    model = ScriptedModel([
        _action("task_note", {"note": "ordinary path works"}),
        "Thought: done.\nFinal Answer: noted",
    ])
    agent = build_effgen_agent(model, runtime=runtime)
    response = agent.run("write a note")
    assert response.success is True
    assert session.task_notes == ("ordinary path works",)
    assert "Saved task note" in model.prompts[1]

    page_model = ScriptedModel([
        _action("web_click", {"target": "2"}),
        "Thought: could not act.\nFinal Answer: the page tool did not run",
    ])
    page_agent = build_effgen_agent(page_model, runtime=runtime)
    page_response = page_agent.run("click the search button")
    assert page_response.success is True
    assert backend.executed_actions == []
    assert "executes in the environment" in page_model.prompts[1]
    session.close()


def test_web_observe_does_not_re_observe(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    runtime = HarnessToolRuntime(session)
    first = runtime.observe()
    second = runtime.observe()
    assert first == second
    assert "observation_id=obs-0" in first
    assert backend.current_observation is not None
    assert backend.current_observation.observation_id == "obs-0"


def test_evidence_save_copies_the_artifact_already_captured(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = Session(RUN, runtime_dir=tmp_path / "runtime", clock=clock)
    session.start_task(TASK)
    html = session.evidence.save_text("html", "<html>fixture</html>", observation_id="obs-0")
    observation = dataclasses.replace(
        backend.current_observation, artifacts=ArtifactSet(html=html)
    )
    session.ingest_observation(observation)
    runtime = HarnessToolRuntime(session)
    model = ScriptedModel([
        _action("evidence_save", {"kind": "html", "label": "a2"}),
        "Thought: saved.\nFinal Answer: evidence saved",
    ])
    agent = build_effgen_agent(model, runtime=runtime)
    response = agent.run("save the captured html")
    assert response.success is True
    assert "Saved html evidence" in model.prompts[1]
    saved_refs = [
        path for path in (session.evidence.run_dir / "html").iterdir() if path.is_file()
    ]
    assert len(saved_refs) == 2, "the captured file and the labelled copy"
    session.close()


# --------------------------------------------------------------------- A07


def test_two_callback_scopes_advance_one_generator_without_leaking(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScopeCheckingModel([
        _action("web_click", {"target": "2"}),
        _action("web_click", {"target": "1"}),
        "Thought: opened.\nFinal Answer: detail open",
    ])
    driver = _driver(session, model)
    agent = driver.agent
    threads_before = threading.active_count()
    observed: list[dict[str, bool]] = []

    def callback(next_step: Any) -> Any:
        def run() -> Any:
            outcome = next_step()
            observed.append({
                "scope_installed_during_call": all(model.scoped),
                "scope_left_after_advance": effgen_agent._run_scope_var.get() is None,
                "call_state_left_after_advance": effgen_agent._call_state_var.get() is None,
                "agent_session_left_after_advance": agent.session is None,
            })
            return outcome
        return run

    context_a = contextvars.Context()
    step = context_a.run(callback(driver.start))
    assert step.status is DriverStatus.ACTION

    action = driver.take_action()
    result = backend.execute(action)
    context_b = contextvars.Context()
    step = context_b.run(callback(lambda: driver.accept_observation(result.observation)))
    assert step.status is DriverStatus.ACTION

    action = driver.take_action()
    result = backend.execute(action)
    context_c = contextvars.Context()
    step = context_c.run(callback(lambda: driver.accept_observation(result.observation)))
    assert step.status is DriverStatus.COMPLETED

    assert observed == [
        {
            "scope_installed_during_call": True,
            "scope_left_after_advance": True,
            "call_state_left_after_advance": True,
            "agent_session_left_after_advance": True,
        }
    ] * 3
    assert model.scoped == [True, True, True]
    # The test's own context saw none of the callbacks' bindings.
    assert effgen_agent._run_scope_var.get() is None
    assert effgen_agent._call_state_var.get() is None
    calls = len(model.calls)
    driver.close()
    session.close()
    assert len(model.calls) == calls, "closing the loop calls no model"
    assert threading.active_count() == threads_before, "the pause starts no background thread"


# --------------------------------------------------------------------- A08


def test_more_than_ten_page_actions_are_not_cut_and_identical_scrolls_both_run(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScriptedModel(
        [_action("web_scroll", {"direction": "down"}) for _ in range(12)]
        + ["Thought: done.\nFinal Answer: scrolled twelve times"]
    )
    driver = _driver(session, model)
    step = driver.start()
    while step.status is DriverStatus.ACTION:
        action = driver.take_action()
        assert action.op is ActionOp.SCROLL
        result = backend.execute(action)
        step = driver.accept_observation(result.observation)

    assert step.status is DriverStatus.COMPLETED
    assert len(backend.executed_actions) == 12
    assert [action.args["direction"] for action in backend.executed_actions] == ["down"] * 12
    assert len({action.observation_id for action in backend.executed_actions}) == 12
    assert len([event for event in _events(session) if event["event_type"] == "dispatched"]) == 12
    assert len([event for event in _events(session) if event["event_type"] == "feedback"]) == 12
    assert len(model.calls) == 13
    session.close()


def test_deadline_stops_model_calls_and_actions(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend, deadline_seconds=60)
    model = ScriptedModel([
        _action("web_click", {"target": "2"}),
        _action("web_click", {"target": "1"}),
        "Thought: done.\nFinal Answer: never reached",
    ])
    driver = _driver(session, model)
    step = driver.start()
    assert step.status is DriverStatus.ACTION
    action = driver.take_action()

    clock.advance(120)
    result = backend.execute(action)
    step = driver.accept_observation(result.observation)

    assert step.status is DriverStatus.DEADLINE
    assert len(model.calls) == 1, "no model call after the deadline"
    assert len(backend.executed_actions) == 1, "no action produced after the deadline"
    assert "deadline" in _types(session)
    assert session.finish_result is not None
    assert session.finish_result.reason is TerminationReason.DEADLINE
    session.close()


def test_internal_spin_ends_at_the_deadline(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend, deadline_seconds=50)
    model = DeadlineAdvancingModel(
        [_action("task_note", {"note": f"note {index}"}) for index in range(20)],
        clock,
        seconds=20,
    )
    driver = _driver(session, model)
    step = driver.start()

    assert step.status is DriverStatus.DEADLINE
    assert len(model.calls) <= 4, "the deadline bounds an internal-only loop"
    assert backend.executed_actions == []
    assert session.finish_result is not None
    assert session.finish_result.reason is TerminationReason.DEADLINE
    assert "deadline" in _types(session)
    assert len([event for event in _events(session) if event["event_type"] == "proposed"]) == 0
    session.close()


def test_cancel_stops_before_the_environment_executes(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([_action("web_click", {"target": "2"})])
    driver = _driver(session, model)
    step = driver.start()
    assert step.status is DriverStatus.ACTION

    cancelled = driver.cancel("user takeover")
    assert cancelled.status is DriverStatus.CANCELLED
    assert backend.executed_actions == []
    assert session.state is SessionState.CANCELLED
    assert "agent_claim" not in _types(session)
    driver.close()
    assert driver.close() is None
    session.close()


# ---------------------------------------------------------- bridge contract


def test_backend_driver_requires_a_running_bound_session(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    runtime = HarnessToolRuntime(session)
    agent = build_effgen_agent(ScriptedModel([]), runtime=runtime)
    other = HarnessToolRuntime(Session(RUN + "-other", runtime_dir=tmp_path / "other", clock=clock))
    with pytest.raises(BridgeError):
        EffGenLoopDriver(session, agent=agent, runtime=other)

    driver = EffGenLoopDriver(session, agent=agent, runtime=runtime)
    with pytest.raises(BridgeError):
        driver.take_action()

    driver2 = create_driver(session, ScriptedModel([_action("web_click", {"target": "2"})]))
    driver2.start()
    driver2.close()
    session.close()


def test_more_than_one_page_action_from_the_guard_is_reported_not_queued(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    """The guard the batch hook calls is the same one the request path uses."""

    session = _session(tmp_path, clock, backend)
    driver = _driver(session, ScriptedModel([]))
    calls = (("web_click", {"target": "1"}), ("web_type", {"target": "2", "text": "x"}))
    correction = driver._batch_correction(calls)  # noqa: SLF001 - the hook contract
    assert "multiple_actions" in correction
    assert "one model output contained 2 web actions" in correction
    assert "sent alone" not in correction
    assert backend.executed_actions == []
    session.close()


# ---------------------------------------------------------- B04 pause/resume


def test_resume_discards_old_binding_and_continues_the_same_loop(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    """Pause before input, re-observe, and never replay the old action."""

    session = _session(tmp_path, clock, backend)
    model = ScriptedModel([
        _action("web_type", {"target": "1", "text": "OLD"}),
        _action("web_type", {"target": "1", "text": "NEW"}),
        "Thought: resumed safely.\nFinal Answer: resumed and typed NEW",
    ])
    driver = _driver(session, model)
    agent = driver.agent
    step = driver.start()
    assert step.status is DriverStatus.ACTION
    old_action = driver.pending_action
    assert old_action is not None and old_action.args["text"] == "OLD"
    assert backend.executed_actions == []

    session.pause()
    assert session.state is SessionState.PAUSED
    assert session.proposed_action is None
    fresh = backend.observe()
    assert fresh.observation_id == "obs-1"
    step = driver.resume(fresh)

    assert step.status is DriverStatus.ACTION
    assert session.observation is not None
    assert session.observation.observation_id == fresh.observation_id
    assert old_action.observation_id == "obs-0"
    assert "NOT EXECUTED" in model.prompts[1]
    assert fresh.observation_id in model.prompts[1]
    assert backend.executed_actions == [], "the paused action must not be replayed"

    action = driver.take_action()
    result = backend.execute(action)
    step = driver.accept_observation(result.observation)
    assert step.status is DriverStatus.COMPLETED
    assert [item.args["text"] for item in backend.executed_actions] == ["NEW"]
    assert action.observation_id == fresh.observation_id
    assert driver.agent is agent
    session.close()


def test_resume_makes_an_old_action_rejected(
    backend: FakeBackend, clock: FakeClock, tmp_path: Path
) -> None:
    session = _session(tmp_path, clock, backend)
    old = build_web_action(ToolRequest("web_click", {"target": "2"}), session.observation)
    session.pause()
    fresh = backend.observe()
    session.resume(fresh)
    with pytest.raises(ActionRejected) as error:
        session.propose(old)
    assert error.value.reason.value == "stale_observation"
    assert backend.executed_actions == []
    session.close()

