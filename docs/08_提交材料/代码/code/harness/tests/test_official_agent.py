"""C01-C03 integration: OfficialAgent driven by fake SDK callbacks only.

The fake platform supplies request dictionaries and consumes the returned action
text; the fake gateway returns scripted ReAct responses. No platform, network,
credential file or browser is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from webagent.backends.official import TextMode
from webagent.clock import FakeClock
from webagent.config import (
    AppConfig,
    BudgetConfig,
    LimitsConfig,
    ModelConfig,
    OfficialSettings,
)
from webagent.events import read_events
from webagent.official_agent import OfficialAgent, plan_type_text
from webagent.protocol import ActionOp

RUN = "official-run-1"
TASK = "click search, type the note, then finish"


class FakeGateway:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> Any:
        self.calls.append(messages)
        if not self.responses:
            raise RuntimeError("fake official gateway exhausted")
        return SimpleNamespace(text=self.responses.pop(0), request_id=f"req-{len(self.calls)}")


def _config(
    tmp_path: Path,
    *,
    text_mode: str = "decompose",
    deadline_margin_seconds: float = 0.0,
    max_messages: int = 256,
    max_request_bytes: int = 1024 * 1024,
) -> AppConfig:
    return AppConfig(
        profile="official",
        backend="official",
        runtime_dir=str(tmp_path / "runtime"),
        model=ModelConfig(
            provider="official",
            name="platform-unified",
            context_length=32768,
            tokenizer="heuristic",
            token_count_source="estimated",
            base_url_env=None,
            api_key_env=None,
        ),
        budget=BudgetConfig(deadline_margin_seconds=deadline_margin_seconds),
        limits=LimitsConfig(
            max_messages=max_messages,
            max_request_bytes=max_request_bytes,
        ),
        official=OfficialSettings(text_mode=text_mode),
    )


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return (
        "Thought: use the current page.\n"
        f"Action: {tool}\n"
        f"Action Input: {json.dumps(arguments, ensure_ascii=False)}"
    )


def _request(
    *,
    step: int,
    text: str,
    url: str,
    deadline: float,
    task: str | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "protocol_version": "1.0",
        "run_id": RUN,
        "step": step,
        "observation": {
            "type": "accessibility_tree",
            "url": url,
            "text": text,
        },
        "history": [],
        "action_history": [],
        "execution_deadline": deadline,
    }
    if task is not None:
        request["task"] = task
    return request


def _make_agent(
    tmp_path: Path,
    clock: FakeClock,
    gateway: FakeGateway,
    *,
    config: AppConfig | None = None,
) -> OfficialAgent:
    return OfficialAgent(
        app_config=config or _config(tmp_path),
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        model_api=gateway,
        download_artifacts=frozenset(),
    )


def test_reset_step_close_with_one_generator_and_one_action_per_step(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_click", {"target": "2"}),
            _action(
                "web_type",
                {"target": "1", "text": "价格（含税）\n第二行 [备注]", "submit": False},
            ),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    assert agent.session is None and agent.driver is None

    first = agent.step(
        _request(
            step=0,
            task=TASK,
            deadline=clock.now() + 120,
            url="http://forum.test/search",
            text="[1] textbox 'Query'\n[2] button 'Search'",
        )
    )
    assert first == {"action": "click [2]"}
    driver = agent.driver
    assert driver is not None
    assert agent.text_mode is TextMode.DECOMPOSE
    same_agent = driver.agent
    assert agent.session is not None
    assert agent.session.state.value == "running"

    # The multi-line type proposal is expanded locally: one sub-action per
    # platform step, with the line break carried by an explicit Enter press.
    second = agent.step(
        _request(
            step=1,
            deadline=clock.now() + 120,
            url="http://forum.test/result",
            text="[1] textbox 'Note'\n[2] button 'Save'",
        )
    )
    assert second == {"action": "type [1] [价格（含税）] [0]"}
    assert "\n" not in second["action"]

    third = agent.step(
        _request(
            step=2,
            deadline=clock.now() + 120,
            url="http://forum.test/result",
            text="[1] textbox 'Note'\n[2] button 'Save'",
        )
    )
    assert third == {"action": "press [Enter]"}

    fourth = agent.step(
        _request(
            step=3,
            deadline=clock.now() + 120,
            url="http://forum.test/result",
            text="[1] textbox 'Note'\n[2] button 'Save'",
        )
    )
    assert fourth == {"action": "type [1] [第二行 [备注]] [0]"}
    assert agent.driver is driver and driver.agent is same_agent

    fifth = agent.step(
        _request(
            step=4,
            deadline=clock.now() + 120,
            url="http://forum.test/result",
            text="[1] link 'Done'",
        )
    )
    assert fifth == {"action": "stop [done]"}
    assert agent.finished

    session = agent.session
    assert session is not None
    local_run_id = agent.local_run_id
    agent.close()
    agent.close()
    assert agent.session is None and agent.driver is None
    # Only the final sub-action's observation is sent back to the model, so the
    # three proposals (click, multiline type, final answer) are three calls.
    assert len(gateway.calls) == 3
    final_prompt = gateway.calls[2][0]["content"]
    assert "obs-4" in final_prompt
    assert "obs-2" not in final_prompt and "obs-3" not in final_prompt

    events = read_events(tmp_path / "runtime" / "runs" / str(local_run_id) / "events.jsonl")
    proposed = [event for event in events if event["event_type"] == "proposed"]
    dispatched = [event for event in events if event["event_type"] == "dispatched"]
    feedback = [event for event in events if event["event_type"] == "feedback"]
    model_calls = [event for event in events if event["event_type"] == "model_call"]
    claims = [event for event in events if event["event_type"] == "agent_claim"]
    expanded = [event for event in events if event["event_type"] == "expanded"]
    subactions = [event for event in events if event["event_type"] == "subaction"]
    assert len(expanded) == 1
    assert expanded[0]["result"]["sub_action_count"] == 3
    assert [event["result"]["delivered_to_model"] for event in subactions] == [
        False,
        False,
        True,
    ]
    # Five proposals: click, the model's multi-line proposal (expanded, never
    # dispatched), and its three sub-actions. Only the sub-actions dispatch.
    assert len(proposed) == 5
    assert len(dispatched) == 4
    assert len(feedback) == 4
    parent_id = expanded[0]["result"]["parent_action_id"]
    assert parent_id not in {event["action_id"] for event in dispatched}
    parent_proposal = next(
        event for event in proposed if event["action_id"] == parent_id
    )
    assert "\n" in parent_proposal["action_arguments"]["text"]
    assert len(model_calls) == 3
    assert len(claims) == 1
    assert claims[0]["result"]["reason"] == "completed"
    assert "score" not in claims[0]["result"]
    assert all(event["model_input_tokens"] is None for event in model_calls)
    assert all(event["model_output_tokens"] is None for event in model_calls)
    assert all(event["model_request_id"].startswith("req-") for event in model_calls)
    result = json.loads(
        (tmp_path / "runtime" / "runs" / str(local_run_id) / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["termination"] == "completed"
    assert "score" not in result
    assert result["summary"]["steps_completed"] == 4
    assert result["summary"]["last_action"]["op"] == "type"
    assert result["summary"]["model_calls"] == {
        "platform_calls": 3,
        "local_attempts": 3,
        "failures": 0,
    }
    run_end = [event for event in events if event["event_type"] == "run_end"]
    assert len(run_end) == 1 and run_end[0]["result"]["termination"] == "completed"


def test_internal_tool_runs_inside_the_same_step_before_one_page_action(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("task_note", {"note": "the first page has a search button"}),
            _action("web_click", {"target": "2"}),
            _action("web_type", {"target": "1", "text": "notes", "submit": False}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'",
            )
        )
        assert first == {"action": "click [2]"}
        assert len(gateway.calls) == 2
        session = agent.session
        assert session is not None
        assert session.task_notes == ("the first page has a search button",)
        events = read_events(session.evidence.events_path)
        assert len([event for event in events if event["event_type"] == "proposed"]) == 1
        assert len([event for event in events if event["event_type"] == "model_call"]) == 2

        second = agent.step(
            _request(
                step=1,
                deadline=clock.now() + 120,
                url="http://forum.test/result",
                text="[1] textbox 'Note'",
            )
        )
        assert second == {"action": "type [1] [notes] [0]"}
        third = agent.step(
            _request(
                step=2,
                deadline=clock.now() + 120,
                url="http://forum.test/result",
                text="[1] link 'Done'",
            )
        )
        assert third == {"action": "stop [done]"}
    finally:
        agent.close()


def test_deadline_before_first_step_does_no_model_or_download_work(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(["must not be called"])
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    result = agent.step(
        _request(
            step=0,
            task=TASK,
            deadline=clock.now() - 1,
            url="http://forum.test/search",
            text="[1] textbox 'Query'",
        )
    )
    assert result["action"].startswith("stop [")
    assert "deadline" in result["action"]
    assert gateway.calls == []
    assert agent.session is None
    agent.close()
    agent.close()


def test_run_mismatch_returns_a_visible_stop_not_a_platform_action(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(["must not be called"])
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    result = agent.step(
        {
            **_request(
                step=0,
                task=TASK,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'",
            ),
            "run_id": "different-run",
        }
    )
    assert result["action"].startswith("stop [")
    assert "local error" in result["action"]
    assert gateway.calls == []
    agent.close()

# ------------------------------------------------- C batch interface fixes
def test_plan_type_text_keeps_line_break_semantics() -> None:
    parts = plan_type_text("A\nB", submit=False)
    assert [(part.op, dict(part.args)) for part in parts] == [
        (ActionOp.TYPE, {"text": "A", "submit": False}),
        (ActionOp.PRESS, {"keys": "Enter"}),
        (ActionOp.TYPE, {"text": "B", "submit": False}),
    ]
    submitted = plan_type_text("A\r\nB", submit=True)
    assert [part.op for part in submitted] == [ActionOp.TYPE, ActionOp.PRESS, ActionOp.TYPE]
    assert submitted[-1].args["submit"] is True
    # A trailing line break plus submit needs its own Enter after the empty line.
    trailing = plan_type_text("A\n", submit=True)
    assert [part.op for part in trailing] == [ActionOp.TYPE, ActionOp.PRESS, ActionOp.PRESS]


def test_raw_text_mode_sends_the_line_break_in_one_type_command(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_type", {"target": "1", "text": "first\nsecond", "submit": False}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway, config=_config(tmp_path, text_mode="raw"))
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Note'",
            )
        )
        assert first == {"action": "type [1] [first\nsecond] [0]"}
        assert "\n" in first["action"] and r"\n" not in first["action"]
        assert agent.text_mode is TextMode.RAW
    finally:
        agent.close()


def test_split_stops_at_the_deadline_recording_only_the_executed_prefix(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [_action("web_type", {"target": "1", "text": "A\nB", "submit": False})]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Note'",
            )
        )
        assert first == {"action": "type [1] [A] [0]"}
        local_run_id = agent.local_run_id
        clock.advance(130)
        second = agent.step(
            _request(
                step=1,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Note'",
            )
        )
        assert second["action"].startswith("stop [")
        assert "1 of 3" in second["action"]
        assert len(gateway.calls) == 1
    finally:
        agent.close()

    run_dir = tmp_path / "runtime" / "runs" / str(local_run_id)
    events = read_events(run_dir / "events.jsonl")
    aborted = [
        event
        for event in events
        if event["event_type"] == "subaction" and event["result"].get("aborted")
    ]
    assert len(aborted) == 1
    assert aborted[0]["result"]["executed"] == 1
    assert aborted[0]["result"]["total"] == 3
    assert aborted[0]["result"]["remaining"] == 2
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["termination"] == "deadline"
    assert result["summary"]["steps_completed"] == 1
    assert "1 of 3" in result["summary"]["unfinished"]
    run_end = [event for event in events if event["event_type"] == "run_end"]
    assert len(run_end) == 1 and run_end[0]["result"]["termination"] == "deadline"


def test_split_stops_on_an_error_and_keeps_the_executed_prefix(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [_action("web_type", {"target": "1", "text": "A\nB", "submit": False})]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Note'",
            )
        )
        assert first == {"action": "type [1] [A] [0]"}
        # The element the rest of the text targets is gone from the next page.
        second = agent.step(
            _request(
                step=1,
                deadline=clock.now() + 120,
                url="http://forum.test/next",
                text="[2] button 'Other'",
            )
        )
        assert second == {"action": "press [Enter]"}
        local_run_id = agent.local_run_id
        third = agent.step(
            _request(
                step=2,
                deadline=clock.now() + 120,
                url="http://forum.test/next",
                text="[2] button 'Other'",
            )
        )
        assert third["action"].startswith("stop [")
        assert "2 of 3" in third["action"]
    finally:
        agent.close()

    run_dir = tmp_path / "runtime" / "runs" / str(local_run_id)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["termination"] == "error"
    assert "2 of 3" in result["summary"]["unfinished"]
    assert "executed 2 of 3" in result["finish"]["claim"]
    events = read_events(run_dir / "events.jsonl")
    aborted = [
        event
        for event in events
        if event["event_type"] == "subaction" and event["result"].get("aborted")
    ]
    assert len(aborted) == 1
    assert aborted[0]["result"]["executed"] == 2
    assert aborted[0]["result"]["remaining"] == 1
    assert aborted[0]["result"]["reason"] == "error"


def test_default_allows_model_call_until_platform_deadline(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway([_action("web_click", {"target": "2"})])
    agent = _make_agent(tmp_path, clock, gateway)
    deadline = clock.now() + 1
    agent.reset({"run_id": RUN, "execution_deadline": deadline})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=deadline,
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'",
            )
        )
        assert first == {"action": "click [2]"}
        assert len(gateway.calls) == 1
        local_run_id = agent.local_run_id
        clock.advance(1)
        second = agent.step(
            _request(
                step=1,
                deadline=deadline,
                url="http://forum.test/result",
                text="[1] link 'Done'",
            )
        )
        assert second["action"].startswith("stop [")
        assert "deadline" in second["action"]
        assert len(gateway.calls) == 1, "no new model call at the hard deadline"
    finally:
        agent.close()
    run_dir = tmp_path / "runtime" / "runs" / str(local_run_id)
    events = read_events(run_dir / "events.jsonl")
    assert not any(event["event_type"] == "deadline_margin" for event in events)


def test_deadline_margin_closes_out_without_a_new_model_call(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_click", {"target": "2"}),
            _action("web_click", {"target": "1"}),
        ]
    )
    agent = _make_agent(
        tmp_path, clock, gateway, config=_config(tmp_path, deadline_margin_seconds=30.0)
    )
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 300})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 300,
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'",
            )
        )
        assert first == {"action": "click [2]"}
        local_run_id = agent.local_run_id
        clock.advance(285)  # 15 s left, below the 30 s margin
        second = agent.step(
            _request(
                step=1,
                deadline=clock.now() + 15,
                url="http://forum.test/result",
                text="[1] link 'Done'",
            )
        )
        assert second["action"].startswith("stop [")
        assert "margin" in second["action"]
        assert len(gateway.calls) == 1, "no model call inside the deadline margin"
    finally:
        agent.close()

    run_dir = tmp_path / "runtime" / "runs" / str(local_run_id)
    events = read_events(run_dir / "events.jsonl")
    assert any(event["event_type"] == "deadline_margin" for event in events)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["termination"] == "deadline"
    assert result["summary"]["steps_completed"] == 1
    assert "deadline margin" in result["summary"]["unfinished"]
    run_end = [event for event in events if event["event_type"] == "run_end"]
    assert len(run_end) == 1 and run_end[0]["result"]["termination"] == "deadline"
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["termination"] == "deadline"
    assert status["model_calls"] == {
        "platform_calls": 1,
        "local_attempts": 1,
        "failures": 0,
    }


def test_hard_kill_leaves_a_status_file_for_reconciliation(tmp_path: Path) -> None:
    clock = FakeClock()

    class InterruptingGateway(FakeGateway):
        def generate(self, messages: list[dict[str, str]]) -> Any:
            self.calls.append(messages)
            if len(self.calls) == 2:
                raise KeyboardInterrupt("simulated SDK kill at the deadline")
            return SimpleNamespace(text=self.responses.pop(0), request_id=f"req-{len(self.calls)}")

    gateway = InterruptingGateway([_action("web_click", {"target": "2"})])
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 300})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 300,
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'",
            )
        )
        assert first == {"action": "click [2]"}
        local_run_id = agent.local_run_id
        # The process dies inside the second model call: nothing runs afterwards.
        with pytest.raises(KeyboardInterrupt):
            agent.step(
                _request(
                    step=1,
                    deadline=clock.now() + 300,
                    url="http://forum.test/result",
                    text="[1] link 'Done'",
                )
            )
        run_dir = tmp_path / "runtime" / "runs" / str(local_run_id)
        assert not (run_dir / "result.json").exists(), "a killed run has no terminal result"
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        assert status["state"] == "running"
        assert status["termination"] is None
        assert status["steps"] == {"completed": 1, "proposed": 1, "dispatched": 1}
        assert status["last_action"]["op"] == "click"
        assert status["last_event"]["event_type"] in {"observation", "feedback"}
        assert status["model_calls"] == {
            "platform_calls": 1,
            "local_attempts": 1,
            "failures": 0,
        }
        # The interrupted call is not counted: no record was produced for it.
        assert len(gateway.calls) == 2
    finally:
        agent.close()

def test_official_agent_refuses_an_oversized_request_before_the_sdk(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway([_action("web_click", {"target": "2"})])
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        config=_config(tmp_path, max_request_bytes=120),
    )
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 300})
    try:
        result = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 300,
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'\n" + "long AX text " * 40,
            )
        )
        assert result["action"].startswith("stop [")
        assert "max_request_bytes" in result["action"]
        assert gateway.calls == [], "the final SDK exit refuses before the call"
        local_run_id = agent.local_run_id
    finally:
        agent.close()
    run_dir = tmp_path / "runtime" / "runs" / str(local_run_id)
    events = read_events(run_dir / "events.jsonl")
    errors = [event for event in events if event["event_type"] == "error"]
    assert errors and errors[-1]["result"]["error_type"] in {
        "MandatoryContentOverflow",
        "RequestLimitExceeded",
    }
    assert "max_request_bytes" in errors[-1]["result"]["message"]
    result_json = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result_json["termination"] == "error"
    # One refused attempt, no platform-accepted call, one failure.
    assert result_json["summary"]["model_calls"] == {
        "platform_calls": 0,
        "local_attempts": 1,
        "failures": 1,
    }
