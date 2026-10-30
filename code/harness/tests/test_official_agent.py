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

from webagent.clock import FakeClock
from webagent.config import AppConfig, ModelConfig
from webagent.events import read_events
from webagent.official_agent import OfficialAgent

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


def _config(tmp_path: Path) -> AppConfig:
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


def _make_agent(tmp_path: Path, clock: FakeClock, gateway: FakeGateway) -> OfficialAgent:
    return OfficialAgent(
        app_config=_config(tmp_path),
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
    same_agent = driver.agent
    assert agent.session is not None
    assert agent.session.state.value == "running"

    second = agent.step(
        _request(
            step=1,
            deadline=clock.now() + 120,
            url="http://forum.test/result",
            text="[1] textbox 'Note'\n[2] button 'Save'",
        )
    )
    assert second == {"action": r"type [1] [价格（含税）\n第二行 \[备注\]] [0]"}
    assert "\n" not in second["action"]
    assert agent.driver is driver and driver.agent is same_agent

    third = agent.step(
        _request(
            step=2,
            deadline=clock.now() + 120,
            url="http://forum.test/result",
            text="[1] link 'Done'",
        )
    )
    assert third == {"action": "stop [done]"}
    assert agent.finished

    session = agent.session
    assert session is not None
    local_run_id = agent.local_run_id
    agent.close()
    agent.close()
    assert agent.session is None and agent.driver is None
    assert len(gateway.calls) == 3

    events = read_events(tmp_path / "runtime" / "runs" / str(local_run_id) / "events.jsonl")
    proposed = [event for event in events if event["event_type"] == "proposed"]
    dispatched = [event for event in events if event["event_type"] == "dispatched"]
    feedback = [event for event in events if event["event_type"] == "feedback"]
    model_calls = [event for event in events if event["event_type"] == "model_call"]
    claims = [event for event in events if event["event_type"] == "agent_claim"]
    assert len(proposed) == 2
    assert len(dispatched) == 2
    assert len(feedback) == 2
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
