"""P03/P04 production-path integration: official entry only, fake SDK callbacks.

The fake platform supplies request dictionaries and consumes the returned action
text; the fake gateway returns scripted ReAct turns. No platform, network,
credential file or browser is used. The screenshot path uses a fake
``third_party_agent.browser`` API so evidence binding is real.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from webagent.clock import FakeClock
from webagent.config import (
    AppConfig,
    BudgetConfig,
    ConfigError,
    ModelConfig,
    OfficialBehaviors,
    OfficialSettings,
    load_config,
    parse_config,
)
from webagent.events import read_events
from webagent.official_agent import GOAL_DECLARE_TOOL, OfficialAgent
from webagent.protocol import TerminationReason

CONFIGS = Path(__file__).parents[2] / "configs"


# --------------------------------------------------------------------- fixtures
def _png(width: int = 4, height: int = 3) -> bytes:
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
    return header + struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00\x00"


_PNG = _png()


class FakeBrowser:
    def __init__(self, png: bytes) -> None:
        self.png = png

    def save_screenshot(self, observation: dict[str, Any], directory: str) -> Path:
        image_id = observation["screenshot"]["image_id"]
        target = Path(directory) / f"{image_id}.png"
        target.write_bytes(self.png)
        return target

    def get_screenshot(self, image_id: str) -> bytes:  # pragma: no cover - save path wins
        return self.png


def _screenshot_metadata(png: bytes) -> dict[str, Any]:
    width, height = struct.unpack(">II", png[16:24])
    return {
        "image_id": "a" * 32,
        "width": width,
        "height": height,
        "mime_type": "image/png",
        "step": 1,
        "byte_size": len(png),
        "sha256": hashlib.sha256(png).hexdigest(),
    }


class FakeGateway:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> Any:
        self.calls.append(messages)
        if not self.responses:
            raise RuntimeError("fake official gateway exhausted")
        return SimpleNamespace(text=self.responses.pop(0), request_id=f"req-{len(self.calls)}")

    def prompt(self, index: int) -> str:
        return "\n".join(message["content"] for message in self.calls[index])


def _config(
    tmp_path: Path,
    *,
    behaviors: OfficialBehaviors | None = None,
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
        ),
        budget=BudgetConfig(deadline_margin_seconds=0.0),
        official=OfficialSettings(behaviors=behaviors or OfficialBehaviors()),
    )


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return (
        "Thought: use the current page.\n"
        f"Action: {tool}\n"
        f"Action Input: {json.dumps(arguments, ensure_ascii=False)}"
    )


def _ax(value: str) -> str:
    """Official-style AX: attribute suffix on the field, value in child StaticText."""

    return (
        "[1] textbox 'Biography' required: False describedby: user_biography_help\n"
        f"  [2] StaticText {json.dumps(value, ensure_ascii=False)}"
    )


def _request(
    *,
    step: int,
    value: str = "Initial",
    screenshot: bool = False,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "type": "accessibility_tree",
        "url": "http://forum.test/profile",
        "text": _ax(value),
    }
    if screenshot:
        observation["screenshot"] = _screenshot_metadata(_PNG)
    return {
        "protocol_version": "1.0",
        "run_id": "official-run-1",
        "step": step,
        "task": "replace the biography with the requested text",
        "observation": observation,
        "history": [],
        "action_history": [],
        "execution_deadline": 1_793_000_300.0,
    }


def _biography_goal() -> str:
    return _action(
        GOAL_DECLARE_TOOL,
        {
            "goal_id": "g1",
            "requirement": "the biography must be exactly the requested text",
            "conditions": [
                {
                    "id": "value",
                    "check": "visible_value_equals",
                    "description": "the biography field equals 'I am a robot'",
                    "expected": "I am a robot",
                    "role": "textbox",
                    "name": "Biography",
                }
            ],
        },
    )


def _make_agent(
    tmp_path: Path,
    clock: FakeClock,
    gateway: FakeGateway,
    *,
    behaviors: OfficialBehaviors | None = None,
    browser: FakeBrowser | None = None,
) -> OfficialAgent:
    return OfficialAgent(
        app_config=_config(tmp_path, behaviors=behaviors),
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        model_api=gateway,
        browser_api=browser or FakeBrowser(_PNG),
    )


# ------------------------------------------------------------------------ P03
def test_request_finish_refusal_is_returned_to_the_same_loop_and_then_passes(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _biography_goal(),
            _action("web_type", {"target": "1", "text": "wrong ", "submit": False}),
            _action("request_finish", {}),
            _action("web_type", {"target": "1", "text": "I am a robot", "submit": False}),
            _action("request_finish", {}),
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        first = agent.step(_request(step=0, value="Initial", screenshot=True))
        assert first == {"action": "type [1] [wrong ] [0]"}
        driver = agent.driver
        assert driver is not None
        same_agent = driver.agent

        # The appended (wrong) value is refused by the completion gate inside the
        # same step; the generator continues and proposes the corrective action.
        second = agent.step(
            _request(step=1, value="wrong Initial", screenshot=True)
        )
        assert second == {"action": "type [1] [I am a robot] [0]"}
        assert agent.driver is driver and driver.agent is same_agent
        assert driver.finish_rejections == 1

        third = agent.step(_request(step=2, value="I am a robot", screenshot=True))
        assert third["action"].startswith("stop [completion verified")
        assert agent.finished
    finally:
        agent.close()

    # The refusal text entered the next model turn, with the remaining requirement.
    assert "completion is NOT confirmed" in gateway.prompt(3)
    assert "g1" in gateway.prompt(3)

    session = driver.session
    events = read_events(session.evidence.events_path)
    request_finish_events = [
        event
        for event in events
        if event["event_type"] == "verification"
        and event["result"].get("stage") == "request_finish"
    ]
    assert [
        (event["result"]["accepted"], event["result"]["continued"])
        for event in request_finish_events
    ] == [(False, True), (True, False)]
    assert request_finish_events[0]["result"]["remaining_goal_ids"] == ["g1"]
    assert request_finish_events[0]["result"]["remaining_conditions"] == ["g1:value"]
    assert len([event for event in events if event["event_type"] == "proposed"]) == 2
    assert len([event for event in events if event["event_type"] == "dispatched"]) == 2
    claims = [event for event in events if event["event_type"] == "agent_claim"]
    assert len(claims) == 1
    assert claims[0]["result"]["reason"] == TerminationReason.COMPLETED.value

    state = json.loads(session.evidence.run_dir.joinpath("state", "state.json").read_text("utf-8"))
    assert state["goals"]["g1"]["status"] == "verified"
    assert state["goals"]["g1"]["evidence"][0]["evidence_id"] == "ev-obs-2-screenshot"
    result = json.loads(session.evidence.run_dir.joinpath("result.json").read_text("utf-8"))
    assert result["termination"] == "completed"
    assert result["summary"]["runtime"]["behaviors"] == {
        "verification": True,
        "state_tools": True,
        "recovery": False,
    }


def test_completion_uses_the_saved_observation_when_no_screenshot_exists(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action(
                GOAL_DECLARE_TOOL,
                {
                    "goal_id": "g1",
                    "requirement": "the biography must be exactly 'Initial'",
                    "conditions": [
                        {
                            "id": "value",
                            "check": "visible_value_equals",
                            "expected": "Initial",
                            "role": "textbox",
                            "name": "Biography",
                        }
                    ],
                },
            ),
            _action("request_finish", {}),
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(_request(step=0, value="Initial", screenshot=False))
        assert result["action"].startswith("stop [completion verified")
        session = agent.session
    finally:
        agent.close()

    state = json.loads(
        session.evidence.run_dir.joinpath("state", "state.json").read_text("utf-8")
    )
    assert state["goals"]["g1"]["status"] == "verified"
    evidence = state["goals"]["g1"]["evidence"][0]
    assert evidence["evidence_id"] == "ev-obs-0-observation"
    assert evidence["kind"] is None
    assert session.evidence.resolve_reference(evidence["reference"]).is_file()


def test_empty_type_refusal_enters_the_next_model_turn_and_the_run_continues(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_type", {"target": "1", "text": "", "submit": False}),
            _action("web_click", {"target": "2"}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        first = agent.step(
            {
                **_request(step=0),
                "observation": {
                    "type": "accessibility_tree",
                    "url": "http://forum.test/search",
                    "text": "[1] textbox 'Query'\n[2] button 'Search'",
                },
            }
        )
        assert first == {"action": "click [2]"}
        driver = agent.driver
        assert driver is not None and len(driver.corrections) == 1
        assert driver.corrections[0]["reason"] == "invalid_arguments"
        second = agent.step(
            {
                **_request(step=1),
                "observation": {
                    "type": "accessibility_tree",
                    "url": "http://forum.test/result",
                    "text": "[1] link 'Done'",
                },
            }
        )
        assert second == {"action": "stop [done]"}
        assert agent.finished
    finally:
        agent.close()

    assert "web_type needs non-empty text" in gateway.prompt(1)
    events = read_events(driver.session.evidence.events_path)
    corrections = [
        event for event in events if event["event_type"] == "format_correction"
    ]
    assert any(item["result"]["reason"] == "empty_type_text" for item in corrections)
    assert any(
        item["result"].get("corrections_so_far") == 1 for item in corrections
    )
    assert len([event for event in events if event["event_type"] == "dispatched"]) == 1


def test_empty_type_corrections_reach_the_format_error_cap(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_type", {"target": "1", "text": "", "submit": False}),
            _action("web_type", {"target": "1", "text": "", "submit": False}),
            _action("web_type", {"target": "1", "text": "", "submit": False}),
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(
            {
                **_request(step=0),
                "observation": {
                    "type": "accessibility_tree",
                    "url": "http://forum.test/search",
                    "text": "[1] textbox 'Query'",
                },
            }
        )
        assert result["action"].startswith("stop [")
        assert "did not become valid" in result["action"]
        driver = agent.driver
        assert driver is not None
        assert driver.terminal is not None
        assert driver.terminal.status.value == "format_error"
        assert len(driver.corrections) == 3
        events = read_events(driver.session.evidence.events_path)
        assert not any(event["event_type"] == "dispatched" for event in events)
    finally:
        agent.close()


def test_finish_refusals_are_bounded_and_end_honestly_as_incomplete(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action(
                GOAL_DECLARE_TOOL,
                {
                    "goal_id": "g1",
                    "requirement": "the biography must be exactly 'never'",
                    "conditions": [
                        {
                            "id": "value",
                            "check": "visible_value_equals",
                            "expected": "never",
                            "role": "textbox",
                            "name": "Biography",
                        }
                    ],
                },
            ),
            _action("request_finish", {}),
            _action("request_finish", {}),
            _action("request_finish", {}),
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(_request(step=0, value="Initial", screenshot=True))
        assert result["action"].startswith("stop [completion was refused 3 times")
        assert "ending honestly as incomplete" in result["action"]
        driver = agent.driver
        assert driver is not None
        assert driver.finish_rejections == 3
        assert driver.terminal is not None
        assert driver.terminal.status.value == "incomplete"
        session = driver.session
    finally:
        agent.close()

    events = read_events(session.evidence.events_path)
    gate_events = [
        event
        for event in events
        if event["event_type"] == "verification"
        and event["result"].get("stage") == "request_finish"
    ]
    assert [event["result"]["continued"] for event in gate_events] == [
        True,
        True,
        False,
    ]
    claims = [event for event in events if event["event_type"] == "agent_claim"]
    assert len(claims) == 1
    assert claims[0]["result"]["reason"] == TerminationReason.INCOMPLETE.value
    assert "partial progress is not complete" in claims[0]["result"]["claim"]
    state = json.loads(
        session.evidence.run_dir.joinpath("state", "state.json").read_text("utf-8")
    )
    assert state["goals"]["g1"]["status"] == "running"
    result_json = json.loads(
        session.evidence.run_dir.joinpath("result.json").read_text("utf-8")
    )
    assert result_json["termination"] == "incomplete"


def test_state_tools_bind_engineering_fields_and_reject_a_stale_revision(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action(
                "state_update",
                {
                    "facts": [
                        {
                            "id": "f1",
                            "object": "profile",
                            "field": "note",
                            "value": "observed by the model",
                            "kind": "inferred",
                            "source": "model turn",
                        }
                    ]
                },
            ),
            _action(
                "state_update",
                {
                    "base_revision": 0,
                    "facts": [
                        {
                            "id": "f1",
                            "object": "profile",
                            "field": "note",
                            "value": "stale rewrite",
                            "kind": "inferred",
                            "source": "model turn",
                            "revision": 2,
                        }
                    ],
                },
            ),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(_request(step=0))
        assert result == {"action": "stop [done]"}
        store = agent.store
        assert store is not None
        assert store.state.revision == 1
        assert store.state.facts["f1"].value == "observed by the model"
    finally:
        agent.close()

    # The first update supplied no revision; the harness bound it. The second
    # supplied a stale one and the ledger refused it with the current state.
    second_prompt = gateway.prompt(2)
    assert "stale_base_revision" in second_prompt
    assert '"revision":1' in second_prompt


# ------------------------------------------------------------------------ P04
def test_all_behavior_switches_off_remove_tools_schemas_and_the_gate(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(["Thought: done.\nFinal Answer: done"])
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(
            verification=False, state_tools=False, recovery=False
        ),
    )
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(_request(step=0))
        assert result == {"action": "stop [done]"}
        assert agent.store is None
        assert agent.driver is not None and agent.driver.completion_gate is None
        session = agent.session
    finally:
        agent.close()

    prompt = gateway.prompt(0)
    for name in (
        "state_read",
        "state_update",
        "plan_update",
        GOAL_DECLARE_TOOL,
        "request_finish",
    ):
        assert name not in prompt
    assert session is not None
    events = read_events(session.evidence.events_path)
    behaviors = [event for event in events if event["event_type"] == "behaviors"]
    assert len(behaviors) == 1
    assert behaviors[0]["result"]["behaviors"] == {
        "verification": False,
        "state_tools": False,
        "recovery": False,
    }
    assert behaviors[0]["result"]["tools"] == []
    assert not session.evidence.run_dir.joinpath("state", "state.json").exists()


def test_verification_on_without_state_tools_keeps_the_gate_and_passes_with_no_goals(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway([_action("request_finish", {})])
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(
            verification=True, state_tools=False, recovery=False
        ),
    )
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(_request(step=0))
        assert result["action"].startswith("stop [completion verified")
        driver = agent.driver
        assert driver is not None and driver.completion_gate is not None
        assert agent.store is not None
    finally:
        agent.close()

    prompt = gateway.prompt(0)
    assert "request_finish" in prompt
    for name in ("state_read", "state_update", "plan_update", GOAL_DECLARE_TOOL):
        assert name not in prompt


def test_state_tools_on_with_verification_off_keeps_ledger_tools_and_no_gate(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(["Thought: done.\nFinal Answer: done"])
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(
            verification=False, state_tools=True, recovery=False
        ),
    )
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(_request(step=0))
        assert result == {"action": "stop [done]"}
        driver = agent.driver
        assert driver is not None and driver.completion_gate is None
        assert agent.store is not None
    finally:
        agent.close()

    prompt = gateway.prompt(0)
    for name in ("state_read", "state_update", "plan_update", GOAL_DECLARE_TOOL):
        assert name in prompt
    assert "request_finish" not in prompt


def test_official_config_exposes_the_explicit_behavior_group() -> None:
    official = load_config(CONFIGS / "official.yaml")
    assert official.official.behaviors == OfficialBehaviors(
        verification=True, state_tools=True, recovery=False
    )
    assert official.summary()["official"]["behaviors"] == {
        "verification": True,
        "state_tools": True,
        "recovery": False,
    }
    # Early close-out remains disabled.
    assert official.budget.deadline_margin_seconds == 0.0

    raw = yaml.safe_load((CONFIGS / "official.yaml").read_text("utf-8"))
    raw["official"]["behaviors"] = {
        "verification": False,
        "state_tools": False,
        "recovery": False,
    }
    parsed = parse_config(raw)
    assert parsed.official.behaviors.verification is False
    assert parsed.official.behaviors.state_tools is False
    assert parsed.official.behaviors.recovery is False

    raw["official"]["behaviors"]["verification"] = "yes"
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = yaml.safe_load((CONFIGS / "official.yaml").read_text("utf-8"))
    raw["official"]["behaviors"]["unknown"] = True
    with pytest.raises(ConfigError):
        parse_config(raw)

