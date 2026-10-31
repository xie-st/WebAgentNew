"""G1 production-path integration: the official entry with the context switch.

The fake platform supplies request dictionaries and the fake gateway records the
exact messages it is asked to send, so "the final send" in these tests is the
same list ``third_party_agent.model.generate`` would receive. No model, network,
credential file or browser is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from webagent.budget import render_observation
from webagent.clock import FakeClock
from webagent.config import (
    AppConfig,
    BudgetConfig,
    ContextSettings,
    ModelConfig,
    OfficialBehaviors,
    OfficialSettings,
)
from webagent.context import RECORDS_FILE_NAME, SectionKind
from webagent.events import read_events
from webagent.official_agent import GOAL_DECLARE_TOOL, OfficialAgent
from webagent.state.records import EffectStatus


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
    context: ContextSettings | None = None,
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
        context=context or ContextSettings(),
    )


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return (
        "Thought: use the current page.\n"
        f"Action: {tool}\n"
        f"Action Input: {json.dumps(arguments, ensure_ascii=False)}"
    )


def _ax(value: str) -> str:
    return (
        "[1] textbox 'Biography' required: False\n"
        f"  [2] StaticText {json.dumps(value, ensure_ascii=False)}"
    )


def _request(*, step: int, value: str = "Initial", task: str = "replace the biography"):
    return {
        "protocol_version": "1.0",
        "run_id": "official-run-1",
        "step": step,
        "task": task,
        "observation": {
            "type": "accessibility_tree",
            "url": "http://forum.test/profile",
            "text": _ax(value),
        },
        "history": [],
        "action_history": [],
        "execution_deadline": 1_793_000_300.0,
    }


GOAL_ACTION = _action(
    GOAL_DECLARE_TOOL,
    {
        "goal_id": "g1",
        "requirement": "the biography must be exactly 'I am a robot'",
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
    context: ContextSettings | None = None,
) -> OfficialAgent:
    return OfficialAgent(
        app_config=_config(tmp_path, behaviors=behaviors, context=context),
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        model_api=gateway,
    )


def _records(session: Any) -> list[dict[str, Any]]:
    """The per-send context records the protector wrote for this run."""

    path = session.evidence.run_dir / RECORDS_FILE_NAME
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _script(task: str) -> list[str]:
    """One ledger update, one page action, then a final answer."""

    return [
        GOAL_ACTION,
        _action(
            "web_type",
            {"target": "1", "text": "I am a robot", "submit": False},
        ),
        "Thought: done.\nFinal Answer: done",
    ]


# ---------------------------------------------------------------- switch on
def test_enabled_switch_appends_and_checks_every_final_send(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(_script("replace the biography"))
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        context=ContextSettings(enabled=True, max_input_tokens=24_000),
    )
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        first = agent.step(_request(step=0))
        assert first["action"] == "type [1] [I am a robot] [0]"
        second = agent.step(_request(step=1, value="I am a robot"))
        assert second["action"] == "stop [done]"
        session = agent.session
        protector = agent.context_protector
        assert protector is not None
        assert protector.checks == len(gateway.calls) == 3
        report = protector.last_report
        assert report is not None
        assert report.mandatory_intact and report.verdict == "ok"
        assert report.revision == 1
    finally:
        agent.close()

    # Every send carried the four necessary regions, appended by the harness.
    for index in range(len(gateway.calls)):
        prompt = gateway.prompt(index)
        assert "replace the biography" in prompt
        assert "CURRENT OBSERVATION" in prompt
        assert "LEDGER VIEW (harness-maintained" in prompt
        assert "PENDING SIDE EFFECTS (unconfirmed" in prompt
        # The current send is exactly one user message; the official byte and
        # message limits are still satisfied.
        assert [message["role"] for message in gateway.calls[index]] == ["user"]

    # The ledger short view actually entered the prompt with the recorded goal.
    # It is refreshed when the next step's tool result is assembled.
    ledger_prompt = gateway.prompt(2)
    assert "- g1 [running]" in ledger_prompt
    assert "revision=1 plan_version=0" in ledger_prompt
    assert "ELEMENTS:" in ledger_prompt
    assert "none recorded" in ledger_prompt

    events = read_events(session.evidence.events_path)
    assert not any(event["event_type"] == "context_pack" for event in events)

    records = _records(session)
    assert len(records) == len(gateway.calls)
    kinds = [item["kind"] for item in records[0]["sections"]]
    assert kinds == [kind.value for kind in SectionKind]
    for record in records:
        assert record["verdict"] == "ok"
        assert record["mandatory_intact"] is True
        for section in record["sections"]:
            assert section["present"] is True
            assert section["matched_bytes"] == section["expected_bytes"]
        assert record["message_count"] == 1
        assert record["max_messages"] == 256
        assert record["token_count_source"].startswith("estimate:chars_div_4")

    summary = json.loads(
        session.evidence.run_dir.joinpath("result.json").read_text("utf-8")
    )["summary"]["runtime"]
    assert summary["behaviors"] == {
        "verification": True,
        "state_tools": True,
        "recovery": False,
    }
    assert summary["context"]["enabled"] is True
    assert summary["context"]["native_context_budget_tokens"] == 24_000
    # Every measured send is recorded next to the run's other evidence; the
    # persisted switch block names the same ceilings the check used.
    assert summary["context"]["max_input_tokens"] == 24_000
    assert summary["context"]["ledger_max_chars"] == 2_000


def test_pending_effect_records_reach_the_prompt_and_are_protected(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(_script("replace the biography"))
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        context=ContextSettings(enabled=True, max_input_tokens=24_000),
    )
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        agent.step(_request(step=0))
        store = agent.store
        assert store is not None
        store.prepare_effect(
            action_id="act-pending-1",
            target="click [2]",
            object_id="biography",
            branch_id="root",
            observation_id="obs-1",
            note="save pressed; result not seen yet",
        )
        agent.step(_request(step=1, value="I am a robot"))
        session = agent.session
        protector = agent.context_protector
        assert protector is not None
    finally:
        agent.close()

    prompt = gateway.prompt(2)
    assert "PENDING SIDE EFFECTS (unconfirmed" in prompt
    assert "act-pending-1" in prompt
    assert "status=pending" in prompt
    records = _records(session)
    assert records
    effects = [
        section
        for section in records[-1]["sections"]
        if section["kind"] == SectionKind.EFFECTS.value
    ][0]
    assert effects["byte_complete"] is True
    assert effects["expected_bytes"] > 0


def test_context_switch_is_independent_of_the_official_behavior_group(
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
        context=ContextSettings(enabled=True),
    )
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(_request(step=0))
        assert result["action"] == "stop [done]"
        assert agent.store is None
        protector = agent.context_protector
        assert protector is not None and protector.store is None
        session = agent.session
    finally:
        agent.close()

    prompt = gateway.prompt(0)
    # No ledger tools are registered, so the block states that plainly instead of
    # pretending a ledger exists; the switch still protects the send.
    assert "no task ledger is active for this task" in prompt
    records = _records(session)
    assert len(records) == 1 and records[0]["verdict"] == "ok"
    assert records[0]["branch"] == "root"


# --------------------------------------------------------------- switch off
def test_disabled_switch_sends_the_identical_prompt_and_adds_no_call(
    tmp_path: Path,
) -> None:
    def run(directory: Path, *, explicit_off: bool) -> tuple[list[str], OfficialAgent]:
        clock = FakeClock()
        gateway = FakeGateway(_script("replace the biography"))
        context = ContextSettings(enabled=False) if explicit_off else None
        agent = _make_agent(directory, clock, gateway, context=context)
        agent.reset(
            {"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0}
        )
        actions = [
            agent.step(_request(step=0))["action"],
            agent.step(_request(step=1, value="I am a robot"))["action"],
        ]
        return [message for call in gateway.calls for message in call], agent

    explicit = tmp_path / "explicit"
    default = tmp_path / "default"
    explicit_messages, explicit_agent = run(explicit, explicit_off=True)
    try:
        default_messages, default_agent = run(default, explicit_off=False)
        try:
            # With the switch off the final send is byte-identical whether the
            # section is written out or left at its default.
            assert explicit_messages == default_messages
            assert explicit_agent.context_protector is None
            assert default_agent.context_protector is None
            for agent in (explicit_agent, default_agent):
                driver = agent.driver
                assert driver is not None
                assert driver.context_note_provider is None
                assert driver.final_send_hook is None
                session = agent.session
                assert session is not None
                events = read_events(session.evidence.events_path)
                assert not any(
                    event["event_type"] == "context_pack" for event in events
                )
        finally:
            default_agent.close()
    finally:
        explicit_agent.close()

    for message in explicit_messages:
        assert "LEDGER VIEW" not in message["content"]
        assert "PENDING SIDE EFFECTS" not in message["content"]
        assert "harness-maintained" not in message["content"]


def test_disabled_switch_keeps_the_p_wave_finish_gate_behavior(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            GOAL_ACTION,
            _action("web_type", {"target": "1", "text": "wrong ", "submit": False}),
            _action("request_finish", {}),
            _action("web_type", {"target": "1", "text": "I am a robot", "submit": False}),
            _action("request_finish", {}),
        ]
    )
    agent = _make_agent(tmp_path, clock, gateway)
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        first = agent.step(_request(step=0))
        second = agent.step(_request(step=1, value="wrong Initial"))
        third = agent.step(_request(step=2, value="I am a robot"))
        assert first["action"] == "type [1] [wrong ] [0]"
        assert second["action"] == "type [1] [I am a robot] [0]"
        assert third["action"].startswith("stop [completion verified")
        driver = agent.driver
        assert driver is not None and driver.finish_rejections == 1
        assert agent.context_protector is None
        session = agent.session
    finally:
        agent.close()
    assert "completion is NOT confirmed" in gateway.prompt(3)
    assert _records(session) == []


# ------------------------------------------------------------------ overflow
def test_over_budget_final_send_is_refused_before_the_model_is_called(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(_script("replace the biography"))
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        context=ContextSettings(enabled=True, max_input_tokens=200),
    )
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        result = agent.step(_request(step=0))
        assert result["action"].startswith("stop [")
        assert "project context budget" in result["action"]
        assert "every necessary region is intact" in result["action"]
        driver = agent.driver
        assert driver is not None
        assert driver.terminal is not None
        assert driver.terminal.status.value == "error"
        protector = agent.context_protector
        assert protector is not None
        assert protector.checks == 1
        report = protector.last_report
        assert report is not None and report.over_limit and report.mandatory_intact
        session = agent.session
    finally:
        agent.close()

    # Nothing was sent: the refusal is the explicit error, not a shortened prompt.
    assert gateway.calls == []
    records = _records(session)
    assert len(records) == 1
    payload = records[0]
    assert payload["verdict"] == "over_limit"
    assert payload["mandatory_intact"] is True
    assert payload["tokens"] > payload["max_input_tokens"]
    for section in payload["sections"]:
        assert section["byte_complete"] is True
        assert section["matched_bytes"] == section["expected_bytes"]


def test_config_rejects_an_unknown_context_key_before_the_entry_starts() -> None:
    from webagent.config import ConfigError, parse_config

    raw = {
        "schema_version": "webagent.config.v1",
        "profile": "official",
        "backend": "official",
        "model": {"provider": "official", "name": "platform-unified"},
        "context": {"enabled": True, "step_summary": True},
    }
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_render_observation_matches_the_block_builder_for_the_same_observation(
    tmp_path: Path,
) -> None:
    """The AX region the guard protects is the same view the loop appends."""

    clock = FakeClock()
    gateway = FakeGateway(["Thought: done.\nFinal Answer: done"])
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        context=ContextSettings(enabled=True, max_input_tokens=24_000),
    )
    agent.reset({"run_id": "official-run-1", "execution_deadline": 1_793_000_300.0})
    try:
        agent.step(_request(step=0, value="Initial"))
        session = agent.session
        observation_rendered = render_observation(session.observation)
        protector = agent.context_protector
        assert protector is not None and protector.last_report is not None
        protected = protector.last_report.section(SectionKind.OBSERVATION)
        assert protected.byte_complete
        assert protected.matched_bytes == len(observation_rendered.encode("utf-8"))
        assert observation_rendered in gateway.prompt(0)
    finally:
        agent.close()
