"""Offline tests for the scripted official character probe.

Only fake SDK callbacks, a fake browser reader and a fake clock are used. No
platform, network, credential file or real model is touched.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
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
from webagent.official_probe import (
    PROBE_NO_TYPEABLE_MESSAGE,
    PROBE_STOP_MESSAGE,
    PROBE_TEXT_A,
    PROBE_TEXT_B,
    OfficialProbeAgent,
)

RUN = "official-probe-run-1"
TASK = "this task must not be executed by the probe"


class ProbeBrowser:
    """Only the screenshot readers the backend may call."""

    def __init__(self, png: bytes) -> None:
        self.png = png
        self.saved: list[tuple[dict[str, Any], str]] = []
        self.screenshot_ids: list[str] = []

    def save_screenshot(self, observation: dict[str, Any], directory: str) -> Path:
        self.saved.append((observation, directory))
        image_id = observation["screenshot"]["image_id"]
        target = Path(directory) / f"{image_id}.png"
        target.write_bytes(self.png)
        return target

    def get_screenshot(self, image_id: str) -> bytes:
        self.screenshot_ids.append(image_id)
        return self.png


def _png(width: int = 5, height: int = 4) -> bytes:
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
    return header + struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00" + b"probe"


def _screenshot_metadata(png: bytes, *, image_id: str = "a" * 32) -> dict[str, Any]:
    width, height = struct.unpack(">II", png[16:24])
    return {
        "image_id": image_id,
        "width": width,
        "height": height,
        "mime_type": "image/png",
        "step": 1,
        "byte_size": len(png),
        "sha256": hashlib.sha256(png).hexdigest(),
    }


def _config(
    tmp_path: Path,
    *,
    text_mode: str = "decompose",
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
        budget=BudgetConfig(deadline_margin_seconds=30.0),
        limits=LimitsConfig(),
        official=OfficialSettings(text_mode=text_mode),
    )


def _request(
    *,
    step: int,
    text: str,
    url: str,
    deadline: float,
    task: str | None = None,
    screenshot: dict[str, Any] | None = None,
    action_error: Any = None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "type": "accessibility_tree",
        "url": url,
        "text": text,
    }
    if screenshot is not None:
        observation["screenshot"] = screenshot
    request: dict[str, Any] = {
        "protocol_version": "1.0",
        "run_id": RUN,
        "step": step,
        "observation": observation,
        "history": [],
        "action_history": [],
        "execution_deadline": deadline,
    }
    if task is not None:
        request["task"] = task
    if action_error is not None:
        request["action_error"] = action_error
    return request


def _agent(
    tmp_path: Path,
    clock: FakeClock,
    *,
    browser: ProbeBrowser | None = None,
    config: AppConfig | None = None,
    probe: bool | None = True,
    text_a: str = PROBE_TEXT_A,
    text_b: str = PROBE_TEXT_B,
) -> OfficialProbeAgent:
    return OfficialProbeAgent(
        app_config=config or _config(tmp_path),
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        browser_api=browser,
        download_artifacts=frozenset(),
        probe=probe,
        probe_text_a=text_a,
        probe_text_b=text_b,
    )


def _run_dir(tmp_path: Path, agent: OfficialProbeAgent) -> Path:
    assert agent.local_run_id is not None
    return tmp_path / "runtime" / "runs" / str(agent.local_run_id)


def test_probe_script_is_type_press_type_stop_and_uses_raw_wire_text(tmp_path: Path) -> None:
    clock = FakeClock()
    png = _png()
    browser = ProbeBrowser(png)
    # The task config says decompose: the probe must ignore it and use raw.
    agent = _agent(tmp_path, clock, browser=browser, config=_config(tmp_path, text_mode="decompose"))
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[2] button 'Search'\n[1] textbox 'Query'",
            )
        )
        expected_a = "type [1] [P[1]\\a] [0]"
        assert first == {"action": expected_a}
        command_a = first["action"]
        # Real bracket pair and one real backslash; no escape literals.
        assert "[" in command_a and "]" in command_a and "\\" in command_a
        assert r"\[" not in command_a and r"\]" not in command_a
        assert r"\n" not in command_a and r"\t" not in command_a
        assert "\\\\" not in command_a
        assert agent.text_mode is TextMode.RAW
        assert agent.configured_text_mode is TextMode.DECOMPOSE
        backend = agent.backend
        assert backend is not None and backend.text_mode is TextMode.RAW

        second = agent.step(
            _request(
                step=1,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'",
            )
        )
        assert second == {"action": "press [Enter]"}

        third = agent.step(
            _request(
                step=2,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'",
            )
        )
        expected_b = "type [1] [L2\tL3] [0]"
        assert third == {"action": expected_b}
        command_b = third["action"]
        # A real TAB character, not the two characters backslash+t.
        assert "\t" in command_b
        assert r"\t" not in command_b

        fourth = agent.step(
            _request(
                step=3,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'",
                screenshot=_screenshot_metadata(png),
            )
        )
        assert fourth == {"action": f"stop [{PROBE_STOP_MESSAGE}]"}
        assert agent.finished
        run_dir = _run_dir(tmp_path, agent)
    finally:
        agent.close()

    events = read_events(run_dir / "events.jsonl")
    observations = [event for event in events if event["event_type"] == "observation"]
    dispatched = [event for event in events if event["event_type"] == "dispatched"]
    feedback = [event for event in events if event["event_type"] == "feedback"]
    proposed = [event for event in events if event["event_type"] == "proposed"]
    artifact_saved = [event for event in events if event["event_type"] == "artifact_saved"]
    claims = [event for event in events if event["event_type"] == "agent_claim"]
    assert [event["action_type"] for event in dispatched] == ["type", "press", "type"]
    assert len(observations) == 4
    assert len(feedback) == 3
    assert len(proposed) == 3
    assert [event["result"]["kind"] for event in artifact_saved] == ["screenshot"]
    assert len(claims) == 1
    assert "character probe" in claims[0]["result"]["claim"]
    assert "task not executed" in claims[0]["result"]["claim"]

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["termination"] == "incomplete"
    assert result["finish"]["claim"].startswith("character probe:")
    assert "task not executed" in result["finish"]["claim"]
    summary = result["summary"]
    assert summary["steps_completed"] == 3
    assert summary["observations"] == 4
    assert summary["actions_proposed"] == 3
    assert summary["actions_dispatched"] == 3
    assert summary["model_calls"] == {
        "platform_calls": 0,
        "local_attempts": 0,
        "failures": 0,
    }
    assert PROBE_STOP_MESSAGE in summary["unfinished"]
    refs = result["finish"]["evidence_refs"]
    assert refs and (run_dir / refs[0]).is_file()
    run_end = [event for event in events if event["event_type"] == "run_end"]
    assert len(run_end) == 1 and run_end[0]["result"]["termination"] == "incomplete"


def test_missing_typeable_element_stops_honestly_without_actions(tmp_path: Path) -> None:
    clock = FakeClock()
    agent = _agent(tmp_path, clock)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 120,
                url="http://forum.test/home",
                text="[1] button 'Go'\n[2] link 'Home'",
            )
        )
        assert first == {"action": f"stop [{PROBE_NO_TYPEABLE_MESSAGE}]"}
        assert agent.finished
        run_dir = _run_dir(tmp_path, agent)
    finally:
        agent.close()

    events = read_events(run_dir / "events.jsonl")
    assert len([event for event in events if event["event_type"] == "observation"]) == 1
    assert [event for event in events if event["event_type"] == "dispatched"] == []
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["termination"] == "incomplete"
    unfinished = result["summary"]["unfinished"]
    assert PROBE_NO_TYPEABLE_MESSAGE in unfinished
    assert "task not executed" in unfinished
    assert result["summary"]["actions_dispatched"] == 0
    assert "task not executed" in result["finish"]["claim"]


def test_platform_action_error_is_summarised_and_does_not_stop_the_script(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    agent = _agent(tmp_path, clock)
    agent.reset({"run_id": RUN, "execution_deadline": clock.now() + 120})
    try:
        first = agent.step(
            _request(
                step=0,
                task=TASK,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'",
            )
        )
        assert first == {"action": "type [1] [P[1]\\a] [0]"}

        # The platform refused the bracketed text. The probe records the error
        # and still presses Enter (no retry of the same action).
        second = agent.step(
            _request(
                step=1,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'",
                action_error={
                    "type": "invalid_action",
                    "message": "brackets are not supported",
                },
            )
        )
        assert second == {"action": "press [Enter]"}

        third = agent.step(
            _request(
                step=2,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'",
            )
        )
        assert third == {"action": "type [1] [L2\tL3] [0]"}

        fourth = agent.step(
            _request(
                step=3,
                deadline=clock.now() + 120,
                url="http://forum.test/search",
                text="[1] textbox 'Query'",
            )
        )
        assert fourth["action"] == f"stop [{PROBE_STOP_MESSAGE}; action errors: type: invalid_action: brackets are not supported]"
        run_dir = _run_dir(tmp_path, agent)
    finally:
        agent.close()

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    unfinished = result["summary"]["unfinished"]
    assert "action errors:" in unfinished
    assert "invalid_action" in unfinished and "brackets are not supported" in unfinished
    events = read_events(run_dir / "events.jsonl")
    assert [
        event["result"] for event in events if event["event_type"] == "feedback"
    ][0] == "error"


def test_probe_is_opt_in_via_environment_or_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.delenv("WEBAGENT_PROBE", raising=False)
    off = _agent(tmp_path, clock, probe=None, config=_config(tmp_path, text_mode="decompose"))
    assert off.probe_enabled is False
    assert off.text_mode is TextMode.DECOMPOSE
    off.close()

    monkeypatch.setenv("WEBAGENT_PROBE", "1")
    on = _agent(tmp_path, clock, probe=None, config=_config(tmp_path, text_mode="decompose"))
    assert on.probe_enabled is True
    assert on.text_mode is TextMode.RAW
    on.close()

    forced_off = _agent(tmp_path, clock, probe=False, config=_config(tmp_path))
    assert forced_off.probe_enabled is False
    assert forced_off.text_mode is TextMode.DECOMPOSE
    forced_off.close()


def test_probe_text_with_a_real_line_break_is_not_escaped(tmp_path: Path) -> None:
    clock = FakeClock()
    # The default script has no line break (A is one line, B uses one TAB);
    # a caller can still probe a raw line break through the constructor.
    agent = _agent(tmp_path, clock, text_a="A\n[1]\\b", text_b="L2\tL3")
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
        command = first["action"]
        assert "\n" in command
        assert r"\n" not in command
        assert r"\[" not in command and r"\]" not in command
    finally:
        agent.close()
