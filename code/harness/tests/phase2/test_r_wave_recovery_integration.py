"""R wave (E03/E04/E05/E06/E07/E08): recovery and takeover through the real entry.

Every test drives ``OfficialAgent`` with a scripted official gateway and a fake
``third_party_agent.browser`` API. No platform, network, credential file, real
model or real browser is used. The fake platform supplies request mappings and
consumes the returned action text, so the tests really exercise the production
step path, the control layer, the ledger and the recovery wiring.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from webagent.clock import FakeClock
from webagent.config import (
    AppConfig,
    BudgetConfig,
    ModelConfig,
    OfficialBehaviors,
    OfficialSettings,
    RecoverySettings,
)
from webagent.control import (
    ControlState,
    MemoryControlSource,
    detect_takeover,
    read_control_events,
)
from webagent.events import read_events
from webagent.official_agent import GOAL_DECLARE_TOOL, OfficialAgent
from webagent.protocol import Feedback, FeedbackStatus, TerminationReason, ToolRequest
from webagent.recovery.wiring import OfficialRecoveryRuntime
from webagent.state.ledger import StateUpdate
from webagent.state.records import Fact, FactKind
from webagent.state.store import StateStore
from webagent.testing import FakeBackend, FakeElement, FakePage, FakeSite
from webagent.tools import build_web_action
from webagent.verification.types import VerificationStatus

from e1_helpers import (
    bootstrap_store,
    evidence_ref,
    make_goal,
    make_observation,
    verified_goal,
)

RUN_ID = "official-run-1"
DEADLINE = 1_793_000_300.0


# --------------------------------------------------------------------- fixtures
def _png(width: int = 4, height: int = 3) -> bytes:
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
    return header + struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00\x00"


_PNG = _png()


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


def _config(tmp_path: Path, *, behaviors: OfficialBehaviors) -> AppConfig:
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
        official=OfficialSettings(behaviors=behaviors),
        recovery=RecoverySettings(
            stagnation_window=6, stagnation_repeats=2, max_recoveries_per_goal=2
        ),
    )


def _make_agent(
    tmp_path: Path,
    clock: FakeClock,
    gateway: FakeGateway,
    *,
    behaviors: OfficialBehaviors,
    control_source: Any | None = None,
    control_wait_timeout: float | None = None,
) -> OfficialAgent:
    return OfficialAgent(
        app_config=_config(tmp_path, behaviors=behaviors),
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        model_api=gateway,
        browser_api=FakeBrowser(_PNG),
        control_source=control_source,
        control_wait_timeout=control_wait_timeout,
    )


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return (
        "Thought: use the current page.\n"
        f"Action: {tool}\n"
        f"Action Input: {json.dumps(arguments, ensure_ascii=False)}"
    )


def _ax(biography: str, *, city: str = "Oslo", note: str = "Saved") -> str:
    return (
        "[1] textbox 'Biography' required: False describedby: user_biography_help\n"
        f"  [2] StaticText {json.dumps(biography, ensure_ascii=False)}\n"
        "[3] button 'Search'\n"
        "[4] textbox 'City' required: False\n"
        f"  [5] StaticText {json.dumps(city, ensure_ascii=False)}\n"
        f"[6] StaticText {json.dumps(note, ensure_ascii=False)}"
    )


def _login_ax() -> str:
    return (
        "[1] button 'Sign in'\n"
        '[2] StaticText "Your session has expired. Sign in in this browser to '
        'restore the page."'
    )


def _restart_ax() -> str:
    return (
        "[1] textbox 'Biography' required: False\n"
        '  [2] StaticText "Alice"\n'
        "[3] button 'Cancel draft'\n"
        "[4] StaticText \"Saved\""
    )


def _request(
    *,
    step: int,
    text: str,
    value: str = "",
    screenshot: bool = True,
    url: str = "http://forum.test/profile",
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "type": "accessibility_tree",
        "url": url,
        "text": text,
    }
    if screenshot:
        observation["screenshot"] = _screenshot_metadata(_PNG)
    return {
        "protocol_version": "1.0",
        "run_id": run_id,
        "step": step,
        "task": "update the profile the user asked for",
        "observation": observation,
        "history": [],
        "action_history": [],
        "execution_deadline": DEADLINE,
    }


def _goal(
    goal_id: str,
    check: str,
    expected: str,
    *,
    role: str | None = None,
    name: str | None = None,
    depends_on: list[str] | None = None,
) -> str:
    condition: dict[str, Any] = {"id": "c1", "check": check, "expected": expected}
    if role is not None:
        condition["role"] = role
    if name is not None:
        condition["name"] = name
    arguments: dict[str, Any] = {
        "goal_id": goal_id,
        "requirement": f"requirement {goal_id}",
        "conditions": [condition],
    }
    if depends_on:
        arguments["depends_on"] = depends_on
    return _action(GOAL_DECLARE_TOOL, arguments)


def _run_dir(tmp_path: Path, run_id: str = RUN_ID) -> Path:
    return tmp_path / "runtime" / "runs" / run_id


def _events(tmp_path: Path, run_id: str = RUN_ID) -> list[dict[str, Any]]:
    return read_events(_run_dir(tmp_path, run_id) / "events.jsonl")


def _control_events(tmp_path: Path, run_id: str = RUN_ID) -> list[dict[str, Any]]:
    path = _run_dir(tmp_path, run_id) / "control_events.jsonl"
    assert path.is_file(), "the control layer wrote no control event log"
    return read_control_events(path)


def _state_json(tmp_path: Path, run_id: str = RUN_ID) -> dict[str, Any]:
    return json.loads((_run_dir(tmp_path, run_id) / "state" / "state.json").read_text("utf-8"))


def _invalidations(tmp_path: Path, run_id: str = RUN_ID) -> list[dict[str, Any]]:
    path = _run_dir(tmp_path, run_id) / "state" / "invalidations.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


# ----------------------------------------------------------------- control sources
class HandoffProbeSource(MemoryControlSource):
    """Wait one poll, then advance the fake clock and answer an explicit resume."""

    def __init__(self, clock: FakeClock, *, wait_seconds: float = 12.0) -> None:
        super().__init__()
        self.clock = clock
        self.wait_seconds = wait_seconds
        self.coordinator_getter: Callable[[], Any] | None = None
        self.probe: Callable[[], dict[str, Any]] | None = None
        self.polls = 0
        self.sent = False
        self.probe_at_handoff: dict[str, Any] = {}
        self.probe_after_wait: dict[str, Any] = {}

    def poll(self) -> str | None:
        command = super().poll()
        if command is not None:
            return command
        coordinator = None if self.coordinator_getter is None else self.coordinator_getter()
        if (
            coordinator is not None
            and coordinator.state is ControlState.USER_CONTROL
            and not self.sent
        ):
            self.polls += 1
            if self.polls < 2:
                return None
            self.sent = True
            if self.probe is not None:
                self.probe_at_handoff = dict(self.probe())
            self.clock.advance(self.wait_seconds)
            if self.probe is not None:
                self.probe_after_wait = dict(self.probe())
            return "resume"
        return None


class CancelDuringHandoffSource(MemoryControlSource):
    def __init__(self) -> None:
        super().__init__()
        self.coordinator_getter: Callable[[], Any] | None = None
        self.sent = False

    def poll(self) -> str | None:
        command = super().poll()
        if command is not None:
            return command
        coordinator = None if self.coordinator_getter is None else self.coordinator_getter()
        if coordinator is not None and coordinator.state is ControlState.USER_CONTROL:
            self.sent = True
            return "cancel"
        return None


# ---------------------------------------------------------------- E05/E06/E07 (a)
def test_login_overlay_pauses_the_whole_task_and_resumes_on_a_user_signal(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_click", {"target": "3"}),
            _action("web_click", {"target": "3"}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    source = HandoffProbeSource(clock, wait_seconds=12.0)
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(verification=True, state_tools=True, recovery=True),
        control_source=source,
    )
    source.coordinator_getter = lambda: agent.control
    source.probe = lambda: {
        "model_calls": len(gateway.calls),
        "events": len(_events(tmp_path)),
    }
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    try:
        first = agent.step(_request(step=0, text=_ax("Initial")))
        assert first == {"action": "click [3]"}
        assert len(gateway.calls) == 1
        driver = agent.driver
        assert driver is not None
        same_agent = driver.agent
        dispatched_action_id = next(
            event["action_id"]
            for event in _events(tmp_path)
            if event["event_type"] == "dispatched"
        )

        # The click's observation is a visible login overlay. The task pauses
        # before the generator advances: no second model call happens here.
        second = agent.step(_request(step=1, text=_login_ax(), url="http://forum.test/login"))
        assert second == {"action": "goto [http://forum.test/login]"}
        assert len(gateway.calls) == 1, "a model call ran while the user owned the browser"
        assert source.polls == 2, "the wait needs an explicit signal, not a timer"
        assert source.probe_at_handoff == source.probe_after_wait
        assert agent.driver is driver and driver.agent is same_agent

        handoff = next(
            event
            for event in _control_events(tmp_path)
            if event["event_type"] == "handoff"
        )
        assert handoff["reason"] == "login_overlay_visible"
        assert handoff["details"]["browser_drained"] is True
        assert handoff["details"]["executed_action_ids"] == [dispatched_action_id]
        control_types = [event["event_type"] for event in _control_events(tmp_path)]
        for expected in ("takeover_detected", "pause_requested", "draining", "handoff", "resume_signal"):
            assert expected in control_types, control_types
        assert control_types.index("takeover_detected") < control_types.index("handoff")

        recovery_events = [
            event for event in _events(tmp_path) if event["event_type"] == "recovery"
        ]
        handoff_event = next(
            event for event in recovery_events if event["result"]["stage"] == "handoff"
        )
        assert handoff_event["result"]["waiting_reason"] == "login_overlay_visible"
        assert handoff_event["result"]["waiting_for"] == "explicit_user_resume"
        assert handoff_event["result"]["old_element_numbers_valid"] is False
        reobserve = next(
            event for event in recovery_events if event["result"]["stage"] == "reobserve"
        )
        assert reobserve["result"]["control_state"] == "refreshing"
        assert reobserve["result"]["page_epoch"] == 0

        # The explicit resume re-observes the affected page. The model sees the
        # truthful result of the executed click plus the fresh observation; it
        # must use the new element numbers.
        third = agent.step(_request(step=2, text=_ax("Alice")))
        assert third == {"action": "click [3]"}
        assert len(gateway.calls) == 2
        resumed_prompt = gateway.prompt(1)
        assert "was executed before the browser was handed to the user" in resumed_prompt
        assert "Fresh observation after the user resumed" in resumed_prompt
        assert "no longer valid" in resumed_prompt
        assert '"Initial"' in resumed_prompt and '"Alice"' in resumed_prompt

        fourth = agent.step(_request(step=3, text=_ax("Alice")))
        assert fourth["action"].startswith("stop [done")
        assert agent.finished
        session = agent.session
    finally:
        agent.close()

    assert session is not None
    events = _events(tmp_path)
    resumed = next(
        event
        for event in events
        if event["event_type"] == "recovery" and event["result"]["stage"] == "resumed"
    )
    assert resumed["result"]["page_epoch"] == 1
    assert resumed["result"]["control_epoch"] >= 3
    assert resumed["result"]["old_element_numbers_valid"] is False
    # No page action was dispatched during the wait: the re-observe action is the
    # first dispatch after the handoff.
    dispatched = [event for event in events if event["event_type"] == "dispatched"]
    assert [event["action_type"] for event in dispatched] == ["click", "goto", "click"]

    result = json.loads(
        (session.evidence.run_dir / "result.json").read_text("utf-8")
    )
    assert result["termination"] == TerminationReason.COMPLETED.value
    control = result["summary"]["runtime"]["control"]
    timing = control["timing"]
    assert timing["human_wait_seconds"] == pytest.approx(12.0)
    assert timing["total_seconds"] == pytest.approx(12.0)
    assert timing["total_seconds"] == pytest.approx(
        timing["human_wait_seconds"] + timing["execution_seconds"]
    )
    assert timing["platform_deadline_epoch"] == pytest.approx(DEADLINE)
    assert timing["deadline_extended"] is False
    assert control["pause_reason"] == "login_overlay_visible"
    assert control["page_epoch"] == 1
    runtime_recovery = result["summary"]["runtime"]["recovery"]
    assert runtime_recovery["enabled"] is True
    assert runtime_recovery["limits"]["deadline_epoch"] == pytest.approx(DEADLINE)
    assert result["summary"]["runtime"]["behaviors"]["recovery"] is True
    behaviors_event = next(
        event for event in events if event["event_type"] == "behaviors"
    )
    assert behaviors_event["result"]["behaviors"]["recovery"] is True
    assert behaviors_event["result"]["control"]["gate_installed"] is True
    assert behaviors_event["result"]["recovery"]["takeover"] is True
    # The control event log keeps the same reconciled timing as result.json.
    control_end = next(
        event
        for event in _control_events(tmp_path)
        if event["event_type"] == "control_end"
    )
    assert control_end["details"]["timing"]["human_wait_seconds"] == pytest.approx(12.0)
    assert control_end["details"]["timing"]["deadline_extended"] is False


def test_cancel_during_the_handoff_ends_the_task_without_a_reobserve_action(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway([_action("web_click", {"target": "3"})])
    source = CancelDuringHandoffSource()
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(recovery=True),
        control_source=source,
    )
    source.coordinator_getter = lambda: agent.control
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    try:
        assert agent.step(_request(step=0, text=_ax("Initial"))) == {"action": "click [3]"}
        stopped = agent.step(_request(step=1, text=_login_ax()))
        assert stopped["action"].startswith("stop [")
        assert source.sent is True
        assert agent.control is not None
        assert agent.control.state is ControlState.CANCELLED
        assert len(gateway.calls) == 1
        events = _events(tmp_path)
        assert not [event for event in events if event["event_type"] == "dispatched"][1:]
        assert not [
            event
            for event in events
            if event["event_type"] == "recovery" and event["result"]["stage"] == "reobserve"
        ]
    finally:
        agent.close()

    result_path = _run_dir(tmp_path) / "result.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text("utf-8"))
    assert payload["termination"] == TerminationReason.CANCELLED.value


# ------------------------------------------------------------------------ E03 (b)
def test_completion_refusal_withdraws_the_affected_claim_and_keeps_unrelated_progress(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _goal("g1", "visible_value_equals", "Alice", role="textbox", name="Biography"),
            _goal("g2", "page_text_contains", "Saved", depends_on=["g1"]),
            _goal("g3", "visible_value_equals", "Oslo", role="textbox", name="City"),
            _action("web_click", {"target": "3"}),
            _action("request_finish", {}),
            _action("web_type", {"target": "1", "text": "Alice", "submit": False}),
            _action("request_finish", {}),
        ]
    )
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(recovery=True),
    )
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    try:
        # Step 0 declares three requirements and proposes one read-only click.
        first = agent.step(_request(step=0, text=_ax("Initial")))
        assert first == {"action": "click [3]"}
        store = agent.store
        assert store is not None

        # Step 1: the fresh page still shows the wrong biography. The completion
        # check refuses, verifies g2/g3 and then the recovery wiring withdraws the
        # affected g2 (verified -> awaiting_verification, no new conflicting
        # evidence because this observation re-verified it) and preserves g3.
        second = agent.step(_request(step=1, text=_ax("Wrong append")))
        assert second == {"action": "type [1] [Alice] [0]"}
        state = _state_json(tmp_path)
        assert state["goals"]["g1"]["status"] == "running"
        assert state["goals"]["g2"]["status"] == "awaiting_verification"
        assert state["goals"]["g3"]["status"] == "verified"
        records = _invalidations(tmp_path)
        assert len(records) == 1
        record = records[0]
        assert record["goal_id"] == "g2"
        assert record["from_status"] == "verified"
        assert record["to_status"] == "awaiting_verification"
        assert record["changed"] is True
        assert record["evidence_ids"] == record["previous_evidence_ids"]
        assert record["appended_evidence_ids"] == []

        refusal_prompt = gateway.prompt(5)
        assert "completion is NOT confirmed" in refusal_prompt
        assert "Recovery:" in refusal_prompt
        assert "withdrawn completion claims: g2" in refusal_prompt
        assert "preserved unrelated verified requirements: g3" in refusal_prompt
        assert "Redo only the withdrawn or remaining requirements" in refusal_prompt

        # Step 2: the corrected value makes g1 pass; g2 is re-verified from the
        # fresh evidence, g3 is already verified and is never re-checked.
        third = agent.step(_request(step=2, text=_ax("Alice")))
        assert third["action"].startswith("stop [completion verified")
        assert agent.finished
        session = agent.session
    finally:
        agent.close()

    state = _state_json(tmp_path)
    assert state["goals"]["g1"]["status"] == "verified"
    assert state["goals"]["g2"]["status"] == "verified"
    assert state["goals"]["g3"]["status"] == "verified"
    # The withdrawn claim keeps its requirement, criteria and old evidence; the
    # later re-verification appends the fresh evidence.
    assert state["goals"]["g2"]["requirement"] == "requirement g2"
    g2_evidence = [item["evidence_id"] for item in state["goals"]["g2"]["evidence"]]
    assert len(set(g2_evidence)) >= 2
    g3_evidence = [item["evidence_id"] for item in state["goals"]["g3"]["evidence"]]
    assert g3_evidence == ["ev-obs-1-screenshot"]

    recovery_events = [
        event
        for event in _events(tmp_path)
        if event["event_type"] == "recovery"
        and event["result"].get("stage") == "completion_refusal"
    ]
    assert len(recovery_events) == 1
    payload = recovery_events[0]["result"]
    assert payload["applied"] is True
    assert payload["withdrawn_goal_ids"] == ["g2"]
    assert payload["preserved_verified_goal_ids"] == ["g3"]
    assert payload["allowed"] is True
    assert payload["attempt"]["total_attempts"] == 1

    dispatched = [event for event in _events(tmp_path) if event["event_type"] == "dispatched"]
    assert [event["action_type"] for event in dispatched] == ["click", "type"]
    assert session is not None
    result = json.loads((session.evidence.run_dir / "result.json").read_text("utf-8"))
    assert result["termination"] == TerminationReason.COMPLETED.value
    summary = result["summary"]["runtime"]["recovery"]
    assert summary["applied"] == 1
    assert summary["attempts"]["per_goal"] == {"g1": 1, "g2": 1}


# ------------------------------------------------------------------------ E04 (c)
def test_stagnation_recovery_is_bounded_and_then_ends_honestly(tmp_path: Path) -> None:
    clock = FakeClock()
    responses = [_goal("g1", "visible_value_equals", "Never", role="textbox", name="Biography")]
    responses.extend(_action("web_click", {"target": "3"}) for _ in range(8))
    gateway = FakeGateway(responses)
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(recovery=True),
    )
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    try:
        result = None
        for step in range(8):
            result = agent.step(_request(step=step, text=_ax("Initial")))
            assert result["action"] == "click [3]"
        # The eighth feedback is the third stagnation signal; the per-goal limit
        # refuses the attempt and the run ends honestly instead of looping.
        result = agent.step(_request(step=8, text=_ax("Initial")))
        assert result is not None
        assert result["action"].startswith("stop [")
        assert "per-goal recovery limit" in result["action"]
        assert "Partial progress is not complete" in result["action"]
        driver = agent.driver
        assert driver is not None
        assert driver.terminal is not None
        assert driver.terminal.status.value == "incomplete"
        assert agent.finished
        session = agent.session
        state = agent.store
        assert state is not None
    finally:
        agent.close()

    recovery_events = [
        event for event in _events(tmp_path) if event["event_type"] == "recovery"
    ]
    applied = [
        event
        for event in recovery_events
        if event["result"].get("stage") == "stagnation" and event["result"]["applied"]
    ]
    denied = [
        event
        for event in recovery_events
        if event["result"].get("stage") == "stagnation_limit"
    ]
    assert len(applied) == 2, [event["result"] for event in recovery_events]
    assert len(denied) == 1
    assert denied[0]["result"]["allowed"] is False
    assert "per-goal recovery limit" in denied[0]["result"]["reason"]
    assert state.state.goals["g1"].status.value == "running"
    runtime_summary = json.loads(
        (session.evidence.run_dir / "result.json").read_text("utf-8")
    )["summary"]["runtime"]["recovery"]
    assert runtime_summary["attempts"]["per_goal"] == {"g1": 2}
    assert runtime_summary["denied"] == 1
    result_payload = json.loads(
        (session.evidence.run_dir / "result.json").read_text("utf-8")
    )
    assert result_payload["termination"] == TerminationReason.INCOMPLETE.value
    # A stalled run must not keep consuming model turns.
    assert len(gateway.calls) <= 9


# ------------------------------------------------------------- E04 deadline unit
def test_recovery_deadline_stops_attempts_through_the_wiring_runtime(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    store = bootstrap_store(runtime_dir, [make_goal("g1", status="running")])
    clock = FakeClock()
    runtime = OfficialRecoveryRuntime(
        store,
        None,
        settings=RecoverySettings(
            stagnation_window=2, stagnation_repeats=2, max_recoveries_total=5
        ),
        clock=clock,
        deadline_epoch=clock.now() + 5,
    )
    site = FakeSite(
        [
            FakePage(
                url="http://fixture.local/form",
                title="Fixture form",
                ax_text="A form with one control.",
                elements=(FakeElement(role="button", name="Search"),),
            )
        ]
    )
    backend = FakeBackend("run-a", site, start_url="http://fixture.local/form")

    def one_step() -> tuple[Any, Any]:
        observation = backend.observe()
        action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
        result = backend.execute(action)
        return action, result.observation

    try:
        for _ in range(2):
            action, observation = one_step()
            runtime.after_feedback(action, observation)
        clock.advance(6)
        action, observation = one_step()
        note = runtime.after_feedback(action, observation)
    finally:
        backend.close()
    assert note.terminal is True
    assert "deadline" in (note.terminal_reason or "")
    # The first attempt was authorized before the deadline; the expired one was not.
    assert runtime.controller.total_attempts == 1
    assert store.state.goals["g1"].status.value == "running"


# ------------------------------------------------------------------------ E08 (d)
def test_restart_loads_the_snapshot_associates_a_new_branch_and_does_not_resend(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    first_gateway = FakeGateway(
        [
            _goal("g1", "visible_value_equals", "Alice", role="textbox", name="Biography"),
            _action("web_click", {"target": "3"}),
        ]
    )
    page_with_save = (
        "[1] textbox 'Biography' required: False\n"
        '  [2] StaticText "Initial"\n'
        "[3] button 'Save draft'\n"
        "[4] button 'Cancel draft'"
    )
    agent = _make_agent(
        tmp_path,
        clock,
        first_gateway,
        behaviors=OfficialBehaviors(recovery=True),
    )
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    assert agent.step(_request(step=0, text=page_with_save)) == {"action": "click [3]"}
    store = agent.store
    assert store is not None
    pending = store.state.pending_effects
    assert len(pending) == 1
    effect = next(iter(pending.values()))
    pending_action_id = effect.action_id
    assert effect.status.value == "pending"
    assert effect.note.startswith("draft_save")
    # The process dies here: the action took effect, its feedback was never
    # recorded and the write is still marked pending.
    agent.close()
    assert StateStore(tmp_path / "runtime", RUN_ID).load().pending_effects

    second_gateway = FakeGateway(
        [
            _action("web_click", {"target": "3"}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    restarted = _make_agent(
        tmp_path,
        clock,
        second_gateway,
        behaviors=OfficialBehaviors(recovery=True),
    )
    restarted.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    restarted_run_id = "official-run-1-rs1"
    assert restarted.local_run_id == restarted_run_id
    try:
        resumed = restarted.step(_request(step=0, text=_restart_ax(), run_id=RUN_ID))
        assert resumed == {"action": "click [3]"}
        prompt = second_gateway.prompt(0)
        assert "Restart note" in prompt
        assert "was NOT restored" in prompt
        assert pending_action_id in prompt
        assert "do not resend the same write blindly" in prompt
        state = restarted.store
        assert state is not None
        assert state.state.branches == {
            "root": RUN_ID,
            "restart-1": restarted_run_id,
        }
        assert state.state.goals["g1"].status.value == "running"
        residue = state.state.pending_effects[pending_action_id]
        assert residue.status.value == "pending"
        assert state.state.revision >= 3

        restarted.step(_request(step=1, text=_restart_ax(), value="Alice"))
        assert restarted.finished
        session = restarted.session
    finally:
        restarted.close()

    assert session is not None
    new_events = _events(tmp_path, restarted_run_id)
    assert not any(
        event["event_type"] == "dispatched"
        and event["action_id"] == pending_action_id
        for event in new_events
    )
    assert [event["action_type"] for event in new_events if event["event_type"] == "dispatched"] == ["click"]
    restart_events = [
        event
        for event in new_events
        if event["event_type"] == "recovery" and event["result"].get("stage") == "restart"
    ]
    assert len(restart_events) == 1
    detail = restart_events[0]["result"]
    assert detail["generator_stack_restored"] is False
    assert detail["replayed_actions"] == []
    assert detail["pending_effect_ids"] == [pending_action_id]
    assert detail["branches"] == {"root": RUN_ID, "restart-1": restarted_run_id}
    control_types = [event["event_type"] for event in _control_events(tmp_path, restarted_run_id)]
    assert "restart_loaded" in control_types
    assert "control_start" in control_types
    restart_loaded = next(
        event
        for event in _control_events(tmp_path, restarted_run_id)
        if event["event_type"] == "restart_loaded"
    )
    assert restart_loaded["details"]["generator_restored"] is False
    result = json.loads((session.evidence.run_dir / "result.json").read_text("utf-8"))
    runtime_summary = result["summary"]["runtime"]
    assert runtime_summary["restart"]["branch_id"] == "restart-1"
    assert runtime_summary["restart"]["generator_stack_restored"] is False
    assert runtime_summary["restart"]["pending_effect_ids"] == [pending_action_id]
    # The reloaded snapshot still keeps the residue pending; nothing confirms or
    # re-sends it implicitly.
    reloaded = StateStore(tmp_path / "runtime", RUN_ID)
    reloaded.load()
    assert reloaded.state.pending_effects[pending_action_id].status.value == "pending"
    assert reloaded.state.branches["restart-1"] == restarted_run_id


# ------------------------------------------------------------------------ P04 (e)
def test_recovery_off_keeps_the_pre_rwave_behavior(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_click", {"target": "3"}),
            _action("web_click", {"target": "1"}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(verification=True, state_tools=True, recovery=False),
    )
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    try:
        page_with_save = (
            "[1] textbox 'Biography' required: False\n"
            '  [2] StaticText "Initial"\n'
            "[3] button 'Save draft'"
        )
        first = agent.step(_request(step=0, text=page_with_save))
        assert first == {"action": "click [3]"}
        assert agent.control is None
        assert agent.recovery is None
        state = agent.store
        assert state is not None
        assert state.state.pending_effects == {}

        # The login overlay is advanced exactly like before the R wave: the
        # generator continues with the overlay observation and the model's next
        # action is returned. There is no pause, no handoff and no goto.
        second = agent.step(_request(step=1, text=_login_ax(), url="http://forum.test/login"))
        assert second == {"action": "click [1]"}
        assert len(gateway.calls) == 2

        third = agent.step(_request(step=2, text=_ax("Alice")))
        assert third["action"].startswith("stop [done")
        session = agent.session
    finally:
        agent.close()

    assert session is not None
    events = _events(tmp_path)
    assert not [event for event in events if event["event_type"] == "recovery"]
    assert not [
        event for event in events if event["action_type"] == "goto"
    ]
    assert not (_run_dir(tmp_path) / "control_events.jsonl").exists()
    assert not (_run_dir(tmp_path) / "control.jsonl").exists()

    result = json.loads((session.evidence.run_dir / "result.json").read_text("utf-8"))
    runtime_summary = result["summary"]["runtime"]
    assert runtime_summary["behaviors"] == {
        "verification": True,
        "state_tools": True,
        "recovery": False,
    }
    assert "control" not in runtime_summary
    assert "recovery" not in runtime_summary
    assert "restart" not in runtime_summary
    behaviors_events = [
        event for event in events if event["event_type"] == "behaviors"
    ]
    assert behaviors_events[-1]["result"]["behaviors"]["recovery"] is False
    # The captured final content carries no recovery/handoff prompt or schema.
    for index in range(len(gateway.calls)):
        prompt = gateway.prompt(index)
        assert "Recovery:" not in prompt
        assert "handoff" not in prompt
        assert "was NOT restored" not in prompt
    # Early close-out stays disabled after this wave.
    assert result["summary"]["runtime"]["deadline_margin_seconds"] == 0.0


def test_explicit_user_pause_records_the_in_flight_result_and_resumes_fresh(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_click", {"target": "3"}),
            _action("web_click", {"target": "3"}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    source = HandoffProbeSource(clock, wait_seconds=8.0)
    source.send("pause")
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(recovery=True),
        control_source=source,
    )
    source.coordinator_getter = lambda: agent.control
    source.probe = lambda: {"model_calls": len(gateway.calls)}
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    try:
        assert agent.step(_request(step=0, text=_ax("Initial"))) == {"action": "click [3]"}
        paused = agent.step(_request(step=1, text=_ax("Initial")))
        assert paused == {"action": "goto [http://forum.test/profile]"}
        # The action really ran, so its result is recorded truthfully before the
        # handoff; the wait spends no model call and no page action.
        feedback_events = [
            event
            for event in _events(tmp_path)
            if event["event_type"] == "feedback"
        ]
        assert len(feedback_events) == 1
        assert feedback_events[0]["result"] == "observed"
        assert len(gateway.calls) == 1
        assert source.probe_at_handoff == source.probe_after_wait
        pause_requested = next(
            event
            for event in _control_events(tmp_path)
            if event["event_type"] == "pause_requested"
        )
        assert pause_requested["reason"] == "user_pause"
        assert pause_requested["source"] == "user"

        resumed = agent.step(_request(step=2, text=_ax("Alice")))
        assert resumed == {"action": "click [3]"}
        assert "Fresh observation after the user resumed" in gateway.prompt(1)
        agent.step(_request(step=3, text=_ax("Alice")))
        assert agent.finished
        session = agent.session
    finally:
        agent.close()

    assert session is not None
    resumed_event = next(
        event
        for event in _events(tmp_path)
        if event["event_type"] == "recovery" and event["result"]["stage"] == "resumed"
    )
    assert resumed_event["result"]["page_epoch"] == 1
    result = json.loads((session.evidence.run_dir / "result.json").read_text("utf-8"))
    control = result["summary"]["runtime"]["control"]
    assert control["pause_reason"] == "user_pause"
    assert control["timing"]["human_wait_seconds"] == pytest.approx(8.0)


def test_action_effect_anomaly_runs_the_bounded_recovery_path(tmp_path: Path) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _goal("g1", "visible_value_equals", "Alice", role="textbox", name="Biography"),
            _action("web_click", {"target": "3"}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(recovery=True),
    )
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    try:
        page = (
            "[1] textbox 'Biography' required: False\n"
            '  [2] StaticText "Initial"\n'
            "[3] button 'Save draft'"
        )
        assert agent.step(_request(step=0, text=page)) == {"action": "click [3]"}
        failing = _request(step=1, text=page)
        failing["action_error"] = {
            "type": "action_failed",
            "message": "the save did not take effect",
        }
        agent.step(failing)
        store = agent.store
        assert store is not None
        effects = list(store.state.pending_effects.values())
        assert len(effects) == 1
        assert effects[0].status.value == "failed"
        assert agent.finished
        session = agent.session
    finally:
        agent.close()

    assert session is not None
    recovery_events = [
        event
        for event in _events(tmp_path)
        if event["event_type"] == "recovery"
        and event["result"].get("stage") == "action_failure"
    ]
    assert len(recovery_events) == 1
    payload = recovery_events[0]["result"]
    assert payload["applied"] is True
    assert payload["allowed"] is True
    assert payload["plan"]["affected_goal_ids"] == ["g1"]
    assert payload["attempt"]["total_attempts"] == 1
    # The model receives one bounded correction, not a replay of the failed write.
    assert "Recovery:" in gateway.prompt(2)
    assert "unknown_effect" in gateway.prompt(2)
    result = json.loads((session.evidence.run_dir / "result.json").read_text("utf-8"))
    assert result["termination"] == TerminationReason.INCOMPLETE.value


def test_takeover_visible_on_the_first_observation_pauses_before_any_model_turn(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_click", {"target": "3"}),
            "Thought: done.\nFinal Answer: done",
        ]
    )
    source = HandoffProbeSource(clock, wait_seconds=5.0)
    agent = _make_agent(
        tmp_path,
        clock,
        gateway,
        behaviors=OfficialBehaviors(recovery=True),
        control_source=source,
    )
    source.coordinator_getter = lambda: agent.control
    source.probe = lambda: {"model_calls": len(gateway.calls)}
    agent.reset({"run_id": RUN_ID, "execution_deadline": DEADLINE})
    try:
        first = agent.step(_request(step=0, text=_login_ax(), url="http://forum.test/login"))
        assert first == {"action": "goto [http://forum.test/login]"}
        assert gateway.calls == []
        assert source.probe_at_handoff == source.probe_after_wait
        driver = agent.driver
        assert driver is not None
        assert driver.agent is not None
        assert not driver.started

        second = agent.step(_request(step=1, text=_ax("Initial")))
        assert second == {"action": "click [3]"}
        assert len(gateway.calls) == 1
        # The loop opens on the fresh page only; the login overlay is not part of
        # the model's first prompt.
        prompt = gateway.prompt(0)
        assert "Your session has expired" not in prompt
        assert '"Initial"' in prompt
        agent.step(_request(step=2, text=_ax("Initial")))
        assert agent.finished
        session = agent.session
    finally:
        agent.close()

    assert session is not None
    control_types = [event["event_type"] for event in _control_events(tmp_path)]
    assert "takeover_detected" in control_types
    assert "handoff" in control_types
    resumed_event = next(
        event
        for event in _events(tmp_path)
        if event["event_type"] == "recovery" and event["result"]["stage"] == "resumed"
    )
    assert resumed_event["result"]["page_epoch"] == 1
    # No model call event exists before the handoff completed.
    model_calls = [event for event in _events(tmp_path) if event["event_type"] == "model_call"]
    assert len(model_calls) >= 1


# ------------------------------------------------------- evidence target (unit)
def test_official_ax_overlay_copy_is_read_from_statictext_lines(make_observation) -> None:
    """The official AX prints overlay copy as numbered StaticText, not TEXT blocks."""

    ax = (
        "[1] button 'Sign in'\n"
        '[2] StaticText "Your session has expired. Sign in in this browser to '
        'restore the page."'
    )
    observation = make_observation(ax_text=ax, elements={"1": ("button", "Sign in")})
    signal = detect_takeover(observation)
    assert signal is not None
    assert signal.kind == "login"
    assert signal.reason == "login_overlay_visible"
    assert signal.matched_copy
    assert signal.matched_controls


def test_visible_sign_in_copy_without_an_action_control_does_not_trigger(
    make_observation,
) -> None:
    """StaticText copy stays visible text; it is not itself an action control."""

    text = "Please sign in to continue to your account."
    ax = (
        f'[1] StaticText "{text}"\n'
        "[2] button 'Apply filters'"
    )
    observation = make_observation(
        ax_text=ax,
        elements={"1": ("StaticText", text), "2": ("button", "Apply filters")},
    )
    assert detect_takeover(observation) is None


def test_withdrawal_with_new_conflicting_evidence_invalidates_and_keeps_sources(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    store = bootstrap_store(
        runtime_dir,
        [
            verified_goal(runtime_dir, "g3", depends_on=("g2",)),
            make_goal("g2", status="running"),
            verified_goal(runtime_dir, "g1"),
        ],
    )
    clock = FakeClock()
    wiring = OfficialRecoveryRuntime(store, None, clock=clock)
    observation = make_observation(runtime_dir, observation_id="obs-conflict")
    ref = evidence_ref(runtime_dir, observation, evidence_id="ev-conflict")
    decision = SimpleNamespace(
        results=[
            SimpleNamespace(
                goal_id="g2",
                status=VerificationStatus.FAIL,
                reason="visible value mismatch",
                evidence=(ref,),
            )
        ]
    )
    note = wiring.after_completion_refusal(decision, observation)
    assert "withdrawn completion claims: g3" in note.text
    assert "preserved unrelated verified requirements: g1" in note.text

    state = store.state
    assert state.goals["g3"].status.value == "invalidated"
    assert state.goals["g2"].status.value == "running"
    assert state.goals["g1"].status.value == "verified"
    assert state.verified_goal_ids == ("g1",)
    records = store.read_invalidations()
    assert len(records) == 1
    record = records[0]
    assert record.from_status.value == "verified"
    assert record.to_status.value == "invalidated"
    assert record.changed is True
    assert record.appended_evidence_ids == ("ev-conflict",)
    assert "ev-conflict" in record.evidence_ids
    assert record.previous_evidence_ids
    # The record, requirement, criteria and old evidence are kept.
    goal = state.goals["g3"]
    assert goal.requirement == "requirement g3"
    assert goal.acceptance_criteria
    assert set(record.previous_evidence_ids) <= set(
        ref.evidence_id for ref in goal.evidence
    )


def test_withdrawal_without_new_evidence_does_not_invent_a_cause(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    store = bootstrap_store(
        runtime_dir,
        [
            verified_goal(runtime_dir, "g3", depends_on=("g2",)),
            make_goal("g2", status="running"),
            verified_goal(runtime_dir, "g1"),
        ],
    )
    wiring = OfficialRecoveryRuntime(store, None, clock=FakeClock())
    observation = make_observation(runtime_dir, observation_id="obs-no-conflict")
    decision = SimpleNamespace(
        results=[
            SimpleNamespace(
                goal_id="g2",
                status=VerificationStatus.FAIL,
                reason="condition could not be read",
                evidence=(),
            )
        ]
    )
    wiring.after_completion_refusal(decision, observation)
    state = store.state
    # The claim is withdrawn but no cause is invented.
    assert state.goals["g3"].status.value == "awaiting_verification"
    assert state.goals["g1"].status.value == "verified"
    record = store.read_invalidations()[0]
    assert record.to_status.value == "awaiting_verification"
    assert record.appended_evidence_ids == ()


def test_repeated_action_with_new_fact_evidence_is_not_stagnation(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    store = bootstrap_store(runtime_dir, [make_goal("g1", status="running")])
    clock = FakeClock()
    wiring = OfficialRecoveryRuntime(
        store,
        None,
        settings=RecoverySettings(stagnation_window=2, stagnation_repeats=2),
        clock=clock,
    )
    site = FakeSite(
        [
            FakePage(
                url="http://fixture.local/form",
                title="Fixture form",
                ax_text="A form with one control.",
                elements=(FakeElement(role="button", name="Search"),),
            )
        ]
    )
    backend = FakeBackend("run-a", site, start_url="http://fixture.local/form")
    try:
        notes = []
        for index in range(4):
            observation = backend.observe()
            action = build_web_action(
                ToolRequest("web_click", {"target": "1"}), observation
            )
            result = backend.execute(action)
            fact = Fact(
                fact_id=f"fact-{index}",
                object_id="form",
                field="note",
                value=index,
                kind=FactKind.INFERRED,
                source="test evidence",
                revision=1,
                branch_id="root",
            )
            state = store.state
            store.apply_update(
                StateUpdate(
                    base_revision=state.revision,
                    plan_version=state.plan_version,
                    source_branch="root",
                    facts=(fact,),
                    note="new observed fact",
                )
            )
            notes.append((action, result.observation))
            wiring.after_feedback(action, result.observation)
    finally:
        backend.close()
    assert not wiring.denied
    assert wiring.detector.evaluate() is None
    assert store.state.goals["g1"].status.value == "running"
    # The repeated action produced new evidence every step, so no bounded
    # recovery attempt was spent.
    assert wiring.controller.total_attempts == 0
