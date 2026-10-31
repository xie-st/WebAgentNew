from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from webagent.errors import ProtocolError
from webagent.protocol import (
    ActionOp,
    Artifact,
    ArtifactSet,
    Capabilities,
    ElementRef,
    ExecutionResult,
    Feedback,
    FeedbackStatus,
    Finish,
    Observation,
    TerminationReason,
    WebAction,
)


def test_observation_round_trip_preserves_localization(fixtures_dir: Path) -> None:
    raw = json.loads((fixtures_dir / "observations" / "search.json").read_text(encoding="utf-8"))
    observation = Observation.from_dict(raw)
    assert observation.elements["1"].localization["selector"] == "#q"
    again = Observation.from_dict(json.loads(json.dumps(observation.to_local_dict())))
    assert again.to_local_dict() == observation.to_local_dict()
    assert again.elements["2"].name == "Search"


def test_prompt_view_hides_backend_localization(fixtures_dir: Path) -> None:
    raw = json.loads((fixtures_dir / "observations" / "search.json").read_text(encoding="utf-8"))
    observation = Observation.from_dict(raw)
    prompt = observation.to_prompt_dict()
    text = json.dumps(prompt, ensure_ascii=False)
    assert "localization" not in text
    assert "cdp_backend_node_id" not in text
    assert prompt["elements"] == [
        {"number": "1", "role": "textbox", "name": "Search query"},
        {"number": "2", "role": "button", "name": "Search"},
    ]


def test_element_must_be_bound_to_its_observation() -> None:
    element = ElementRef(
        page_id="page-1", observation_id="obs-1", number="1", role="button", name="Go"
    )
    with pytest.raises(ProtocolError):
        Observation(
            run_id="run-a",
            page_id="page-1",
            observation_id="obs-2",
            url="http://fixture.local/",
            ax_text="x",
            elements={"1": element},
        )
    with pytest.raises(ProtocolError):
        Observation(
            run_id="run-a",
            page_id="page-1",
            observation_id="obs-1",
            url="http://fixture.local/",
            ax_text="x",
            elements={"2": element},
        )


def test_feedback_status_rules() -> None:
    with pytest.raises(ProtocolError):
        Feedback(status=FeedbackStatus.OBSERVED, action_id="act-1", observation_id=None, message="x")
    with pytest.raises(ProtocolError):
        Feedback(status=FeedbackStatus.ERROR, action_id="act-1", observation_id=None, message="x")
    unknown = Feedback(
        status="unknown", action_id="act-1", observation_id=None, message="no execution evidence"
    )
    assert unknown.status is FeedbackStatus.UNKNOWN


def test_observation_feedback_binding_rules() -> None:
    feedback = Feedback(
        status=FeedbackStatus.OBSERVED,
        action_id="act-1",
        observation_id="obs-2",
        message="clicked",
    )
    with pytest.raises(ProtocolError):
        Observation(
            run_id="run-a",
            page_id="page-1",
            observation_id="obs-2",
            url="http://fixture.local/",
            ax_text="x",
            previous_action_id="act-2",
            feedback=feedback,
        )
    with pytest.raises(ProtocolError):
        Observation(
            run_id="run-a",
            page_id="page-1",
            observation_id="obs-2",
            url="http://fixture.local/",
            ax_text="x",
            previous_action_id=None,
            feedback=feedback,
        )


def test_web_action_argument_validation() -> None:
    action = WebAction(
        action_id="act-1",
        run_id="run-a",
        page_id="page-1",
        observation_id="obs-1",
        op="type",
        args={"target": "1", "text": "hello"},
    )
    assert action.submit is False
    assert action.text == "hello"
    with pytest.raises(ProtocolError):
        WebAction(
            action_id="act-1",
            run_id="run-a",
            page_id="page-1",
            observation_id="obs-1",
            op="type",
            args={"target": "1", "text": "hello", "clear": True},
        )
    with pytest.raises(ProtocolError):
        WebAction(
            action_id="act-1",
            run_id="run-a",
            page_id="page-1",
            observation_id="obs-1",
            op="goto",
            args={"url": "/relative"},
        )
    with pytest.raises(ProtocolError):
        WebAction(
            action_id="act-1",
            run_id="run-a",
            page_id="page-1",
            observation_id="obs-1",
            op="go_back",
            args={"target": "1"},
        )
    scroll = WebAction(
        action_id="act-2",
        run_id="run-a",
        page_id="page-1",
        observation_id="obs-1",
        op="scroll",
        args={"direction": "down"},
    )
    assert scroll.direction == "down"


def test_execution_result_checks_successor_binding(make_observation) -> None:
    feedback = Feedback(
        status=FeedbackStatus.OBSERVED,
        action_id="act-1",
        observation_id="obs-2",
        message="clicked",
    )
    observation = make_observation(
        observation_id="obs-2", previous_action_id="act-1", feedback=feedback
    )
    ExecutionResult("act-1", feedback, observation)
    wrong_feedback = Feedback(
        status=FeedbackStatus.OBSERVED,
        action_id="act-2",
        observation_id="obs-2",
        message="clicked",
    )
    with pytest.raises(ProtocolError):
        ExecutionResult("act-1", wrong_feedback, observation)
    wrong_successor = make_observation(observation_id="obs-3")
    with pytest.raises(ProtocolError):
        ExecutionResult("act-1", feedback, wrong_successor)


def test_unknown_feedback_may_lack_observation_reference(make_observation) -> None:
    unknown = Feedback(
        status=FeedbackStatus.UNKNOWN,
        action_id="act-1",
        observation_id=None,
        message="no execution evidence",
    )
    observation = make_observation(
        observation_id="obs-2", previous_action_id="act-1", feedback=unknown
    )
    ExecutionResult("act-1", unknown, observation)


def test_finish_has_reason_not_official_score() -> None:
    finish = Finish(claim="done", message="finished the offline task", reason="completed")
    payload = finish.to_dict()
    assert payload["reason"] == "completed"
    assert "score" not in payload
    with pytest.raises(ProtocolError):
        Finish(claim="done", message="x", reason="passed")


def test_protocol_values_are_immutable() -> None:
    action = WebAction(
        action_id="act-1",
        run_id="run-a",
        page_id="page-1",
        observation_id="obs-1",
        op="click",
        args={"target": "1"},
    )
    with pytest.raises(TypeError):
        action.args["target"] = "2"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        action.op = ActionOp.TYPE  # type: ignore[misc]
    element = ElementRef(
        page_id="page-1", observation_id="obs-1", number="1", role="button", name="Go"
    )
    with pytest.raises(FrozenInstanceError):
        element.localization = {}  # type: ignore[assignment]


def test_artifacts_distinguish_available_and_missing() -> None:
    unavailable = Artifact(kind="screenshot", available=False, reason="capture_failed")
    assert unavailable.reference is None
    with pytest.raises(ProtocolError):
        Artifact(kind="screenshot", available=False, reason=None)
    with pytest.raises(ProtocolError):
        Artifact(kind="screenshot", available=True, reference="x.png", sha256="bad", byte_size=1)
    available = Artifact(
        kind="screenshot",
        available=True,
        reference="screenshots/x.png",
        sha256="a" * 64,
        byte_size=10,
    )
    artifact_set = ArtifactSet(screenshot=available)
    assert artifact_set.get("screenshot") is available
    with pytest.raises(ProtocolError):
        ArtifactSet(screenshot=available, html=available)


def test_capabilities_default_and_restriction() -> None:
    capabilities = Capabilities()
    assert capabilities.allows(ActionOp.CLICK)
    restricted = Capabilities(actions={ActionOp.CLICK, "scroll"})
    assert restricted.allows(ActionOp.SCROLL)
    assert not restricted.allows(ActionOp.GOTO)
