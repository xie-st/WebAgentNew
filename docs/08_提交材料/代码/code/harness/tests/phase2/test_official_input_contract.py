"""P01: official input contract for empty text and post-submit forensics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from webagent.backends.official import (
    EMPTY_TYPE_CORRECTION,
    TYPE_SUBMIT_ENTER_COMMAND,
    OfficialBackend,
    TextMode,
)
from webagent.clock import FakeClock
from webagent.errors import ActionRejected, RejectReason
from webagent.events import read_events
from webagent.protocol import ToolRequest
from webagent.session import Session
from webagent.tools import build_web_action

START = "http://example.test/start"
NEXT = "http://example.test/next"
EMPTY_TYPE_WIRE = "type [1] [] [1]"


def _payload(
    *,
    url: str = START,
    text: str = "[1] textbox 'Query'\n[2] button 'Search'",
    **extra: Any,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "type": "accessibility_tree",
        "url": url,
        "text": text,
    }
    observation.update(extra)
    return {
        "protocol_version": "1.0",
        "run_id": "run-a",
        "step": 0,
        "observation": observation,
        "history": [],
        "action_history": [],
    }


@pytest.fixture
def official(tmp_path: Path):
    clock = FakeClock()
    session = Session(
        "run-a",
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        page_id="official-page",
    )
    session.start_task("official input contract")
    backend = OfficialBackend(
        "run-a",
        session=session,
        evidence=session.evidence,
        browser_api=None,
        download_artifacts=frozenset(),
    )
    try:
        yield session, backend
    finally:
        session.close()


def _action(backend: OfficialBackend, tool: str, args: dict[str, Any], action_id: str):
    observation = backend.current_observation
    assert observation is not None
    return build_web_action(ToolRequest(tool, args), observation, action_id=action_id)


def _markers(session: Session) -> list[dict[str, Any]]:
    return [
        event
        for event in read_events(session.evidence.events_path)
        if isinstance(event.get("result"), dict)
        and event["result"].get("post_submit_evidence") is True
    ]


def test_empty_type_with_submit_becomes_enter_instead_of_an_empty_type(official) -> None:
    _session, backend = official
    backend.from_request(_payload())
    action = _action(
        backend,
        "web_type",
        {"target": "1", "text": "", "submit": True},
        "act-empty-submit",
    )

    command = backend.serialize_action(action)
    assert command == TYPE_SUBMIT_ENTER_COMMAND == "press [Enter]"
    assert command != EMPTY_TYPE_WIRE

    # Both wire-text modes have the same contract: an empty type field is never
    # sent to the platform with submit=1.
    backend.text_mode = TextMode.RAW
    assert backend.serialize_action(action) == "press [Enter]"
    backend.text_mode = TextMode.DECOMPOSE


def test_empty_type_without_submit_is_refused_locally_with_model_facing_text(
    official,
) -> None:
    session, backend = official
    backend.from_request(_payload())
    action = _action(
        backend,
        "web_type",
        {"target": "1", "text": "", "submit": False},
        "act-empty-type",
    )

    with pytest.raises(ActionRejected) as caught:
        backend.serialize_action(action)

    error = caught.value
    assert error.reason is RejectReason.INVALID_ARGUMENTS
    assert error.correction == EMPTY_TYPE_CORRECTION
    assert "non-empty" in error.correction
    assert "web_press" in error.correction and "Enter" in error.correction
    assert "empty" in "; ".join(error.details)

    corrections = [
        event
        for event in read_events(session.evidence.events_path)
        if event["event_type"] == "format_correction"
    ]
    assert len(corrections) == 1
    assert corrections[0]["result"]["reason"] == "empty_type_text"
    assert corrections[0]["result"]["dispatched"] is False
    assert corrections[0]["result"]["correction"] == EMPTY_TYPE_CORRECTION


def test_non_empty_type_wire_format_is_unchanged(official) -> None:
    _session, backend = official
    backend.from_request(_payload())
    action = _action(
        backend,
        "web_type",
        {"target": "1", "text": "中文 [备注]", "submit": True},
        "act-text",
    )
    assert backend.serialize_action(action) == "type [1] [中文 [备注]] [1]"


def test_submit_observation_records_a_post_submit_evidence_marker(official) -> None:
    session, backend = official
    first = backend.from_request(_payload())
    action = _action(
        backend,
        "web_type",
        {"target": "1", "text": "notes", "submit": True},
        "act-submit",
    )
    assert backend.serialize_action(action) == "type [1] [notes] [1]"

    second = backend.from_request(
        _payload(step=1, url=NEXT, text="[1] textbox 'Query'\n[2] button 'Search'"),
        previous_action=action,
        previous_observation=first,
    )

    markers = _markers(session)
    assert len(markers) == 1
    marker = markers[0]
    assert marker["observation_id"] == second.observation_id
    assert marker["action_id"] == action.action_id
    assert marker["result"]["kind"] == "post_submit_observation"
    assert marker["result"]["submit_action_id"] == action.action_id
    assert marker["result"]["submit_target"] == "1"
    assert marker["result"]["submit_text_chars"] == len("notes")
    assert marker["result"]["submit_was_empty"] is False


def test_empty_text_submit_still_marks_the_resulting_observation(official) -> None:
    session, backend = official
    first = backend.from_request(_payload())
    action = _action(
        backend,
        "web_type",
        {"target": "1", "text": "", "submit": True},
        "act-empty-submit",
    )
    assert backend.serialize_action(action) == "press [Enter]"

    second = backend.from_request(
        _payload(step=1, url=START, text="[1] textbox 'Query'"),
        previous_action=action,
        previous_observation=first,
    )
    markers = _markers(session)
    assert len(markers) == 1
    assert markers[0]["observation_id"] == second.observation_id
    assert markers[0]["result"]["submit_was_empty"] is True
    assert markers[0]["result"]["submit_text_chars"] == 0


def test_non_submit_observation_has_no_post_submit_marker(official) -> None:
    session, backend = official
    first = backend.from_request(_payload())
    action = _action(
        backend,
        "web_type",
        {"target": "1", "text": "notes", "submit": False},
        "act-no-submit",
    )
    assert backend.serialize_action(action) == "type [1] [notes] [0]"

    backend.from_request(
        _payload(step=1, url=NEXT, text="[1] link 'Next'"),
        previous_action=action,
        previous_observation=first,
    )
    assert _markers(session) == []
