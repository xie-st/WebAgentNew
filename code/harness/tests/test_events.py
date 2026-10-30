from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from webagent.clock import FakeClock
from webagent.events import REQUIRED_FIELDS, EventLogger, read_events
from webagent.protocol import Finish, ToolRequest
from webagent.redaction import Redactor
from webagent.session import Session
from webagent.testing import FakeBackend
from webagent.tools import build_web_action


def test_required_fields_and_action_stage_distinction(
    backend: FakeBackend, runtime: Path, clock: FakeClock
) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("type into the query box")
    session.ingest_observation(backend.current_observation)
    action = build_web_action(
        ToolRequest("web_type", {"target": "1", "text": "sample query"}),
        session.observation,
        action_id="act-1",
    )
    session.propose(action)
    session.consume()
    result = backend.execute(action)
    session.ingest_observation(result.observation)
    session.record_model_call(
        request_id="req-1",
        input_tokens=120,
        output_tokens=8,
        token_count_source="estimate:chars_div_4",
        latency_ms=4.5,
    )
    session.finish(Finish(claim="typed", message="typed the query", reason="completed"))
    session.close()
    events = read_events(session.evidence.events_path)
    assert events
    for event in events:
        assert set(REQUIRED_FIELDS) <= set(event)
        assert datetime.fromisoformat(event["time"]).tzinfo is not None
        assert event["run_id"] == "run-a"
    proposed = [event for event in events if event["event_type"] == "proposed"]
    dispatched = [event for event in events if event["event_type"] == "dispatched"]
    feedback = [event for event in events if event["event_type"] == "feedback"]
    claims = [event for event in events if event["event_type"] == "agent_claim"]
    assert proposed and dispatched and feedback and claims
    assert proposed[0]["result"] is None
    assert dispatched[0]["result"] is None
    assert feedback[0]["result"] == "observed"
    assert feedback[0]["action_id"] == "act-1"
    assert claims[0]["result"]["reason"] == "completed"
    assert "score" not in claims[0]["result"]
    model_calls = [event for event in events if event["event_type"] == "model_call"]
    assert model_calls[0]["model_request_id"] == "req-1"
    assert model_calls[0]["model_input_tokens"] == 120
    assert model_calls[0]["model_output_tokens"] == 8
    assert model_calls[0]["token_count_source"] == "estimate:chars_div_4"
    assert model_calls[0]["latency_ms"] == 4.5
    assert len({event["event_id"] for event in events}) == len(events)


def test_null_fields_are_explicit(clock: FakeClock, tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = EventLogger("run-a", path, clock=clock)
    logger.emit("run_start", result={"note": "start"})
    logger.close()
    event = read_events(path)[0]
    for field in (
        "page_id",
        "url",
        "observation_id",
        "action_id",
        "action_type",
        "model_request_id",
    ):
        assert field in event
        assert event[field] is None
    assert event["result"] == {"note": "start"}
    assert event["model_calls_total"] == 0


def test_proposed_and_dispatched_cannot_carry_execution_result(clock: FakeClock, tmp_path: Path) -> None:
    logger = EventLogger("run-a", tmp_path / "events.jsonl", clock=clock)
    with pytest.raises(ValueError):
        logger.emit("proposed", action_id="act-1", action_type="click", result="observed")
    with pytest.raises(ValueError):
        logger.emit("dispatched", action_id="act-1", action_type="click", result="success")
    with pytest.raises(ValueError):
        logger.emit("agent_claim", result={"claim": "done", "reason": "completed", "score": 1})
    logger.close()


def test_logs_redact_fake_credentials(
    backend: FakeBackend, runtime: Path, clock: FakeClock
) -> None:
    secret = "tok_live_ABC123456"
    task = (
        "Find the item; password=hunter2; api_key=sk-REAL-SECRET-123456; "
        "Authorization: Bearer abcdefghijklmnop"
    )
    session = Session(
        "run-a",
        runtime_dir=runtime,
        clock=clock,
        redactor=Redactor(secrets=(secret,)),
    )
    session.start_task(task)
    session.ingest_observation(backend.current_observation)
    action = build_web_action(
        ToolRequest(
            "web_type",
            {"target": "1", "text": f"draft {secret} password=hunter2"},
        ),
        session.observation,
    )
    session.propose(action)
    session.consume()
    result = backend.execute(action)
    session.ingest_observation(result.observation)
    session.record_model_call(
        request_id="req-secret",
        input_tokens=10,
        output_tokens=2,
        token_count_source="estimate:chars_div_4",
        latency_ms=1.0,
        model_name="unified-model",
    )
    session.log_exception(
        "offline",
        RuntimeError(f"failed with token={secret} password=hunter2 sk-REAL-SECRET-123456"),
    )
    session.finish(Finish(claim="done", message="offline", reason="completed"))
    session.close()
    text = session.evidence.events_path.read_text(encoding="utf-8")
    for forbidden in (
        "hunter2",
        "sk-REAL-SECRET-123456",
        "Bearer abcdefghijklmnop",
        secret,
    ):
        assert forbidden not in text
    assert "[REDACTED]" in text
    result_text = session.evidence.result_path.read_text(encoding="utf-8")
    assert "hunter2" not in result_text
    assert secret not in result_text
