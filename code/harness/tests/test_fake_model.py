from __future__ import annotations

from pathlib import Path

import pytest
from effgen.models.base import BaseModel, GenerationResult

from webagent.backend import Backend
from webagent.clock import FakeClock
from webagent.errors import ProtocolError
from webagent.events import read_events
from webagent.models.fake import FakeResponse, ScriptedModel
from webagent.protocol import Finish, ToolRequest
from webagent.session import Session
from webagent.testing import FakeBackend, FakeElement, FakePage, FakeSite
from webagent.tools import build_web_action


def test_scripted_model_implements_framework_contract(fixtures_dir: Path) -> None:
    model = ScriptedModel.from_script(fixtures_dir / "fake_scripts" / "basic.json")
    assert isinstance(model, BaseModel)
    model.load()
    result = model.generate("Task: search")
    assert isinstance(result, GenerationResult)
    assert result.text.startswith("Thought:")
    assert result.model_name == "fake-scripted"
    assert result.tokens_used > 0
    assert model.call_count == 1
    assert model.prompts[0] == "Task: search"
    assert model.count_tokens("one two").count > 0
    assert model.get_context_length() == 32_768
    model.unload()
    assert model.is_loaded() is False


def test_stream_and_error_and_exhaustion() -> None:
    streamed = ScriptedModel(["abcdef"])
    chunks = list(streamed.generate_stream("prompt"))
    assert "".join(chunks) == "abcdef"
    assert streamed.call_count == 1

    failing = ScriptedModel([FakeResponse(error="offline failure")])
    with pytest.raises(RuntimeError, match="offline failure"):
        failing.generate("prompt")

    empty = ScriptedModel([])
    with pytest.raises(RuntimeError) as excinfo:
        empty.generate("a prompt containing a secret token=abc123")
    assert "script exhausted" in str(excinfo.value)
    # The exhaustion error must not echo the prompt.
    assert "secret" not in str(excinfo.value)


def test_fake_backend_observes_executes_and_binds_feedback(site: FakeSite) -> None:
    backend = FakeBackend("run-a", site, start_url=site.default_url)
    assert isinstance(backend, Backend)
    first = backend.current_observation
    assert first is not None
    action = build_web_action(ToolRequest("web_click", {"target": "2"}), first)
    result = backend.execute(action)
    assert result.feedback.status.value == "observed"
    assert result.observation.url == "http://fixture.local/results"
    assert result.observation.previous_action_id == action.action_id
    back = build_web_action(ToolRequest("web_back", {}), result.observation)
    back_result = backend.execute(back)
    assert back_result.observation.url == "http://fixture.local/"
    forward = build_web_action(ToolRequest("web_forward", {}), back_result.observation)
    forward_result = backend.execute(forward)
    assert forward_result.observation.url == "http://fixture.local/results"
    backend.close()
    backend.close()


def test_offline_session_backend_flow(backend: FakeBackend, runtime: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=runtime, clock=clock)
    session.start_task("type a query and search")
    session.ingest_observation(backend.current_observation)
    typed = build_web_action(
        ToolRequest("web_type", {"target": "1", "text": "sample"}), session.observation
    )
    session.propose(typed)
    session.consume()
    typed_result = backend.execute(typed)
    session.ingest_observation(typed_result.observation)
    clicked = build_web_action(
        ToolRequest("web_click", {"target": "2"}), session.observation
    )
    session.propose(clicked)
    session.consume()
    clicked_result = backend.execute(clicked)
    session.ingest_observation(clicked_result.observation)
    session.finish(Finish(claim="done", message="offline flow", reason="completed"))
    session.close()
    events = read_events(session.evidence.events_path)
    assert len([event for event in events if event["event_type"] == "dispatched"]) == 2
    assert len([event for event in events if event["event_type"] == "feedback"]) == 2
    assert session.evidence.result_path.is_file()


def test_fake_site_validation() -> None:
    with pytest.raises(ProtocolError):
        FakeSite([])
    with pytest.raises(ProtocolError):
        FakePage(url="/relative", title="x", ax_text="x")
    with pytest.raises(ProtocolError):
        FakeElement(role="link", name="x", href="/relative")
    with pytest.raises(ProtocolError):
        FakeSite(
            [
                FakePage(url="http://a.test/", title="x", ax_text="x"),
                FakePage(url="http://a.test/", title="y", ax_text="y"),
            ]
        )
