"""C01: the official model contract, offline against a fake gateway.

No platform connection, credential or model service is used. The real frozen
``third_party_agent`` package is imported from its checked-in directory exactly
as A1 did, while calls are made through an in-memory fake gateway.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from effgen.models.base import BaseModel, GenerationConfig

from webagent.clock import FakeClock
from webagent.config import ModelConfig
from webagent.events import read_events
from webagent.models.official import OfficialModel, OfficialModelError
from webagent.session import Session

OFFICIAL_CLIENT = Path(__file__).resolve().parents[2] / "official_client"


def _config(**overrides: Any) -> ModelConfig:
    values = {
        "provider": "official",
        "name": "platform-unified",
        "context_length": 32768,
        "tokenizer": "heuristic",
        "token_count_source": "estimated",
        "base_url_env": None,
        "api_key_env": None,
    }
    values.update(overrides)
    return ModelConfig(**values)


class FakeGateway:
    """A fake ``third_party_agent.model`` module with call capture."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, str]] = []
        self.call_args: list[tuple[Any, ...]] = []
        self.call_kwargs: list[dict[str, Any]] = []

    def generate(self, messages: dict[str, str]) -> Any:
        self.calls.append(messages)
        self.call_args.append((messages,))
        self.call_kwargs.append({})
        if not self.responses:
            raise RuntimeError("fake gateway script exhausted")
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(text=value, request_id=f"platform-req-{len(self.calls)}")


def test_frozen_sdk_import_matches_a1_and_adapter_is_base_model() -> None:
    sys.path.insert(0, str(OFFICIAL_CLIENT))
    try:
        from third_party_agent import model as sdk_model
    finally:
        sys.path.remove(str(OFFICIAL_CLIENT))
    assert callable(sdk_model.generate)
    assert hasattr(sdk_model, "ModelResult")
    assert issubclass(OfficialModel, BaseModel)


def test_only_role_content_text_and_no_generation_parameters() -> None:
    gateway = FakeGateway(["Final Answer: ok"])
    model = OfficialModel(
        _config(),
        model_api=gateway,
        system_prompt="SYSTEM ROLE",
    )
    result = model.generate(
        "FULL EFFGEN PROMPT\nAction: web_click",
        GenerationConfig(temperature=0.93, max_tokens=77, top_p=0.1),
    )
    assert gateway.calls == [
        [
            {"role": "system", "content": "SYSTEM ROLE"},
            {"role": "user", "content": "FULL EFFGEN PROMPT\nAction: web_click"},
        ]
    ]
    assert gateway.call_args[0] == (gateway.calls[0],)
    assert gateway.call_kwargs == [{}]
    body = gateway.calls[0]
    assert all(set(item) == {"role", "content"} for item in body)
    assert all(item["role"] in {"system", "user", "assistant"} for item in body)
    assert all(isinstance(item["content"], str) for item in body)
    assert "tool" not in json.dumps(body)
    assert result.text == "Final Answer: ok"
    assert result.metadata["request_id"] == "platform-req-1"
    assert result.metadata["usage_source"] == "missing"
    assert result.metadata["token_count_source"] == "platform_usage_unavailable"
    assert result.metadata["input_tokens"] is None
    assert result.metadata["output_tokens"] is None
    assert result.metadata["total_tokens"] is None
    assert isinstance(result.metadata["estimated_input_tokens"], int)
    assert isinstance(result.metadata["estimated_output_tokens"], int)
    assert len(model.calls) == 1
    record = model.calls[0]
    assert record.request_id == "platform-req-1"
    assert record.input_tokens is None and record.output_tokens is None
    assert record.usage_source == "missing"
    assert record.has_system_message is True


def test_complete_prompt_is_one_user_message_without_an_invented_system_turn() -> None:
    gateway = FakeGateway(["answer"])
    model = OfficialModel(_config(), model_api=gateway)
    model.generate("complete prompt")
    assert gateway.calls == [[{"role": "user", "content": "complete prompt"}]]


def test_supplied_message_list_keeps_system_role_and_refuses_tool_or_image_content() -> None:
    gateway = FakeGateway(["answer"])
    model = OfficialModel(_config(), model_api=gateway)
    model.generate(
        [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ]
    )
    assert gateway.calls == [
        [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ]
    ]
    with pytest.raises(Exception):
        model.generate([{"role": "tool", "content": "tool result"}])
    with pytest.raises(Exception):
        model.generate(
            [{"role": "user", "content": [{"type": "image", "data": "..."}]}]
        )


def test_each_explicit_attempt_is_counted_even_when_the_sdk_raises() -> None:
    gateway = FakeGateway([RuntimeError("transport failed"), "recovered"])
    model = OfficialModel(_config(), model_api=gateway)
    with pytest.raises(OfficialModelError) as error:
        model.generate("first")
    assert "RuntimeError" in str(error.value)
    assert len(model.calls) == 1
    assert model.calls[0].status == "error"
    assert model.calls[0].request_id is None
    assert model.calls[0].failure_type == "RuntimeError"

    result = model.generate("second")
    assert result.text == "recovered"
    assert len(model.calls) == 2
    assert [record.index for record in model.calls] == [0, 1]
    assert model.calls[1].request_id == "platform-req-2"


def test_missing_usage_is_null_in_session_model_event(tmp_path: Path, clock: FakeClock) -> None:
    session = Session("run-a", runtime_dir=tmp_path / "runtime", clock=clock)
    gateway = FakeGateway(["answer"])
    model = OfficialModel(
        _config(),
        model_api=gateway,
        call_sink=lambda record: session.record_model_call(
            request_id=record.request_id,
            input_tokens=None,
            output_tokens=None,
            token_count_source="platform_usage_unavailable",
            latency_ms=record.latency_ms,
            failure_type=record.failure_type,
            model_name="platform-unified",
        ),
    )
    model.generate("prompt")
    session.close()
    events = read_events(session.evidence.events_path)
    calls = [event for event in events if event["event_type"] == "model_call"]
    assert len(calls) == 1
    assert calls[0]["model_request_id"] == "platform-req-1"
    assert calls[0]["model_input_tokens"] is None
    assert calls[0]["model_output_tokens"] is None
    assert calls[0]["token_count_source"] == "platform_usage_unavailable"
    assert calls[0]["model_calls_total"] == 1


def test_prompt_validation_and_single_chunk_stream() -> None:
    gateway = FakeGateway(["streamed"])
    model = OfficialModel(_config(), model_api=gateway)
    with pytest.raises(ValueError):
        model.generate("")
    with pytest.raises(ValueError):
        model.generate(123)  # type: ignore[arg-type]
    assert list(model.generate_stream("prompt")) == ["streamed"]
    assert len(gateway.calls) == 1


def test_unload_is_safe_to_repeat_and_blocks_further_calls() -> None:
    gateway = FakeGateway(["first"])
    model = OfficialModel(_config(), model_api=gateway)
    model.generate("one")
    model.unload()
    model.unload()
    with pytest.raises(OfficialModelError):
        model.generate("two")

def test_final_sdk_exit_enforces_project_message_and_byte_limits() -> None:
    from webagent.budget import BudgetLimits
    from webagent.errors import RequestLimitExceeded

    normal_gateway = FakeGateway(["ok"])
    normal = OfficialModel(_config(), model_api=normal_gateway)
    result = normal.generate("short prompt")
    assert result.text == "ok"
    assert result.metadata["request_message_count"] == 1
    assert result.metadata["request_bytes"] > 0

    message_gateway = FakeGateway(["must not be called"])
    message_limited = OfficialModel(
        _config(),
        model_api=message_gateway,
        system_prompt="SYSTEM",
        limits=BudgetLimits(max_messages=1),
    )
    with pytest.raises(RequestLimitExceeded) as error:
        message_limited.generate("prompt")
    assert error.value.limit_name == "max_messages"
    assert error.value.limit == 1 and error.value.actual == 2
    assert message_gateway.calls == [], "an over-limit request never reaches the SDK"
    assert message_limited.calls[-1].status == "rejected"
    assert message_limited.calls[-1].failure_type == "request_limit"

    byte_gateway = FakeGateway(["must not be called"])
    byte_limited = OfficialModel(
        _config(),
        model_api=byte_gateway,
        limits=BudgetLimits(max_request_bytes=64),
    )
    with pytest.raises(RequestLimitExceeded) as error:
        byte_limited.generate("x" * 500)
    assert error.value.limit_name == "max_request_bytes"
    assert byte_gateway.calls == []


def test_mandatory_task_and_observation_overflow_is_reported_separately() -> None:
    from webagent.budget import BudgetLimits, MandatoryContentOverflow
    from webagent.errors import RequestLimitExceeded

    mandatory = [{"role": "user", "content": "TASK:\n" + "x" * 400}]
    gateway = FakeGateway(["must not be called"])
    model = OfficialModel(
        _config(),
        model_api=gateway,
        limits=BudgetLimits(max_request_bytes=200),
        mandatory_provider=lambda: mandatory,
    )
    with pytest.raises(MandatoryContentOverflow) as error:
        model.generate("short prompt")
    assert error.value.limit_name == "max_request_bytes"
    assert error.value.scope == "mandatory"
    assert gateway.calls == []

    # When the must-keep view fits but the whole request does not, the error is
    # the ordinary request limit: the mandatory items themselves are not the
    # problem and are never trimmed.
    small_mandatory = [{"role": "user", "content": "TASK:\nsmall"}]
    other_gateway = FakeGateway(["must not be called"])
    other = OfficialModel(
        _config(),
        model_api=other_gateway,
        limits=BudgetLimits(max_request_bytes=200),
        mandatory_provider=lambda: small_mandatory,
    )
    with pytest.raises(RequestLimitExceeded) as error:
        other.generate("y" * 500)
    assert not isinstance(error.value, MandatoryContentOverflow)
    assert error.value.limit_name == "max_request_bytes"


def test_set_request_limits_installs_the_gate_after_construction() -> None:
    from webagent.budget import BudgetLimits
    from webagent.errors import RequestLimitExceeded

    gateway = FakeGateway(["must not be called"])
    model = OfficialModel(_config(), model_api=gateway)
    model.set_request_limits(BudgetLimits(max_messages=1))
    with pytest.raises(RequestLimitExceeded):
        model.generate([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    assert gateway.calls == []
