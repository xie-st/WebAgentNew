"""Offline real-model adapter tests against a local fake HTTP service.

No public network and no real credentials: the service binds 127.0.0.1 with an
ephemeral port, and every value is a fake placeholder.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from webagent.config import ConfigError, ModelConfig
from webagent.models.real import RealModel, RealModelError

BASE_ENV = "WEBAGENT_TEST_BASE_URL"
KEY_ENV = "WEBAGENT_TEST_API_KEY"
FAKE_KEY = "sk-FAKE-B1-LOCAL-ONLY"


class _FakeEndpoint:
    def __init__(self, responses: list[tuple[int, dict[str, Any]]]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = list(responses)
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                endpoint.requests.append(
                    {
                        "path": self.path,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "body": json.loads(raw),
                    }
                )
                if not endpoint._responses:
                    status, payload = 500, {"error": "no scripted response"}
                else:
                    status, payload = endpoint._responses.pop(0)
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("x-request-id", f"req-{len(endpoint.requests)}")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _success(text: str = "Final Answer: ok", usage: bool = True) -> tuple[int, dict[str, Any]]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-test-1",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }
    if usage:
        payload["usage"] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    return 200, payload


def _config(**overrides: Any) -> ModelConfig:
    values = {
        "provider": "openai_compatible",
        "name": "test-unified-model",
        "context_length": 8192,
        "tokenizer": "heuristic",
        "token_count_source": "estimated",
        "base_url_env": BASE_ENV,
        "api_key_env": KEY_ENV,
    }
    values.update(overrides)
    return ModelConfig(**values)


@pytest.fixture
def service():
    endpoint = _FakeEndpoint([_success()])
    try:
        yield endpoint
    finally:
        endpoint.close()


def test_generate_records_provider_usage_request_id_and_no_secret(service: _FakeEndpoint) -> None:
    model = RealModel(
        _config(),
        environ={BASE_ENV: service.url, KEY_ENV: FAKE_KEY},
        retry_backoff_seconds=0,
        sleep=lambda _: None,
    )
    try:
        model.load()
        result = model.generate("Thought: answer\nFinal Answer: ok")
        assert result.text.startswith("Final Answer")
        assert result.metadata["request_id"].startswith("ireq-")
        assert result.metadata["usage_source"] == "provider"
        assert result.metadata["retries"] == 0
        assert len(model.calls) == 1
        record = model.calls[0]
        assert record.status == "ok"
        assert record.http_status == 200
        assert record.provider_request_id == "req-1"
        assert record.request_id == result.metadata["request_id"]
        assert record.input_tokens == 11
        assert record.output_tokens == 7
        assert record.total_tokens == 18
        assert record.latency_ms >= 0
        assert record.usage_source == "provider"
        assert record.endpoint.endswith("/v1/chat/completions")
        assert service.requests[0]["path"] == "/v1/chat/completions"
        assert service.requests[0]["headers"]["authorization"] == f"Bearer {FAKE_KEY}"
        body = service.requests[0]["body"]
        assert body["model"] == "test-unified-model"
        assert body["messages"] == [
            {"role": "user", "content": "Thought: answer\nFinal Answer: ok"}
        ]
        assert "tools" not in body
        assert "tool_choice" not in body
        serialized = json.dumps([item.to_dict() for item in model.calls], ensure_ascii=False)
        assert FAKE_KEY not in serialized
        assert FAKE_KEY not in repr(model)
        assert model.count_tokens("abcd").count == 1
        assert model.get_context_length() == 8192
        assert model.supports_tool_calling() is False
    finally:
        model.unload()


def test_retries_are_counted_and_missing_usage_is_null() -> None:
    endpoint = _FakeEndpoint(
        [
            (503, {"error": "busy"}),
            _success(usage=False),
        ]
    )
    try:
        model = RealModel(
            _config(),
            environ={BASE_ENV: endpoint.url, KEY_ENV: FAKE_KEY},
            max_retries=1,
            retry_backoff_seconds=0,
            sleep=lambda _: None,
        )
        model.load()
        result = model.generate("retry please")
        assert result.text == "Final Answer: ok"
        assert len(model.calls) == 2
        first, second = model.calls
        assert first.status == "http_error"
        assert first.http_status == 503
        assert first.retries_before == 0
        assert first.usage_source == "missing"
        assert first.input_tokens is None and first.output_tokens is None
        assert second.status == "ok"
        assert second.retries_before == 1
        assert second.usage_source == "missing"
        assert second.input_tokens is None
        assert second.output_tokens is None
        assert result.metadata["usage_source"] == "missing"
        assert result.metadata["retries"] == 1
        assert "[REDACTED]" not in json.dumps([record.to_dict() for record in model.calls])
        assert FAKE_KEY not in json.dumps([record.to_dict() for record in model.calls])
        model.unload()
    finally:
        endpoint.close()


def test_no_retry_on_non_retryable_4xx_and_safe_error() -> None:
    endpoint = _FakeEndpoint([(401, {"error": "bad key"})])
    try:
        model = RealModel(
            _config(),
            environ={BASE_ENV: endpoint.url, KEY_ENV: FAKE_KEY},
            max_retries=3,
            retry_backoff_seconds=0,
            sleep=lambda _: None,
        )
        model.load()
        with pytest.raises(RealModelError) as error:
            model.generate("nope")
        assert "401" in str(error.value)
        assert FAKE_KEY not in str(error.value)
        assert len(model.calls) == 1
        assert model.calls[0].failure_type == "http_401"
        model.unload()
    finally:
        endpoint.close()


def test_load_reports_missing_environment_names_without_values() -> None:
    model = RealModel(_config(), environ={BASE_ENV: "http://127.0.0.1:9/v1"})
    with pytest.raises(ConfigError) as error:
        model.load()
    assert KEY_ENV in str(error.value)
    assert "http://127.0.0.1:9" not in str(error.value)


def test_stream_contract_uses_one_scripted_response() -> None:
    endpoint = _FakeEndpoint([_success("streamed text", usage=True)])
    try:
        model = RealModel(
            _config(),
            environ={BASE_ENV: endpoint.url, KEY_ENV: FAKE_KEY},
            retry_backoff_seconds=0,
            sleep=lambda _: None,
        )
        model.load()
        chunks = list(model.generate_stream("hello"))
        assert chunks == ["streamed text"]
        assert len(model.calls) == 1
        model.unload()
        with pytest.raises(RealModelError):
            model.generate("closed")
    finally:
        endpoint.close()


def test_call_sink_receives_every_attempt() -> None:
    endpoint = _FakeEndpoint([(503, {"error": "busy"}), _success()])
    seen: list[str] = []
    try:
        model = RealModel(
            _config(),
            environ={BASE_ENV: endpoint.url, KEY_ENV: FAKE_KEY},
            max_retries=1,
            retry_backoff_seconds=0,
            sleep=lambda _: None,
            call_sink=lambda record: seen.append(record.status),
        )
        model.load()
        model.generate("hello")
        assert seen == ["http_error", "ok"]
        model.unload()
    finally:
        endpoint.close()
