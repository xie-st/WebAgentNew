"""Real-model adapter driven by the EffGen loop against a local fake endpoint.

This is an offline integration check: the adapter speaks HTTP to 127.0.0.1, the
EffGen generator and A1 session remain the same, and no public service or real
credential is used.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from webagent.budget import Budget
from webagent.clock import SystemClock
from webagent.config import ModelConfig
from webagent.effgen_bridge import DriverStatus, create_driver
from webagent.events import read_events
from webagent.models.real import RealModel
from webagent.session import Session
from webagent.testing import FakeBackend, FakeSite

BASE_ENV = "WEBAGENT_LOOP_BASE_URL"
KEY_ENV = "WEBAGENT_LOOP_API_KEY"
FAKE_KEY = "sk-FAKE-LOOP-ONLY"


class _ReActEndpoint:
    def __init__(self, responses: list[str]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = list(responses)
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                endpoint.requests.append(json.loads(self.rfile.read(length)))
                text = endpoint._responses.pop(0) if endpoint._responses else "Final Answer: empty"
                payload = {
                    "id": f"chatcmpl-{len(endpoint.requests)}",
                    "choices": [
                        {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30},
                }
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("x-request-id", f"loop-{len(endpoint.requests)}")
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


def test_real_model_loop_records_actual_usage_per_http_call(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    endpoint = _ReActEndpoint(
        [
            'Thought: open the search result.\nAction: web_click\nAction Input: {"target": "2"}',
            "Thought: page opened.\nFinal Answer: opened the search result",
        ]
    )
    site = FakeSite.from_json(fixtures_dir / "fake_site.json")
    backend = FakeBackend("real-loop", site, start_url=site.default_url)
    clock = SystemClock()
    session = Session(
        "real-loop",
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        budget=Budget(clock=clock),
    )
    session.start_task("open the search result")
    session.ingest_observation(backend.current_observation)
    model = RealModel(
        ModelConfig(
            provider="openai_compatible",
            name="local-fake-model",
            context_length=8192,
            tokenizer="heuristic",
            base_url_env=BASE_ENV,
            api_key_env=KEY_ENV,
        ),
        environ={BASE_ENV: endpoint.url, KEY_ENV: FAKE_KEY},
        retry_backoff_seconds=0,
        sleep=lambda _: None,
    )
    model.set_deadline_check(session.check_deadline)
    model.set_call_sink(
        lambda record: session.record_model_call(
            request_id=record.request_id,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            token_count_source="provider_usage",
            latency_ms=record.latency_ms,
            failure_type=record.failure_type,
            model_name=record.model,
        )
    )
    driver = None
    try:
        model.load()
        driver = create_driver(session, model)
        step = driver.start()
        while step.status is DriverStatus.ACTION:
            action = driver.take_action()
            result = backend.execute(action)
            step = driver.accept_observation(result.observation)
        assert step.status is DriverStatus.COMPLETED
        assert len(model.calls) == 2
        assert len(endpoint.requests) == 2
        assert "obs-0" in endpoint.requests[0]["messages"][0]["content"]
        assert "obs-1" in endpoint.requests[1]["messages"][0]["content"]

        events = read_events(session.evidence.events_path)
        model_events = [event for event in events if event["event_type"] == "model_call"]
        assert len(model_events) == 2, "one event per HTTP call, no pre-call duplicate"
        assert all(event["model_request_id"] for event in model_events)
        assert all(event["model_input_tokens"] == 21 for event in model_events)
        assert all(event["model_output_tokens"] == 9 for event in model_events)
        assert all(event["token_count_source"] == "provider_usage" for event in model_events)
        assert all(event["latency_ms"] is not None for event in model_events)
        assert all(event["failure_type"] is None for event in model_events)
        assert all(event["model_calls_total"] == index for index, event in enumerate(model_events, 1))
        serialized = (session.evidence.run_dir / "events.jsonl").read_text(encoding="utf-8")
        assert FAKE_KEY not in serialized
    finally:
        if driver is not None:
            driver.close()
        model.unload()
        session.close()
        backend.close()
        endpoint.close()
