"""Pointing effGen at a server that speaks the OpenAI protocol.

The tests run against a stub server implementing ``/v1/models`` and
``/v1/chat/completions`` at the wire level, which is the whole of what vLLM,
SGLang, TGI, llama.cpp, Ollama and LM Studio have in common. Serving real
weights would tie these to a GPU without exercising any more of the client.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from effgen.models import load_model
from effgen.models._base_url import (
    BASE_URL_ENV_VARS,
    PLACEHOLDER_API_KEY,
    resolve_base_url,
)
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_compatible_adapter import OpenAICompatibleAdapter
from effgen.models.registry import ProviderRegistry

SERVED_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ANSWER = "The answer is 42."


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send({
                "object": "list",
                "data": [{"id": SERVED_MODEL, "object": "model", "owned_by": "vllm"}],
            })
            return
        self._send({"id": SERVED_MODEL, "object": "model", "owned_by": "vllm"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        self.server.last_request = request
        self.server.last_authorization = self.headers.get("Authorization", "")

        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for piece in ("The answer ", "is ", "42."):
                chunk = {
                    "id": "chatcmpl-test", "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": request.get("model"),
                    "choices": [
                        {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            done = {
                "id": "chatcmpl-test", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": request.get("model"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._send({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ANSWER},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        })


@pytest.fixture
def compatible_server(monkeypatch):
    """A running OpenAI-protocol server; yields its base URL.

    Clears the endpoint environment variables so a value set on the developer's
    machine cannot decide what these tests measure.
    """
    for var in BASE_URL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.last_request = None
    server.last_authorization = ""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _text(result) -> str:
    return getattr(result, "text", None) or getattr(result, "content", None) or str(result)


class TestEndpointResolution:
    """Where an OpenAI-protocol call is addressed."""

    def test_an_explicit_url_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_BASE_URL", "http://from-env/v1")
        assert resolve_base_url("http://explicit/v1") == "http://explicit/v1"

    def test_the_environment_supplies_a_url_when_no_argument_does(self, monkeypatch):
        for var in BASE_URL_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://from-env/v1")
        assert resolve_base_url() == "http://from-env/v1"

    def test_effgen_s_own_variable_is_consulted_first(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_BASE_URL", "http://effgen/v1")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://openai/v1")
        assert resolve_base_url() == "http://effgen/v1"

    def test_no_url_anywhere_resolves_to_the_provider_s_own(self, monkeypatch):
        for var in BASE_URL_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert resolve_base_url() is None

    def test_a_trailing_slash_is_removed(self):
        assert resolve_base_url("http://host:8000/v1/") == "http://host:8000/v1"

    def test_a_url_without_a_scheme_is_refused(self):
        with pytest.raises(ValueError, match="http://"):
            resolve_base_url("127.0.0.1:8100/v1")

    def test_the_refusal_names_the_variable_the_url_came_from(self, monkeypatch):
        """A URL applied from the environment is the one the caller never chose.

        Left to the HTTP client this surfaces as "Connection error" several
        retries later, carrying advice to check the provider's status page —
        so the message has to say which variable redirected the call.
        """
        for var in BASE_URL_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_BASE", "localhost:11434/v1")
        with pytest.raises(ValueError, match="OPENAI_API_BASE"):
            resolve_base_url()

    def test_an_https_url_is_accepted(self):
        assert resolve_base_url("https://gateway.example/v1") == "https://gateway.example/v1"


class TestABlankOverride:
    """A variable that is present but empty means "no override", everywhere.

    The OpenAI SDK reads ``OPENAI_BASE_URL`` itself and, unlike
    :func:`resolve_base_url`, treats an empty value as a real address: it then
    sends every request to ``''`` and the failure surfaces as a connection
    error advising the caller to check the provider's status page. effGen's own
    project template ships these variables for the user to fill in, so a blank
    one is an ordinary state rather than a corner case, and effGen's resolution
    has to be the only one that decides.
    """

    @staticmethod
    def _client_url(monkeypatch, var: str, value: str) -> str:
        for name in BASE_URL_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(var, value)
        adapter = OpenAIAdapter("gpt-4o-mini", api_key="sk-test", timeout=5, max_retries=0)
        adapter.load()
        return str(adapter.client.base_url)

    @pytest.mark.parametrize("var", BASE_URL_ENV_VARS)
    def test_an_empty_variable_leaves_the_call_addressed_to_openai(self, monkeypatch, var):
        assert self._client_url(monkeypatch, var, "").startswith("https://api.openai.com")

    @pytest.mark.parametrize("var", BASE_URL_ENV_VARS)
    def test_a_whitespace_variable_leaves_the_call_addressed_to_openai(
        self, monkeypatch, var
    ):
        assert self._client_url(monkeypatch, var, "   ").startswith("https://api.openai.com")

    def test_a_real_override_still_redirects_the_call(self, monkeypatch, compatible_server):
        monkeypatch.setenv("OPENAI_BASE_URL", compatible_server)
        adapter = OpenAIAdapter("gpt-4o-mini", api_key="sk-test", timeout=5, max_retries=0)
        adapter.load()
        assert str(adapter.client.base_url).rstrip("/") == compatible_server

    def test_the_catalog_refresh_is_not_redirected_by_a_blank_variable(self, monkeypatch):
        """``_fetch_openai`` builds its own client and had the same hole."""
        from effgen.models import _refresh

        for name in BASE_URL_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        seen: dict[str, object] = {}

        class _Client:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.models = self

            def list(self):
                return []

        monkeypatch.setattr("openai.OpenAI", _Client)
        _refresh._fetch_openai("sk-test")
        assert str(seen.get("base_url", "")).startswith("https://api.openai.com")


class TestAdapterConstruction:
    """What the adapter assumes when the catalog cannot answer."""

    def test_an_endpointless_adapter_says_how_to_supply_one(self, monkeypatch):
        for var in BASE_URL_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatibleAdapter(SERVED_MODEL)

    def test_a_local_server_needs_no_credential(self):
        adapter = OpenAICompatibleAdapter(SERVED_MODEL, base_url="http://h/v1")
        assert adapter.api_key == PLACEHOLDER_API_KEY

    def test_a_supplied_credential_is_used(self):
        adapter = OpenAICompatibleAdapter(
            SERVED_MODEL, base_url="http://h/v1", api_key="sk-real"
        )
        assert adapter.api_key == "sk-real"

    def test_the_caller_sets_the_context_window(self):
        adapter = OpenAICompatibleAdapter(
            SERVED_MODEL, base_url="http://h/v1", context_length=8192
        )
        assert adapter.get_context_length() == 8192

    def test_a_self_hosted_call_reports_no_price(self):
        adapter = OpenAICompatibleAdapter(SERVED_MODEL, base_url="http://h/v1")
        assert adapter._calculate_cost(1_000_000, 1_000_000) is None

    def test_the_full_sampling_surface_is_offered(self):
        adapter = OpenAICompatibleAdapter(SERVED_MODEL, base_url="http://h/v1")
        assert adapter._supports_sampling_params is True

    def test_the_provider_is_registered_under_its_own_name(self):
        assert "openai_compatible" in ProviderRegistry.list_providers()


class TestRouting:
    """Which adapter a given call selects."""

    def test_the_named_provider_selects_the_compatible_adapter(self, compatible_server):
        model = load_model(
            SERVED_MODEL, provider="openai_compatible", base_url=compatible_server
        )
        assert isinstance(model, OpenAICompatibleAdapter)

    def test_a_base_url_alone_selects_it(self, compatible_server):
        model = load_model(SERVED_MODEL, base_url=compatible_server)
        assert isinstance(model, OpenAICompatibleAdapter)

    def test_provider_openai_with_a_base_url_reaches_that_url(self, compatible_server):
        model = load_model(
            SERVED_MODEL, provider="openai", base_url=compatible_server, api_key="EMPTY"
        )
        assert isinstance(model, OpenAICompatibleAdapter)
        assert model.base_url == compatible_server

    def test_provider_openai_without_one_stays_on_openai(self, monkeypatch):
        for var in BASE_URL_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter("gpt-4o-mini")
        assert adapter.base_url is None
        assert type(adapter) is OpenAIAdapter

    def test_the_provider_prefix_form_resolves(self, compatible_server):
        model = load_model(
            f"openai_compatible:{SERVED_MODEL}", base_url=compatible_server
        )
        assert isinstance(model, OpenAICompatibleAdapter)

    @pytest.mark.parametrize(
        "spelling",
        ["openai-compatible", "openai_compat", "compatible", "server", "vllm_server"],
    )
    def test_the_spellings_people_reach_for_all_resolve(self, compatible_server, spelling):
        model = load_model(SERVED_MODEL, provider=spelling, base_url=compatible_server)
        assert isinstance(model, OpenAICompatibleAdapter)


class TestGeneration:
    """Talking to the server."""

    def test_a_prompt_is_answered(self, compatible_server):
        model = load_model(SERVED_MODEL, base_url=compatible_server)
        assert "42" in _text(model.generate("What is 6*7?"))

    def test_the_served_id_is_what_the_server_is_asked_for(self, compatible_server):
        model = load_model(SERVED_MODEL, base_url=compatible_server)
        model.generate("What is 6*7?")
        assert model.client.base_url.host == "127.0.0.1"

    def test_the_answer_arrives_as_more_than_one_delta(self, compatible_server):
        model = load_model(SERVED_MODEL, base_url=compatible_server)
        deltas = list(model.generate_stream("Count to three."))
        assert len(deltas) > 1
        assert "42" in "".join(_text(d) for d in deltas)

    def test_the_server_reports_the_ids_it_serves(self, compatible_server):
        model = load_model(SERVED_MODEL, base_url=compatible_server)
        assert model.list_served_models() == [SERVED_MODEL]

    def test_an_unreachable_endpoint_raises_rather_than_answering(self):
        model = load_model(
            SERVED_MODEL,
            base_url="http://127.0.0.1:1/v1",
            max_retries=0,
            timeout=5,
        )
        with pytest.raises((RuntimeError, OSError, ConnectionError)):
            model.generate("hello")

    def test_an_unreachable_endpoint_is_named_in_the_error(self):
        """The stock advice is to check the provider's status page.

        That is the wrong page when the request went to a server the caller
        runs, so the address it went to has to appear in the message.
        """
        dead = "http://127.0.0.1:1/v1"
        model = load_model(
            SERVED_MODEL, base_url=dead, max_retries=0, timeout=5,
        )
        with pytest.raises(RuntimeError) as excinfo:
            model.generate("hello")
        assert dead in str(excinfo.value)


class TestAgentIntegration:
    """The same endpoint reached through the agent surface."""

    def test_an_agent_configured_with_a_base_url_answers_through_it(
        self, compatible_server
    ):
        from effgen import Agent, AgentConfig

        agent = Agent(AgentConfig(model=SERVED_MODEL, base_url=compatible_server,
                                  max_iterations=2))
        try:
            response = agent.run("What is 6*7?")
            assert response.success
            assert "42" in response.output
        finally:
            agent.close()
