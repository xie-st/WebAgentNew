"""Regressions for what the provider failure matrix found.

The matrix in ``test_failure_matrix.py`` measures every adapter against every
failure class end to end. These are the narrow cases for the individual defects
it surfaced, so a regression names itself instead of showing up as one red cell
in a table of eighty.
"""

from __future__ import annotations

import json

import pytest

from effgen.models.errors import classify_provider_error

KEY = "sk-proj-regression0123456789abcdefghijklmnopqrstuvwxyz0123"


def scrub_exception(exc):
    """The redaction under test, imported per call so each case reports itself."""
    from effgen.models.errors import scrub_exception as impl

    return impl(exc)


def _chain_of(exc):
    """Every exception reachable from *exc* through cause and context."""
    seen, out, current = set(), [], exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        out.append(current)
        current = current.__cause__ or current.__context__
    return out


def retry_after_seconds(exc):
    """The stated-delay reader under test, imported per call for the same reason."""
    from effgen.models.errors import retry_after_seconds as impl

    return impl(exc)


class _SdkError(Exception):
    """Shaped like an OpenAI-family SDK error: text, and the parsed body."""

    def __init__(self, message: str, body: dict) -> None:
        super().__init__(message)
        self.message = message
        self.body = body
        self.status_code = 401


class TestTheCredentialNeverReachesTheCaller:
    """A provider that quotes the submitted key in its 401 must not pass it on."""

    def _chained(self):
        body = {"error": {"message": f"Incorrect API key provided: {KEY}",
                          "code": "invalid_api_key"}}
        inner = _SdkError(f"Error code: 401 - {body}", body)
        try:
            raise inner
        except _SdkError as exc:
            outer = RuntimeError("openai auth error: [REDACTED]")
            outer.__cause__ = exc
            return outer

    def test_the_chained_sdk_message_is_redacted(self):
        error = scrub_exception(self._chained())
        assert KEY not in str(error.__cause__)
        assert KEY not in error.__cause__.message

    def test_the_parsed_error_body_is_redacted(self):
        error = scrub_exception(self._chained())
        assert KEY not in json.dumps(error.__cause__.body)

    def test_a_rendered_traceback_carries_no_key(self):
        import traceback

        error = scrub_exception(self._chained())
        rendered = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        assert KEY not in rendered

    def test_the_failure_itself_survives_redaction(self):
        """Redaction removes the secret, not the diagnosis."""
        error = scrub_exception(self._chained())
        assert "401" in str(error.__cause__)
        assert error.__cause__.status_code == 401

    def test_a_cyclic_chain_terminates(self):
        first = RuntimeError(f"key={KEY}")
        second = RuntimeError("wrapper")
        first.__cause__ = second
        second.__cause__ = first
        scrub_exception(first)
        assert KEY not in str(first)

    def test_an_exception_that_refuses_assignment_is_skipped(self):
        class Stubborn(Exception):
            @property
            def message(self) -> str:
                return f"key={KEY}"

        error = Stubborn("outer")
        scrub_exception(error)  # must not raise
        assert error.message == f"key={KEY}"

    def test_the_boundary_actually_strips_what_it_wraps(self):
        """The wrapper is not enough on its own — it has to do the redaction.

        Checking only that the methods are wrapped leaves the wrapper free to
        become a no-op: an adapter would still report the failure, with the key
        the caller submitted quoted inside it.
        """
        from effgen.models.base import BaseModel, ModelType

        class _QuotesTheKey(BaseModel):
            def __init__(self) -> None:
                super().__init__("probe", ModelType.OPENAI, context_length=8)

            def load(self):
                body = {"error": {"message": f"Incorrect API key provided: {KEY}"}}
                raise _SdkError(f"Error code: 401 - {body}", body)

            def generate(self, prompt, **kwargs):
                self.load()

            def generate_stream(self, prompt, **kwargs):
                self.load()
                yield ""

            def unload(self):
                return None

            def count_tokens(self, text):
                return 0

            def get_context_length(self):
                return 8

        for call in (lambda m: m.load(), lambda m: m.generate("hi")):
            model = _QuotesTheKey()
            with pytest.raises(Exception) as caught:  # noqa: PT011 - any type
                call(model)
            leaked = [
                link
                for link in _chain_of(caught.value)
                if KEY in str(link) or KEY in json.dumps(getattr(link, "body", {}))
            ]
            assert not leaked, (
                "the boundary let the submitted credential through: "
                f"{[type(link).__name__ for link in leaked]}"
            )

    def test_every_adapter_call_is_wrapped_for_redaction(self):
        """The redaction sits at one boundary, so it cannot be forgotten."""
        from effgen.models.groq_adapter import GroqAdapter

        for name in ("generate", "generate_stream", "generate_batch", "load"):
            method = getattr(GroqAdapter, name, None)
            if method is not None:
                assert getattr(method, "__effgen_timed__", False), (
                    f"GroqAdapter.{name} is not wrapped, so an error it raises "
                    f"leaves with the provider's text unredacted"
                )


class TestAnAbsentCredentialIsAnAuthProblem:
    """Providers call it a key or a token; both mean "set a credential"."""

    @pytest.mark.parametrize("message", [
        "HuggingFace API token not found.  Set the HF_TOKEN environment variable.",
        "Replicate API token not found.  Set the REPLICATE_API_TOKEN environment variable.",
        "Groq API key not found. Set the GROQ_API_KEY environment variable.",
        "OpenAI API key not provided. Set OPENAI_API_KEY.",
    ])
    def test_classified_as_auth(self, message):
        assert classify_provider_error(ValueError(message)).category == "auth"

    @pytest.mark.parametrize("message", [
        "HuggingFace API token not found.  Set the HF_TOKEN environment variable.",
        "Replicate API token not found.  Set the REPLICATE_API_TOKEN environment variable.",
    ])
    def test_the_server_reports_a_missing_key_as_configuration(self, message):
        """Not a 404 telling the caller their model id is wrong."""
        from effgen.api.openai_compat_errors import _classify_http

        status, _, code = _classify_http(ValueError(message))
        assert (status, code) == (503, "upstream_key_missing")

    def test_a_genuinely_missing_model_still_reads_as_not_found(self):
        error = ValueError("model 'nope/nope' does not exist")
        assert classify_provider_error(error).category == "not_found"

    def test_a_credential_of_the_wrong_shape_is_not_transient(self):
        """Another provider's key pasted into ``HF_TOKEN``.

        Hugging Face answers 400 rather than 401 for this, and retrying it can
        never succeed, so reporting it as transient sends the caller round a
        loop instead of telling them to fix the token.
        """
        error = RuntimeError(
            "Cannot select auto-router when using non-Hugging Face API key."
        )
        cls = classify_provider_error(error)
        assert cls.category == "auth"
        assert cls.should_retry is False


class TestNoRetryBudgetMeansOneRequestNotNone:
    """``max_retries=0`` asks for no retry, not for no request."""

    @pytest.mark.parametrize("factory", [
        pytest.param(
            lambda: __import__(
                "effgen.models.fireworks_adapter", fromlist=["FireworksAdapter"]
            ).FireworksAdapter(
                model_name="accounts/fireworks/models/gpt-oss-120b",
                api_key="k", max_retries=0, enable_rate_limiting=False,
                warn_unknown_model=False,
            ),
            id="fireworks",
        ),
        pytest.param(
            lambda: __import__(
                "effgen.models.replicate_adapter", fromlist=["ReplicateAdapter"]
            ).ReplicateAdapter(
                model_name="meta/meta-llama-3-8b-instruct",
                api_token="k", max_retries=0, enable_rate_limiting=False,
                warn_unknown_model=False,
            ),
            id="replicate",
        ),
    ])
    def test_at_least_one_attempt_is_budgeted(self, factory):
        assert factory().max_retries >= 1

    def test_fireworks_reports_an_empty_budget_as_a_typed_error(self):
        """Not a bare assertion, which -O removes and which says nothing."""
        import inspect

        from effgen.models import fireworks_adapter

        source = inspect.getsource(fireworks_adapter._do_generate) if hasattr(
            fireworks_adapter, "_do_generate"
        ) else inspect.getsource(fireworks_adapter.FireworksAdapter._do_generate)
        assert "assert _last_exc" not in source


class TestGeminiCallsAreBounded:
    """A Gemini call has to answer to the deadline the caller set."""

    def test_the_deadline_reaches_the_sdk_in_milliseconds(self):
        from effgen.models.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter(model_name="gemini-3.1-flash-lite",
                                api_key="k", timeout=7.5)
        assert adapter._http_options().timeout == 7500

    def test_the_sdk_does_not_retry_underneath_the_adapter(self):
        """Two retry layers multiply: five SDK attempts inside each of ours."""
        from effgen.models.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key="k")
        assert adapter._http_options().retry_options.attempts == 1

    def test_no_retries_means_one_attempt(self):
        from effgen.models.gemini_adapter import GeminiAdapter

        assert GeminiAdapter(
            model_name="gemini-3.1-flash-lite", api_key="k", max_retries=0
        ).max_retries == 0

    def test_the_default_retry_budget_is_unchanged(self):
        from effgen.models.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key="k")
        assert adapter.max_retries + 1 == GeminiAdapter.MAX_RATE_LIMIT_RETRIES


class TestALatencyProbeIsOneRequest:
    """A probe measures latency; a refusing provider must not make it a loop."""

    def test_a_refused_probe_makes_one_request_and_returns(self, monkeypatch):
        """Against a provider answering 429, the probe stops after one try."""
        import time

        from effgen.models.latency_tracker import LatencyTracker
        from effgen.models.routing._probe import _probe_one

        from .failure_injection import FaultServer

        server = FaultServer("rate_limit", "gemini", retry_after_s=1).start()
        monkeypatch.setenv("GOOGLE_API_KEY", "probe-key")
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", server.url)
        started = time.monotonic()
        try:
            _probe_one("gemini", "gemini-3.1-flash-lite", LatencyTracker())
        finally:
            elapsed = time.monotonic() - started
            server.stop()
        assert server.call_count == 1, (
            f"the probe sent {server.call_count} requests to a provider that "
            f"refused; a latency probe is one request"
        )
        assert elapsed < 15.0, f"the probe held {elapsed:.1f}s on a refusal"

    @pytest.mark.parametrize("path,kwargs", [
        ("effgen.models.gemini_adapter.GeminiAdapter", {"api_key": "k"}),
        ("effgen.models.fireworks_adapter.FireworksAdapter",
         {"api_key": "k", "enable_rate_limiting": False,
          "warn_unknown_model": False}),
        ("effgen.models.replicate_adapter.ReplicateAdapter",
         {"api_token": "k", "enable_rate_limiting": False,
          "warn_unknown_model": False}),
    ])
    def test_max_retries_zero_is_exactly_one_attempt(self, path, kwargs):
        """Every adapter reads the budget its own way; all must budget one try."""
        import importlib

        module_name, _, cls_name = path.rpartition(".")
        cls = getattr(importlib.import_module(module_name), cls_name)
        model = {
            "GeminiAdapter": "gemini-3.1-flash-lite",
            "FireworksAdapter": "accounts/fireworks/models/gpt-oss-120b",
            "ReplicateAdapter": "meta/meta-llama-3-8b-instruct",
        }[cls_name]
        adapter = cls(model_name=model, max_retries=0, **kwargs)
        attempts = (adapter.max_retries + 1 if cls_name == "GeminiAdapter"
                    else adapter.max_retries)
        assert attempts == 1, f"{cls_name} budgets {attempts} attempts for max_retries=0"


class TestGeminiHonoursPerCallOverrides:
    """A cap passed to ``generate()`` has to reach the request."""

    def _adapter(self):
        from effgen.models.gemini_adapter import GeminiAdapter

        return GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key="k")

    def test_the_output_token_cap_reaches_the_request(self):
        """Dropped, the caller pays for tokens it capped and sees no truncation."""
        from effgen.models._adapter_utils import merge_call_overrides
        from effgen.models.base import GenerationConfig

        adapter = self._adapter()
        config = merge_call_overrides(GenerationConfig(), {"max_tokens": 37})
        assert adapter._build_config(config).max_output_tokens == 37

    @pytest.mark.parametrize("field,value,attribute", [
        ("temperature", 0.125, "temperature"),
        ("top_p", 0.25, "top_p"),
        ("seed", 4242, "seed"),
    ])
    def test_the_sampling_overrides_reach_the_request(self, field, value, attribute):
        from effgen.models._adapter_utils import merge_call_overrides
        from effgen.models.base import GenerationConfig

        adapter = self._adapter()
        config = merge_call_overrides(GenerationConfig(), {field: value})
        assert getattr(adapter._build_config(config), attribute) == value

    def test_generate_merges_the_call_keywords(self):
        """The merge happens where the call is made, not only in the helper."""
        import inspect

        from effgen.models.gemini_adapter import GeminiAdapter

        for name in ("generate", "generate_stream"):
            source = inspect.getsource(getattr(GeminiAdapter, name))
            assert "merge_call_overrides" in source, (
                f"GeminiAdapter.{name} builds its config without merging the "
                f"per-call keywords, so max_tokens= is dropped"
            )


class TestReplicateFailureClassification:
    def test_a_reply_that_is_not_json_is_worth_retrying(self):
        """A gateway page is the peer misbehaving, not the request being wrong."""
        import inspect

        from effgen.models.replicate_adapter import ReplicateAdapter

        source = inspect.getsource(ReplicateAdapter._do_generate)
        decode_at = source.find("json.JSONDecodeError")
        value_at = source.find("except (ValueError, TypeError)")
        assert decode_at != -1, "a non-JSON reply has no branch of its own"
        assert decode_at < value_at, (
            "json.JSONDecodeError is a ValueError, so its branch has to come "
            "first or a gateway page is reported as an invalid request"
        )

    def test_the_http_client_carries_the_stated_deadline(self):
        pytest.importorskip("replicate")
        from effgen.models.replicate_adapter import ReplicateAdapter

        adapter = ReplicateAdapter(
            model_name="meta/meta-llama-3-8b-instruct", api_token="k",
            timeout=9.0, enable_rate_limiting=False, warn_unknown_model=False,
        )
        adapter.load()
        assert adapter._client._timeout.read == 9.0

    def test_a_request_the_sdk_sends_without_a_deadline_gets_one(self):
        """The SDK creates a prediction with ``timeout=None`` explicitly."""
        import httpx

        from effgen.models.replicate_adapter import _deadline_transport

        transport = _deadline_transport(3.0)
        request = httpx.Request("POST", "http://127.0.0.1:1/v1/predictions")
        request.extensions = {
            "timeout": {"connect": None, "read": None, "write": None, "pool": None}
        }
        with pytest.raises(httpx.ConnectError):
            transport.handle_request(request)
        assert request.extensions["timeout"]["read"] == 3.0

    def test_a_deadline_the_sdk_did_state_is_left_alone(self):
        import httpx

        from effgen.models.replicate_adapter import _deadline_transport

        transport = _deadline_transport(3.0)
        request = httpx.Request("POST", "http://127.0.0.1:1/v1/predictions")
        request.extensions = {"timeout": {"connect": 1.0, "read": 1.0,
                                          "write": 1.0, "pool": 1.0}}
        with pytest.raises(httpx.ConnectError):
            transport.handle_request(request)
        assert request.extensions["timeout"]["read"] == 1.0


class TestASocketThatWentAwayIsTransient:
    """A connection cut mid-call reports as transient on every adapter.

    Only the SDKs that raise a class named after the connection are recognised
    by class name. The adapters that wrap the operating system's error in their
    own message reach the text heuristics, and "connection reset by peer" is
    not the phrase "connection error" — so the same failure used to be reported
    as unexpected on some providers and transient on others.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "Gemini generation failed [will_retry]: [Errno 104] Connection reset by peer.",
            "HuggingFace Inference request failed [will_retry]: Connection aborted.",
            "Replicate request failed: ('Connection aborted.', ConnectionResetError(104))",
            "Together request failed: Server disconnected without sending a response.",
            "request failed: Remote end closed connection without response",
            "write failed: Broken pipe",
        ],
    )
    def test_a_severed_connection_is_transient_and_retryable(self, message):
        verdict = classify_provider_error(RuntimeError(message))
        assert verdict.category == "transient", message
        assert verdict.should_retry

    @pytest.mark.parametrize(
        "message",
        [
            "openai call failed: Connection refused",
            "httpx request failed: All connection attempts failed",
            "urllib3: Failed to establish a new connection",
            "socket.gaierror: Name or service not known",
        ],
    )
    def test_a_connection_that_never_opened_is_unreachable(self, message):
        """Nothing answered, so the endpoint is the caller's problem.

        A severed connection above reached a server and is worth chasing on the
        provider's status page. These never reached one, so they are reported
        against the address instead — and, at the agent surface, raise rather
        than returning a run that looks like it produced an answer.
        """
        verdict = classify_provider_error(RuntimeError(message))
        assert verdict.category == "unreachable", message
        assert verdict.should_retry

    def test_a_rejected_credential_is_still_auth(self):
        """The new phrases must not steal a failure that belongs elsewhere."""
        verdict = classify_provider_error(
            RuntimeError("auth error: invalid api key; the connection was closed")
        )
        assert verdict.category == "auth"
        assert not verdict.should_retry

    def test_a_stated_rate_limit_is_still_a_rate_limit(self):
        verdict = classify_provider_error(
            RuntimeError("Rate limit reached; the connection was closed. Try again in 2.5s")
        )
        assert verdict.category == "rate_limited"


class TestTheStatedRetryDelayIsCarried:
    """A 429 that states a delay must not lose it on the way to the caller."""

    @pytest.mark.parametrize("text,expected", [
        ("Rate limit reached. Please try again in 17.3s.", 17.3),
        ("Rate limit reached, please try again in 1m12s", 72.0),
        ("Please try again in 30 seconds", 30.0),
        (
            (
                "429 RESOURCE_EXHAUSTED. {'error': {'details': [{'@type': "
                "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '16s'}]}}"
            ),
            16.0,
        ),
    ])
    def test_a_delay_stated_in_the_body_is_read(self, text, expected):
        assert retry_after_seconds(RuntimeError(text)) == pytest.approx(expected)

    def test_a_delay_stated_in_a_header_is_read(self):
        class Response:
            headers = {"retry-after": "23"}

        class Error(Exception):
            response = Response()

        assert retry_after_seconds(Error("429")) == 23.0

    def test_the_delay_is_found_through_the_wrapping(self):
        """By the time a caller sees it, the response is two links down."""
        inner = RuntimeError("Rate limit reached. Please try again in 5.5s")
        try:
            try:
                raise inner
            except RuntimeError as exc:
                raise ValueError("groq generate failed [rate_limited]") from exc
        except ValueError as wrapped:
            assert retry_after_seconds(wrapped) == pytest.approx(5.5)

    def test_a_silent_provider_yields_nothing_to_invent(self):
        assert retry_after_seconds(RuntimeError("429 Too Many Requests")) is None

    def test_an_absurd_delay_is_clamped(self):
        from effgen.models.errors import MAX_RETRY_AFTER_SECONDS

        error = RuntimeError("Please try again in 99999s")
        assert retry_after_seconds(error) == MAX_RETRY_AFTER_SECONDS

    def test_the_retry_policy_reads_a_wrapped_hint(self):
        from effgen.reliability.retry import _get_retry_after

        inner = RuntimeError("Please try again in 4s")
        try:
            try:
                raise inner
            except RuntimeError as exc:
                raise ValueError("rate limited") from exc
        except ValueError as wrapped:
            assert _get_retry_after(wrapped) == pytest.approx(4.0)


class TestTheServerPassesTheDelayOn:
    """A 429 caused by the provider tells the client how long to wait."""

    def _client(self, runner):
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        from effgen.server.app import create_app

        return TestClient(create_app(api_key="matrix-key", runner=runner),
                          raise_server_exceptions=False)

    def _post(self, runner):
        client = self._client(runner)
        return client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer matrix-key"},
            json={"model": "openai:gpt-5-nano",
                  "messages": [{"role": "user", "content": "hi"}]},
        )

    def test_the_upstream_delay_becomes_retry_after(self):
        def runner(prompt, **kwargs):
            raise RuntimeError(
                "groq generate failed [rate_limited]: Rate limit reached for "
                "model. Please try again in 17.3s."
            )

        response = self._post(runner)
        assert response.status_code == 429
        assert response.headers["retry-after"] == "17"
        assert response.json()["error"]["type"] == "rate_limit_exceeded"

    def test_a_provider_that_states_nothing_sets_no_header(self):
        """No invented number: the client keeps its own backoff."""
        def runner(prompt, **kwargs):
            raise RuntimeError("groq generate failed [rate_limited]: 429")

        response = self._post(runner)
        assert response.status_code == 429
        assert "retry-after" not in {k.lower() for k in response.headers}

    def test_a_non_429_failure_carries_no_delay(self):
        def runner(prompt, **kwargs):
            raise ValueError("model 'nope' does not exist")

        response = self._post(runner)
        assert response.status_code == 404
        assert "retry-after" not in {k.lower() for k in response.headers}
