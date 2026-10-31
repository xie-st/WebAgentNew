"""Unit tests for provider/model-labeled server metrics.

Covers the wiring introduced so an operator can graph latency, error rate,
and token usage per provider/model straight from ``/metrics`` instead of
only the flat per-``agent_name`` aggregate:

- ``Agent.run()`` feeds ``effgen_model_call_latency_seconds{provider,model,
  outcome}`` and ``effgen_tokens_total{provider,model,kind}`` on every run
  (success and failure), with no double-count when ``raise_on_error`` turns
  a failed run into a raised exception.
- Each cloud adapter class carries the ``_provider`` label
  ``Agent._model_provider`` reads.
- ``effgen_http_requests_total{route,method,status}`` is recorded once per
  HTTP request by the server's status-recording middleware.

These use in-process fake models (no network) — not a mock of live API
behavior, which the project forbids for provider calls.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.models.errors import ModelNotFoundError, ModelTimeoutError
from effgen.observability.metrics import (
    http_requests_total,
    model_call_latency,
    reset_all,
    tokens_total,
)


class _FakeModel(BaseModel):
    """Minimal in-process model: either returns fixed text or raises `exc`."""

    def __init__(
        self, *, text: str = "hi", exc: Exception | None = None, provider: str = "fake",
        usage_metadata: dict | None = None,
    ):
        super().__init__(model_name="fake-model", model_type=ModelType.OPENAI)
        self._text = text
        self._exc = exc
        self._provider = provider
        self._usage_metadata = usage_metadata or {}

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        if self._exc is not None:
            raise self._exc
        return GenerationResult(
            text=self._text, tokens_used=3, finish_reason="stop", model_name=self.model_name,
            metadata=dict(self._usage_metadata),
        )

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self._text

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 4096

    def unload(self) -> None:  # pragma: no cover
        pass

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config) for p in prompts]


def _agent(model: _FakeModel, **cfg_kwargs) -> Agent:
    return Agent(
        AgentConfig(
            name="t",
            model=model,
            tools=[],
            enable_sub_agents=False,
            enable_memory=False,
            **cfg_kwargs,
        )
    )


@pytest.fixture(autouse=True)
def _reset():
    reset_all()
    yield
    reset_all()


class TestAgentRunRecordsModelMetrics:
    def test_success_records_ok_with_tokens(self):
        m = _FakeModel(
            provider="openai",
            usage_metadata={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        _agent(m).run("hi")
        text = model_call_latency.export()
        assert 'provider="openai"' in text
        assert 'model="fake-model"' in text
        assert 'outcome="ok"' in text
        assert tokens_total.get(
            labels={"provider": "openai", "model": "fake-model", "kind": "input"}
        ) == 10
        assert tokens_total.get(
            labels={"provider": "openai", "model": "fake-model", "kind": "output"}
        ) == 5

    def test_failure_without_raise_on_error_records_error_once(self):
        m = _FakeModel(provider="groq", exc=ModelNotFoundError("groq", "bad-model"))
        r = _agent(m, raise_on_error=False).run("hi")
        assert r.success is False
        count = model_call_latency._data
        matches = [
            v for k, v in count.items()
            if dict(k).get("provider") == "groq" and dict(k).get("outcome") in ("error", "not_found")
        ]
        assert len(matches) == 1, f"expected exactly one recorded outcome, got {matches}"

    def test_failure_with_raise_on_error_records_classified_outcome_once(self):
        """The pre-raise success-path block and the except block must not both
        record the same failed request (that would double-count it)."""
        m = _FakeModel(provider="openai", exc=ModelNotFoundError("openai", "bad-model"))
        a = _agent(m, raise_on_error=True)
        with pytest.raises(ModelNotFoundError):
            a.run("hi")
        text = model_call_latency.export()
        assert 'provider="openai"' in text
        assert 'outcome="not_found"' in text
        # Exactly one observation total for this provider/model/outcome key.
        n_lines = [ln for ln in text.splitlines() if ln.startswith("effgen_model_call_latency_seconds_count{")]
        assert len(n_lines) == 1, text
        assert n_lines[0].strip().endswith("1")

    def test_timeout_outcome_classified(self):
        # Timeout is retryable, so the raised type after retries are exhausted
        # is a reconstructed RuntimeError (see Agent._reconstruct_error) — the
        # "timed out" text in its message is what classify_provider_error()
        # uses to label the outcome "timeout".
        m = _FakeModel(provider="cerebras", exc=ModelTimeoutError("cerebras", "fake-model"))
        a = _agent(m, raise_on_error=True)
        with pytest.raises(RuntimeError):
            a.run("hi")
        text = model_call_latency.export()
        assert 'outcome="timeout"' in text

    def test_no_tokens_recorded_when_no_usage(self):
        m = _FakeModel(provider="openai")
        _agent(m).run("hi")
        # No prompt/completion_tokens in the fake's metadata -> no token counter bump.
        assert tokens_total.get(
            labels={"provider": "openai", "model": "fake-model", "kind": "input"}
        ) == 0.0


class TestAdapterProviderLabels:
    """Each cloud adapter class must carry the `_provider` label
    Agent._model_provider() reads — otherwise every server metric for that
    provider silently falls back to "unknown"."""

    @pytest.mark.parametrize(
        "module,cls_name,expected",
        [
            ("effgen.models.openai_adapter", "OpenAIAdapter", "openai"),
            ("effgen.models.anthropic_adapter", "AnthropicAdapter", "anthropic"),
            ("effgen.models.gemini_adapter", "GeminiAdapter", "gemini"),
            ("effgen.models.cerebras_adapter", "CerebrasAdapter", "cerebras"),
            ("effgen.models.groq_adapter", "GroqAdapter", "groq"),
            ("effgen.models.together_adapter", "TogetherAdapter", "together"),
            ("effgen.models.fireworks_adapter", "FireworksAdapter", "fireworks"),
            ("effgen.models.replicate_adapter", "ReplicateAdapter", "replicate"),
        ],
    )
    def test_class_carries_provider_label(self, module, cls_name, expected):
        import importlib

        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        assert getattr(cls, "_provider", None) == expected


class TestHttpRequestsTotal:
    def test_record_and_export(self):
        from effgen.observability.metrics import record_http_request

        record_http_request(route="/v1/chat/completions", method="POST", status=200)
        record_http_request(route="/v1/chat/completions", method="POST", status=404)
        record_http_request(route="/v1/chat/completions", method="POST", status=200)
        assert http_requests_total.get(
            labels={"route": "/v1/chat/completions", "method": "POST", "status": "200"}
        ) == 2.0
        assert http_requests_total.get(
            labels={"route": "/v1/chat/completions", "method": "POST", "status": "404"}
        ) == 1.0

    def test_appears_in_export_metrics(self):
        from effgen.observability.metrics import export_metrics, record_http_request

        record_http_request(route="/health", method="GET", status=200)
        text = export_metrics()
        assert "effgen_http_requests_total" in text
        assert 'route="/health"' in text
        assert 'status="200"' in text
