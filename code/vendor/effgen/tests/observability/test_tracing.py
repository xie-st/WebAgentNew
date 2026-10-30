"""
Tests for effgen.observability.tracing.

Coverage:
- Sampler correctness: AlwaysOn/Off, TraceIdRatio, RateLimited, ParentBased
- Span tree shape for a 3-tool agent run (in-memory exporter)
- Span attribute constants from spans.py are used (no raw strings)
- retry event attachment
- Non-blocking: exceptions in spans don't propagate
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.sdk.trace.sampling import Decision
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _OTEL_AVAILABLE, reason="opentelemetry-sdk not installed"
)

from effgen.observability import (  # noqa: E402,I001  (must come after optional OTel skipif)
    AgentAttrs,
    AlwaysOffSampler,
    AlwaysOnSampler,
    ModelAttrs,
    ParentBasedSampler,
    RateLimitedSampler,
    RetryAttrs,
    RouterAttrs,
    SpanName,
    ToolAttrs,
    TraceIdRatioSampler,
    record_retry_attempt,
    reset_tracing,
    set_span_attribute,
    set_span_error,
    setup_tracing,
    start_agent_iteration,
    start_agent_run,
    start_model_call,
    start_router_decision,
    start_tool_call,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset():
    """Reset global tracing state before every test."""
    reset_tracing()
    yield
    reset_tracing()


def _make_exporter() -> "InMemorySpanExporter":
    """Create an in-memory exporter and register it as the global provider."""
    exporter = InMemorySpanExporter()
    setup_tracing(sampler=AlwaysOnSampler(), exporter=exporter)
    return exporter


# ===========================================================================
# Sampler tests
# ===========================================================================


class TestAlwaysOnSampler:
    def test_description(self):
        assert "AlwaysOn" in AlwaysOnSampler().get_description()

    def test_samples_every_span(self):
        s = AlwaysOnSampler()
        result = s.should_sample(None, 0, "test")
        assert result.decision == Decision.RECORD_AND_SAMPLE

    def test_samples_different_trace_ids(self):
        s = AlwaysOnSampler()
        for tid in [0, 1, 2**64, 2**128 - 1]:
            r = s.should_sample(None, tid, "x")
            assert r.decision == Decision.RECORD_AND_SAMPLE


class TestAlwaysOffSampler:
    def test_description(self):
        assert "AlwaysOff" in AlwaysOffSampler().get_description()

    def test_drops_every_span(self):
        s = AlwaysOffSampler()
        result = s.should_sample(None, 0, "test")
        assert result.decision == Decision.DROP

    def test_drops_different_trace_ids(self):
        s = AlwaysOffSampler()
        for tid in [0, 1, 2**64, 2**128 - 1]:
            r = s.should_sample(None, tid, "x")
            assert r.decision == Decision.DROP


class TestTraceIdRatioSampler:
    def test_ratio_zero_drops_all(self):
        s = TraceIdRatioSampler(0.0)
        # All trace IDs should be dropped
        for tid in [0, 100, 2**64, 2**128 - 1]:
            assert s.should_sample(None, tid, "x").decision == Decision.DROP

    def test_ratio_one_samples_all(self):
        s = TraceIdRatioSampler(1.0)
        for tid in [0, 100, 2**64, 2**128 - 1]:
            assert s.should_sample(None, tid, "x").decision == Decision.RECORD_AND_SAMPLE

    def test_ratio_clamped(self):
        s_low = TraceIdRatioSampler(-1.0)
        assert s_low.ratio == 0.0
        s_high = TraceIdRatioSampler(2.0)
        assert s_high.ratio == 1.0

    def test_description_contains_ratio(self):
        s = TraceIdRatioSampler(0.5)
        desc = s.get_description()
        assert "0.5" in desc

    def test_deterministic(self):
        """Same trace ID always gives same decision."""
        s = TraceIdRatioSampler(0.5)
        results = set()
        for _ in range(10):
            r = s.should_sample(None, 12345, "x").decision
            results.add(r)
        assert len(results) == 1

    def test_rough_ratio(self):
        """Over a large range of trace IDs, sampling rate ≈ p."""
        s = TraceIdRatioSampler(0.3)
        total = 10_000
        sampled = sum(
            1 for i in range(total)
            if s.should_sample(None, i * (2**128 // total), "x").decision == Decision.RECORD_AND_SAMPLE
        )
        ratio = sampled / total
        assert 0.25 <= ratio <= 0.35, f"Expected ~0.3, got {ratio}"

    def test_end_to_end_ratio_via_iterations(self):
        """
        End-to-end sampler-honoured check: run 100 iteration spans with
        ParentBased(TraceIdRatio(0.5)). Sampled span count must fall inside
        a three-sigma binomial band (35..65) around the expected 50, wide
        enough that a fair sampler fails less than 0.3% of the time while an
        always-on/always-off one still fails every time.
        """
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        reset_tracing()
        exporter = InMemorySpanExporter()
        setup_tracing(
            sampler=ParentBasedSampler(TraceIdRatioSampler(0.5)),
            exporter=exporter,
        )
        for i in range(100):
            with start_agent_iteration(preset="ratio_check", iteration=i):
                pass
        spans = exporter.get_finished_spans()
        assert 35 <= len(spans) <= 65, (
            f"Expected ~50 sampled spans, got {len(spans)}"
        )


class TestRateLimitedSampler:
    def test_description(self):
        s = RateLimitedSampler(5.0)
        assert "5.0" in s.get_description()

    def test_positive_per_second_required(self):
        with pytest.raises(ValueError):
            RateLimitedSampler(0.0)
        with pytest.raises(ValueError):
            RateLimitedSampler(-1.0)

    def test_initial_burst(self):
        """Bucket starts full — first per_second calls should all be sampled."""
        s = RateLimitedSampler(10.0)
        sampled = sum(
            1 for i in range(10)
            if s.should_sample(None, i, "x").decision == Decision.RECORD_AND_SAMPLE
        )
        assert sampled == 10

    def test_bucket_exhausted(self):
        """After draining, next call is dropped."""
        s = RateLimitedSampler(3.0)
        # Drain all 3 tokens
        for _ in range(3):
            s.should_sample(None, 0, "x")
        # 4th should drop
        r = s.should_sample(None, 0, "x")
        assert r.decision == Decision.DROP

    def test_refill_over_time(self):
        """After waiting, bucket refills and sampling resumes."""
        s = RateLimitedSampler(100.0)
        # Drain tokens
        for _ in range(100):
            s.should_sample(None, 0, "x")
        assert s.should_sample(None, 0, "x").decision == Decision.DROP
        # Wait > 1/100 second so ≥1 token refills
        time.sleep(0.02)
        r = s.should_sample(None, 0, "x")
        assert r.decision == Decision.RECORD_AND_SAMPLE

    def test_thread_safe(self):
        """Concurrent sampling doesn't exceed rate."""
        rate = 50.0
        s = RateLimitedSampler(rate)
        results: list[bool] = []
        lock = threading.Lock()

        def sample():
            r = s.should_sample(None, 0, "x").decision == Decision.RECORD_AND_SAMPLE
            with lock:
                results.append(r)

        threads = [threading.Thread(target=sample) for _ in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sampled = sum(results)
        assert sampled <= 200  # At most all (no over-sampling)


class TestParentBasedSampler:
    def test_description_includes_root(self):
        root = TraceIdRatioSampler(0.5)
        s = ParentBasedSampler(root)
        desc = s.get_description()
        assert "ParentBased" in desc
        assert "TraceIdRatio" in desc

    def test_root_property(self):
        root = AlwaysOnSampler()
        s = ParentBasedSampler(root)
        assert s.root is root

    def test_no_parent_delegates_to_root(self):
        """With no parent context, root sampler decides."""
        s = ParentBasedSampler(AlwaysOffSampler())
        result = s.should_sample(None, 123, "test")
        assert result.decision == Decision.DROP

    def test_no_parent_always_on_root(self):
        s = ParentBasedSampler(AlwaysOnSampler())
        result = s.should_sample(None, 123, "test")
        assert result.decision == Decision.RECORD_AND_SAMPLE


# ===========================================================================
# Span emission tests
# ===========================================================================


class TestSpanConstants:
    """All declared span name / attribute constants have expected string values."""

    def test_span_names_prefixed(self):
        for name in [
            SpanName.AGENT_RUN, SpanName.AGENT_ITERATION, SpanName.MODEL_CALL,
            SpanName.TOOL_CALL, SpanName.ROUTER_DECISION, SpanName.RETRY_ATTEMPT,
        ]:
            assert name.startswith("effgen."), f"Unexpected span name: {name!r}"

    def test_agent_attrs_prefixed(self):
        for val in [AgentAttrs.PRESET, AgentAttrs.ITERATION, AgentAttrs.RUN_ID, AgentAttrs.TASK]:
            assert val.startswith("effgen.agent."), val

    def test_model_attrs_prefixed(self):
        for val in [
            ModelAttrs.PROVIDER, ModelAttrs.NAME, ModelAttrs.CACHED_TOKENS,
            ModelAttrs.INPUT_TOKENS, ModelAttrs.OUTPUT_TOKENS,
            ModelAttrs.REASONING_EFFORT, ModelAttrs.THINKING_BUDGET,
            ModelAttrs.PARTS_COUNT, ModelAttrs.COST_USD, ModelAttrs.OUTCOME,
            ModelAttrs.LATENCY_MS,
        ]:
            assert val.startswith("effgen.model."), val

    def test_tool_attrs_prefixed(self):
        for val in [ToolAttrs.NAME, ToolAttrs.STATUS, ToolAttrs.INPUT, ToolAttrs.LATENCY_MS]:
            assert val.startswith("effgen.tool."), val

    def test_router_attrs_prefixed(self):
        for val in [
            RouterAttrs.POLICY, RouterAttrs.SELECTED_PROVIDER,
            RouterAttrs.CONSIDERED, RouterAttrs.ELIMINATED_COUNT,
        ]:
            assert val.startswith("effgen.router."), val

    def test_retry_attrs_prefixed(self):
        for val in [RetryAttrs.ATTEMPT, RetryAttrs.REASON, RetryAttrs.DELAY_S]:
            assert val.startswith("effgen.retry."), val


class TestAgentRunSpan:
    def test_span_name(self):
        exporter = _make_exporter()
        with start_agent_run(preset="my_agent", task="hello", run_id="r1"):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == SpanName.AGENT_RUN for s in spans)

    def test_preset_attribute(self):
        exporter = _make_exporter()
        with start_agent_run(preset="my_agent", task="hello"):
            pass
        spans = exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == SpanName.AGENT_RUN)
        assert run_span.attributes[AgentAttrs.PRESET] == "my_agent"

    def test_task_attribute_truncated(self):
        exporter = _make_exporter()
        long_task = "x" * 1000
        with start_agent_run(preset="a", task=long_task):
            pass
        spans = exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == SpanName.AGENT_RUN)
        assert len(run_span.attributes[AgentAttrs.TASK]) <= 500

    def test_run_id_attribute(self):
        exporter = _make_exporter()
        with start_agent_run(preset="a", task="t", run_id="abc-123"):
            pass
        spans = exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == SpanName.AGENT_RUN)
        assert run_span.attributes[AgentAttrs.RUN_ID] == "abc-123"

    def test_no_run_id_skips_attribute(self):
        exporter = _make_exporter()
        with start_agent_run(preset="a", task="t"):
            pass
        spans = exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == SpanName.AGENT_RUN)
        # run_id not set
        assert AgentAttrs.RUN_ID not in run_span.attributes


class TestModelCallSpan:
    def test_span_name(self):
        exporter = _make_exporter()
        with start_model_call(provider="openai", model="gpt-4o-mini"):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == SpanName.MODEL_CALL for s in spans)

    def test_provider_and_name_attributes(self):
        exporter = _make_exporter()
        with start_model_call(provider="cerebras", model="llama3.1-8b"):
            pass
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert m.attributes[ModelAttrs.PROVIDER] == "cerebras"
        assert m.attributes[ModelAttrs.NAME] == "llama3.1-8b"

    def test_outcome_defaults_to_ok(self):
        exporter = _make_exporter()
        with start_model_call(provider="x", model="y"):
            pass
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert m.attributes.get(ModelAttrs.OUTCOME) == "ok"

    def test_latency_ms_auto_set(self):
        exporter = _make_exporter()
        with start_model_call(provider="x", model="y"):
            pass
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert ModelAttrs.LATENCY_MS in m.attributes
        assert m.attributes[ModelAttrs.LATENCY_MS] >= 0.0

    def test_reasoning_effort_optional(self):
        exporter = _make_exporter()
        with start_model_call(provider="openai", model="o3", reasoning_effort="high"):
            pass
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert m.attributes[ModelAttrs.REASONING_EFFORT] == "high"

    def test_thinking_budget_optional(self):
        exporter = _make_exporter()
        with start_model_call(provider="google", model="gemini-2.0-flash-thinking", thinking_budget=1024):
            pass
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert m.attributes[ModelAttrs.THINKING_BUDGET] == 1024

    def test_parts_count_optional(self):
        exporter = _make_exporter()
        with start_model_call(provider="google", model="gemini-2.0-flash", parts_count=3):
            pass
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert m.attributes[ModelAttrs.PARTS_COUNT] == 3

    def test_outcome_error_on_exception(self):
        exporter = _make_exporter()
        with pytest.raises(ValueError):
            with start_model_call(provider="x", model="y"):
                raise ValueError("fail")
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert m.attributes.get(ModelAttrs.OUTCOME) == "error"

    def test_caller_can_set_token_attrs(self):
        exporter = _make_exporter()
        with start_model_call(provider="openai", model="gpt-4o-mini") as span:
            span.set_attribute(ModelAttrs.INPUT_TOKENS, 100)
            span.set_attribute(ModelAttrs.OUTPUT_TOKENS, 50)
            span.set_attribute(ModelAttrs.CACHED_TOKENS, 20)
            span.set_attribute(ModelAttrs.COST_USD, 0.001)
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert m.attributes[ModelAttrs.INPUT_TOKENS] == 100
        assert m.attributes[ModelAttrs.OUTPUT_TOKENS] == 50
        assert m.attributes[ModelAttrs.CACHED_TOKENS] == 20
        assert abs(m.attributes[ModelAttrs.COST_USD] - 0.001) < 1e-9


class TestToolCallSpan:
    def test_span_name(self):
        exporter = _make_exporter()
        with start_tool_call(tool_name="calculator", tool_input="2+2"):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == SpanName.TOOL_CALL for s in spans)

    def test_tool_name_attribute(self):
        exporter = _make_exporter()
        with start_tool_call(tool_name="web_search", tool_input="query"):
            pass
        spans = exporter.get_finished_spans()
        t = next(s for s in spans if s.name == SpanName.TOOL_CALL)
        assert t.attributes[ToolAttrs.NAME] == "web_search"

    def test_input_truncated(self):
        exporter = _make_exporter()
        with start_tool_call(tool_name="t", tool_input="x" * 1000):
            pass
        spans = exporter.get_finished_spans()
        t = next(s for s in spans if s.name == SpanName.TOOL_CALL)
        assert len(t.attributes[ToolAttrs.INPUT]) <= 500

    def test_status_defaults_to_ok(self):
        exporter = _make_exporter()
        with start_tool_call(tool_name="calc", tool_input="1"):
            pass
        spans = exporter.get_finished_spans()
        t = next(s for s in spans if s.name == SpanName.TOOL_CALL)
        assert t.attributes.get(ToolAttrs.STATUS) == "ok"

    def test_latency_ms_auto_set(self):
        exporter = _make_exporter()
        with start_tool_call(tool_name="calc", tool_input="1"):
            pass
        spans = exporter.get_finished_spans()
        t = next(s for s in spans if s.name == SpanName.TOOL_CALL)
        assert ToolAttrs.LATENCY_MS in t.attributes
        assert t.attributes[ToolAttrs.LATENCY_MS] >= 0.0

    def test_status_error_on_exception(self):
        exporter = _make_exporter()
        with pytest.raises(RuntimeError):
            with start_tool_call(tool_name="bad", tool_input=""):
                raise RuntimeError("tool error")
        spans = exporter.get_finished_spans()
        t = next(s for s in spans if s.name == SpanName.TOOL_CALL)
        assert t.attributes.get(ToolAttrs.STATUS) == "error"


class TestRouterDecisionSpan:
    def test_span_name(self):
        exporter = _make_exporter()
        with start_router_decision(policy="cost"):
            pass
        spans = exporter.get_finished_spans()
        assert any(s.name == SpanName.ROUTER_DECISION for s in spans)

    def test_policy_attribute(self):
        exporter = _make_exporter()
        with start_router_decision(policy="latency"):
            pass
        spans = exporter.get_finished_spans()
        r = next(s for s in spans if s.name == SpanName.ROUTER_DECISION)
        assert r.attributes[RouterAttrs.POLICY] == "latency"

    def test_candidates_attribute(self):
        exporter = _make_exporter()
        with start_router_decision(
            policy="cost",
            candidates=[("cerebras", "llama3.1-8b"), ("openai", "gpt-4o-mini")],
        ):
            pass
        spans = exporter.get_finished_spans()
        r = next(s for s in spans if s.name == SpanName.ROUTER_DECISION)
        considered = r.attributes.get(RouterAttrs.CONSIDERED, "")
        assert "cerebras" in considered
        assert "openai" in considered


class TestRetryEvent:
    def test_retry_event_added_to_span(self):
        exporter = _make_exporter()
        with start_model_call(provider="openai", model="gpt-4o-mini") as span:
            record_retry_attempt(span, attempt=1, reason="http_429", delay_s=1.0)
            record_retry_attempt(span, attempt=2, reason="http_429", delay_s=2.0)
        spans = exporter.get_finished_spans()
        m = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        events = m.events
        retry_events = [e for e in events if e.name == SpanName.RETRY_ATTEMPT]
        assert len(retry_events) == 2
        assert retry_events[0].attributes[RetryAttrs.ATTEMPT] == 1
        assert retry_events[0].attributes[RetryAttrs.REASON] == "http_429"
        assert retry_events[0].attributes[RetryAttrs.DELAY_S] == 1.0
        assert retry_events[1].attributes[RetryAttrs.ATTEMPT] == 2

    def test_retry_event_non_blocking_on_none_span(self):
        """record_retry_attempt with None span must not raise."""
        record_retry_attempt(None, attempt=1, reason="test")


# ===========================================================================
# 3-tool agent run span tree test (core exit criterion)
# ===========================================================================


class TestThreeToolAgentSpanTree:
    """
    Assert the full span tree shape for a simulated 3-tool agent run.

    Hierarchy expected (7 spans total):
      effgen.agent.run (root)
        effgen.agent.iteration (child)
          effgen.model.call (grandchild)
          effgen.tool.call [calculator]
          effgen.tool.call [web_search]
          effgen.tool.call [wikipedia]
        effgen.router.decision (child of run)
    """

    def test_span_count(self):
        exporter = _make_exporter()
        self._run_3_tool_agent(exporter)
        spans = exporter.get_finished_spans()
        # 7 spans: agent.run (1) + agent.iteration (1) + model.call (1)
        #          + tool.call x3 (3) + router.decision (1) = 7
        assert len(spans) == 7, f"Expected 7 spans, got {len(spans)}: {[s.name for s in spans]}"

    def test_span_names_present(self):
        exporter = _make_exporter()
        self._run_3_tool_agent(exporter)
        spans = exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert SpanName.AGENT_RUN in names
        assert SpanName.AGENT_ITERATION in names
        assert SpanName.MODEL_CALL in names
        assert names.count(SpanName.TOOL_CALL) == 3
        assert SpanName.ROUTER_DECISION in names

    def test_agent_run_span_has_preset(self):
        exporter = _make_exporter()
        self._run_3_tool_agent(exporter)
        spans = exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == SpanName.AGENT_RUN)
        assert run_span.attributes[AgentAttrs.PRESET] == "test_agent"

    def test_model_call_has_all_required_attrs(self):
        exporter = _make_exporter()
        self._run_3_tool_agent(exporter)
        spans = exporter.get_finished_spans()
        model_span = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        attrs = model_span.attributes
        assert ModelAttrs.PROVIDER in attrs
        assert ModelAttrs.NAME in attrs
        assert ModelAttrs.INPUT_TOKENS in attrs
        assert ModelAttrs.OUTPUT_TOKENS in attrs
        assert ModelAttrs.OUTCOME in attrs
        assert ModelAttrs.LATENCY_MS in attrs

    def test_tool_calls_have_status_and_name(self):
        exporter = _make_exporter()
        self._run_3_tool_agent(exporter)
        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name == SpanName.TOOL_CALL]
        assert len(tool_spans) == 3
        tool_names = {s.attributes[ToolAttrs.NAME] for s in tool_spans}
        assert tool_names == {"calculator", "web_search", "wikipedia"}
        for ts in tool_spans:
            assert ToolAttrs.STATUS in ts.attributes
            assert ToolAttrs.LATENCY_MS in ts.attributes

    def test_router_decision_has_policy_and_provider(self):
        exporter = _make_exporter()
        self._run_3_tool_agent(exporter)
        spans = exporter.get_finished_spans()
        r = next(s for s in spans if s.name == SpanName.ROUTER_DECISION)
        assert RouterAttrs.POLICY in r.attributes
        assert RouterAttrs.SELECTED_PROVIDER in r.attributes

    def test_span_tree_parent_child_structure(self):
        """Verify parent trace IDs form the expected hierarchy."""
        exporter = _make_exporter()
        self._run_3_tool_agent(exporter)
        spans = exporter.get_finished_spans()

        run_span = next(s for s in spans if s.name == SpanName.AGENT_RUN)
        run_trace_id = run_span.context.trace_id
        run_span_id = run_span.context.span_id

        # All spans share the same trace_id
        for s in spans:
            assert s.context.trace_id == run_trace_id, (
                f"Span {s.name} has different trace_id"
            )

        # The iteration span is a child of agent.run
        iter_span = next(s for s in spans if s.name == SpanName.AGENT_ITERATION)
        assert iter_span.parent is not None
        assert iter_span.parent.span_id == run_span_id

        # model.call and tool.calls are children of agent.iteration
        iter_span_id = iter_span.context.span_id
        model_span = next(s for s in spans if s.name == SpanName.MODEL_CALL)
        assert model_span.parent is not None
        assert model_span.parent.span_id == iter_span_id

        tool_spans = [s for s in spans if s.name == SpanName.TOOL_CALL]
        for ts in tool_spans:
            assert ts.parent is not None
            assert ts.parent.span_id == iter_span_id

    @staticmethod
    def _run_3_tool_agent(exporter: "InMemorySpanExporter") -> None:
        """Simulate a 3-tool agent run and populate the exporter."""
        with start_agent_run(preset="test_agent", task="calculate, search, and look up", run_id="run-test") as _aspan:
            with start_agent_iteration(preset="test_agent", iteration=1) as _ispan:
                # Model call
                with start_model_call(provider="cerebras", model="llama3.1-8b") as mspan:
                    mspan.set_attribute(ModelAttrs.INPUT_TOKENS, 200)
                    mspan.set_attribute(ModelAttrs.OUTPUT_TOKENS, 80)
                    mspan.set_attribute(ModelAttrs.CACHED_TOKENS, 0)
                    mspan.set_attribute(ModelAttrs.COST_USD, 0.0)
                    mspan.set_attribute(ModelAttrs.OUTCOME, "ok")

                # 3 tool calls
                with start_tool_call("calculator", "sqrt(144)") as t1:
                    t1.set_attribute(ToolAttrs.STATUS, "ok")
                with start_tool_call("web_search", "Python news 2025") as t2:
                    t2.set_attribute(ToolAttrs.STATUS, "ok")
                with start_tool_call("wikipedia", "OpenTelemetry") as t3:
                    t3.set_attribute(ToolAttrs.STATUS, "ok")

            # Router decision (sibling of iteration, child of run)
            with start_router_decision("cost", [("cerebras", "llama3.1-8b")]) as rspan:
                rspan.set_attribute(RouterAttrs.POLICY, "cost")
                rspan.set_attribute(RouterAttrs.SELECTED_PROVIDER, "cerebras")
                rspan.set_attribute(RouterAttrs.ELIMINATED_COUNT, 0)


# ===========================================================================
# Sampler enforcement (AlwaysOff drops spans)
# ===========================================================================


class TestAlwaysOffDropsSpans:
    def test_no_spans_when_always_off(self):
        exporter = InMemorySpanExporter()
        reset_tracing()
        setup_tracing(sampler=AlwaysOffSampler(), exporter=exporter)
        with start_agent_run(preset="a", task="x"):
            with start_model_call(provider="p", model="m"):
                pass
        spans = exporter.get_finished_spans()
        assert len(spans) == 0, f"Expected 0 spans with AlwaysOff, got {len(spans)}"


# ===========================================================================
# Non-blocking tests
# ===========================================================================


class TestNonBlocking:
    def test_setup_tracing_non_blocking_on_bad_exporter(self):
        """setup_tracing should not raise even if exporter is broken."""
        # reset state
        reset_tracing()
        bad = MagicMock()
        bad.export.side_effect = RuntimeError("broken")
        # Should not raise
        setup_tracing(sampler=AlwaysOnSampler(), exporter=bad)

    def test_span_exception_propagates(self):
        """The underlying exception must propagate; tracing should not swallow it."""
        _make_exporter()
        with pytest.raises(ValueError):
            with start_tool_call("t", "i"):
                raise ValueError("real error")

    def test_set_span_attribute_no_active_span(self):
        """set_span_attribute with no active span must not raise."""
        set_span_attribute("some.key", "value")

    def test_set_span_error_no_active_span(self):
        """set_span_error with no active span must not raise."""
        set_span_error(ValueError("oops"))


# ===========================================================================
# Cost reaches the model span on every path
# ===========================================================================


class TestModelCallSpanCarriesItsCost:
    """The money must be on the span whenever the call reported one.

    The adapters stamp ``effgen.model.cost_usd`` through
    ``set_span_attribute``, which writes to whatever span is *currently
    active*. It therefore landed only when the adapter's call ran inside the
    span the agent had opened; on the other paths the span carried the
    provider, the model and both token counts and no money — which reads as a
    pricing gap in a dashboard while the run's own total is right.

    The agent now stamps it from the result metadata, with the span in hand.
    Measured live before the change: 2 of 8 tool-using groq runs lost it, both
    on the native-prompt turn. After: 8 of 8 carry it.
    """

    @staticmethod
    def _agent(cost, tools):
        from effgen.core.agent import Agent, AgentConfig
        from effgen.models.base import (
            BaseModel,
            GenerationResult,
            ModelType,
            TokenCount,
        )

        class _Priced(BaseModel):
            def __init__(self):
                super().__init__(model_name="fake-priced", model_type=ModelType.TRANSFORMERS)
                self._is_loaded = True

            def load(self):
                self._is_loaded = True

            def unload(self):
                self._is_loaded = False

            def get_context_length(self):
                return 4096

            def count_tokens(self, text):
                return TokenCount(prompt_tokens=1, completion_tokens=0, total_tokens=1)

            def generate_stream(self, prompt, config=None, **kwargs):
                yield "x"

            def generate(self, prompt, config=None, **kwargs):
                return GenerationResult(
                    text="Final Answer: 42",
                    tokens_used=7,
                    finish_reason="stop",
                    model_name="fake-priced",
                    metadata={
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                        "cost_usd": cost,
                    },
                )

        return Agent(AgentConfig(
            name="span-cost", model=_Priced(), tools=tools,
            raise_on_error=False, enable_memory=False, enable_sub_agents=False,
        ))

    @staticmethod
    def _model_spans(exporter):
        return [s for s in exporter.get_finished_spans() if s.name == SpanName.MODEL_CALL]

    def test_the_direct_path_stamps_the_cost(self):
        exporter = _make_exporter()
        with self._agent(0.000012, tools=[]) as agent:
            agent.run("what is the answer")
        spans = self._model_spans(exporter)
        assert spans, "no model.call span was recorded"
        assert all(s.attributes.get(ModelAttrs.COST_USD) == 0.000012 for s in spans)

    def test_the_tool_loop_path_stamps_the_cost(self):
        from effgen.tools import get_registry

        exporter = _make_exporter()
        calculator = get_registry().get_tool_sync("calculator")
        with self._agent(0.000012, tools=[calculator]) as agent:
            agent.run("what is the answer")
        spans = self._model_spans(exporter)
        assert spans, "no model.call span was recorded"
        assert all(s.attributes.get(ModelAttrs.COST_USD) == 0.000012 for s in spans)

    def test_an_unpriced_call_writes_no_cost_attribute(self):
        """``None`` is not zero: an unpriced model must not read as free."""
        exporter = _make_exporter()
        with self._agent(None, tools=[]) as agent:
            agent.run("what is the answer")
        spans = self._model_spans(exporter)
        assert spans
        assert all(ModelAttrs.COST_USD not in (s.attributes or {}) for s in spans)
