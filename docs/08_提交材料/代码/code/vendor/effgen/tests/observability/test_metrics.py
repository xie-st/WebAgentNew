"""
Tests for effgen.observability.metrics

Verifies:
- LabeledHistogram bucket accumulation and Prometheus text export
- LabeledCounter increment and Prometheus text export
- record_model_call / record_tool_call / record_agent_iteration / record_tokens helpers
- export_metrics() produces valid Prometheus text format
- Prometheus format correctness: HELP, TYPE, _bucket, _sum, _count lines
- Thread-safety (basic)
"""

from __future__ import annotations

import math
import re
import threading

import pytest

from effgen.observability.metrics import (
    LabeledCounter,
    LabeledGauge,
    LabeledHistogram,
    agent_iteration_latency,
    bulkhead_active,
    bulkhead_queued,
    bulkhead_utilization_pct,
    circuit_breaker_state,
    export_metrics,
    http_requests_total,
    model_call_latency,
    record_agent_iteration,
    record_http_request,
    record_model_call,
    record_tokens,
    record_tool_call,
    reset_all,
    tokens_total,
    tool_call_latency,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prom_blocks(text: str) -> dict[str, list[str]]:
    """Parse Prometheus text into {metric_family_name: [lines]} blocks."""
    blocks: dict[str, list[str]] = {}
    current_name: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# HELP "):
            current_name = line.split()[2]
            blocks.setdefault(current_name, [])
            blocks[current_name].append(line)
        elif line.startswith("# TYPE "):
            if current_name:
                blocks[current_name].append(line)
        elif current_name:
            blocks[current_name].append(line)
    return blocks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset():
    """Reset all metrics before and after each test."""
    reset_all()
    yield
    reset_all()


# ---------------------------------------------------------------------------
# LabeledHistogram unit tests
# ---------------------------------------------------------------------------

class TestLabeledHistogram:
    def test_observe_no_labels(self):
        h = LabeledHistogram(
            name="test_hist",
            help="test",
            label_names=(),
            buckets=(0.1, 0.5, 1.0),
        )
        h.observe(0.05)
        h.observe(0.2)
        h.observe(0.8)

        text = h.export()
        assert "# HELP test_hist test" in text
        assert "# TYPE test_hist histogram" in text
        # +Inf bucket should have count 3
        assert 'le="+Inf"' in text

    def test_buckets_always_have_inf(self):
        h = LabeledHistogram(
            name="no_inf",
            help="x",
            label_names=(),
            buckets=(0.1, 0.5),
        )
        assert math.isinf(h.buckets[-1])

    def test_cumulative_buckets(self):
        h = LabeledHistogram(
            name="cumul",
            help="x",
            label_names=(),
            buckets=(0.1, 0.5, 1.0),
        )
        h.observe(0.05)   # <= 0.1
        h.observe(0.3)    # <= 0.5
        h.observe(0.9)    # <= 1.0
        h.observe(2.0)    # > 1.0, only +Inf

        text = h.export()
        lines = text.splitlines()
        # Build a map: le_value -> cumulative count
        bucket_lines = [ln for ln in lines if "_bucket" in ln]
        counts = {}
        for bl in bucket_lines:
            m = re.search(r'le="([^"]+)".*?}\s+(\d+)', bl)
            if m:
                counts[m.group(1)] = int(m.group(2))

        assert counts.get("0.1") == 1    # only 0.05
        assert counts.get("0.5") == 2    # 0.05 + 0.3
        assert counts.get("1.0") == 3    # 0.05 + 0.3 + 0.9
        assert counts.get("+Inf") == 4   # all 4

    def test_labels_in_export(self):
        h = LabeledHistogram(
            name="labeled_hist",
            help="x",
            label_names=("provider", "model", "outcome"),
            buckets=(0.5, 1.0),
        )
        h.observe(0.3, labels={"provider": "cerebras", "model": "llama3.1-8b", "outcome": "ok"})

        text = h.export()
        assert 'provider="cerebras"' in text
        assert 'model="llama3.1-8b"' in text
        assert 'outcome="ok"' in text

    def test_sum_and_count(self):
        h = LabeledHistogram(
            name="labeled_sc",
            help="x",
            label_names=("tag",),
            buckets=(1.0, 2.0),
        )
        h.observe(0.5, labels={"tag": "a"})
        h.observe(1.5, labels={"tag": "a"})

        text = h.export()
        # _sum should be 2.0, _count should be 2
        # Use exact suffix matching: lines ending in "_sum{...} value"
        sum_lines = [ln for ln in text.splitlines()
                     if re.match(r'labeled_sc_sum\{', ln)]
        count_lines = [ln for ln in text.splitlines()
                       if re.match(r'labeled_sc_count\{', ln)]
        assert sum_lines, f"No _sum lines found in:\n{text}"
        assert count_lines, f"No _count lines found in:\n{text}"
        assert "2.0" in sum_lines[0], f"Expected 2.0 in: {sum_lines[0]}"
        assert count_lines[0].endswith("} 2"), f"Expected count 2 in: {count_lines[0]}"

    def test_reset(self):
        h = LabeledHistogram(
            name="reset_hist",
            help="x",
            label_names=(),
            buckets=(1.0,),
        )
        h.observe(0.5)
        h.reset()
        text = h.export()
        # After reset, no data lines (only HELP + TYPE)
        data_lines = [ln for ln in text.splitlines()
                      if ln and not ln.startswith("#")]
        assert data_lines == []

    def test_thread_safety(self):
        h = LabeledHistogram(
            name="thread_hist",
            help="x",
            label_names=("t",),
            buckets=(0.1, 0.5, 1.0),
        )
        errors = []

        def worker(i):
            try:
                for _ in range(50):
                    h.observe(0.05 * (i % 5), labels={"t": str(i % 3)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# LabeledCounter unit tests
# ---------------------------------------------------------------------------

class TestLabeledCounter:
    def test_increment(self):
        c = LabeledCounter(name="test_cnt", help="test counter")
        c.inc(1.0, labels={"kind": "input"})
        c.inc(2.0, labels={"kind": "input"})
        c.inc(1.0, labels={"kind": "output"})
        assert c.get(labels={"kind": "input"}) == 3.0
        assert c.get(labels={"kind": "output"}) == 1.0
        assert c.get(labels={"kind": "cached"}) == 0.0

    def test_export_format(self):
        c = LabeledCounter(name="effgen_tokens_test", help="test")
        c.inc(10, labels={"provider": "cerebras", "model": "llama3.1-8b", "kind": "input"})
        text = c.export()
        assert "# HELP effgen_tokens_test test" in text
        assert "# TYPE effgen_tokens_test counter" in text
        assert 'kind="input"' in text
        assert "10" in text

    def test_reset(self):
        c = LabeledCounter(name="test_reset_cnt", help="x")
        c.inc(5)
        c.reset()
        assert c.get() == 0.0

    def test_no_labels(self):
        c = LabeledCounter(name="plain_cnt", help="plain")
        c.inc(3.0)
        text = c.export()
        assert "plain_cnt 3.0" in text


# ---------------------------------------------------------------------------
# LabeledGauge unit tests
# ---------------------------------------------------------------------------

class TestLabeledGauge:
    def test_set_and_get(self):
        g = LabeledGauge(name="test_gauge", help="test gauge")
        g.set(2, labels={"provider": "openai"})
        assert g.get(labels={"provider": "openai"}) == 2
        assert g.get(labels={"provider": "groq"}) is None

    def test_set_overwrites_not_accumulates(self):
        g = LabeledGauge(name="test_gauge2", help="test")
        g.set(1, labels={"provider": "openai"})
        g.set(5, labels={"provider": "openai"})
        assert g.get(labels={"provider": "openai"}) == 5

    def test_export_format(self):
        g = LabeledGauge(name="effgen_test_state", help="test state")
        g.set(2, labels={"provider": "openai"})
        text = g.export()
        assert "# HELP effgen_test_state test state" in text
        assert "# TYPE effgen_test_state gauge" in text
        assert 'provider="openai"' in text
        assert "2" in text

    def test_reset(self):
        g = LabeledGauge(name="test_reset_gauge", help="x")
        g.set(3, labels={"provider": "openai"})
        g.reset()
        assert g.get(labels={"provider": "openai"}) is None

    def test_no_labels(self):
        g = LabeledGauge(name="plain_gauge", help="plain")
        g.set(7)
        text = g.export()
        assert "plain_gauge 7" in text


# ---------------------------------------------------------------------------
# Circuit-breaker / bulkhead reliability gauges
# ---------------------------------------------------------------------------

class TestReliabilityGauges:
    def test_export_includes_gauge_help_type_even_when_empty(self):
        reset_all()
        text = export_metrics()
        assert "# HELP effgen_circuit_breaker_state" in text
        assert "# TYPE effgen_circuit_breaker_state gauge" in text
        assert "# HELP effgen_bulkhead_active" in text
        assert "# HELP effgen_bulkhead_queued" in text
        assert "# HELP effgen_bulkhead_utilization_pct" in text

    def test_circuit_breaker_state_reflects_live_registry(self):
        from effgen.models.registry import ProviderRegistry

        provider = "_test_reliability_gauge_provider"
        ProviderRegistry._providers.setdefault(provider, {})
        try:
            cb = ProviderRegistry.get_circuit_breaker(
                provider, failure_threshold=2, recovery_timeout=30.0
            )
            cb.on_failure()
            cb.on_failure()
            assert cb.state.value == "open"

            text = export_metrics()
            assert f'effgen_circuit_breaker_state{{provider="{provider}"}} 2' in text
        finally:
            ProviderRegistry._providers.pop(provider, None)
            circuit_breaker_state.reset()

    def test_bulkhead_gauges_reflect_live_registry(self):
        from effgen.models.registry import ProviderRegistry

        provider = "_test_reliability_bulkhead_provider"
        ProviderRegistry._providers.setdefault(provider, {})
        try:
            bh = ProviderRegistry.get_bulkhead(provider, max_concurrency=2, queue_size=2)
            with bh.acquire():
                text = export_metrics()
                assert f'effgen_bulkhead_active{{provider="{provider}"}} 1' in text
                assert f'effgen_bulkhead_utilization_pct{{provider="{provider}"}} 50.0' in text
        finally:
            ProviderRegistry._providers.pop(provider, None)
            bulkhead_active.reset()
            bulkhead_queued.reset()
            bulkhead_utilization_pct.reset()

    def test_untouched_provider_reports_no_gauge_value(self):
        from effgen.models.registry import ProviderRegistry

        provider = "_test_reliability_untouched_provider"
        ProviderRegistry._providers.setdefault(provider, {})
        try:
            # No get_circuit_breaker()/get_bulkhead() call for this provider —
            # it must not appear in the gauge output at all.
            text = export_metrics()
            assert f'provider="{provider}"' not in text
        finally:
            ProviderRegistry._providers.pop(provider, None)


# ---------------------------------------------------------------------------
# Recording helper tests
# ---------------------------------------------------------------------------

class TestRecordingHelpers:
    def test_record_model_call_ok(self):
        record_model_call(
            provider="cerebras",
            model="llama3.1-8b",
            outcome="ok",
            latency=0.3,
        )
        text = model_call_latency.export()
        assert 'provider="cerebras"' in text
        assert 'outcome="ok"' in text

    def test_record_model_call_error(self):
        record_model_call(
            provider="openai",
            model="gpt-4o-mini",
            outcome="error",
            latency=1.2,
        )
        text = model_call_latency.export()
        assert 'outcome="error"' in text

    def test_record_tool_call(self):
        record_tool_call(tool="calculator", outcome="ok", latency=0.01)
        text = tool_call_latency.export()
        assert 'tool="calculator"' in text
        assert 'outcome="ok"' in text

    def test_record_agent_iteration(self):
        record_agent_iteration(preset="default", latency=0.5)
        text = agent_iteration_latency.export()
        assert 'preset="default"' in text

    def test_record_tokens_all_kinds(self):
        record_tokens(
            provider="cerebras",
            model="llama3.1-8b",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=20,
        )
        assert tokens_total.get(labels={"provider": "cerebras", "model": "llama3.1-8b", "kind": "input"}) == 100
        assert tokens_total.get(labels={"provider": "cerebras", "model": "llama3.1-8b", "kind": "output"}) == 50
        assert tokens_total.get(labels={"provider": "cerebras", "model": "llama3.1-8b", "kind": "cached"}) == 20

    def test_record_tokens_partial(self):
        record_tokens(provider="openai", model="gpt-4o-mini", input_tokens=10)
        assert tokens_total.get(labels={"provider": "openai", "model": "gpt-4o-mini", "kind": "input"}) == 10
        # output/cached should be 0 (not incremented)
        assert tokens_total.get(labels={"provider": "openai", "model": "gpt-4o-mini", "kind": "output"}) == 0.0

    def test_record_tokens_accumulated(self):
        for _ in range(5):
            record_tokens(provider="cerebras", model="llama3.1-8b", input_tokens=10)
        assert tokens_total.get(
            labels={"provider": "cerebras", "model": "llama3.1-8b", "kind": "input"}
        ) == 50.0

    def test_record_http_request(self):
        record_http_request(route="/v1/chat/completions", method="POST", status=200)
        record_http_request(route="/v1/chat/completions", method="POST", status=429)
        assert http_requests_total.get(
            labels={"route": "/v1/chat/completions", "method": "POST", "status": "200"}
        ) == 1.0
        assert http_requests_total.get(
            labels={"route": "/v1/chat/completions", "method": "POST", "status": "429"}
        ) == 1.0


# ---------------------------------------------------------------------------
# export_metrics() Prometheus format validity
# ---------------------------------------------------------------------------

class TestExportMetrics:
    def test_produces_text(self):
        record_model_call(
            provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.5
        )
        text = export_metrics()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_prometheus_help_and_type_lines(self):
        record_tool_call(tool="search", outcome="ok", latency=0.1)
        text = export_metrics()
        lines = text.splitlines()
        help_lines = [ln for ln in lines if ln.startswith("# HELP ")]
        type_lines = [ln for ln in lines if ln.startswith("# TYPE ")]
        assert len(help_lines) >= 4, "Expected at least 4 HELP lines"
        assert len(type_lines) >= 4, "Expected at least 4 TYPE lines"

    def test_histogram_names_present(self):
        text = export_metrics()
        # These names appear in HELP/TYPE lines even with no data
        # (but only in non-empty exports — let's add one observation)
        record_model_call(
            provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.1
        )
        record_tool_call(tool="calc", outcome="ok", latency=0.02)
        record_agent_iteration(preset="default", latency=0.5)
        record_tokens(provider="cerebras", model="llama3.1-8b", input_tokens=5)
        record_http_request(route="/v1/chat/completions", method="POST", status=200)
        text = export_metrics()
        assert "effgen_model_call_latency_seconds" in text
        assert "effgen_tool_call_latency_seconds" in text
        assert "effgen_agent_iteration_latency_seconds" in text
        assert "effgen_tokens_total" in text
        assert "effgen_http_requests_total" in text

    def test_bucket_lines_format(self):
        record_model_call(
            provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.5
        )
        text = export_metrics()
        bucket_lines = [ln for ln in text.splitlines() if "_bucket{" in ln]
        assert len(bucket_lines) > 0, "No _bucket lines found"
        # Each bucket line should match: name{labels,le="..."} count
        patt = re.compile(r'^\w+_bucket\{.*le="[^"]+".*\}\s+\d+')
        for line in bucket_lines:
            assert patt.match(line), f"Malformed bucket line: {line!r}"

    def test_sum_and_count_lines(self):
        record_model_call(
            provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.25
        )
        text = export_metrics()
        # Match lines that are exactly <name>_sum{...} or <name>_count{...}
        sum_lines = [ln for ln in text.splitlines()
                     if re.match(r'\w+_sum\{', ln)]
        count_lines = [ln for ln in text.splitlines()
                       if re.match(r'\w+_count\{', ln)]
        assert sum_lines, "No _sum lines in Prometheus output"
        assert count_lines, "No _count lines in Prometheus output"

    def test_counter_type_line(self):
        record_tokens(provider="cerebras", model="llama3.1-8b", input_tokens=1)
        text = export_metrics()
        assert "# TYPE effgen_tokens_total counter" in text

    def test_no_secrets_in_output(self):
        """Metrics export must not leak any secret-like strings."""
        record_model_call(
            provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.1
        )
        text = export_metrics()
        # No API key patterns
        assert "sk-" not in text
        assert "csk-" not in text
        assert "Bearer " not in text

    def test_empty_export_is_valid(self):
        """Even with no observations, export_metrics should return valid Prometheus text."""
        text = export_metrics()
        # Should still have HELP + TYPE for the histogram names (via legacy metrics)
        # At minimum must be a non-empty string ending with newline
        assert text.endswith("\n")

    def test_multiple_providers_in_output(self):
        record_model_call(provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.1)
        record_model_call(provider="openai", model="gpt-4o-mini", outcome="ok", latency=0.2)
        text = export_metrics()
        assert 'provider="cerebras"' in text
        assert 'provider="openai"' in text
