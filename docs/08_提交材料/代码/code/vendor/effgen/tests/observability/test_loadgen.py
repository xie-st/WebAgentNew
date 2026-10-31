"""
Tests for effgen.tools.loadgen.

Covers:
- LoadConfig defaults and validation
- LoadScenario enum values
- _percentile helper
- RequestResult / LoadReport dataclasses
- LoadGenerator with the built-in mock (no real provider)
- Scenario prompt factories (fixed / synthetic / multi_tool)
- Ramp-up and think-time wiring
- report.to_dict() shape + p50/p95/p99 presence
- Zero-result edge case
- JSON output file writing
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from effgen.tools.loadgen import (
    LoadConfig,
    LoadGenerator,
    LoadReport,
    LoadScenario,
    RequestResult,
    _classify_loadtest_error,
    _mock_target,
    _percentile,
)

# ---------------------------------------------------------------------------
# _percentile helper
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([0.5], 50) == 0.5
        assert _percentile([0.5], 0) == 0.5
        assert _percentile([0.5], 100) == 0.5

    def test_p50_of_even_list(self):
        data = sorted([1.0, 2.0, 3.0, 4.0])
        p50 = _percentile(data, 50)
        assert 2.0 <= p50 <= 3.0

    def test_p0_is_min(self):
        data = sorted([3.0, 1.0, 4.0, 1.5, 2.5])
        assert _percentile(data, 0) == pytest.approx(1.0)

    def test_p100_is_max(self):
        data = sorted([3.0, 1.0, 4.0, 1.5, 2.5])
        assert _percentile(data, 100) == pytest.approx(4.0)

    def test_p95_of_100_elements(self):
        data = sorted(float(i) for i in range(100))
        p95 = _percentile(data, 95)
        assert 93.0 <= p95 <= 95.0


# ---------------------------------------------------------------------------
# LoadScenario enum
# ---------------------------------------------------------------------------

class TestLoadScenario:
    def test_values(self):
        assert LoadScenario.FIXED == "fixed"
        assert LoadScenario.SYNTHETIC == "synthetic"
        assert LoadScenario.MULTI_TOOL == "multi_tool"

    def test_from_string(self):
        assert LoadScenario("fixed") == LoadScenario.FIXED
        assert LoadScenario("synthetic") == LoadScenario.SYNTHETIC
        assert LoadScenario("multi_tool") == LoadScenario.MULTI_TOOL


# ---------------------------------------------------------------------------
# LoadConfig
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_defaults(self):
        cfg = LoadConfig()
        assert cfg.concurrency == 10
        assert cfg.duration == 30.0
        assert cfg.scenario == LoadScenario.FIXED
        assert cfg.request_timeout == 60.0   # explicit, never None
        assert cfg.think_time == 0.0
        assert cfg.ramp_up == 0.0
        assert cfg.provider is None
        assert cfg.model is None

    def test_custom_values(self):
        cfg = LoadConfig(concurrency=5, duration=10.0, scenario=LoadScenario.SYNTHETIC)
        assert cfg.concurrency == 5
        assert cfg.duration == 10.0
        assert cfg.scenario == LoadScenario.SYNTHETIC

    def test_request_timeout_is_never_none(self):
        """Explicit timeout is required — we never allow None."""
        cfg = LoadConfig()
        assert cfg.request_timeout is not None
        assert isinstance(cfg.request_timeout, float)


# ---------------------------------------------------------------------------
# RequestResult / LoadReport dataclasses
# ---------------------------------------------------------------------------

class TestRequestResult:
    def test_success_result(self):
        r = RequestResult(latency=0.05, success=True, prompt="hello", response="world")
        assert r.latency == 0.05
        assert r.success is True
        assert r.error is None

    def test_failure_result(self):
        r = RequestResult(latency=61.0, success=False, error="TimeoutError")
        assert r.success is False
        assert r.error == "TimeoutError"

    def test_error_category_defaults_none(self):
        r = RequestResult(latency=0.01, success=True)
        assert r.error_category is None

    def test_error_category_set_on_failure(self):
        r = RequestResult(latency=0.01, success=False, error="x", error_category="timeout")
        assert r.error_category == "timeout"


class TestClassifyLoadtestError:
    def test_auth_error_classified(self):
        from effgen.models.errors import ModelAuthError

        assert _classify_loadtest_error(ModelAuthError("openai")) == "auth"

    def test_not_found_error_classified(self):
        from effgen.models.errors import ModelNotFoundError

        assert _classify_loadtest_error(ModelNotFoundError("groq", "bad")) == "not_found"

    def test_generic_exception_classified_unknown_or_something(self):
        # Falls through classify_provider_error's heuristics; must never raise.
        result = _classify_loadtest_error(RuntimeError("something odd"))
        assert isinstance(result, str) and result


class TestLoadReport:
    def _make_report(self, **kwargs) -> LoadReport:
        defaults = {
            "scenario": "fixed",
            "concurrency": 10,
            "duration": 30.0,
            "total_requests": 100,
            "successful_requests": 98,
            "failed_requests": 2,
            "error_rate": 0.02,
            "throughput": 3.33,
            "p50_latency": 0.005,
            "p95_latency": 0.010,
            "p99_latency": 0.015,
            "min_latency": 0.001,
            "max_latency": 0.020,
            "mean_latency": 0.006,
            "stdev_latency": 0.002,
            "provider": None,
            "model": None,
        }
        defaults.update(kwargs)
        return LoadReport(**defaults)

    def test_to_dict_has_latency_sub_keys(self):
        report = self._make_report()
        d = report.to_dict()
        lat = d["latency"]
        for key in ("p50", "p95", "p99", "min", "max", "mean", "stdev"):
            assert key in lat, f"Missing latency.{key}"

    def test_to_dict_has_top_keys(self):
        report = self._make_report()
        d = report.to_dict()
        for key in ("scenario", "concurrency", "duration_s", "total_requests",
                    "successful_requests", "failed_requests", "error_rate",
                    "throughput_rps"):
            assert key in d, f"Missing top-level key '{key}'"

    def test_to_dict_include_raw_false(self):
        r = RequestResult(latency=0.01, success=True)
        report = self._make_report(raw_results=[r])
        d = report.to_dict(include_raw=False)
        assert "raw_results" not in d

    def test_to_dict_include_raw_true(self):
        r = RequestResult(latency=0.01, success=True, prompt="hi")
        report = self._make_report(raw_results=[r])
        d = report.to_dict(include_raw=True)
        assert "raw_results" in d
        assert len(d["raw_results"]) == 1

    def test_to_dict_include_raw_has_error_category(self):
        r = RequestResult(latency=0.01, success=False, error="x", error_category="timeout")
        report = self._make_report(raw_results=[r])
        d = report.to_dict(include_raw=True)
        assert d["raw_results"][0]["error_category"] == "timeout"

    def test_to_dict_has_requested_duration_and_drain(self):
        report = self._make_report(duration=35.6, requested_duration=10.0)
        d = report.to_dict()
        assert d["requested_duration_s"] == 10.0
        assert d["duration_s"] == 35.6
        assert d["drain_s"] == pytest.approx(25.6)

    def test_to_dict_drain_zero_when_no_overshoot(self):
        report = self._make_report(duration=10.0, requested_duration=10.0)
        d = report.to_dict()
        assert d["drain_s"] == 0.0

    def test_to_dict_drain_never_negative(self):
        """A run that finishes early (rare, but possible) must not report a
        negative drain time."""
        report = self._make_report(duration=9.5, requested_duration=10.0)
        d = report.to_dict()
        assert d["drain_s"] == 0.0

    def test_to_dict_error_breakdown_present_and_sorted(self):
        report = self._make_report(error_breakdown={"timeout": 2, "auth": 1})
        d = report.to_dict()
        assert d["error_breakdown"] == {"auth": 1, "timeout": 2}


# ---------------------------------------------------------------------------
# LoadGenerator — mock runs
# ---------------------------------------------------------------------------

class TestLoadGeneratorMock:
    def test_mock_run_produces_report(self):
        cfg = LoadConfig(concurrency=2, duration=2.0, scenario=LoadScenario.FIXED)
        gen = LoadGenerator(cfg)
        report = gen.run()
        assert isinstance(report, LoadReport)

    def test_mock_run_requests_positive(self):
        cfg = LoadConfig(concurrency=3, duration=2.0)
        gen = LoadGenerator(cfg)
        report = gen.run()
        assert report.total_requests > 0

    def test_mock_run_error_rate_zero(self):
        """Mock never fails — error rate should be 0."""
        cfg = LoadConfig(concurrency=4, duration=2.0)
        gen = LoadGenerator(cfg)
        report = gen.run()
        assert report.error_rate == 0.0

    def test_mock_run_p50_p95_p99_present(self):
        cfg = LoadConfig(concurrency=4, duration=2.0)
        gen = LoadGenerator(cfg)
        report = gen.run()
        assert report.p50_latency >= 0.0
        assert report.p95_latency >= report.p50_latency
        assert report.p99_latency >= report.p95_latency

    def test_mock_run_throughput_positive(self):
        cfg = LoadConfig(concurrency=4, duration=2.0)
        gen = LoadGenerator(cfg)
        report = gen.run()
        assert report.throughput > 0.0

    def test_mock_run_dict_is_json_serialisable(self):
        cfg = LoadConfig(concurrency=2, duration=2.0)
        gen = LoadGenerator(cfg)
        report = gen.run()
        d = report.to_dict()
        json.dumps(d)   # must not raise

    def test_concurrency_10_duration_5s(self):
        """Standard smoke: c=10, d=5s — used as canary."""
        cfg = LoadConfig(concurrency=10, duration=5.0)
        gen = LoadGenerator(cfg)
        report = gen.run()
        assert report.concurrency == 10
        assert report.total_requests > 10  # at least 1 req per VU
        assert report.error_rate == 0.0


# ---------------------------------------------------------------------------
# Scenario prompt variety
# ---------------------------------------------------------------------------

class TestScenarioPrompts:
    def test_fixed_always_same_prompt(self):
        gen = LoadGenerator(LoadConfig(scenario=LoadScenario.FIXED))
        prompts = {gen._make_prompt(vu, req) for vu in range(5) for req in range(5)}
        assert len(prompts) == 1  # all the same

    def test_synthetic_varies_by_vu_and_req(self):
        gen = LoadGenerator(LoadConfig(scenario=LoadScenario.SYNTHETIC))
        prompts = {gen._make_prompt(vu, req) for vu in range(3) for req in range(5)}
        assert len(prompts) > 1  # different prompts

    def test_multi_tool_varies(self):
        gen = LoadGenerator(LoadConfig(scenario=LoadScenario.MULTI_TOOL))
        prompts = {gen._make_prompt(vu, req) for vu in range(3) for req in range(5)}
        assert len(prompts) > 1

    def test_multi_tool_contains_calculate(self):
        gen = LoadGenerator(LoadConfig(scenario=LoadScenario.MULTI_TOOL))
        for vu in range(4):
            prompt = gen._make_prompt(vu, 0)
            assert "Calculate" in prompt


# ---------------------------------------------------------------------------
# Zero-result edge case
# ---------------------------------------------------------------------------

class TestZeroResults:
    def test_aggregate_zero_results(self):
        cfg = LoadConfig(concurrency=0, duration=0.001)
        gen = LoadGenerator(cfg)
        # Force _aggregate with empty list
        report = gen._aggregate([], actual_duration=0.001)
        assert report.total_requests == 0
        assert report.error_rate == 0.0
        assert report.throughput == 0.0
        assert report.p50_latency == 0.0
        assert report.p95_latency == 0.0
        assert report.p99_latency == 0.0
        assert report.requested_duration == 0.001
        assert report.error_breakdown == {}


# ---------------------------------------------------------------------------
# Custom target callable
# ---------------------------------------------------------------------------

class TestCustomTarget:
    def test_custom_sync_style_async_target(self):
        call_log: list[str] = []

        async def my_target(prompt: str) -> str:
            call_log.append(prompt)
            return f"ok: {prompt}"

        cfg = LoadConfig(concurrency=2, duration=1.0, scenario=LoadScenario.FIXED)
        gen = LoadGenerator(cfg, target=my_target)
        report = gen.run()
        assert report.total_requests > 0
        assert report.error_rate == 0.0
        assert len(call_log) == report.total_requests

    def test_failing_target_increments_error_count(self):
        async def bad_target(prompt: str) -> str:
            raise RuntimeError("always fails")

        cfg = LoadConfig(concurrency=2, duration=1.0)
        gen = LoadGenerator(cfg, target=bad_target)
        report = gen.run()
        assert report.failed_requests == report.total_requests
        assert report.error_rate == 1.0

    def test_timeout_target_increments_error_count(self):
        async def slow_target(prompt: str) -> str:
            await asyncio.sleep(999)
            return "never"

        cfg = LoadConfig(concurrency=2, duration=1.5, request_timeout=0.1)
        gen = LoadGenerator(cfg, target=slow_target)
        report = gen.run()
        assert report.failed_requests > 0

    def test_timeout_target_error_breakdown_labeled_timeout(self):
        async def slow_target(prompt: str) -> str:
            await asyncio.sleep(999)
            return "never"

        cfg = LoadConfig(concurrency=2, duration=1.0, request_timeout=0.1)
        gen = LoadGenerator(cfg, target=slow_target)
        report = gen.run()
        assert report.error_breakdown == {"timeout": report.failed_requests}

    def test_on_finish_awaited_inside_the_same_event_loop(self):
        # A target that opens loop-bound resources (e.g. an httpx.AsyncClient
        # connection pool) needs its cleanup awaited on the same loop the run
        # used, not a second asyncio.run() after this one returns.
        seen_loop: list[object] = []

        async def echo_target(prompt: str) -> str:
            return prompt

        async def on_finish() -> None:
            seen_loop.append(asyncio.get_running_loop())

        cfg = LoadConfig(concurrency=1, duration=0.2)
        gen = LoadGenerator(cfg, target=echo_target)
        report = gen.run(on_finish=on_finish)
        assert report.total_requests > 0
        assert len(seen_loop) == 1

    def test_run_without_on_finish_still_works(self):
        async def echo_target(prompt: str) -> str:
            return prompt

        cfg = LoadConfig(concurrency=1, duration=0.2)
        gen = LoadGenerator(cfg, target=echo_target)
        report = gen.run()
        assert report.total_requests > 0

    def test_failing_target_error_breakdown_populated(self):
        from effgen.models.errors import ModelAuthError

        async def bad_target(prompt: str) -> str:
            raise ModelAuthError("groq")

        cfg = LoadConfig(concurrency=2, duration=1.0)
        gen = LoadGenerator(cfg, target=bad_target)
        report = gen.run()
        assert report.error_breakdown == {"auth": report.failed_requests}

    def test_drain_note_reflects_timeout_overshoot(self):
        """When in-flight requests must wait out request_timeout after the
        window closes, the wall duration exceeds the requested duration and
        drain_s reports the gap."""
        async def slow_target(prompt: str) -> str:
            await asyncio.sleep(0.6)
            return "ok"

        cfg = LoadConfig(concurrency=1, duration=0.1, request_timeout=5.0)
        gen = LoadGenerator(cfg, target=slow_target)
        report = gen.run()
        d = report.to_dict()
        assert d["requested_duration_s"] == pytest.approx(0.1, abs=1e-6)
        assert d["duration_s"] > d["requested_duration_s"]
        assert d["drain_s"] > 0.0


# ---------------------------------------------------------------------------
# Output file writing
# ---------------------------------------------------------------------------

class TestOutputFileWriting:
    def test_report_written_to_file(self, tmp_path):
        out = tmp_path / "report.json"
        cfg = LoadConfig(
            concurrency=2,
            duration=1.0,
            output_path=out,
        )
        gen = LoadGenerator(cfg)
        gen.run()
        assert out.exists()
        d = json.loads(out.read_text())
        assert "latency" in d
        assert "p50" in d["latency"]
        assert "p95" in d["latency"]
        assert "p99" in d["latency"]

    def test_report_json_has_required_keys(self, tmp_path):
        out = tmp_path / "report.json"
        cfg = LoadConfig(concurrency=2, duration=1.0, output_path=out)
        gen = LoadGenerator(cfg)
        gen.run()
        d = json.loads(out.read_text())
        for key in ("scenario", "concurrency", "duration_s", "total_requests",
                    "error_rate", "throughput_rps", "latency"):
            assert key in d, f"Missing key: {key}"

    def test_no_output_path_no_file(self, tmp_path):
        cfg = LoadConfig(concurrency=2, duration=1.0, output_path=None)
        gen = LoadGenerator(cfg)
        gen.run()
        # No file created in tmp_path (nothing to assert; just must not crash)

    def test_nested_output_path_created(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "report.json"
        cfg = LoadConfig(concurrency=2, duration=1.0, output_path=out)
        gen = LoadGenerator(cfg)
        gen.run()
        assert out.exists()


# ---------------------------------------------------------------------------
# Mock target sanity check
# ---------------------------------------------------------------------------

class TestMockTarget:
    def test_mock_target_returns_string(self):
        result = asyncio.run(_mock_target("hello"))
        assert isinstance(result, str)
        assert "hello" in result or "Mock" in result

    def test_mock_target_is_fast(self):
        t0 = time.monotonic()
        asyncio.run(_mock_target("prompt"))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1  # must complete in under 100 ms

    def test_mock_target_accepts_timeout_kwarg(self):
        result = asyncio.run(_mock_target("prompt", timeout=30.0))
        assert isinstance(result, str)
