"""
effGen Load Generator — synthetic load harness for agent/model benchmarking.

Runs concurrent "virtual users" against a target callable (default: a local
mock that returns immediately), collects per-request latencies, and reports
aggregate statistics: throughput, p50 / p95 / p99 latency, error rate.

Usage (library)
---------------
    from effgen.tools.loadgen import LoadGenerator, LoadConfig, LoadScenario

    cfg = LoadConfig(concurrency=10, duration=30, scenario=LoadScenario.FIXED)
    lg  = LoadGenerator(cfg)
    report = lg.run()
    print(report.to_dict())

Usage (CLI)
-----------
    effgen loadtest --concurrency 10 --duration 30 --scenario fixed
    effgen loadtest --provider cerebras --model gpt-oss-120b --concurrency 5

Scenarios
---------
- ``fixed``       — every VU sends the same short prompt ("Hello, what is 2+2?")
- ``synthetic``   — prompts are varied algorithmically (cycling through a library)
- ``multi_tool``  — VUs invoke the local calculator tool on varied expressions

The target callable signature: ``(prompt: str) -> str``.  For --provider runs
the CLI wires up a real model adapter.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..observability import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

_FIXED_PROMPT = "Hello, what is 2+2?"

_SYNTHETIC_PROMPTS: list[str] = [
    "Summarise the key principles of machine learning in two sentences.",
    "What is the capital of France?",
    "Convert 100 USD to EUR assuming 1 USD = 0.93 EUR.",
    "List three famous Python libraries for data science.",
    "Explain what a neural network is in simple terms.",
    "What year did the first moon landing occur?",
    "Give me a one-line Python snippet to reverse a string.",
    "Name the top 5 largest countries by area.",
    "What is the boiling point of water at sea level in Celsius?",
    "Translate 'Good morning' into Spanish, French, and German.",
]

_MULTI_TOOL_EXPRESSIONS: list[str] = [
    "Calculate: 17 * 34",
    "Calculate: sqrt(144)",
    "Calculate: (2 ** 10) - 1",
    "Calculate: 3.14159 * 5 ** 2",
    "Calculate: 1000 / 7",
    "Calculate: log(1000)",
    "Calculate: factorial(10)",
    "Calculate: abs(-42) + 58",
]


class LoadScenario(StrEnum):
    """Prompt mix used by the load generator."""

    FIXED = "fixed"
    SYNTHETIC = "synthetic"
    MULTI_TOOL = "multi_tool"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoadConfig:
    """Configuration for a load-test run."""

    concurrency: int = 10
    """Number of concurrent virtual users."""

    duration: float = 30.0
    """Test wall-clock duration in seconds."""

    scenario: LoadScenario = LoadScenario.FIXED
    """Workload scenario."""

    ramp_up: float = 0.0
    """Linear ramp-up period in seconds (VUs added gradually).  0 = instant."""

    request_timeout: float = 60.0
    """Per-request timeout in seconds (must be explicit — never None)."""

    think_time: float = 0.0
    """Simulated think-time between requests per VU in seconds."""

    provider: str | None = None
    """Provider name for live runs (None = mock)."""

    model: str | None = None
    """Model id for live runs (None = mock)."""

    output_path: Path | None = None
    """Path to write the JSON report (None = no file written)."""


# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Outcome of a single request."""

    latency: float
    """Wall time in seconds from send to first byte / completion."""

    success: bool
    """True if request completed without error."""

    error: str | None = None
    """Error message (if not success)."""

    error_category: str | None = None
    """Failure class for a failed request (e.g. "timeout", "rate_limited",
    "auth", "not_found", "invalid_request", "transient", "unknown"). ``None``
    for a successful request. Matches
    :func:`effgen.models.errors.classify_provider_error`'s ``category``."""

    prompt: str = ""
    """Prompt that was sent."""

    response: str = ""
    """Truncated response (first 120 chars)."""


# ---------------------------------------------------------------------------
# Load report
# ---------------------------------------------------------------------------

@dataclass
class LoadReport:
    """Aggregate statistics from a completed load run."""

    scenario: str
    concurrency: int
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float           # 0.0–1.0
    throughput: float           # requests / second
    p50_latency: float          # seconds
    p95_latency: float          # seconds
    p99_latency: float          # seconds
    min_latency: float
    max_latency: float
    mean_latency: float
    stdev_latency: float
    provider: str | None
    model: str | None
    requested_duration: float = 0.0
    """The ``--duration`` asked for. ``duration`` is the actual wall time,
    which can exceed this while in-flight requests drain up to
    ``request_timeout`` after the requested window closes."""
    error_breakdown: dict[str, int] = field(default_factory=dict)
    """Failed-request count by :attr:`RequestResult.error_category`
    (e.g. ``{"timeout": 3, "rate_limited": 1}``), so a failing run is
    triageable from the report alone instead of only a single error rate."""
    raw_results: list[RequestResult] = field(default_factory=list, repr=False)

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        """Return the report as a JSON-serializable dict (raw samples optional)."""
        drain_s = round(self.duration - self.requested_duration, 3)
        d = {
            "scenario": self.scenario,
            "concurrency": self.concurrency,
            "duration_s": round(self.duration, 3),
            "requested_duration_s": round(self.requested_duration, 3),
            # Positive when the run took longer than --duration because
            # in-flight requests were drained after the window closed.
            "drain_s": drain_s if drain_s > 0 else 0.0,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": round(self.error_rate, 4),
            "error_breakdown": dict(sorted(self.error_breakdown.items())),
            "throughput_rps": round(self.throughput, 4),
            "latency": {
                "p50": round(self.p50_latency, 4),
                "p95": round(self.p95_latency, 4),
                "p99": round(self.p99_latency, 4),
                "min": round(self.min_latency, 4),
                "max": round(self.max_latency, 4),
                "mean": round(self.mean_latency, 4),
                "stdev": round(self.stdev_latency, 4),
            },
            "provider": self.provider,
            "model": self.model,
        }
        if include_raw:
            d["raw_results"] = [
                {
                    "latency": round(r.latency, 4),
                    "success": r.success,
                    "error": r.error,
                    "error_category": r.error_category,
                    "prompt": r.prompt[:80],
                }
                for r in self.raw_results
            ]
        return d


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _classify_loadtest_error(exc: Exception) -> str:
    """Return a short failure-class label for a load-test request exception.

    Reuses the same provider-error taxonomy the server and ``Agent`` use
    (``classify_provider_error``), so a failed live run reports *why* it
    failed (rate limited vs. auth vs. transient upstream error) instead of
    only a bare error-rate percentage.
    """
    from ..models.errors import classify_provider_error

    try:
        return classify_provider_error(exc).category
    except Exception:  # pragma: no cover - classification must never crash the run
        return "unknown"


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *sorted_data* (0–100)."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * pct / 100.0
    lo = int(k)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    frac = k - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ---------------------------------------------------------------------------
# Mock target (used when no real provider is configured)
# ---------------------------------------------------------------------------

async def _mock_target(prompt: str, timeout: float = 60.0) -> str:  # noqa: ARG001
    """
    Fast local mock that simulates a minimal response delay (1–5 ms jitter).

    This is the default target for ``--scenario fixed / synthetic / multi_tool``
    when no ``--provider`` is specified.  It never calls a network and always
    succeeds.
    """
    # Simulate tiny processing time (1–5 ms) without sleeping the event loop
    await asyncio.sleep(0.001 + (hash(prompt) % 5) * 0.001)
    return f"Mock response to: {prompt[:40]}"


# ---------------------------------------------------------------------------
# Core load generator
# ---------------------------------------------------------------------------

class LoadGenerator:
    """
    Concurrent load generator.

    Parameters
    ----------
    config:
        ``LoadConfig`` describing the workload.
    target:
        Async callable ``(prompt: str) -> str``.  Defaults to the built-in
        mock if not provided.
    """

    def __init__(
        self,
        config: LoadConfig,
        target: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self._target: Callable[[str], Any] = target or _mock_target
        self._results: list[RequestResult] = []

    # ------------------------------------------------------------------
    # Prompt factories
    # ------------------------------------------------------------------

    def _make_prompt(self, vu_id: int, request_idx: int) -> str:
        """Return a prompt appropriate for the active scenario."""
        scenario = self.config.scenario
        if scenario == LoadScenario.FIXED:
            return _FIXED_PROMPT
        elif scenario == LoadScenario.SYNTHETIC:
            idx = (vu_id * 17 + request_idx) % len(_SYNTHETIC_PROMPTS)
            return _SYNTHETIC_PROMPTS[idx]
        else:  # MULTI_TOOL
            idx = (vu_id * 7 + request_idx) % len(_MULTI_TOOL_EXPRESSIONS)
            return _MULTI_TOOL_EXPRESSIONS[idx]

    # ------------------------------------------------------------------
    # Virtual user coroutine
    # ------------------------------------------------------------------

    async def _virtual_user(
        self,
        vu_id: int,
        start_ts: float,
        end_ts: float,
        results_queue: asyncio.Queue,
    ) -> None:
        """Run a single virtual user until *end_ts*."""
        req_idx = 0
        while time.monotonic() < end_ts:
            prompt = self._make_prompt(vu_id, req_idx)
            t0 = time.monotonic()
            success = False
            error_msg = None
            error_category = None
            response = ""
            try:
                raw = await asyncio.wait_for(
                    self._target(prompt),  # type: ignore[arg-type]
                    timeout=self.config.request_timeout,
                )
                response = str(raw)[:120]
                success = True
            except TimeoutError:
                # asyncio.TimeoutError is an alias of the builtin from 3.11 on,
                # which is the declared floor, so one name covers both.
                error_msg = f"TimeoutError after {self.config.request_timeout}s"
                error_category = "timeout"
            except Exception as exc:
                error_msg = repr(exc)[:200]
                error_category = _classify_loadtest_error(exc)

            latency = time.monotonic() - t0
            await results_queue.put(RequestResult(
                latency=latency,
                success=success,
                error=error_msg,
                error_category=error_category,
                prompt=prompt,
                response=response,
            ))

            req_idx += 1
            if self.config.think_time > 0:
                await asyncio.sleep(self.config.think_time)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, *, on_finish: Callable[[], Any] | None = None) -> LoadReport:
        """Execute the load test synchronously (blocks until done).

        ``on_finish``, if given, is an async callable awaited inside the same
        event loop the run itself used, right before ``run()`` returns -- so a
        target that opened loop-bound resources (e.g. an ``httpx.AsyncClient``
        connection pool) can close them on the loop that created them, instead
        of a separate ``asyncio.run()`` call after this one returns, which
        would hand the resources to an event loop that never opened them.
        """
        return asyncio.run(self._arun(on_finish=on_finish))

    async def _arun(self, *, on_finish: Callable[[], Any] | None = None) -> LoadReport:
        cfg = self.config
        results_queue: asyncio.Queue[RequestResult] = asyncio.Queue()

        log.event(
            "loadgen.start",
            scenario=cfg.scenario.value,
            concurrency=cfg.concurrency,
            duration=cfg.duration,
            provider=cfg.provider or "mock",
            model=cfg.model or "mock",
        )

        start_ts = time.monotonic()
        end_ts = start_ts + cfg.duration

        # Ramp-up: stagger VU start times
        tasks: list[asyncio.Task] = []
        for vu_id in range(cfg.concurrency):
            if cfg.ramp_up > 0 and cfg.concurrency > 1:
                delay = (cfg.ramp_up / (cfg.concurrency - 1)) * vu_id
                await asyncio.sleep(delay)
            task = asyncio.create_task(
                self._virtual_user(vu_id, start_ts, end_ts, results_queue)
            )
            tasks.append(task)

        # Wait for all VUs to finish
        await asyncio.gather(*tasks, return_exceptions=True)

        # Drain queue
        results: list[RequestResult] = []
        while not results_queue.empty():
            results.append(results_queue.get_nowait())

        actual_duration = time.monotonic() - start_ts
        report = self._aggregate(results, actual_duration)

        log.event(
            "loadgen.done",
            total_requests=report.total_requests,
            error_rate=round(report.error_rate, 4),
            throughput_rps=round(report.throughput, 4),
            p95_latency=round(report.p95_latency, 4),
        )

        # Write output file if configured
        if cfg.output_path:
            cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.output_path.write_text(
                json.dumps(report.to_dict(include_raw=False), indent=2)
            )
            log.event("loadgen.report_written", path=str(cfg.output_path))

        if on_finish is not None:
            await on_finish()

        return report

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        results: list[RequestResult],
        actual_duration: float,
    ) -> LoadReport:
        cfg = self.config

        if not results:
            # Return a zero report — never raise
            return LoadReport(
                scenario=cfg.scenario.value,
                concurrency=cfg.concurrency,
                duration=actual_duration,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=0.0,
                throughput=0.0,
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                min_latency=0.0,
                max_latency=0.0,
                mean_latency=0.0,
                stdev_latency=0.0,
                provider=cfg.provider,
                model=cfg.model,
                requested_duration=cfg.duration,
                error_breakdown={},
                raw_results=[],
            )

        total = len(results)
        successes = sum(1 for r in results if r.success)
        failures = total - successes
        latencies = sorted(r.latency for r in results)

        error_breakdown: dict[str, int] = {}
        for r in results:
            if not r.success:
                category = r.error_category or "unknown"
                error_breakdown[category] = error_breakdown.get(category, 0) + 1

        throughput = total / actual_duration if actual_duration > 0 else 0.0
        mean_lat = statistics.mean(latencies)
        stdev_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        return LoadReport(
            scenario=cfg.scenario.value,
            concurrency=cfg.concurrency,
            duration=actual_duration,
            total_requests=total,
            successful_requests=successes,
            failed_requests=failures,
            error_rate=failures / total,
            throughput=throughput,
            p50_latency=_percentile(latencies, 50),
            p95_latency=_percentile(latencies, 95),
            p99_latency=_percentile(latencies, 99),
            min_latency=latencies[0],
            max_latency=latencies[-1],
            mean_latency=mean_lat,
            stdev_latency=stdev_lat,
            provider=cfg.provider,
            model=cfg.model,
            requested_duration=cfg.duration,
            error_breakdown=error_breakdown,
            raw_results=results,
        )
