"""Resource sampling and bounded-growth analysis for long-running soaks.

A soak answers one question: does a resource grow with the number of
iterations, or does it settle? A single before/after reading cannot tell those
apart — a process that allocates a cache on the first iteration and one that
leaks a little on every iteration both end higher than they started. So a soak
here samples across the whole run and fits a trend line to the steady-state
window, and the assertion is on the *slope* as well as the total delta.

Sampled per reading: resident memory, open file descriptors, thread count,
direct child processes and (when torch reports a CUDA device) the device memory
still held once the allocator's reusable cache is released.

Usage::

    sampler = ResourceSampler()
    for i in range(iterations):
        do_one_iteration()
        sampler.sample(i)          # cheap; call every iteration or every Nth
    report = sampler.report(warmup_fraction=0.25)
    assert_bounded(report, GrowthBudget(rss_mb_per_1k=25.0, rss_mb_total=64.0))

``report.table()`` renders the trend as text and ``report.to_dict()`` as JSON,
so the same scenario can be run as a short test or as an hours-long driver and
produce the same numbers in both.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil

__all__ = [
    "GrowthBudget",
    "ResourceSample",
    "ResourceSampler",
    "SoakReport",
    "TrendLine",
    "assert_bounded",
    "scale_iterations",
    "soak_scale",
]


# ---------------------------------------------------------------------------
# Scale knob
# ---------------------------------------------------------------------------


def soak_scale() -> float:
    """Iteration multiplier from ``EFFGEN_SOAK_SCALE`` (default ``1.0``).

    The short form of every scenario runs at scale 1 so a test lane stays
    minutes; the long form is the same code at a higher scale.
    """
    raw = os.environ.get("EFFGEN_SOAK_SCALE", "1")
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return value if value > 0 else 1.0


def scale_iterations(base: int, minimum: int = 8) -> int:
    """Scale *base* iterations by :func:`soak_scale`, never below *minimum*."""
    return max(minimum, int(round(base * soak_scale())))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _gpu_reserved_mb(*, release_cache: bool = True) -> float:
    """Device memory this process still holds after the allocator cache is freed.

    Reading ``memory_reserved`` on its own counts the blocks the caching
    allocator keeps for reuse. That figure rises and falls with the longest
    sequence the process has seen rather than with anything retained, so a run
    whose prompt length varies shows a rising reserve while the memory actually
    held is flat — a trend line drawn through it reports a leak that is not
    there. Freeing the cache first leaves the memory that is genuinely still
    held, which is the question a soak is asking.

    Returns ``0.0`` when torch is absent or reports no CUDA device, so a
    scenario that never touches a GPU carries the column without branching.
    """
    try:
        import torch
    except Exception:
        return 0.0
    try:
        if not torch.cuda.is_available():
            return 0.0
        if release_cache:
            torch.cuda.empty_cache()
        return sum(
            torch.cuda.memory_reserved(d) for d in range(torch.cuda.device_count())
        ) / 1024**2
    except Exception:
        return 0.0


@dataclass(frozen=True)
class ResourceSample:
    """One reading of the process's resource usage."""

    iteration: int
    elapsed_s: float
    rss_mb: float
    fds: int
    threads: int
    children: int
    gpu_reserved_mb: float
    gc_objects: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrendLine:
    """Least-squares fit of one metric against the iteration number."""

    metric: str
    first: float
    last: float
    minimum: float
    maximum: float
    slope_per_iteration: float
    per_1k_iterations: float
    delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fit(metric: str, xs: list[float], ys: list[float]) -> TrendLine:
    n = len(xs)
    if n == 0:
        return TrendLine(metric, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if n == 1:
        return TrendLine(metric, ys[0], ys[0], ys[0], ys[0], 0.0, 0.0, 0.0)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if denominator == 0 else sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator
    return TrendLine(
        metric=metric,
        first=ys[0],
        last=ys[-1],
        minimum=min(ys),
        maximum=max(ys),
        slope_per_iteration=slope,
        per_1k_iterations=slope * 1000.0,
        delta=ys[-1] - ys[0],
    )


@dataclass
class SoakReport:
    """Trend lines over the steady-state window of one soak."""

    scenario: str
    iterations: int
    duration_s: float
    samples: list[ResourceSample]
    warmup_samples: int
    trends: dict[str, TrendLine] = field(default_factory=dict)

    @property
    def steady(self) -> list[ResourceSample]:
        return self.samples[self.warmup_samples :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "iterations": self.iterations,
            "duration_s": round(self.duration_s, 2),
            "warmup_samples": self.warmup_samples,
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "samples": [s.to_dict() for s in self.samples],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def table(self) -> str:
        """Render the steady-state trend as a fixed-width table."""
        header = (
            f"scenario={self.scenario} iterations={self.iterations} "
            f"duration={self.duration_s:.1f}s samples={len(self.samples)} "
            f"(warmup {self.warmup_samples} discarded)"
        )
        rows = [
            (
                f"{'metric':<18}{'first':>12}{'last':>12}{'min':>12}"
                f"{'max':>12}{'delta':>12}{'per 1k iter':>14}"
            )
        ]
        rows.append("-" * len(rows[0]))
        for trend in self.trends.values():
            rows.append(
                f"{trend.metric:<18}{trend.first:>12.2f}{trend.last:>12.2f}"
                f"{trend.minimum:>12.2f}{trend.maximum:>12.2f}"
                f"{trend.delta:>12.2f}{trend.per_1k_iterations:>14.2f}"
            )
        return header + "\n" + "\n".join(rows)

    def samples_csv(self) -> str:
        """Every reading as CSV, for plotting the trend outside the run."""
        fields = [
            "iteration", "elapsed_s", "rss_mb", "fds", "threads",
            "children", "gpu_reserved_mb", "gc_objects",
        ]
        lines = [",".join(fields)]
        for s in self.samples:
            d = s.to_dict()
            lines.append(",".join(str(d[f]) for f in fields))
        return "\n".join(lines)


class ResourceSampler:
    """Samples the calling process's resource usage across a run."""

    #: Metrics carried on every report, in table order.
    METRICS = ("rss_mb", "fds", "threads", "children", "gpu_reserved_mb", "gc_objects")

    def __init__(
        self, scenario: str = "soak", *, collect: bool = True, pid: int | None = None
    ) -> None:
        self.scenario = scenario
        #: Sampling another process cannot walk its heap, so the object count
        #: and the collect-first step apply to the calling process only.
        self.is_self = pid is None or pid == os.getpid()
        self.collect = collect and self.is_self
        self._process = psutil.Process(pid)
        self._started = time.monotonic()
        self.samples: list[ResourceSample] = []

    def _children(self) -> int:
        try:
            return len(self._process.children())
        except psutil.Error:
            return -1

    def _fds(self) -> int:
        try:
            return self._process.num_fds()
        except (psutil.Error, AttributeError):
            return -1

    def sample(self, iteration: int) -> ResourceSample:
        """Take one reading and append it to the run.

        ``collect=True`` runs a full ``gc.collect()`` first so the reading
        reflects what is still reachable rather than what has not been
        collected yet — a leak has to survive that to be reported.
        """
        if self.collect:
            gc.collect()
        try:
            rss = self._process.memory_info().rss / 1024**2
            threads = self._process.num_threads()
        except psutil.Error:  # pragma: no cover - process always alive here
            rss, threads = -1.0, -1
        reading = ResourceSample(
            iteration=iteration,
            elapsed_s=time.monotonic() - self._started,
            rss_mb=round(rss, 3),
            fds=self._fds(),
            threads=threads,
            children=self._children(),
            gpu_reserved_mb=round(
                _gpu_reserved_mb(release_cache=self.collect) if self.is_self else 0.0, 3
            ),
            gc_objects=len(gc.get_objects()) if self.is_self else -1,
        )
        self.samples.append(reading)
        return reading

    def report(self, *, warmup_fraction: float = 0.25) -> SoakReport:
        """Fit a trend line per metric over the post-warmup window.

        The first *warmup_fraction* of samples is discarded: import work, lazy
        caches and the allocator's first growth all land there and would
        otherwise be read as a slope.
        """
        warmup = int(len(self.samples) * warmup_fraction)
        if self.samples and warmup >= len(self.samples):
            warmup = len(self.samples) - 1
        report = SoakReport(
            scenario=self.scenario,
            iterations=self.samples[-1].iteration + 1 if self.samples else 0,
            duration_s=time.monotonic() - self._started,
            samples=list(self.samples),
            warmup_samples=warmup,
        )
        steady = report.steady
        xs = [float(s.iteration) for s in steady]
        for metric in self.METRICS:
            ys = [float(getattr(s, metric)) for s in steady]
            report.trends[metric] = _fit(metric, xs, ys)
        return report


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrowthBudget:
    """What a scenario is allowed to grow by, as numbers decided up front.

    ``*_per_1k`` bounds the fitted slope extrapolated to a thousand iterations
    — the shape a leak has. ``*_total`` bounds the steady-state delta, which
    catches a step change the slope averages away. ``None`` states that a
    scenario does not assert that bound (a run of a dozen heavy iterations has
    no meaningful per-1k slope), and the metric is still reported.
    """

    rss_mb_per_1k: float | None = 25.0
    rss_mb_total: float | None = 64.0
    fds_per_1k: float | None = 4.0
    fds_total: float | None = 8.0
    threads_per_1k: float | None = 2.0
    threads_total: float | None = 4.0
    children_per_1k: float | None = 1.0
    children_total: float | None = 2.0
    gpu_mb_per_1k: float | None = 64.0
    gpu_mb_total: float | None = 64.0
    gc_objects_per_1k: float | None = 20_000.0
    gc_objects_total: float | None = 50_000.0

    def limits(self) -> dict[str, tuple[float | None, float | None]]:
        return {
            "rss_mb": (self.rss_mb_per_1k, self.rss_mb_total),
            "fds": (self.fds_per_1k, self.fds_total),
            "threads": (self.threads_per_1k, self.threads_total),
            "children": (self.children_per_1k, self.children_total),
            "gpu_reserved_mb": (self.gpu_mb_per_1k, self.gpu_mb_total),
            "gc_objects": (self.gc_objects_per_1k, self.gc_objects_total),
        }


def check_bounded(report: SoakReport, budget: GrowthBudget) -> list[str]:
    """Return one message per metric that exceeded its budget (empty if none)."""
    failures: list[str] = []
    for metric, (per_1k, total) in budget.limits().items():
        trend = report.trends.get(metric)
        if trend is None or trend.first < 0:  # unreadable metric on this platform
            continue
        if per_1k is not None and trend.per_1k_iterations > per_1k:
            failures.append(
                f"{report.scenario}: {metric} grows {trend.per_1k_iterations:.2f} "
                f"per 1k iterations, budget {per_1k:.2f} "
                f"({trend.first:.2f} -> {trend.last:.2f} over "
                f"{len(report.steady)} samples)"
            )
        if total is not None and trend.delta > total:
            failures.append(
                f"{report.scenario}: {metric} rose {trend.delta:.2f} across the "
                f"steady-state window, budget {total:.2f} "
                f"({trend.first:.2f} -> {trend.last:.2f})"
            )
    return failures


def assert_bounded(report: SoakReport, budget: GrowthBudget) -> None:
    """Fail with the trend table when any metric exceeded its budget."""
    failures = check_bounded(report, budget)
    if failures:
        raise AssertionError("\n".join([*failures, "", report.table()]))
