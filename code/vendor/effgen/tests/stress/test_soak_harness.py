"""Self-checks for the soak measurement itself.

A soak that cannot detect a leak passes on a leaking tree, so the detector is
tested against planted growth here: a series that rises by a known amount per
iteration must be rejected, and a flat one accepted. These cases are fast and
carry no ``expensive`` marker, so they run in the ordinary lane and confirm the
long-run scenarios are measuring what they claim to.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .soak import (
    GrowthBudget,
    ResourceSample,
    ResourceSampler,
    SoakReport,
    _gpu_reserved_mb,
    assert_bounded,
    check_bounded,
    scale_iterations,
    soak_scale,
)
from .test_endurance_soak import BUDGETS, ITERATIONS, WARMUP_FRACTION


def _series(values: list[float], metric: str = "rss_mb") -> SoakReport:
    """Build a report whose *metric* follows *values*, everything else flat."""
    sampler = ResourceSampler("planted")
    for i, value in enumerate(values):
        fields = {
            "iteration": i,
            "elapsed_s": float(i),
            "rss_mb": 100.0,
            "fds": 10,
            "threads": 5,
            "children": 0,
            "gpu_reserved_mb": 0.0,
            "gc_objects": 1000,
        }
        fields[metric] = value
        sampler.samples.append(ResourceSample(**fields))  # type: ignore[arg-type]
    return sampler.report(warmup_fraction=0.0)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_flat_series_has_zero_slope() -> None:
    report = _series([200.0] * 40)
    trend = report.trends["rss_mb"]
    assert trend.slope_per_iteration == pytest.approx(0.0, abs=1e-9)
    assert trend.delta == pytest.approx(0.0)


def test_rising_series_reports_its_slope() -> None:
    """A series rising 0.5 per iteration is reported as 500 per 1k."""
    report = _series([100.0 + 0.5 * i for i in range(40)])
    trend = report.trends["rss_mb"]
    assert trend.slope_per_iteration == pytest.approx(0.5, rel=1e-6)
    assert trend.per_1k_iterations == pytest.approx(500.0, rel=1e-6)
    assert trend.delta == pytest.approx(19.5, rel=1e-6)


def test_warmup_window_is_discarded() -> None:
    """Growth confined to the warmup window does not count as a slope."""
    values = [100.0 + 10.0 * i for i in range(10)] + [190.0] * 30
    report = ResourceSampler("warmup")
    for i, value in enumerate(values):
        report.samples.append(
            ResourceSample(i, float(i), value, 10, 5, 0, 0.0, 1000)
        )
    fitted = report.report(warmup_fraction=0.25)
    assert fitted.warmup_samples == 10
    assert fitted.trends["rss_mb"].slope_per_iteration == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Planted violations — the detector must catch each one
# ---------------------------------------------------------------------------


def test_planted_memory_leak_is_caught() -> None:
    """0.5 MB per iteration is rejected by a 25 MB-per-1k budget."""
    report = _series([100.0 + 0.5 * i for i in range(40)])
    failures = check_bounded(report, GrowthBudget())
    assert any("rss_mb" in f and "per 1k" in f for f in failures), failures
    with pytest.raises(AssertionError, match="rss_mb"):
        assert_bounded(report, GrowthBudget())


def test_planted_step_change_is_caught_without_a_slope() -> None:
    """A single jump larger than the total budget fails even at a low slope.

    Half the readings sit at 100 and half at 200: spread over 10 000
    iterations the fitted slope stays inside the per-1k budget, so only the
    delta check rejects it.
    """
    values = [100.0] * 5_000 + [200.0] * 5_000
    report = _series(values)
    trend = report.trends["rss_mb"]
    assert trend.per_1k_iterations < GrowthBudget().rss_mb_per_1k
    failures = check_bounded(report, GrowthBudget())
    assert any("rose" in f for f in failures), failures


@pytest.mark.parametrize(
    "metric,values",
    [
        ("fds", [10 + i for i in range(40)]),
        ("threads", [5 + i for i in range(40)]),
        ("children", list(range(40))),
        ("gpu_reserved_mb", [0.0 + 40.0 * i for i in range(40)]),
        ("gc_objects", [1000 + 5000 * i for i in range(40)]),
    ],
)
def test_planted_growth_is_caught_on_every_metric(metric: str, values: list) -> None:
    report = _series(values, metric=metric)
    failures = check_bounded(report, GrowthBudget())
    assert any(metric in f for f in failures), (metric, failures)


def test_flat_series_passes_every_budget() -> None:
    for metric in ResourceSampler.METRICS:
        report = _series([7.0] * 40, metric=metric)
        assert check_bounded(report, GrowthBudget()) == [], metric


def test_shrinking_series_is_not_a_failure() -> None:
    """Releasing memory is not growth."""
    report = _series([500.0 - 5.0 * i for i in range(40)])
    assert check_bounded(report, GrowthBudget()) == []


def test_unreadable_metric_is_skipped_not_failed() -> None:
    """A metric the platform will not report (-1) is not read as a trend."""
    report = _series([-1.0] * 40, metric="gc_objects")
    assert check_bounded(report, GrowthBudget()) == []


def test_none_budget_is_not_asserted() -> None:
    """A bound stated as ``None`` is reported but never fails the scenario."""
    report = _series([100.0 + 5.0 * i for i in range(40)])
    unbounded = GrowthBudget(rss_mb_per_1k=None, rss_mb_total=None)
    assert check_bounded(report, unbounded) == []
    assert report.trends["rss_mb"].per_1k_iterations > 0


# ---------------------------------------------------------------------------
# Sampling and reporting
# ---------------------------------------------------------------------------


def test_sampler_reads_this_process() -> None:
    sampler = ResourceSampler("selftest")
    reading = sampler.sample(0)
    assert reading.rss_mb > 0
    assert reading.fds > 0
    assert reading.threads >= 1
    assert reading.children >= 0
    assert reading.gc_objects > 0


def test_the_device_reading_excludes_the_allocator_cache() -> None:
    """The device figure is what is still held, not what is cached for reuse.

    A caching allocator's reserve tracks the largest allocation the process has
    made, so a run whose input sizes vary shows a rising reserve while nothing
    is retained. The reading releases the cache first, which makes it equal to
    the reserve measured straight after a release.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    scratch = torch.empty(64 * 1024 * 1024 // 4, device="cuda")
    del scratch
    sampled = _gpu_reserved_mb()
    torch.cuda.empty_cache()
    released = sum(
        torch.cuda.memory_reserved(d) for d in range(torch.cuda.device_count())
    ) / 1024**2
    assert sampled == pytest.approx(released, abs=1.0), (
        f"sampled {sampled:.1f} MB against {released:.1f} MB after a release"
    )
    assert _gpu_reserved_mb(release_cache=False) >= released


def test_sampling_another_process_omits_the_object_count() -> None:
    """Another process's heap cannot be walked, so the count is a sentinel."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        sampler = ResourceSampler("other", pid=proc.pid)
        reading = sampler.sample(0)
        assert reading.rss_mb > 0
        assert reading.gc_objects == -1
        assert reading.gpu_reserved_mb == 0.0
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def test_a_process_that_dies_mid_run_reads_as_unreadable_not_as_flat() -> None:
    """Readings taken after the sampled process exits are all sentinels.

    An unreadable metric is skipped rather than failed, so a scenario sampling
    another process cannot rely on the budget check alone to notice that the
    process it was measuring went away — it has to assert the process is still
    alive and that the readings were real.
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sampler = ResourceSampler("dying", pid=proc.pid)
    sampler.sample(0)
    assert sampler.samples[0].rss_mb > 0
    proc.terminate()
    proc.wait(timeout=20)
    for i in range(1, 8):
        sampler.sample(i)
    report = sampler.report(warmup_fraction=0.25)
    assert report.trends["rss_mb"].last == -1.0
    assert check_bounded(report, GrowthBudget()) == []


def test_report_serialises_to_json() -> None:
    sampler = ResourceSampler("selftest")
    for i in range(4):
        sampler.sample(i)
    restored = json.loads(sampler.report().to_json())
    assert restored["scenario"] == "selftest"
    assert len(restored["samples"]) == 4
    assert set(restored["trends"]) == set(ResourceSampler.METRICS)


def test_report_renders_a_table_and_csv() -> None:
    report = _series([100.0] * 12)
    table = report.table()
    assert "scenario=planted" in table
    for metric in ResourceSampler.METRICS:
        assert metric in table
    csv = report.samples_csv()
    assert csv.splitlines()[0].startswith("iteration,elapsed_s,rss_mb")
    assert len(csv.splitlines()) == 13


# ---------------------------------------------------------------------------
# The scale knob and the scenario table
# ---------------------------------------------------------------------------


def test_scale_multiplies_iterations(monkeypatch) -> None:
    monkeypatch.delenv("EFFGEN_SOAK_SCALE", raising=False)
    assert soak_scale() == 1.0
    assert scale_iterations(100) == 100
    monkeypatch.setenv("EFFGEN_SOAK_SCALE", "20")
    assert scale_iterations(100) == 2000
    monkeypatch.setenv("EFFGEN_SOAK_SCALE", "0.1")
    assert scale_iterations(100, minimum=8) == 10
    assert scale_iterations(10, minimum=8) == 8


def test_scale_falls_back_on_an_unusable_value(monkeypatch) -> None:
    for value in ("", "abc", "-3", "0"):
        monkeypatch.setenv("EFFGEN_SOAK_SCALE", value)
        assert soak_scale() == 1.0


def test_budget_table_covers_every_scenario() -> None:
    assert set(ITERATIONS) == set(BUDGETS), (
        f"iterations and budgets differ: {set(ITERATIONS) ^ set(BUDGETS)}"
    )
    for name, budget in BUDGETS.items():
        assert ITERATIONS[name] > 0, name
        assert set(budget.limits()) == set(ResourceSampler.METRICS), name
    assert 0.0 <= WARMUP_FRACTION < 1.0, WARMUP_FRACTION


# ---------------------------------------------------------------------------
# Lane selection
# ---------------------------------------------------------------------------


#: The only module under ``tests/stress/`` the ordinary lane may collect: these
#: self-checks are fast by construction. Everything else in the directory runs
#: for minutes to hours and must be deselected.
FAST_STRESS_MODULE = "tests/stress/test_soak_harness.py"


def test_stress_lane_collects_nothing_but_the_harness_self_checks() -> None:
    """The CI lane selector deselects every long-running case in this directory.

    Covers the whole directory rather than one file, so a scenario added in a
    new module without the marker is caught too.
    """
    selector = "not gpu and not api and not live and not docker and not expensive"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/stress/",
         "-m", selector, "--collect-only", "-q", "--no-header",
         "-p", "no:randomly", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=root, timeout=300,
    )
    collected = [
        line for line in result.stdout.splitlines()
        if line.startswith("tests/stress/") and "::" in line
    ]
    unmarked = [line for line in collected if not line.startswith(FAST_STRESS_MODULE)]
    assert not unmarked, (
        "the lane selector collected long-running stress cases — every case "
        "outside " + FAST_STRESS_MODULE + " needs the `expensive` marker:\n"
        + "\n".join(unmarked)
    )
    assert "deselected" in result.stdout, result.stdout[-2000:]


def test_the_soak_cases_are_not_cut_off_by_the_suite_timeout() -> None:
    """A soak outlives the suite-wide per-test cap, so it opts out of it.

    Without the opt-out a scenario at a raised scale is failed the moment it
    passes the cap, and a run cut off part-way reads as a defect that is not
    there. The cap is what the rest of the suite runs under and stays.
    """
    import re

    from . import test_endurance_soak as endurance

    root = Path(__file__).resolve().parents[2]
    match = re.search(
        r"^timeout\s*=\s*(\d+)", (root / "pyproject.toml").read_text(), re.MULTILINE
    )
    assert match, "the suite no longer configures a per-test timeout"
    configured = int(match.group(1))
    overrides = [m for m in endurance.pytestmark if m.name == "timeout"]
    assert overrides and overrides[0].args == (0,), (
        f"the soak cases still run under the {configured}s per-test cap: "
        f"{[m.name for m in endurance.pytestmark]}"
    )


def test_every_scenario_in_the_table_is_driven_by_a_case() -> None:
    """Every budget row names a scenario some case actually runs.

    A row added to the table without a case that drives it reads as covered —
    the budget is there and the tables agree — while nothing measures it. The
    scenario names passed to ``run_soak`` and ``assert_scenario`` are read out
    of the module source, so the check holds without running the soaks.
    """
    import ast

    module = Path(__file__).with_name("test_endurance_soak.py")
    tree = ast.parse(module.read_text())
    driven: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in {"run_soak", "assert_scenario"}:
            continue
        for value in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                driven.add(value.value)
    missing = set(ITERATIONS) - driven
    assert not missing, (
        f"scenarios with a budget but no case that drives them: {sorted(missing)}"
    )


def test_every_soak_scenario_carries_the_expensive_marker() -> None:
    """A new scenario cannot ship without the marker that keeps it out of CI."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/stress/test_endurance_soak.py",
         "-m", "expensive and slow", "--collect-only", "-q", "--no-header",
         "-p", "no:randomly", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=root, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:]
    collected = [
        line for line in result.stdout.splitlines()
        if line.startswith("tests/stress/test_endurance_soak.py::")
    ]
    assert len(collected) >= len(ITERATIONS), (
        f"only {len(collected)} marked cases for {len(ITERATIONS)} scenarios:\n"
        + result.stdout[-2000:]
    )
