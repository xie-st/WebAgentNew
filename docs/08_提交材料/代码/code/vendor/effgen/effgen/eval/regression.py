"""
Regression testing for agent evaluations.

Compare evaluation results across versions and alert on regressions.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .evaluator import SuiteResults

logger = logging.getLogger(__name__)

# Default thresholds
ACCURACY_DROP_THRESHOLD = 0.05   # >5% accuracy drop
LATENCY_INCREASE_THRESHOLD = 0.20  # >20% latency increase

# Baselines saved by effGen versions before ``--baseline-dir`` existed live
# under the package checkout. New baselines are written under the caller's
# own working directory instead (see ``RegressionTracker.__init__``); this
# path is read-only, used as a fallback when the configured directory has no
# baseline for a suite.
_LEGACY_BASELINES_DIR = Path(__file__).resolve().parents[2] / "tests" / "benchmarks"


class RegressionAlert:
    """A single regression alert.

    ``blocking`` distinguishes a genuine quality regression (accuracy,
    tool-accuracy) — which should fail CI and open an issue — from an
    advisory *performance notice* (latency). Latency on a shared free-tier
    API endpoint is dominated by upstream queue/network jitter rather than
    effGen code, so a latency increase is reported but never blocks. See
    :pyattr:`ComparisonReport.has_regressions`.
    """

    def __init__(
        self,
        metric: str,
        baseline_value: float,
        current_value: float,
        threshold: float,
        suite: str,
        blocking: bool = True,
    ) -> None:
        self.metric = metric
        self.baseline_value = baseline_value
        self.current_value = current_value
        self.threshold = threshold
        self.suite = suite
        self.blocking = blocking

    @property
    def delta(self) -> float:
        """Relative change from the baseline value (0.0 when the baseline is 0)."""
        if self.baseline_value == 0:
            return 0.0
        return (self.current_value - self.baseline_value) / abs(self.baseline_value)

    @property
    def severity(self) -> str:
        """``warning``/``high``/``critical`` by how far the delta exceeds the threshold."""
        d = abs(self.delta)
        if d > self.threshold * 3:
            return "critical"
        if d > self.threshold * 2:
            return "high"
        return "warning"

    def __str__(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        tag = self.severity.upper() if self.blocking else "NOTICE"
        return (
            f"[{tag}] {self.suite}/{self.metric}: "
            f"{self.baseline_value:.4f} -> {self.current_value:.4f} "
            f"({sign}{self.delta * 100:.1f}%, threshold: {self.threshold * 100:.0f}%)"
        )


class ComparisonReport:
    """Report comparing current results against a baseline."""

    def __init__(
        self,
        suite: str,
        baseline_version: str,
        current_version: str,
        alerts: list[RegressionAlert],
        baseline_summary: dict[str, Any],
        current_summary: dict[str, Any],
    ) -> None:
        self.suite = suite
        self.baseline_version = baseline_version
        self.current_version = current_version
        self.alerts = alerts
        self.baseline_summary = baseline_summary
        self.current_summary = current_summary

    @property
    def blocking_alerts(self) -> list[RegressionAlert]:
        """Quality regressions that should fail CI / open an issue."""
        return [a for a in self.alerts if a.blocking]

    @property
    def notices(self) -> list[RegressionAlert]:
        """Advisory (non-blocking) alerts, e.g. latency drift."""
        return [a for a in self.alerts if not a.blocking]

    @property
    def has_regressions(self) -> bool:
        """True only when a *blocking* quality regression is present.

        A latency-only notice does not count as a regression: it is
        advisory and must not fail the nightly job (see issue history of
        chronic false-positive "regression" reports driven by free-tier
        API latency jitter).
        """
        return any(a.blocking for a in self.alerts)

    def to_markdown(self) -> str:
        """Render the report as Markdown: status, metric table, alerts."""
        lines = [
            f"# Regression Report: {self.suite}",
            "",
            f"**Baseline:** {self.baseline_version}  ",
            f"**Current:** {self.current_version}  ",
            f"**Status:** {'REGRESSION DETECTED' if self.has_regressions else 'PASS'}",
            "",
            "## Metrics",
            "",
            "| Metric | Baseline | Current | Change |",
            "|--------|----------|---------|--------|",
        ]
        metrics = ["accuracy", "avg_latency", "total_tokens", "avg_tool_accuracy"]
        for m in metrics:
            bv = self.baseline_summary.get(m, 0)
            cv = self.current_summary.get(m, 0)
            if bv != 0:
                delta = (cv - bv) / abs(bv) * 100
                sign = "+" if delta >= 0 else ""
                change = f"{sign}{delta:.1f}%"
            else:
                change = "N/A"
            lines.append(f"| {m} | {bv:.4f} | {cv:.4f} | {change} |")

        if self.blocking_alerts:
            lines.extend(["", "## Regressions", ""])
            for a in self.blocking_alerts:
                lines.append(f"- {a}")

        if self.notices:
            lines.extend([
                "",
                "## Performance notices (advisory — do not fail CI)",
                "",
            ])
            for a in self.notices:
                lines.append(f"- {a}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "suite": self.suite,
            "baseline_version": self.baseline_version,
            "current_version": self.current_version,
            "has_regressions": self.has_regressions,
            "alerts": [str(a) for a in self.alerts],
            "regressions": [str(a) for a in self.blocking_alerts],
            "notices": [str(a) for a in self.notices],
            "baseline": self.baseline_summary,
            "current": self.current_summary,
        }


class RegressionTracker:
    """Compare eval results across versions and detect regressions.

    Usage::

        tracker = RegressionTracker()
        tracker.save_baseline("math", results, version="0.1.3")

        # Later...
        report = tracker.compare("math", new_results, version="0.2.0")
        if report.has_regressions:
            for alert in report.alerts:
                print(alert)
    """

    def __init__(
        self,
        baselines_dir: str | Path | None = None,
        accuracy_threshold: float = ACCURACY_DROP_THRESHOLD,
        latency_threshold: float = LATENCY_INCREASE_THRESHOLD,
    ) -> None:
        """
        Args:
            baselines_dir: Where baseline files are read from and written to.
                Defaults to ``./.effgen/baselines`` under the current working
                directory, so baselines live in the caller's own repository
                (committable, diffable in CI) instead of the installed
                package tree. A baseline saved under the package tree by an
                older effGen version is still found by :meth:`load_baseline`
                when the configured directory has no file for that suite.
        """
        self.baselines_dir = Path(baselines_dir) if baselines_dir is not None else Path.cwd() / ".effgen" / "baselines"
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self.accuracy_threshold = accuracy_threshold
        self.latency_threshold = latency_threshold

    def _baseline_path(self, suite: str) -> Path:
        # Sanitize so a suite name that is a filesystem path (e.g. a custom
        # dataset's raw CLI argument) can never introduce a directory
        # separator into the baseline filename.
        safe = suite.replace("/", "_").replace("\\", "_")
        return self.baselines_dir / f"eval_baseline_{safe}.json"

    def save_baseline(
        self,
        suite: str,
        results: SuiteResults,
        version: str = "",
    ) -> Path:
        """Persist a baseline for the given suite.

        Args:
            suite: The suite the baseline belongs to.
            results: The results to record as the baseline.
            version: A label stored with the baseline, such as a release.

        Returns:
            The path the baseline was written to.
        """
        path = self._baseline_path(suite)
        data = {
            "version": version,
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": results.summary(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved baseline for suite %r at %s", suite, path)
        return path

    def load_baseline(self, suite: str) -> dict[str, Any] | None:
        """Load the stored baseline for *suite*, or ``None`` if absent.

        Falls back to the legacy package-tree location when the configured
        ``baselines_dir`` has no file for *suite*, so a baseline saved before
        ``--baseline-dir`` existed is still found.
        """
        path = self._baseline_path(suite)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        safe = suite.replace("/", "_").replace("\\", "_")
        legacy_path = _LEGACY_BASELINES_DIR / f"eval_baseline_{safe}.json"
        if legacy_path != path and legacy_path.exists():
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        return None

    def compare(
        self,
        suite: str,
        current: SuiteResults,
        version: str = "",
    ) -> ComparisonReport:
        """Compare *current* results against the stored baseline.

        Args:
            suite: The suite whose baseline is compared against.
            current: The results from this run.
            version: A label stored with the comparison, such as a release.

        Returns:
            The comparison, naming every metric that moved.
        """
        baseline_data = self.load_baseline(suite)
        if baseline_data is None:
            logger.warning("No baseline for suite %r — saving current as baseline", suite)
            self.save_baseline(suite, current, version)
            return ComparisonReport(
                suite=suite,
                baseline_version=version,
                current_version=version,
                alerts=[],
                baseline_summary=current.summary(),
                current_summary=current.summary(),
            )

        baseline_summary = baseline_data["summary"]
        current_summary = current.summary()
        alerts = self._check_thresholds(suite, baseline_summary, current_summary)

        return ComparisonReport(
            suite=suite,
            baseline_version=baseline_data.get("version", "unknown"),
            current_version=version,
            alerts=alerts,
            baseline_summary=baseline_summary,
            current_summary=current_summary,
        )

    def _check_thresholds(
        self,
        suite: str,
        baseline: dict[str, Any],
        current: dict[str, Any],
    ) -> list[RegressionAlert]:
        alerts: list[RegressionAlert] = []

        # Accuracy drop (lower is worse)
        b_acc = baseline.get("accuracy", 0)
        c_acc = current.get("accuracy", 0)
        if b_acc > 0 and (b_acc - c_acc) / b_acc > self.accuracy_threshold:
            alerts.append(RegressionAlert(
                metric="accuracy",
                baseline_value=b_acc,
                current_value=c_acc,
                threshold=self.accuracy_threshold,
                suite=suite,
            ))

        # Latency increase (higher is worse). Advisory only: latency on a
        # shared free-tier API endpoint is dominated by upstream queue/network
        # jitter, so it is reported as a non-blocking notice rather than a
        # regression that fails CI and opens an issue.
        b_lat = baseline.get("avg_latency", 0)
        c_lat = current.get("avg_latency", 0)
        if b_lat > 0 and (c_lat - b_lat) / b_lat > self.latency_threshold:
            alerts.append(RegressionAlert(
                metric="avg_latency",
                baseline_value=b_lat,
                current_value=c_lat,
                threshold=self.latency_threshold,
                suite=suite,
                blocking=False,
            ))

        # Tool accuracy drop
        b_ta = baseline.get("avg_tool_accuracy", 0)
        c_ta = current.get("avg_tool_accuracy", 0)
        if b_ta > 0 and (b_ta - c_ta) / b_ta > self.accuracy_threshold:
            alerts.append(RegressionAlert(
                metric="avg_tool_accuracy",
                baseline_value=b_ta,
                current_value=c_ta,
                threshold=self.accuracy_threshold,
                suite=suite,
            ))

        return alerts
