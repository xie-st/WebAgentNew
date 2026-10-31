"""Per-lane timing for a pytest session.

A full run of the suite is one number; a lane that has quietly grown to a third of it
is not visible in that number. This plugin buckets every test's setup/call/teardown
time by :mod:`lane <tests._harness.lanes>` and writes a JSON report with each lane's
wall time, its outcome counts and its slowest tests.

It is opt-in and inert otherwise: set ``EFFGEN_LANE_TIMING`` to the path the report
should be written to. Setting ``EFFGEN_LANE_TIMING_TABLE=1`` additionally prints the
table in the terminal summary. With neither set, the plugin is not registered and the
session's output is unchanged.

The variables are read rather than a command-line option added, so the same setting
reaches a suite that is driven through a wrapper script or a CI matrix without every
layer having to forward a flag.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests._harness.lanes import lane_of_nodeid

REPORT_PATH_VAR = "EFFGEN_LANE_TIMING"
PRINT_TABLE_VAR = "EFFGEN_LANE_TIMING_TABLE"
PLUGIN_NAME = "effgen-lane-timing"

#: How many of a lane's slowest tests the report keeps.
SLOWEST_PER_LANE = 10
#: How many of the session's slowest tests the report keeps.
SLOWEST_OVERALL = 40


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class _LaneTotals:
    lane: str
    duration_s: float = 0.0
    setup_s: float = 0.0
    call_s: float = 0.0
    teardown_s: float = 0.0
    tests: int = 0
    passed: int = 0
    failed: int = 0
    error: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    durations: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        slowest = sorted(self.durations.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "lane": self.lane,
            "duration_s": round(self.duration_s, 3),
            "setup_s": round(self.setup_s, 3),
            "call_s": round(self.call_s, 3),
            "teardown_s": round(self.teardown_s, 3),
            "tests": self.tests,
            "passed": self.passed,
            "failed": self.failed,
            "error": self.error,
            "skipped": self.skipped,
            "xfailed": self.xfailed,
            "xpassed": self.xpassed,
            "slowest": [
                {"nodeid": nodeid, "duration_s": round(seconds, 3)}
                for nodeid, seconds in slowest[:SLOWEST_PER_LANE]
            ],
        }


class LaneTimingPlugin:
    """Accumulate per-lane durations and write them out at the end of the session."""

    def __init__(self, report_path: Path, print_table: bool = False) -> None:
        self.report_path = report_path
        self.print_table = print_table
        self._lanes: dict[str, _LaneTotals] = {}
        self._started = time.time()
        self._session_duration_s = 0.0

    # -- collection ------------------------------------------------------

    def _lane(self, nodeid: str) -> _LaneTotals:
        lane = lane_of_nodeid(nodeid)
        totals = self._lanes.get(lane)
        if totals is None:
            totals = _LaneTotals(lane=lane)
            self._lanes[lane] = totals
        return totals

    def pytest_runtest_logreport(self, report: Any) -> None:
        totals = self._lane(report.nodeid)
        duration = float(getattr(report, "duration", 0.0) or 0.0)
        totals.duration_s += duration
        if report.when == "setup":
            totals.setup_s += duration
        elif report.when == "call":
            totals.call_s += duration
        elif report.when == "teardown":
            totals.teardown_s += duration
        totals.durations[report.nodeid] = totals.durations.get(report.nodeid, 0.0) + duration

        if report.when == "setup":
            if report.failed:
                totals.tests += 1
                totals.error += 1
            elif report.skipped:
                totals.tests += 1
                totals.skipped += 1
        elif report.when == "call":
            totals.tests += 1
            if report.passed:
                if getattr(report, "wasxfail", None) is not None:
                    totals.xpassed += 1
                else:
                    totals.passed += 1
            elif report.failed:
                if getattr(report, "wasxfail", None) is not None:
                    totals.xfailed += 1
                else:
                    totals.failed += 1
            elif report.skipped:
                if getattr(report, "wasxfail", None) is not None:
                    totals.xfailed += 1
                else:
                    totals.skipped += 1
        elif report.when == "teardown" and report.failed:
            totals.error += 1

    # -- reporting -------------------------------------------------------

    def _document(self, config: Any) -> dict[str, Any]:
        accounted = sum(lane.duration_s for lane in self._lanes.values())
        session = self._session_duration_s or (time.time() - self._started)
        overall: dict[str, float] = {}
        for lane in self._lanes.values():
            overall.update(lane.durations)
        slowest = sorted(overall.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._started)),
            "session_duration_s": round(session, 3),
            "accounted_s": round(accounted, 3),
            "accounted_fraction": round(accounted / session, 4) if session else 0.0,
            "unaccounted_s": round(max(session - accounted, 0.0), 3),
            "order": {
                "randomly_seed": getattr(config.option, "randomly_seed", None),
                "reversed": os.environ.get("EFFGEN_TEST_REVERSE_ORDER") == "1",
            },
            "hermetic": _truthy(os.environ.get("EFFGEN_TEST_HERMETIC")),
            "lanes": [
                self._lanes[name].as_dict()
                for name in sorted(self._lanes, key=lambda n: self._lanes[n].duration_s, reverse=True)
            ],
            "slowest_tests": [
                {"nodeid": nodeid, "duration_s": round(seconds, 3)}
                for nodeid, seconds in slowest[:SLOWEST_OVERALL]
            ],
        }

    def pytest_sessionfinish(self, session: Any) -> None:
        start = getattr(session.config, "_effgen_session_start", None)
        if start is not None:
            self._session_duration_s = time.time() - start
        document = self._document(session.config)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        self._document_cache = document

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if not self.print_table:
            return
        document = getattr(self, "_document_cache", None)
        if document is None:
            document = self._document(terminalreporter.config)
        write = terminalreporter.write_line
        write("")
        write("per-lane timing")
        write(f"{'lane':<16}{'seconds':>10}{'share':>8}{'tests':>8}{'failed':>8}{'skipped':>9}")
        session = document["session_duration_s"] or 1.0
        for lane in document["lanes"]:
            share = 100.0 * lane["duration_s"] / session
            write(
                f"{lane['lane']:<16}{lane['duration_s']:>10.1f}{share:>7.1f}%"
                f"{lane['tests']:>8}{lane['failed'] + lane['error']:>8}{lane['skipped']:>9}"
            )
        write(
            f"{'total':<16}{document['accounted_s']:>10.1f}"
            f"{100.0 * document['accounted_fraction']:>7.1f}%"
        )
        write(f"per-lane timing written to {self.report_path}")


def maybe_register(config: Any) -> LaneTimingPlugin | None:
    """Register the plugin when ``EFFGEN_LANE_TIMING`` names a report path."""
    destination = (os.environ.get(REPORT_PATH_VAR) or "").strip()
    if not destination:
        return None
    if config.pluginmanager.has_plugin(PLUGIN_NAME):
        return None
    plugin = LaneTimingPlugin(
        report_path=Path(destination).expanduser(),
        print_table=_truthy(os.environ.get(PRINT_TABLE_VAR)),
    )
    config.pluginmanager.register(plugin, PLUGIN_NAME)
    return plugin
