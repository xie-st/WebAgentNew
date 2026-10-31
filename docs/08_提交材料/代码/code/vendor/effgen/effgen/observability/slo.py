"""
effGen Observability — SLO (Service Level Objective) tracker.

Provides:
- :class:`SLO` — definition of an objective (name, target %, window, query tag).
- :class:`SLOTracker` — in-memory rolling-window recorder + burn-rate calculator.

Burn-rate definition
--------------------
The error budget for an SLO of target ``T`` over window ``W`` is the fraction
``(100 - T) / 100`` of events that may fail.  If the *actual* bad-event fraction
over the current rolling window is ``B``, then the **burn rate** is:

    burn_rate = B / ((100 - T) / 100)

A burn rate of 1.0 means you're consuming exactly your error budget at the
intended pace.  > 1.0 means you're burning faster than budget (alert).

Usage
-----
    from effgen.observability.slo import SLO, SLOTracker

    tracker = SLOTracker()
    tracker.register(SLO(name="model_call_success", target_pct=99.0,
                         window_seconds=3600, query="outcome=ok"))

    # After each model call:
    tracker.record("model_call_success", ok=True)

    # At any time:
    rate = tracker.burn_rate("model_call_success")  # 1.0 = on budget
    status = tracker.status("model_call_success")   # dict with full details
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# SLO definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SLO:
    """
    Definition of a Service Level Objective.

    Attributes:
        name: Unique identifier for this SLO.
        target_pct: Target success percentage (e.g. 99.0 means 99 % success).
        window_seconds: Rolling time window in seconds (e.g. 3600 = 1 hour).
        query: Informal label / filter tag documenting which outcome counts as
               "good" (e.g. ``'outcome="ok"'``).  Not evaluated programmatically —
               :meth:`SLOTracker.record` takes a plain ``ok=True/False`` flag.
    """

    name: str
    target_pct: float
    window_seconds: float
    query: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.target_pct <= 100.0:
            raise ValueError(
                f"SLO target_pct must be in (0, 100], got {self.target_pct!r}"
            )
        if self.window_seconds <= 0:
            raise ValueError(
                f"SLO window_seconds must be > 0, got {self.window_seconds!r}"
            )

    @property
    def error_budget_fraction(self) -> float:
        """Fraction of events that may fail: ``(100 - target_pct) / 100``."""
        return (100.0 - self.target_pct) / 100.0


# ---------------------------------------------------------------------------
# Rolling window event
# ---------------------------------------------------------------------------


@dataclass
class _Event:
    """Single event in the rolling window."""

    ts: float  # wall-clock seconds (time.monotonic())
    ok: bool


# ---------------------------------------------------------------------------
# SLO Tracker
# ---------------------------------------------------------------------------


class SLOTracker:
    """
    In-memory rolling-window SLO tracker.

    Records success/failure events per named SLO and computes:
    - good ratio (successes / total)
    - bad ratio  (failures / total)
    - burn rate  (bad_ratio / error_budget_fraction)

    All operations are thread-safe through a per-tracker lock.

    Example::

        tracker = SLOTracker()
        tracker.register(SLO("api_success", 99.5, 3600))

        tracker.record("api_success", ok=True)
        tracker.record("api_success", ok=False)

        print(tracker.burn_rate("api_success"))  # float
        print(tracker.status("api_success"))     # dict
    """

    def __init__(self) -> None:
        import threading

        self._slos: dict[str, SLO] = {}
        self._windows: dict[str, deque[_Event]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, slo: SLO) -> None:
        """Register an SLO definition.  Idempotent — safe to call twice."""
        with self._lock:
            self._slos[slo.name] = slo
            if slo.name not in self._windows:
                self._windows[slo.name] = deque()

    def get_slo(self, name: str) -> SLO:
        """Return the registered SLO for *name* (raises KeyError if missing)."""
        return self._slos[name]

    def list_slos(self) -> list[str]:
        """Return sorted list of registered SLO names."""
        return sorted(self._slos.keys())

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        name: str,
        *,
        ok: bool,
        ts: float | None = None,
    ) -> None:
        """
        Record one event for SLO *name*.

        Args:
            name: SLO name (must be registered first via :meth:`register`).
            ok: True if the event is "good" (counts towards success).
            ts: Optional override for event timestamp (``time.monotonic()`` by
                default).  Useful in tests.
        """
        if name not in self._slos:
            raise KeyError(f"SLO {name!r} not registered")
        ts = ts if ts is not None else time.monotonic()
        with self._lock:
            self._windows[name].append(_Event(ts=ts, ok=ok))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def _purge_old(self, name: str, now: float) -> None:
        """Remove events outside the rolling window.  Caller holds lock."""
        slo = self._slos[name]
        cutoff = now - slo.window_seconds
        window = self._windows[name]
        while window and window[0].ts < cutoff:
            window.popleft()

    def _counts(self, name: str, now: float | None = None) -> tuple[int, int]:
        """Return (good, bad) counts within the current window."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            self._purge_old(name, now)
            events = list(self._windows[name])
        good = sum(1 for e in events if e.ok)
        bad = len(events) - good
        return good, bad

    def good_count(self, name: str) -> int:
        """Number of good events within the rolling window."""
        g, _ = self._counts(name)
        return g

    def bad_count(self, name: str) -> int:
        """Number of bad events within the rolling window."""
        _, b = self._counts(name)
        return b

    def total_count(self, name: str) -> int:
        """Total events within the rolling window."""
        g, b = self._counts(name)
        return g + b

    def good_ratio(self, name: str) -> float:
        """
        Fraction of events that are good (0.0 – 1.0).

        Returns 1.0 (full SLO met) when no events have been recorded yet.
        """
        g, b = self._counts(name)
        total = g + b
        return g / total if total else 1.0

    def bad_ratio(self, name: str) -> float:
        """
        Fraction of events that are bad (0.0 – 1.0).

        Returns 0.0 (no errors) when no events have been recorded yet.
        """
        g, b = self._counts(name)
        total = g + b
        return b / total if total else 0.0

    def burn_rate(self, name: str) -> float:
        """
        Error-budget burn rate over the current rolling window.

        - **1.0** — consuming budget exactly at pace (bad_ratio == budget_fraction).
        - **> 1.0** — burning faster than budget → alert condition.
        - **0.0** — zero errors in window.
        - Returns **0.0** when ``error_budget_fraction == 0`` (100 % target) and
          there are no errors.  Returns ``float("inf")`` when budget is 0 but there
          are errors (impossible to satisfy the SLO).

        Returns:
            float: burn rate (non-negative).
        """
        slo = self._slos[name]
        bad = self.bad_ratio(name)
        budget = slo.error_budget_fraction

        if budget == 0.0:
            return 0.0 if bad == 0.0 else float("inf")

        return bad / budget

    def status(self, name: str) -> dict[str, Any]:
        """
        Return a JSON-serialisable status dictionary for *name*.

        Keys:
            - ``name``
            - ``target_pct``
            - ``window_seconds``
            - ``query``
            - ``total_events``
            - ``good_events``
            - ``bad_events``
            - ``good_ratio``
            - ``bad_ratio``
            - ``burn_rate``
            - ``within_budget`` (bool)
        """
        slo = self._slos[name]
        g, b = self._counts(name)
        total = g + b
        gr = g / total if total else 1.0
        br = b / total if total else 0.0
        budget = slo.error_budget_fraction
        if budget == 0.0:
            burn = 0.0 if br == 0.0 else float("inf")
        else:
            burn = br / budget
        return {
            "name": name,
            "target_pct": slo.target_pct,
            "window_seconds": slo.window_seconds,
            "query": slo.query,
            "total_events": total,
            "good_events": g,
            "bad_events": b,
            "good_ratio": round(gr, 6),
            "bad_ratio": round(br, 6),
            "burn_rate": round(burn, 6),
            "within_budget": burn <= 1.0,
        }

    def all_statuses(self) -> list[dict[str, Any]]:
        """Return :meth:`status` for every registered SLO, sorted by name."""
        return [self.status(n) for n in self.list_slos()]


# ---------------------------------------------------------------------------
# Global singleton tracker
# ---------------------------------------------------------------------------

_global_tracker: SLOTracker | None = None
_tracker_lock = None


def get_tracker() -> SLOTracker:
    """
    Return the global :class:`SLOTracker` singleton (thread-safe).

    Initialised on first call.  Register your SLOs once at application startup::

        from effgen.observability.slo import get_tracker, SLO
        tracker = get_tracker()
        tracker.register(SLO("model_call_success", 99.0, 3600, 'outcome="ok"'))
    """
    global _global_tracker, _tracker_lock
    import threading

    if _tracker_lock is None:
        _tracker_lock = threading.Lock()

    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = SLOTracker()
        return _global_tracker


def _reset_global_tracker() -> None:
    """Reset the global tracker (used in tests only)."""
    global _global_tracker
    _global_tracker = None


#: Explanation attached to an empty ``/slo`` response. A tracker with nothing
#: registered is the normal state, and an unqualified empty list reads as "this
#: server has no SLO data" next to a dashboard showing populated percentiles.
#: Shared by every ``/slo`` endpoint so the wording cannot drift between them.
EMPTY_SLO_DETAIL = (
    "No SLO objectives are registered in this process. This endpoint reports "
    "registered objectives only; measured latency percentiles and availability "
    "for served traffic are in the 'slo' block of /dashboard/data.json."
)


__all__ = [
    "SLO",
    "EMPTY_SLO_DETAIL",
    "SLOTracker",
    "get_tracker",
    "_reset_global_tracker",
]
