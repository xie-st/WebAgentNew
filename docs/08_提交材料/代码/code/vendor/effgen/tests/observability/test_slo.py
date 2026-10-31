"""
Tests for effgen.observability.slo

Verifies:
- SLO validation (target_pct range, window_seconds > 0)
- SLOTracker.record() / good_count / bad_count / total_count
- SLOTracker.good_ratio / bad_ratio
- SLOTracker.burn_rate() math spot-checked against known inputs
- Rolling window eviction (old events removed after window expires)
- status() and all_statuses() structure
- Edge cases: 100% target, zero events, all-bad events
- Global tracker singleton
- Thread-safety
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from effgen.observability.slo import (
    SLO,
    SLOTracker,
    _reset_global_tracker,
    get_tracker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_global():
    """Reset global tracker before/after each test."""
    _reset_global_tracker()
    yield
    _reset_global_tracker()


@pytest.fixture
def tracker():
    t = SLOTracker()
    t.register(SLO("model_call_success", target_pct=99.0, window_seconds=3600,
                    query='outcome="ok"'))
    return t


# ---------------------------------------------------------------------------
# SLO validation
# ---------------------------------------------------------------------------

class TestSLOValidation:
    def test_valid_slo(self):
        slo = SLO("test", 99.0, 3600)
        assert slo.name == "test"
        assert slo.target_pct == 99.0
        assert slo.window_seconds == 3600

    def test_error_budget_fraction(self):
        slo = SLO("x", 99.0, 3600)
        assert abs(slo.error_budget_fraction - 0.01) < 1e-9

        slo2 = SLO("y", 99.9, 3600)
        assert abs(slo2.error_budget_fraction - 0.001) < 1e-9

        slo3 = SLO("z", 95.0, 3600)
        assert abs(slo3.error_budget_fraction - 0.05) < 1e-9

    def test_target_pct_too_low(self):
        with pytest.raises(ValueError, match="target_pct"):
            SLO("bad", 0.0, 3600)

    def test_target_pct_negative(self):
        with pytest.raises(ValueError, match="target_pct"):
            SLO("bad", -1.0, 3600)

    def test_target_pct_max_ok(self):
        slo = SLO("perfect", 100.0, 3600)
        assert slo.target_pct == 100.0
        assert slo.error_budget_fraction == 0.0

    def test_window_zero(self):
        with pytest.raises(ValueError, match="window_seconds"):
            SLO("bad", 99.0, 0)

    def test_window_negative(self):
        with pytest.raises(ValueError, match="window_seconds"):
            SLO("bad", 99.0, -1)

    def test_slo_immutable(self):
        slo = SLO("test", 99.0, 3600)
        with pytest.raises((AttributeError, TypeError)):
            slo.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Recording and counting
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_ok(self, tracker):
        tracker.record("model_call_success", ok=True)
        assert tracker.good_count("model_call_success") == 1
        assert tracker.bad_count("model_call_success") == 0
        assert tracker.total_count("model_call_success") == 1

    def test_record_bad(self, tracker):
        tracker.record("model_call_success", ok=False)
        assert tracker.good_count("model_call_success") == 0
        assert tracker.bad_count("model_call_success") == 1

    def test_record_mixed(self, tracker):
        for _ in range(99):
            tracker.record("model_call_success", ok=True)
        tracker.record("model_call_success", ok=False)
        assert tracker.good_count("model_call_success") == 99
        assert tracker.bad_count("model_call_success") == 1
        assert tracker.total_count("model_call_success") == 100

    def test_record_unknown_slo_raises(self):
        t = SLOTracker()
        with pytest.raises(KeyError):
            t.record("nonexistent", ok=True)

    def test_register_idempotent(self, tracker):
        slo = SLO("model_call_success", 99.0, 3600)
        tracker.register(slo)  # second call, same name
        tracker.record("model_call_success", ok=True)
        assert tracker.total_count("model_call_success") == 1


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------

class TestRatios:
    def test_good_ratio_all_good(self, tracker):
        for _ in range(10):
            tracker.record("model_call_success", ok=True)
        assert tracker.good_ratio("model_call_success") == 1.0
        assert tracker.bad_ratio("model_call_success") == 0.0

    def test_good_ratio_all_bad(self, tracker):
        for _ in range(5):
            tracker.record("model_call_success", ok=False)
        assert tracker.good_ratio("model_call_success") == 0.0
        assert tracker.bad_ratio("model_call_success") == 1.0

    def test_ratio_99_pct(self, tracker):
        for _ in range(99):
            tracker.record("model_call_success", ok=True)
        tracker.record("model_call_success", ok=False)
        gr = tracker.good_ratio("model_call_success")
        br = tracker.bad_ratio("model_call_success")
        assert abs(gr - 0.99) < 1e-9
        assert abs(br - 0.01) < 1e-9

    def test_ratio_empty(self, tracker):
        """No events → good_ratio=1.0, bad_ratio=0.0 (assume all good)."""
        assert tracker.good_ratio("model_call_success") == 1.0
        assert tracker.bad_ratio("model_call_success") == 0.0


# ---------------------------------------------------------------------------
# Burn-rate math — spot-checked
# ---------------------------------------------------------------------------

class TestBurnRate:
    """
    Burn-rate formula: burn_rate = bad_ratio / error_budget_fraction

    For SLO target=99%, error_budget=0.01.
      - 0 bad / 100 total  → bad_ratio=0.0 → burn_rate=0.0
      - 1 bad / 100 total  → bad_ratio=0.01 → burn_rate=1.0 (exactly on budget)
      - 2 bad / 100 total  → bad_ratio=0.02 → burn_rate=2.0 (burning 2x)
      - 14 bad / 100 total → bad_ratio=0.14 → burn_rate=14.0 (fast burn)
    """

    def test_burn_rate_zero_errors(self, tracker):
        for _ in range(100):
            tracker.record("model_call_success", ok=True)
        assert tracker.burn_rate("model_call_success") == pytest.approx(0.0, abs=1e-9)

    def test_burn_rate_exactly_on_budget(self, tracker):
        """1 bad out of 100 for 99% SLO → burn_rate=1.0."""
        for _ in range(99):
            tracker.record("model_call_success", ok=True)
        tracker.record("model_call_success", ok=False)
        rate = tracker.burn_rate("model_call_success")
        assert rate == pytest.approx(1.0, rel=1e-6), f"Expected 1.0 got {rate}"

    def test_burn_rate_double(self, tracker):
        """2 bad out of 100 for 99% SLO → burn_rate=2.0."""
        for _ in range(98):
            tracker.record("model_call_success", ok=True)
        tracker.record("model_call_success", ok=False)
        tracker.record("model_call_success", ok=False)
        rate = tracker.burn_rate("model_call_success")
        assert rate == pytest.approx(2.0, rel=1e-6), f"Expected 2.0 got {rate}"

    def test_burn_rate_fast_burn(self, tracker):
        """14 bad out of 100 for 99% SLO → burn_rate=14.0."""
        for _ in range(86):
            tracker.record("model_call_success", ok=True)
        for _ in range(14):
            tracker.record("model_call_success", ok=False)
        rate = tracker.burn_rate("model_call_success")
        assert rate == pytest.approx(14.0, rel=1e-6), f"Expected 14.0 got {rate}"

    def test_burn_rate_99_9_slo(self):
        """For 99.9% SLO (budget=0.001): 1 bad / 1000 total → burn_rate=1.0."""
        t = SLOTracker()
        t.register(SLO("api_99_9", 99.9, 3600))
        for _ in range(999):
            t.record("api_99_9", ok=True)
        t.record("api_99_9", ok=False)
        rate = t.burn_rate("api_99_9")
        assert rate == pytest.approx(1.0, rel=1e-6)

    def test_burn_rate_95_slo(self):
        """For 95% SLO (budget=0.05): 10 bad / 100 total → burn_rate=2.0."""
        t = SLOTracker()
        t.register(SLO("loose", 95.0, 3600))
        for _ in range(90):
            t.record("loose", ok=True)
        for _ in range(10):
            t.record("loose", ok=False)
        rate = t.burn_rate("loose")
        assert rate == pytest.approx(2.0, rel=1e-6)

    def test_burn_rate_empty(self, tracker):
        """No events → burn_rate=0.0."""
        assert tracker.burn_rate("model_call_success") == pytest.approx(0.0)

    def test_burn_rate_100_pct_slo_no_errors(self):
        """100% SLO, no errors → burn_rate=0.0 (zero budget, zero errors)."""
        t = SLOTracker()
        t.register(SLO("perfect", 100.0, 3600))
        for _ in range(10):
            t.record("perfect", ok=True)
        assert t.burn_rate("perfect") == pytest.approx(0.0)

    def test_burn_rate_100_pct_slo_with_errors(self):
        """100% SLO with any error → burn_rate=inf (budget depleted)."""
        t = SLOTracker()
        t.register(SLO("perfect", 100.0, 3600))
        t.record("perfect", ok=False)
        rate = t.burn_rate("perfect")
        assert math.isinf(rate)


# ---------------------------------------------------------------------------
# Rolling window eviction
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_old_events_purged(self):
        """Events older than window_seconds should not count."""
        t = SLOTracker()
        t.register(SLO("fresh", 99.0, window_seconds=2.0))

        now = time.monotonic()
        # Record an "old" bad event (3 seconds ago → outside 2 s window)
        t.record("fresh", ok=False, ts=now - 3.0)
        # Record a recent good event
        t.record("fresh", ok=True, ts=now)

        # Only the recent good event should be in the window
        assert t.total_count("fresh") == 1
        assert t.good_count("fresh") == 1
        assert t.bad_count("fresh") == 0

    def test_recent_events_kept(self):
        t = SLOTracker()
        t.register(SLO("fresh2", 99.0, window_seconds=60.0))

        now = time.monotonic()
        t.record("fresh2", ok=True, ts=now - 30)
        t.record("fresh2", ok=False, ts=now - 10)
        t.record("fresh2", ok=True, ts=now)

        assert t.total_count("fresh2") == 3

    def test_all_events_expire(self):
        t = SLOTracker()
        t.register(SLO("expire_all", 99.0, window_seconds=1.0))

        now = time.monotonic()
        for _ in range(10):
            t.record("expire_all", ok=False, ts=now - 5.0)

        assert t.total_count("expire_all") == 0
        assert t.burn_rate("expire_all") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# status() and all_statuses()
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_structure(self, tracker):
        tracker.record("model_call_success", ok=True)
        tracker.record("model_call_success", ok=False)
        s = tracker.status("model_call_success")

        required_keys = {
            "name", "target_pct", "window_seconds", "query",
            "total_events", "good_events", "bad_events",
            "good_ratio", "bad_ratio", "burn_rate", "within_budget",
        }
        assert required_keys.issubset(s.keys())
        assert s["name"] == "model_call_success"
        assert s["target_pct"] == 99.0
        assert s["total_events"] == 2
        assert s["good_events"] == 1
        assert s["bad_events"] == 1

    def test_within_budget_true(self, tracker):
        for _ in range(99):
            tracker.record("model_call_success", ok=True)
        tracker.record("model_call_success", ok=False)
        s = tracker.status("model_call_success")
        assert s["within_budget"] is True  # burn_rate=1.0 ≤ 1.0

    def test_within_budget_false(self, tracker):
        for _ in range(90):
            tracker.record("model_call_success", ok=True)
        for _ in range(10):
            tracker.record("model_call_success", ok=False)
        s = tracker.status("model_call_success")
        assert s["within_budget"] is False  # burn_rate=10.0 > 1.0

    def test_all_statuses(self):
        t = SLOTracker()
        t.register(SLO("slo_a", 99.0, 3600))
        t.register(SLO("slo_b", 99.9, 1800))
        statuses = t.all_statuses()
        names = [s["name"] for s in statuses]
        assert "slo_a" in names
        assert "slo_b" in names

    def test_all_statuses_sorted(self):
        t = SLOTracker()
        t.register(SLO("z_slo", 99.0, 3600))
        t.register(SLO("a_slo", 99.0, 3600))
        names = [s["name"] for s in t.all_statuses()]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Global tracker singleton
# ---------------------------------------------------------------------------

class TestGlobalTracker:
    def test_singleton(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_resets_between_tests(self):
        """After _reset_global_tracker, a new instance is returned."""
        _reset_global_tracker()
        t1 = get_tracker()
        _reset_global_tracker()
        t2 = get_tracker()
        assert t1 is not t2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record(self, tracker):
        errors = []

        def worker(n):
            try:
                for _ in range(100):
                    tracker.record("model_call_success", ok=(n % 2 == 0))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f"Thread errors: {errors}"
        assert tracker.total_count("model_call_success") == 1000

    def test_concurrent_burn_rate_read(self, tracker):
        """burn_rate reads while writes are happening should not raise."""
        errors = []
        stop = threading.Event()

        def writer():
            while not stop.is_set():
                try:
                    tracker.record("model_call_success", ok=True)
                except Exception as e:
                    errors.append(e)

        def reader():
            while not stop.is_set():
                try:
                    tracker.burn_rate("model_call_success")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for th in threads:
            th.start()
        time.sleep(0.2)
        stop.set()
        for th in threads:
            th.join()
        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# list_slos
# ---------------------------------------------------------------------------

class TestListSLOs:
    def test_list_empty(self):
        t = SLOTracker()
        assert t.list_slos() == []

    def test_list_after_register(self):
        t = SLOTracker()
        t.register(SLO("beta", 99.0, 3600))
        t.register(SLO("alpha", 99.0, 3600))
        assert t.list_slos() == ["alpha", "beta"]
