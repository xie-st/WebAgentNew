"""
Tests for CircuitBreaker state machine.

Coverage:
- CLOSED → OPEN on failure_threshold consecutive failures
- OPEN → HALF_OPEN after recovery_timeout
- HALF_OPEN → CLOSED on half_open_probes successes
- HALF_OPEN → OPEN on failure
- is_call_permitted() returns False when OPEN
- CircuitBreakerOpen raised on blocked calls
- Stats snapshot
- Manual reset() / trip()
- CircuitBreakerRegistry
- ProviderRegistry.get_circuit_breaker() integration
- Thread-safety under concurrent load
"""

from __future__ import annotations

import threading
import time

import pytest

from effgen.reliability.circuit import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
)

pytestmark = pytest.mark.reliability


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestCircuitBreakerStateTransitions:
    def _new_cb(
        self,
        name: str = "test",
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_probes: int = 1,
    ) -> CircuitBreaker:
        return CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_probes=half_open_probes,
        )

    def test_initial_state_is_closed(self):
        cb = self._new_cb()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_call_permitted()

    def test_closed_to_open_on_threshold(self):
        cb = self._new_cb(failure_threshold=3)
        cb.on_failure()
        assert cb.state == CircuitState.CLOSED
        cb.on_failure()
        assert cb.state == CircuitState.CLOSED
        cb.on_failure()  # 3rd failure → OPEN
        assert cb.state == CircuitState.OPEN

    def test_open_blocks_calls(self):
        cb = self._new_cb(failure_threshold=1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.is_call_permitted()

    def test_open_to_half_open_after_timeout(self):
        """After recovery_timeout, state should auto-advance to HALF_OPEN."""
        cb = self._new_cb(failure_threshold=1, recovery_timeout=0.05)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.07)  # Let recovery_timeout elapse

        # Querying is_call_permitted() triggers the advance
        assert cb.is_call_permitted()
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = self._new_cb(failure_threshold=1, recovery_timeout=0.05, half_open_probes=2)
        cb.on_failure()
        time.sleep(0.07)

        # Advance to HALF_OPEN
        assert cb.is_call_permitted()
        assert cb.state == CircuitState.HALF_OPEN

        cb.on_success()  # 1st probe success
        assert cb.state == CircuitState.HALF_OPEN  # need 2

        cb.on_success()  # 2nd probe → CLOSED
        assert cb.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = self._new_cb(failure_threshold=1, recovery_timeout=0.05)
        cb.on_failure()
        time.sleep(0.07)

        assert cb.is_call_permitted()
        assert cb.state == CircuitState.HALF_OPEN

        cb.on_failure()  # Probe failed → back to OPEN immediately
        assert cb.state == CircuitState.OPEN

    def test_success_in_closed_resets_failure_count(self):
        cb = self._new_cb(failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        assert cb.state == CircuitState.CLOSED
        cb.on_success()  # Reset counter
        cb.on_failure()  # 1st failure again
        cb.on_failure()  # 2nd failure again
        assert cb.state == CircuitState.CLOSED  # Not OPEN yet (counter reset)

    def test_multiple_failures_beyond_threshold(self):
        """Additional failures beyond threshold should keep state OPEN."""
        cb = self._new_cb(failure_threshold=2)
        for _ in range(10):
            cb.on_failure()
        assert cb.state == CircuitState.OPEN

    def test_full_cycle(self):
        """CLOSED → OPEN → HALF_OPEN → CLOSED full cycle."""
        cb = self._new_cb(failure_threshold=2, recovery_timeout=0.05, half_open_probes=1)

        # Step 1: CLOSED → OPEN
        cb.on_failure()
        cb.on_failure()
        assert cb.state == CircuitState.OPEN

        # Step 2: OPEN → HALF_OPEN
        time.sleep(0.07)
        assert cb.is_call_permitted()
        assert cb.state == CircuitState.HALF_OPEN

        # Step 3: HALF_OPEN → CLOSED
        cb.on_success()
        assert cb.state == CircuitState.CLOSED

        # Step 4: normal operation again
        assert cb.is_call_permitted()


# ---------------------------------------------------------------------------
# CircuitBreakerOpen exception
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpen:
    def test_circuit_breaker_open_attributes(self):
        exc = CircuitBreakerOpen("my_provider")
        assert exc.name == "my_provider"
        assert "my_provider" in str(exc)
        assert "OPEN" in str(exc)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestCircuitBreakerStats:
    def test_initial_stats(self):
        cb = CircuitBreaker("stats_test", failure_threshold=5)
        s = cb.stats()
        assert s["name"] == "stats_test"
        assert s["state"] == "closed"
        assert s["consecutive_failures"] == 0
        assert s["total_failures"] == 0
        assert s["total_successes"] == 0
        assert s["total_rejected"] == 0
        assert s["opened_at"] is None

    def test_stats_after_failures(self):
        cb = CircuitBreaker("stats_test2", failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        s = cb.stats()
        assert s["consecutive_failures"] == 2
        assert s["total_failures"] == 2

    def test_stats_after_open(self):
        cb = CircuitBreaker("stats_test3", failure_threshold=1)
        cb.on_failure()
        s = cb.stats()
        assert s["state"] == "open"
        assert s["opened_at"] is not None

    def test_stats_tracks_rejected(self):
        cb = CircuitBreaker("stats_test4", failure_threshold=1, recovery_timeout=60.0)
        cb.on_failure()  # → OPEN
        cb.is_call_permitted()  # → rejected
        cb.is_call_permitted()  # → rejected
        s = cb.stats()
        assert s["total_rejected"] >= 2

    def test_repr(self):
        cb = CircuitBreaker("repr_test", failure_threshold=5)
        r = repr(cb)
        assert "repr_test" in r
        assert "closed" in r


# ---------------------------------------------------------------------------
# Manual controls
# ---------------------------------------------------------------------------


class TestManualControls:
    def test_reset_from_open(self):
        cb = CircuitBreaker("manual_test", failure_threshold=1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_call_permitted()

    def test_trip_from_closed(self):
        cb = CircuitBreaker("manual_test2", failure_threshold=100)
        assert cb.state == CircuitState.CLOSED
        cb.trip()
        assert cb.state == CircuitState.OPEN
        assert not cb.is_call_permitted()

    def test_invalid_recovery_timeout(self):
        with pytest.raises(ValueError):
            CircuitBreaker("bad", recovery_timeout=None)  # type: ignore[arg-type]

    def test_invalid_recovery_timeout_zero(self):
        with pytest.raises(ValueError):
            CircuitBreaker("bad", recovery_timeout=0.0)


# ---------------------------------------------------------------------------
# Custom is_failure predicate
# ---------------------------------------------------------------------------


class TestCustomIsFailurePredicate:
    def test_only_matching_exceptions_count(self):
        """Only ValueError should count toward failure threshold."""
        cb = CircuitBreaker(
            "custom_pred",
            failure_threshold=2,
            is_failure=lambda exc: isinstance(exc, ValueError),
        )
        cb.on_failure(RuntimeError("ignored"))
        cb.on_failure(RuntimeError("ignored"))
        assert cb.state == CircuitState.CLOSED  # RuntimeErrors don't count

        cb.on_failure(ValueError("counts"))
        cb.on_failure(ValueError("counts"))
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# CircuitBreakerRegistry
# ---------------------------------------------------------------------------


class TestCircuitBreakerRegistry:
    def setup_method(self):
        self.registry = CircuitBreakerRegistry()

    def test_get_or_create_new(self):
        cb = self.registry.get_or_create("provider_a")
        assert cb.name == "provider_a"

    def test_get_or_create_returns_same_instance(self):
        cb1 = self.registry.get_or_create("provider_b")
        cb2 = self.registry.get_or_create("provider_b")
        assert cb1 is cb2

    def test_get_returns_none_for_unknown(self):
        assert self.registry.get("nonexistent") is None

    def test_all_stats(self):
        self.registry.get_or_create("p1")
        self.registry.get_or_create("p2")
        stats = self.registry.all_stats()
        names = {s["name"] for s in stats}
        assert "p1" in names
        assert "p2" in names

    def test_reset_all(self):
        cb = self.registry.get_or_create("p3", failure_threshold=1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        self.registry.reset_all()
        assert cb.state == CircuitState.CLOSED

    def test_clear(self):
        self.registry.get_or_create("p4")
        self.registry.clear()
        assert self.registry.get("p4") is None


# ---------------------------------------------------------------------------
# ProviderRegistry integration
# ---------------------------------------------------------------------------


class TestProviderRegistryIntegration:
    def setup_method(self):
        from effgen.models.registry import ProviderRegistry

        ProviderRegistry.clear()

        # Register a fake provider
        class FakeAdapter:
            pass

        ProviderRegistry.register(
            "fake_provider",
            FakeAdapter,
            {"fake-model-1": {"name": "Fake Model 1"}},
            env_keys=["FAKE_API_KEY"],
        )

    def teardown_method(self):
        from effgen.models.registry import ProviderRegistry

        ProviderRegistry.reset()

    def test_get_circuit_breaker_creates(self):
        from effgen.models.registry import ProviderRegistry

        cb = ProviderRegistry.get_circuit_breaker("fake_provider")
        assert cb.name == "fake_provider"
        assert cb.state == CircuitState.CLOSED

    def test_get_circuit_breaker_same_instance(self):
        from effgen.models.registry import ProviderRegistry

        cb1 = ProviderRegistry.get_circuit_breaker("fake_provider")
        cb2 = ProviderRegistry.get_circuit_breaker("fake_provider")
        assert cb1 is cb2

    def test_get_circuit_breaker_unknown_provider(self):
        from effgen.models.registry import ProviderRegistry

        with pytest.raises(KeyError, match="unknown_provider"):
            ProviderRegistry.get_circuit_breaker("unknown_provider")

    def test_get_bulkhead_creates(self):
        from effgen.models.registry import ProviderRegistry

        bh = ProviderRegistry.get_bulkhead("fake_provider")
        assert bh.name == "fake_provider"

    def test_reliability_stats(self):
        from effgen.models.registry import ProviderRegistry

        ProviderRegistry.get_circuit_breaker("fake_provider")
        ProviderRegistry.get_bulkhead("fake_provider")
        stats = ProviderRegistry.reliability_stats()
        assert "fake_provider" in stats
        assert stats["fake_provider"]["circuit_breaker"] is not None
        assert stats["fake_provider"]["bulkhead"] is not None


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestCircuitBreakerThreadSafety:
    def test_concurrent_failures_and_successes(self):
        """Multiple threads recording failures/successes should not corrupt state."""
        cb = CircuitBreaker("thread_test", failure_threshold=50, recovery_timeout=0.05)
        errors: list[Exception] = []

        def record_failures(n: int):
            try:
                for _ in range(n):
                    cb.on_failure()
            except Exception as e:
                errors.append(e)

        def record_successes(n: int):
            try:
                for _ in range(n):
                    cb.on_success()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_failures, args=(20,))
            for _ in range(5)
        ] + [
            threading.Thread(target=record_successes, args=(10,))
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # State should be consistent (not corrupted)
        s = cb.stats()
        assert s["state"] in ("closed", "open", "half_open")
        assert s["consecutive_failures"] >= 0
