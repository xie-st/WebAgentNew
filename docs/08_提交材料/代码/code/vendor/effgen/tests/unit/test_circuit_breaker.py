"""Tests for CircuitBreaker — open/closed/half-open states."""

import time

from effgen.utils.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerClosed:
    """Test normal (closed) operation."""

    def test_new_tool_is_available(self):
        cb = CircuitBreaker()
        assert cb.is_available("my_tool") is True

    def test_state_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.get_state("my_tool") == CircuitState.CLOSED

    def test_success_keeps_closed(self):
        cb = CircuitBreaker()
        cb.record_success("my_tool")
        assert cb.get_state("my_tool") == CircuitState.CLOSED
        assert cb.is_available("my_tool") is True

    def test_failures_below_threshold_stay_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("my_tool")
        cb.record_failure("my_tool")
        assert cb.get_state("my_tool") == CircuitState.CLOSED
        assert cb.is_available("my_tool") is True


class TestCircuitBreakerOpen:
    """Test that circuit opens after threshold failures."""

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        for _ in range(3):
            cb.record_failure("my_tool")
        assert cb.get_state("my_tool") == CircuitState.OPEN
        assert cb.is_available("my_tool") is False

    def test_opens_above_threshold(self):
        cb = CircuitBreaker(failure_threshold=2)
        for _ in range(5):
            cb.record_failure("t")
        assert cb.get_state("t") == CircuitState.OPEN

    def test_open_tool_not_available(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=9999)
        cb.record_failure("t")
        assert cb.is_available("t") is False


class TestCircuitBreakerHalfOpen:
    """Test half-open recovery after cooldown."""

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure("t")
        assert cb.get_state("t") == CircuitState.OPEN
        assert cb.is_available("t") is False

        time.sleep(0.15)
        assert cb.is_available("t") is True
        assert cb.get_state("t") == CircuitState.HALF_OPEN

    def test_success_in_half_open_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
        cb.record_failure("t")
        time.sleep(0.1)
        cb.is_available("t")  # triggers transition to HALF_OPEN
        cb.record_success("t")
        assert cb.get_state("t") == CircuitState.CLOSED
        assert cb.is_available("t") is True

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
        cb.record_failure("t")
        time.sleep(0.1)
        cb.is_available("t")  # HALF_OPEN
        cb.record_failure("t")
        assert cb.get_state("t") == CircuitState.OPEN


class TestCircuitBreakerReset:
    """Test manual reset."""

    def test_reset_single_tool(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("t")
        assert cb.get_state("t") == CircuitState.OPEN
        cb.reset("t")
        assert cb.get_state("t") == CircuitState.CLOSED
        assert cb.is_available("t") is True

    def test_reset_all(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("a")
        cb.record_failure("b")
        cb.reset_all()
        assert cb.is_available("a") is True
        assert cb.is_available("b") is True

    def test_reset_nonexistent_tool_is_noop(self):
        cb = CircuitBreaker()
        cb.reset("nonexistent")  # should not raise


class TestCircuitBreakerMultipleTools:
    """Test that circuits are independent per tool."""

    def test_independent_circuits(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("a")
        cb.record_failure("a")
        assert cb.get_state("a") == CircuitState.OPEN
        assert cb.get_state("b") == CircuitState.CLOSED
        assert cb.is_available("b") is True


class TestCircuitBreakerPersistence:
    """Test save/restore of circuit state across instances (persist_path)."""

    def test_open_state_round_trips(self, tmp_path):
        path = str(tmp_path / "circuits.json")
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60, persist_path=path)
        cb.record_failure("t")
        cb.record_failure("t")
        assert cb.get_state("t") == CircuitState.OPEN

        # A fresh instance (simulating a process restart) restores the OPEN state.
        cb2 = CircuitBreaker(failure_threshold=2, cooldown_seconds=60, persist_path=path)
        assert cb2.get_state("t") == CircuitState.OPEN
        assert cb2.is_available("t") is False

    def test_restored_open_circuit_recovers_not_stuck(self, tmp_path):
        # Regression: persisted recovery timestamps are monotonic-clock based and
        # meaningless across restarts. A restored OPEN circuit must still cool
        # down to HALF_OPEN rather than getting stuck open forever.
        path = str(tmp_path / "circuits.json")
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1, persist_path=path)
        cb.record_failure("t")
        assert cb.get_state("t") == CircuitState.OPEN

        cb2 = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1, persist_path=path)
        assert cb2.is_available("t") is False  # still cooling down right after load
        time.sleep(0.15)
        assert cb2.is_available("t") is True
        assert cb2.get_state("t") == CircuitState.HALF_OPEN
