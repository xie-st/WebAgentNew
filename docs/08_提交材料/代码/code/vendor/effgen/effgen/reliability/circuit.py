"""
Circuit breaker for effGen provider calls.

Implements the classic three-state machine:

    CLOSED  ──(failure_threshold reached)──►  OPEN
      ▲                                          │
      │                                    (recovery_timeout elapsed)
      │                                          ▼
      └─────(half_open_probes successes)───  HALF_OPEN

Usage::

    from effgen.reliability.circuit import CircuitBreaker, CircuitBreakerOpen, CircuitState

    cb = CircuitBreaker(
        name="openai",
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_probes=1,
    )

    if cb.is_call_permitted():
        try:
            result = call_model()
            cb.on_success()
        except Exception as exc:
            cb.on_failure(exc)
            raise
    else:
        raise CircuitBreakerOpen("openai")

Integration with :class:`~effgen.models.registry.ProviderRegistry`
is done by :meth:`ProviderRegistry.get_circuit_breaker`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "CircuitState",
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitBreakerRegistry",
    "get_circuit_breaker",
]


class CircuitState(Enum):
    """States of the circuit breaker finite state machine."""

    CLOSED = "closed"
    """Normal operation — calls flow through."""

    OPEN = "open"
    """Circuit is open — all calls are blocked (fast-fail)."""

    HALF_OPEN = "half_open"
    """Recovery probing — a limited number of test calls are allowed."""


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted on an OPEN circuit.

    Attributes:
        name:  The circuit breaker name (usually the provider name).
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Circuit breaker '{name}' is OPEN — refusing call to prevent "
            "cascade failure.  Wait for recovery_timeout or call reset()."
        )
        self.name = name


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Args:
        name:               Human-readable label (e.g. provider name).
        failure_threshold:  Number of consecutive failures before → OPEN.
        recovery_timeout:   Seconds to stay OPEN before → HALF_OPEN.
        half_open_probes:   Number of consecutive successes in HALF_OPEN
                            before → CLOSED.
        is_failure:         Optional callable ``(exc) -> bool``.  If provided,
                            only exceptions for which it returns True count
                            toward the failure counter.  Defaults to all
                            exceptions counting as failures.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_probes: int = 1,
        is_failure: Callable[[BaseException], bool] | None = None,
    ) -> None:
        if recovery_timeout is None or recovery_timeout <= 0:
            raise ValueError(f"recovery_timeout must be > 0, got {recovery_timeout!r}")

        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_probes = half_open_probes
        self._is_failure = is_failure  # None → all exceptions count

        self._lock = threading.RLock()
        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0  # tracked in HALF_OPEN
        self._opened_at: float = 0.0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._total_rejected: int = 0

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current state (may auto-advance OPEN→HALF_OPEN on timeout)."""
        with self._lock:
            return self._check_and_advance()

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive failures recorded since the last success.

        Read-only view of the value also reported under ``stats()``.
        """
        with self._lock:
            return self._consecutive_failures

    def is_call_permitted(self) -> bool:
        """Return True if a call should be attempted.

        - CLOSED → True (always)
        - HALF_OPEN → True (probe calls allowed)
        - OPEN → False (unless recovery_timeout elapsed, then → HALF_OPEN)
        """
        with self._lock:
            state = self._check_and_advance()
            if state == CircuitState.OPEN:
                self._total_rejected += 1
                return False
            return True

    def _check_and_advance(self) -> CircuitState:
        """Internal: advance OPEN → HALF_OPEN if timeout elapsed."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def on_success(self) -> None:
        """Record a successful call outcome."""
        with self._lock:
            self._total_successes += 1
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.half_open_probes:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                pass  # normal

    def on_failure(self, exc: BaseException | None = None) -> None:
        """Record a failed call outcome.

        Args:
            exc:  The exception.  If ``is_failure`` was provided at
                  construction time, only matching exceptions increment
                  the counter.
        """
        with self._lock:
            # Check if this exception type counts
            if exc is not None and self._is_failure is not None:
                if not self._is_failure(exc):
                    return  # not a circuit-relevant failure

            self._total_failures += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed → immediately back to OPEN
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: CircuitState) -> None:
        """Perform a state transition (lock must be held)."""
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            self._consecutive_failures = 0
            log.warning(
                "Circuit breaker '%s' OPENED after %d consecutive failures.",
                self.name,
                self.failure_threshold,
            )
        elif new_state == CircuitState.HALF_OPEN:
            self._consecutive_successes = 0
            log.info(
                "Circuit breaker '%s' → HALF_OPEN (recovery timeout elapsed).",
                self.name,
            )
        elif new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            log.info(
                "Circuit breaker '%s' CLOSED after %d probe successes.",
                self.name,
                self.half_open_probes,
            )

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Force circuit to CLOSED state (e.g. after infrastructure fix)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            log.info("Circuit breaker '%s' manually RESET to CLOSED.", self.name)

    def trip(self) -> None:
        """Force circuit to OPEN state (e.g. proactive isolation)."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            log.info("Circuit breaker '%s' manually TRIPPED to OPEN.", self.name)

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of current statistics."""
        with self._lock:
            state = self._check_and_advance()
            return {
                "name": self.name,
                "state": state.value,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "total_rejected": self._total_rejected,
                "opened_at": self._opened_at if state == CircuitState.OPEN else None,
            }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"CircuitBreaker(name={self.name!r}, state={s['state']!r}, "
            f"failures={s['consecutive_failures']}/{self.failure_threshold})"
        )


# ---------------------------------------------------------------------------
# Registry — one breaker per provider, shared globally
# ---------------------------------------------------------------------------


class CircuitBreakerRegistry:
    """Thread-safe registry of :class:`CircuitBreaker` instances keyed by name.

    Typically used as a module-level singleton so the same breaker instance
    is shared across the entire process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_probes: int = 1,
        is_failure: Callable[[BaseException], bool] | None = None,
    ) -> CircuitBreaker:
        """Return an existing :class:`CircuitBreaker` or create one.

        Parameters match :class:`CircuitBreaker.__init__`.

        Args:
            name: The breaker's name in the registry.
            failure_threshold: Failures before the circuit opens.
            recovery_timeout: Seconds the circuit stays open before a probe.
            half_open_probes: Successful probes before the circuit closes again.
            is_failure: Decides which exceptions count as a failure.
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    half_open_probes=half_open_probes,
                    is_failure=is_failure,
                )
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Return the breaker for *name*, or None if not registered."""
        with self._lock:
            return self._breakers.get(name)

    def all_stats(self) -> list[dict[str, Any]]:
        """Return stats snapshots for all registered breakers."""
        with self._lock:
            return [cb.stats() for cb in self._breakers.values()]

    def reset_all(self) -> None:
        """Reset all breakers to CLOSED."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()

    def clear(self) -> None:
        """Remove all breakers (useful for testing)."""
        with self._lock:
            self._breakers.clear()


# Module-level default registry
_default_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    *,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_probes: int = 1,
) -> CircuitBreaker:
    """Get or create a :class:`CircuitBreaker` from the default registry.

    Args:
        name:              Provider or component name.
        failure_threshold: Failures before OPEN.
        recovery_timeout:  Seconds in OPEN before HALF_OPEN.
        half_open_probes:  Successes in HALF_OPEN before CLOSED.

    Returns:
        The registered circuit breaker.
    """
    return _default_registry.get_or_create(
        name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_probes=half_open_probes,
    )
