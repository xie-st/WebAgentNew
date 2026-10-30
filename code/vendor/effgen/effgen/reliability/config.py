"""
Reliability configuration dataclasses.

``ReliabilityConfig`` is the single source of truth for all timeout values,
retry defaults, circuit-breaker thresholds, and bulkhead capacities.  It can
be embedded inside ``ObservabilityConfig`` or used standalone.

Example::

    from effgen.reliability.config import ReliabilityConfig, TimeoutConfig

    cfg = ReliabilityConfig(
        timeouts=TimeoutConfig(model_call=45, tool_call=20, http=15),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeoutConfig:
    """Explicit timeouts (in seconds) for every I/O boundary.

    No field defaults to ``None`` — the build enforces that every path has an
    explicit upper bound.

    Attributes:
        model_call:  Maximum wall-clock time for a single model inference call.
        tool_call:   Maximum wall-clock time for a single tool execution.
        http:        Default httpx / requests connect + read timeout.
        agent_loop:  Maximum wall-clock time for the full agent.run() call
                     (sum of iterations; safety valve only).
        queue:       Maximum time to wait for a Bulkhead permit before raising
                     :class:`~effgen.reliability.bulkhead.BulkheadFull`.
    """

    model_call: float = 60.0
    tool_call: float = 30.0
    http: float = 20.0
    agent_loop: float = 600.0
    queue: float = 5.0

    def as_dict(self) -> dict[str, float]:
        """Return a plain dict of all timeout values."""
        return {
            "model_call": self.model_call,
            "tool_call": self.tool_call,
            "http": self.http,
            "agent_loop": self.agent_loop,
            "queue": self.queue,
        }


@dataclass
class RetryDefaults:
    """Default retry policy parameters."""

    max_attempts: int = 3
    base_delay: float = 0.5      # seconds
    max_delay: float = 30.0      # seconds
    jitter: bool = True
    # Retry on these HTTP status codes
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class CircuitBreakerDefaults:
    """Default circuit-breaker parameters (per-provider)."""

    failure_threshold: int = 5        # consecutive failures → OPEN
    recovery_timeout: float = 30.0    # seconds in OPEN before → HALF_OPEN
    half_open_probes: int = 1         # successful probes to close circuit


@dataclass
class BulkheadDefaults:
    """Default bulkhead parameters (per-provider)."""

    max_concurrency: int = 20
    queue_size: int = 100


@dataclass
class ReliabilityConfig:
    """Top-level reliability configuration.

    Embed this in your application-level config or pass it directly to
    :class:`~effgen.models.registry.ProviderRegistry` middleware helpers.

    Attributes:
        timeouts:         Explicit timeout values.
        retry:            Default retry policy.
        circuit_breaker:  Default circuit-breaker thresholds.
        bulkhead:         Default bulkhead capacities.
        extra:            Arbitrary extension dict for future fields.
    """

    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    retry: RetryDefaults = field(default_factory=RetryDefaults)
    circuit_breaker: CircuitBreakerDefaults = field(default_factory=CircuitBreakerDefaults)
    bulkhead: BulkheadDefaults = field(default_factory=BulkheadDefaults)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def defaults(cls) -> "ReliabilityConfig":
        """Return a ``ReliabilityConfig`` with all defaults applied."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "timeouts": self.timeouts.as_dict(),
            "retry": {
                "max_attempts": self.retry.max_attempts,
                "base_delay": self.retry.base_delay,
                "max_delay": self.retry.max_delay,
                "jitter": self.retry.jitter,
                "retryable_status_codes": list(self.retry.retryable_status_codes),
            },
            "circuit_breaker": {
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "recovery_timeout": self.circuit_breaker.recovery_timeout,
                "half_open_probes": self.circuit_breaker.half_open_probes,
            },
            "bulkhead": {
                "max_concurrency": self.bulkhead.max_concurrency,
                "queue_size": self.bulkhead.queue_size,
            },
        }
