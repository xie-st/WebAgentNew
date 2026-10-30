"""RetryPolicy — exponential-backoff + jitter retry logic for the effGen router.

Retry semantics
---------------
Retry on (transient, recoverable):
  - ``RateLimitExceeded``  — provider rate-limited; back off and try again.
  - ``ProviderTransientError`` — 5xx from provider; may resolve on retry.
  - ``ModelTimeoutError``  — prediction timed out; worth retrying.
  - ``BudgetExceededError`` — not retried against the *same* provider, but
    treated as retriable here because the router's meaning of "retriable" is
    "worth trying the next candidate" (e.g. a free-tier fallback), not "the
    same call might succeed if repeated."

Do NOT retry on (permanent, operator-action required):
  - ``ModelAuthError``     — bad/missing API key.
  - ``ModelRefusalError``  — model refused; retrying will produce the same result.
  - ``InvalidRequestError`` — malformed request; retrying will always fail.

Relationship to ``effgen.models.errors.classify_provider_error`` /
``effgen.reliability.retry.is_transient_error``: those answer "is retrying
*this exact call* worth it", so an unrecognized exception defaults to
retryable (a genuine transient blip should not become a hard failure) and
``BudgetExceededError`` defaults to *not* retryable (repeating the same call
against the same provider/budget will not succeed). ``RetryPolicy`` answers a
related but distinct question — "should the router try the next candidate" —
so it deliberately keeps its own explicit allow/deny lists rather than
delegating: an unrecognized exception here defaults to *not* retriable
(fail closed on an unknown failure rather than failing over blindly), and
``BudgetExceededError`` is retriable here specifically because failing over
to a different, cheaper/free-tier candidate is exactly the useful response.

Usage::

    from effgen.models.routing.retry import RetryPolicy

    policy = RetryPolicy(max_retries=3, backoff_base=1.0, jitter=0.5)

    for attempt in policy.attempts():
        try:
            result = call_provider(...)
            break
        except Exception as exc:
            if not policy.is_retriable(exc):
                raise
            policy.record_failure(attempt, exc)
    else:
        raise AllCandidatesExhaustedError(...)
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from effgen.models._rate_limit import RateLimitExceeded
from effgen.models.errors import (
    BudgetExceededError,
    InvalidRequestError,
    ModelAuthError,
    ModelRefusalError,
    ModelTimeoutError,
    ProviderTransientError,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Exceptions that are safe to retry (with failover to another provider).
_RETRIABLE = (RateLimitExceeded, ProviderTransientError, ModelTimeoutError, BudgetExceededError)

# Exceptions that must NEVER be retried.
_NON_RETRIABLE = (ModelAuthError, ModelRefusalError, InvalidRequestError)


@dataclass
class RetryState:
    """Mutable state for a single retry sequence."""
    attempt: int = 0
    total_sleep_seconds: float = 0.0
    failures: list[tuple[int, Exception]] = field(default_factory=list)


class RetryPolicy:
    """Exponential-backoff retry policy with full-jitter.

    The sleep between attempt *n* and *n+1* is::

        sleep = min(cap, backoff_base * 2**n + U(0, jitter))

    where ``U(0, jitter)`` is a uniform random variable in seconds.

    Args:
        max_retries:  Maximum number of *retries* (not total attempts).
                      Total attempts = max_retries + 1.
        backoff_base: Base delay in seconds.
        jitter:       Maximum random additive delay in seconds.
                      Set to 0.0 for deterministic backoff.
        backoff_cap:  Maximum sleep cap in seconds.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        jitter: float = 0.5,
        backoff_cap: float = 30.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.jitter = jitter
        self.backoff_cap = backoff_cap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_retriable(self, exc: Exception) -> bool:
        """Return True if *exc* is worth trying the next candidate for.

        Uses this module's own allow/deny lists rather than
        ``classify_provider_error`` — see the module docstring for why the
        two intentionally disagree on ``BudgetExceededError`` and on
        unrecognized exceptions.
        """
        if isinstance(exc, _NON_RETRIABLE):
            return False
        return isinstance(exc, _RETRIABLE)

    def sleep_seconds(self, attempt: int) -> float:
        """Compute sleep duration before attempt *attempt* (0-indexed retry count).

        attempt=0 → first retry (after initial failure), etc.
        """
        base = min(self.backoff_cap, self.backoff_base * (2 ** attempt))
        return min(self.backoff_cap, base + random.uniform(0.0, max(self.jitter, 0.0)))

    def execute_with_retry(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[T, RetryState]:
        """Call *fn(*args, **kwargs)* with retry on retriable exceptions.

        Returns the function's return value and a ``RetryState`` with
        diagnostics.

        Raises:
            The original exception if it is not retriable, or if
            ``max_retries`` is exhausted (re-raises the last retriable error).
        """
        state = RetryState()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            state.attempt = attempt
            try:
                result = fn(*args, **kwargs)
                return result, state
            except Exception as exc:  # noqa: BLE001
                if not self.is_retriable(exc):
                    logger.debug(
                        "RetryPolicy: non-retriable %s on attempt %d — raising immediately",
                        type(exc).__name__, attempt,
                    )
                    raise

                state.failures.append((attempt, exc))
                last_exc = exc

                if attempt < self.max_retries:
                    sleep = self.sleep_seconds(attempt)
                    state.total_sleep_seconds += sleep
                    logger.info(
                        "RetryPolicy: retriable %s on attempt %d/%d — sleeping %.3fs",
                        type(exc).__name__, attempt, self.max_retries, sleep,
                    )
                    time.sleep(sleep)
                else:
                    logger.warning(
                        "RetryPolicy: max_retries=%d exhausted after %s",
                        self.max_retries, type(exc).__name__,
                    )

        raise last_exc  # type: ignore[misc]
