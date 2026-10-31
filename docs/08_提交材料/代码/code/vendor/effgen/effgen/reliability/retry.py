"""
Retry with jittered exponential backoff.

Usage::

    from effgen.reliability.retry import Retry, retryable, is_transient_error

    policy = Retry(
        max_attempts=4,
        base_delay=0.5,
        max_delay=30.0,
        jitter=True,
        retryable=is_transient_error,
    )

    @retryable(policy)
    def call_api():
        ...

    # Async version:
    @retryable(policy)
    async def call_api_async():
        ...

Each retry attempt emits an ``effgen.retry.attempt`` span *event* on the
currently active OTel span (no-op if tracing is not configured).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from effgen.errors import quote_for_message, with_next_step

log = logging.getLogger(__name__)

__all__ = [
    "Retry",
    "RetryExhausted",
    "retryable",
    "is_transient_error",
    "DEFAULT_RETRY",
]


# ---------------------------------------------------------------------------
# Retryable error classification
# ---------------------------------------------------------------------------


def is_transient_error(exc: BaseException) -> bool:
    """Return True for errors that are safe to retry.

    Network-level errors (``ConnectionError``, ``OSError``, ``TimeoutError``,
    ``asyncio.TimeoutError``, effGen's own :class:`~effgen.reliability.timeouts.TimeoutError`)
    are always transient and are checked first.

    Anything carrying a structured ``.error_context`` — effGen's own typed
    provider errors (``effgen.models.errors``) and the generic ``RuntimeError``
    every provider adapter wraps a real SDK failure into (see
    ``effgen.models._adapter_utils.provider_runtime_error``) — was already
    classified from the *original* SDK exception before wrapping could lose
    its status code or class name, so we defer to that classification. This
    is what makes a live 429/5xx that a provider adapter has wrapped retry
    correctly, not just a raw, never-wrapped SDK exception.

    Everything else (a raw SDK exception effGen never touched, or a plain
    application exception) falls back to:
    - HTTP 429 / 5xx responses (detected by status code attribute)
    - httpx network/timeout errors
    A plain, unclassified exception (e.g. an application bug) is *not*
    retried — retrying it cannot change the outcome.
    """
    from effgen.reliability.timeouts import TimeoutError as EffGenTimeout

    # Network / transient errors — cheapest check, no imports needed.
    transient_types = (
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        TimeoutError,
        asyncio.TimeoutError,
        EffGenTimeout,
        OSError,
    )
    if isinstance(exc, transient_types):
        return True

    # A structured error_context (attached by provider_runtime_error()/
    # attach_error_context(), or set directly by effGen's typed provider
    # errors) is authoritative — the error taxonomy is the single source of
    # truth for retry-worthiness whenever it is present.
    if isinstance(exc, Exception) and isinstance(getattr(exc, "error_context", None), dict):
        from effgen.models.errors import classify_provider_error

        return classify_provider_error(exc).should_retry

    # Check for HTTP status codes on the exception
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status == 429 or (500 <= status < 600):
            return True

    # Check response attribute (httpx / requests style)
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int):
            if resp_status == 429 or (500 <= resp_status < 600):
                return True

    # httpx-specific errors if available
    try:
        import httpx
        if isinstance(exc, httpx.NetworkError | httpx.TimeoutException | httpx.RemoteProtocolError):
            return True
    except ImportError:
        pass

    return False


def _get_retry_after(exc: BaseException) -> float | None:
    """Seconds the provider asked for before the next attempt, if it said.

    Delegates to the shared reader so a wrapped provider error — the shape a
    caller actually receives, with the response two ``__cause__`` links down —
    is read the same way here as it is on the HTTP surface, including the delay
    Gemini and the OpenAI-compatible providers state in the body rather than in
    a header.
    """
    try:
        from effgen.models.errors import retry_after_seconds

        return retry_after_seconds(exc)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Retry dataclass
# ---------------------------------------------------------------------------


@dataclass
class Retry:
    """Retry policy with jittered exponential backoff.

    Attributes:
        max_attempts:  Total number of attempts (including the first).
                       ``1`` means no retries.
        base_delay:    Initial backoff delay in seconds.
        max_delay:     Cap on the computed backoff delay.
        jitter:        When True, add ``uniform(0, base_delay)`` to each delay.
        retryable:     Callable ``(exc) -> bool`` that decides whether an
                       exception is worth retrying.  Defaults to
                       :func:`is_transient_error`.
        honour_retry_after:  When True, sleep the Retry-After value from a 429
                             response instead of the computed backoff.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: bool = True
    retryable: Callable[[BaseException], bool] = field(
        default=is_transient_error
    )
    honour_retry_after: bool = True

    def compute_delay(self, attempt: int, exc: BaseException | None = None) -> float:
        """Compute the wait duration before the next attempt.

        Args:
            attempt:  0-based attempt index (0 = first retry).
            exc:      The exception that triggered this retry.  Used to
                      extract Retry-After header when present.

        Returns:
            Seconds to sleep before the next attempt.
        """
        # Honour Retry-After from 429 responses
        if self.honour_retry_after and exc is not None:
            ra = _get_retry_after(exc)
            if ra is not None:
                return min(ra, self.max_delay)

        # Exponential backoff: base * 2^attempt
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay += random.uniform(0.0, self.base_delay)
            delay = min(delay, self.max_delay)

        return delay


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RetryExhausted(Exception):
    """Raised when all retry attempts have been consumed.

    Attributes:
        attempts:   Number of attempts made.
        last_error: The last exception that triggered a retry.
    """

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(
            with_next_step(
                f"All {attempts} retry attempts exhausted. "
                f"Last error: {type(last_error).__name__}: "
                f"{quote_for_message(last_error)}",
                "Fix the cause named above — retrying more will not clear it; "
                "raise max_attempts only for a failure that is genuinely "
                "transient.",
            )
        )
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# OTel span event helper (non-blocking)
# ---------------------------------------------------------------------------


def _emit_retry_event(attempt: int, reason: str, delay_s: float) -> None:
    """Attach a retry event to the current OTel span (no-op if no span)."""
    try:
        from opentelemetry import trace

        from effgen.observability.spans import RetryAttrs, SpanName

        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(
                SpanName.RETRY_ATTEMPT,
                attributes={
                    RetryAttrs.ATTEMPT: attempt,
                    RetryAttrs.REASON: reason,
                    RetryAttrs.DELAY_S: delay_s,
                },
            )
    except Exception:
        pass  # telemetry is non-blocking


# ---------------------------------------------------------------------------
# retryable decorator
# ---------------------------------------------------------------------------


DEFAULT_RETRY = Retry()


def retryable(policy: Retry = DEFAULT_RETRY) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory that wraps a sync or async function with *policy*.

    Usage::

        @retryable(Retry(max_attempts=4, base_delay=1.0))
        def call_api() -> str:
            ...

        @retryable(Retry(max_attempts=3))
        async def async_call() -> str:
            ...

    Raises:
        :class:`RetryExhausted`: When *max_attempts* is reached.
        The original exception is re-raised if it is not retryable.
    """

    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: BaseException | None = None
                for attempt in range(policy.max_attempts):
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if not policy.retryable(exc):
                            raise
                        if attempt + 1 >= policy.max_attempts:
                            break
                        reason = type(exc).__name__
                        delay = policy.compute_delay(attempt, exc)
                        log.warning(
                            "retry attempt %d/%d for %s after %.2fs "
                            "(reason: %s: %s)",
                            attempt + 1,
                            policy.max_attempts,
                            fn.__qualname__,
                            delay,
                            reason,
                            exc,
                        )
                        _emit_retry_event(attempt + 1, reason, delay)
                        await asyncio.sleep(delay)
                raise RetryExhausted(policy.max_attempts, last_exc)

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: BaseException | None = None
                for attempt in range(policy.max_attempts):
                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if not policy.retryable(exc):
                            raise
                        if attempt + 1 >= policy.max_attempts:
                            break
                        reason = type(exc).__name__
                        delay = policy.compute_delay(attempt, exc)
                        log.warning(
                            "retry attempt %d/%d for %s after %.2fs "
                            "(reason: %s: %s)",
                            attempt + 1,
                            policy.max_attempts,
                            fn.__qualname__,
                            delay,
                            reason,
                            exc,
                        )
                        _emit_retry_event(attempt + 1, reason, delay)
                        time.sleep(delay)
                raise RetryExhausted(policy.max_attempts, last_exc)

            return sync_wrapper

    return decorator
