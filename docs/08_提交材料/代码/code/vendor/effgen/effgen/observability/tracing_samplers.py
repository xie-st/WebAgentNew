"""Sampler implementations for effGen tracing.

Five samplers, each a valid ``Sampler`` object for the OTel SDK:

- ``AlwaysOnSampler``   — sample every span
- ``AlwaysOffSampler``  — drop every span
- ``TraceIdRatioSampler(p)``  — probabilistic, 0 ≤ p ≤ 1
- ``RateLimitedSampler(per_second)``  — token-bucket rate limiter
- ``ParentBasedSampler(root)``  — honour the parent sampling decision and
  delegate root spans to *root*

When the OTel SDK is not installed, each name is bound to a stand-in that
carries the same constructor, properties and ``get_description``, so a caller
that builds a sampler still works and the description says ``(no-op)``.

Import them from :mod:`effgen.observability.tracing`; this module is the
implementation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from effgen.observability.tracing_otel import (
    _OTEL_AVAILABLE,
    Context,
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
    SpanKind,
)

# Pinned rather than derived from ``__name__``: the tracing layer logs under one
# name whichever module the log site lives in.
logger = logging.getLogger("effgen.observability.tracing")


# ---------------------------------------------------------------------------
# Sampler implementations
# ---------------------------------------------------------------------------


if _OTEL_AVAILABLE:

    class AlwaysOnSampler(Sampler):
        """Sample every span.  Wraps the SDK ``ALWAYS_ON`` singleton."""

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes: Any = None,
            links: Any = None,
            trace_state: Any = None,
        ) -> SamplingResult:
            """Record and sample every span.

            Args:
                parent_context: Context of the parent span, when there is one.
                trace_id: Id of the trace the span belongs to.
                name: The span's name.
                kind: The span kind OpenTelemetry reported.
                attributes: Attributes the span was started with.
                links: Links to other spans.
                trace_state: Vendor state carried on the trace.

            Returns:
                The sampling decision, with the attributes and trace state to keep.
            """
            return SamplingResult(
                decision=Decision.RECORD_AND_SAMPLE,
                attributes=attributes,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            """Return the sampler's description string."""
            return "AlwaysOnSampler"

    class AlwaysOffSampler(Sampler):
        """Drop every span.  Useful for disabling tracing at runtime."""

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes: Any = None,
            links: Any = None,
            trace_state: Any = None,
        ) -> SamplingResult:
            """Drop every span.

            Args:
                parent_context: Context of the parent span, when there is one.
                trace_id: Id of the trace the span belongs to.
                name: The span's name.
                kind: The span kind OpenTelemetry reported.
                attributes: Attributes the span was started with.
                links: Links to other spans.
                trace_state: Vendor state carried on the trace.

            Returns:
                The sampling decision, with the attributes and trace state to keep.
            """
            return SamplingResult(
                decision=Decision.DROP,
                attributes=attributes,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            """Return the sampler's description string."""
            return "AlwaysOffSampler"

    class TraceIdRatioSampler(Sampler):
        """
        Probabilistic sampler based on trace-ID hash.

        Deterministic: the same trace ID always produces the same sampling
        decision, so all spans in a trace share the same fate.

        Args:
            ratio: Sampling ratio in [0.0, 1.0].  ``0.0`` → never; ``1.0`` →
                   always.  Clamped to [0, 1].
        """

        def __init__(self, ratio: float) -> None:
            self._ratio = max(0.0, min(1.0, float(ratio)))
            # Convert to integer threshold in [0, 2^128)
            self._threshold = int(self._ratio * (2**128))

        @property
        def ratio(self) -> float:
            """The configured sampling ratio in [0.0, 1.0]."""
            return self._ratio

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes: Any = None,
            links: Any = None,
            trace_state: Any = None,
        ) -> SamplingResult:
            """Sample when the trace ID falls under the ratio threshold.

            Args:
                parent_context: Context of the parent span, when there is one.
                trace_id: Id of the trace the span belongs to, which the ratio is
                    applied to.
                name: The span's name.
                kind: The span kind OpenTelemetry reported.
                attributes: Attributes the span was started with.
                links: Links to other spans.
                trace_state: Vendor state carried on the trace.

            Returns:
                The sampling decision, with the attributes and trace state to keep.
            """
            decision = (
                Decision.RECORD_AND_SAMPLE
                if trace_id < self._threshold
                else Decision.DROP
            )
            return SamplingResult(
                decision=decision,
                attributes=attributes,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            """Return the sampler's description string."""
            return f"TraceIdRatioSampler({self._ratio:.4f})"

    class RateLimitedSampler(Sampler):
        """
        Token-bucket rate-limited sampler.

        Guarantees at most *per_second* sampled traces per second regardless of
        traffic volume.  Implemented as a thread-safe token bucket.

        Args:
            per_second: Maximum number of traces to sample per second.  Must be
                        > 0.  Fractional values are supported (e.g. ``0.5``
                        for one trace every 2 seconds).
        """

        def __init__(self, per_second: float) -> None:
            if per_second <= 0:
                raise ValueError(f"per_second must be > 0, got {per_second!r}")
            self._per_second = float(per_second)
            self._tokens: float = per_second  # start full
            self._last_refill: float = time.monotonic()
            self._lock = threading.Lock()

        @property
        def per_second(self) -> float:
            """The maximum number of sampled traces per second."""
            return self._per_second

        def _refill(self, now: float) -> None:
            """Add tokens proportional to elapsed time (call under lock)."""
            elapsed = now - self._last_refill
            self._tokens = min(
                self._per_second,
                self._tokens + elapsed * self._per_second,
            )
            self._last_refill = now

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes: Any = None,
            links: Any = None,
            trace_state: Any = None,
        ) -> SamplingResult:
            """Sample while a token is available; drop once the budget is spent.

            Args:
                parent_context: Context of the parent span, when there is one.
                trace_id: Id of the trace the span belongs to.
                name: The span's name.
                kind: The span kind OpenTelemetry reported.
                attributes: Attributes the span was started with.
                links: Links to other spans.
                trace_state: Vendor state carried on the trace.

            Returns:
                The sampling decision, with the attributes and trace state to keep.
            """
            now = time.monotonic()
            with self._lock:
                self._refill(now)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    decision = Decision.RECORD_AND_SAMPLE
                else:
                    decision = Decision.DROP
            return SamplingResult(
                decision=decision,
                attributes=attributes,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            """Return the sampler's description string."""
            return f"RateLimitedSampler({self._per_second}/s)"

    class ParentBasedSampler(Sampler):
        """
        Honour the parent span's sampling decision.

        - If there is **no parent** (root span), delegate to *root*.
        - If the parent was **sampled**, sample this span too.
        - If the parent was **not sampled**, drop.

        This mirrors the behaviour of the SDK's built-in ``ParentBased``
        but wraps our custom root samplers.

        Args:
            root: Sampler to use for root spans (no parent trace).
        """

        def __init__(self, root: Sampler) -> None:
            self._root = root
            # Use SDK's ParentBased to correctly read parent context
            self._sdk_parent_based = ParentBased(root=root)

        @property
        def root(self) -> Sampler:
            """The sampler used for root spans."""
            return self._root

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes: Any = None,
            links: Any = None,
            trace_state: Any = None,
        ) -> SamplingResult:
            """Follow the parent span's decision (the root sampler for roots).

            Args:
                parent_context: Context of the parent span, whose decision is followed.
                trace_id: Id of the trace the span belongs to.
                name: The span's name.
                kind: The span kind OpenTelemetry reported.
                attributes: Attributes the span was started with.
                links: Links to other spans.
                trace_state: Vendor state carried on the trace.

            Returns:
                The sampling decision, with the attributes and trace state to keep.
            """
            return self._sdk_parent_based.should_sample(
                parent_context,
                trace_id,
                name,
                kind=kind,
                attributes=attributes,
                links=links,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            """Return the sampler's description string."""
            return f"ParentBasedSampler(root={self._root.get_description()})"

else:
    # Stub classes when OTel is not installed

    class AlwaysOnSampler:  # type: ignore[no-redef]
        """No-op stand-in when OpenTelemetry is not installed."""

        def get_description(self) -> str:
            """Return the sampler's description string."""
            return "AlwaysOnSampler(no-op)"

    class AlwaysOffSampler:  # type: ignore[no-redef]
        """No-op stand-in when OpenTelemetry is not installed."""

        def get_description(self) -> str:
            """Return the sampler's description string."""
            return "AlwaysOffSampler(no-op)"

    class TraceIdRatioSampler:  # type: ignore[no-redef]
        """No-op stand-in when OpenTelemetry is not installed."""

        def __init__(self, ratio: float) -> None:
            self._ratio = ratio

        @property
        def ratio(self) -> float:
            """The configured sampling ratio in [0.0, 1.0]."""
            return self._ratio

        def get_description(self) -> str:
            """Return the sampler's description string."""
            return f"TraceIdRatioSampler({self._ratio:.4f})"

    class RateLimitedSampler:  # type: ignore[no-redef]
        """No-op stand-in when OpenTelemetry is not installed."""

        def __init__(self, per_second: float) -> None:
            self._per_second = per_second

        @property
        def per_second(self) -> float:
            """The maximum number of sampled traces per second."""
            return self._per_second

        def get_description(self) -> str:
            """Return the sampler's description string."""
            return f"RateLimitedSampler({self._per_second}/s)"

    class ParentBasedSampler:  # type: ignore[no-redef]
        """No-op stand-in when OpenTelemetry is not installed."""

        def __init__(self, root: Any) -> None:
            self._root = root

        @property
        def root(self) -> Any:
            """The sampler used for root spans."""
            return self._root

        def get_description(self) -> str:
            """Return the sampler's description string."""
            return f"ParentBasedSampler(root={self._root.get_description()})"
