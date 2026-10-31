"""Optional OpenTelemetry imports shared by the tracing modules.

The OTel SDK is an optional dependency. This module makes the single import
attempt the tracing layer needs and publishes the outcome as
``_OTEL_AVAILABLE`` together with the SDK names its siblings use, so no other
module repeats the ``try``/``except ImportError`` block.

When the SDK is absent, ``Sampler`` falls back to ``object`` and the remaining
SDK names are bound to ``None``. Every one of them is read only under
``if _OTEL_AVAILABLE:``, so those fallbacks are never used at run time; they
exist so importing this module always yields the same set of names.

``_OTEL_AVAILABLE`` is read by value wherever it is imported: rebinding it on
one module does not change what another module does.
"""

from __future__ import annotations

import logging

# Pinned rather than derived from ``__name__``: the tracing layer logs under one
# name whichever module the log site lives in.
logger = logging.getLogger("effgen.observability.tracing")

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.context import Context
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.sampling import (
        Decision,
        ParentBased,
        Sampler,
        SamplingResult,
    )
    from opentelemetry.trace import SpanKind, StatusCode
    from opentelemetry.trace.propagation import get_current_span as _get_current_span

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    Sampler = object  # type: ignore[misc,assignment]
    Context = None  # type: ignore[misc,assignment]
    Decision = None  # type: ignore[misc,assignment]
    ParentBased = None  # type: ignore[misc,assignment]
    SamplingResult = None  # type: ignore[misc,assignment]
    Resource = None  # type: ignore[misc,assignment]
    TracerProvider = None  # type: ignore[misc,assignment]
    SimpleSpanProcessor = None  # type: ignore[misc,assignment]
    SpanKind = None  # type: ignore[misc,assignment]
    StatusCode = None  # type: ignore[misc,assignment]
    _otel_trace = None  # type: ignore[assignment]
    _get_current_span = None  # type: ignore[assignment]
