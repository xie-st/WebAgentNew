"""Tracer-provider lifecycle for effGen tracing.

Owns the process-wide OTel ``TracerProvider`` and the no-op stand-ins used when
the SDK is absent: ``setup_tracing`` registers a provider (with a sampler and an
optional exporter), ``get_tracer`` hands out tracers, ``shutdown_tracing``
flushes and clears it, and ``reset_tracing`` additionally clears the SDK's
set-once lock so a fresh provider can be registered.

``_provider`` and ``_initialized`` are module globals that these functions
rebind. Read them from this module: :mod:`effgen.observability.tracing` forwards
both by attribute lookup rather than binding a copy, so a reader there sees the
live values.

Import the public names from :mod:`effgen.observability.tracing`; this module is
the implementation.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from effgen.observability.tracing_otel import (
    _OTEL_AVAILABLE,
    Resource,
    SimpleSpanProcessor,
    TracerProvider,
    _otel_trace,
)
from effgen.observability.tracing_samplers import ParentBasedSampler, TraceIdRatioSampler

# Pinned rather than derived from ``__name__``: the tracing layer logs under one
# name whichever module the log site lives in.
logger = logging.getLogger("effgen.observability.tracing")


# ---------------------------------------------------------------------------
# Global provider state
# ---------------------------------------------------------------------------

_provider: Any = None
_initialized: bool = False
_provider_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------


def setup_tracing(
    service_name: str | None = None,
    sampler: Any | None = None,
    exporter: Any | None = None,
    export_to_console: bool = False,
) -> None:
    """
    Initialise the OTel ``TracerProvider`` for effGen.

    Safe to call more than once — only the first call takes effect.

    Args:
        service_name:   OTel ``service.name`` resource attribute
                        (default: env ``OTEL_SERVICE_NAME`` or ``"effgen"``).
        sampler:        A sampler instance (default:
                        ``ParentBasedSampler(TraceIdRatioSampler(1.0))``).
        exporter:       A :class:`SpanExporter` to send spans to
                        (default: no-op, i.e. spans are recorded in-memory
                        only when an ``InMemorySpanExporter`` is passed via
                        tests).
        export_to_console: Also add a ``ConsoleSpanExporter`` processor.

    Notes:
        Telemetry is non-blocking — a failure here is logged but never raised.
    """
    global _provider, _initialized

    if not _OTEL_AVAILABLE:
        logger.debug(
            "opentelemetry-sdk not installed — tracing disabled. "
            "Install with: pip install opentelemetry-sdk"
        )
        return

    with _provider_lock:
        if _initialized:
            return

        try:
            svc = service_name or os.environ.get("OTEL_SERVICE_NAME", "effgen")
            resource = Resource.create({"service.name": svc})

            _sampler = sampler or ParentBasedSampler(TraceIdRatioSampler(1.0))

            provider = TracerProvider(resource=resource, sampler=_sampler)

            if exporter is not None:
                provider.add_span_processor(SimpleSpanProcessor(exporter))

            if export_to_console:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

            _otel_trace.set_tracer_provider(provider)
            _provider = provider
            _initialized = True
            logger.debug(
                "effGen tracing initialised: service=%s sampler=%s",
                svc,
                _sampler.get_description(),
            )
        except Exception as exc:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
            logger.warning("Failed to initialise tracing (non-fatal): %s", exc)


def get_tracer(name: str = "effgen") -> Any:
    """
    Return the OTel tracer for *name*.

    If the SDK is not installed or ``setup_tracing`` has not been called, a
    no-op context manager is returned so callers do not need guard clauses.
    """
    if _OTEL_AVAILABLE:
        return _otel_trace.get_tracer(name)
    return _NoOpTracer()


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the provider."""
    global _provider, _initialized
    with _provider_lock:
        if _provider is not None and hasattr(_provider, "shutdown"):
            try:
                _provider.shutdown()
            except Exception as exc:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
                logger.warning("Error shutting down tracing (non-fatal): %s", exc)
        _provider = None
        _initialized = False


def reset_tracing() -> None:
    """
    Reset global tracing state (for use in tests between test cases).

    Shuts down any existing provider and clears the OTel global provider lock
    so that the next ``setup_tracing`` call can register a fresh provider.

    .. warning::
        This function is intended for test isolation only.  It uses an
        internal OTel API to reset the "set-once" provider lock.  Do not
        call it in production code.
    """
    global _provider, _initialized
    with _provider_lock:
        if _provider is not None and hasattr(_provider, "shutdown"):
            try:
                _provider.shutdown()
            except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
                pass
        _provider = None
        _initialized = False
    if _OTEL_AVAILABLE:
        try:
            # Force-reset the OTel SDK's "set-once" lock so the next call to
            # setup_tracing() can register a fresh TracerProvider.
            # This is the idiomatic approach for OTel test isolation.
            _otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
            _otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
            pass


# ---------------------------------------------------------------------------
# No-op fallback objects (when OTel not installed)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """No-op span context manager."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass

    def set_status(self, status: Any, description: str | None = None) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    @property
    def is_recording(self) -> bool:
        return False

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    @contextmanager
    def start_span(self, name: str, **kwargs: Any) -> Generator[_NoOpSpan]:
        yield _NoOpSpan()
