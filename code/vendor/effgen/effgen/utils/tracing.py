"""
OpenTelemetry tracing integration for effGen.

.. note::
   ``effgen.observability.tracing`` is the canonical tracing home and what the
   provider adapters use (OTel samplers + ``start_agent_run`` / ``start_model_call``
   / ``start_tool_call`` span context managers). This module provides the older
   ``setup_tracing`` / ``get_tracer`` helpers and stays available for backwards
   compatibility; prefer ``effgen.observability`` in new code.

Provides distributed tracing with span hierarchy for agent execution,
tool calls, model inference, and ReAct iterations. Uses no-op tracer
if OpenTelemetry is not installed.

Span hierarchy:
    agent.run -> agent.iterate -> tool.execute -> model.generate

Configuration via environment variables:
    OTEL_EXPORTER_TYPE: "otlp" | "jaeger" | "zipkin" | "console" (default: "otlp")
    OTEL_EXPORTER_ENDPOINT: Exporter endpoint URL
    OTEL_SERVICE_NAME: Service name (default: "effgen")

Usage:
    from effgen.utils.tracing import get_tracer, setup_tracing
    setup_tracing()  # Configure once at startup
    tracer = get_tracer()
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# --- Optional OpenTelemetry imports ---
try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.trace import StatusCode
    from opentelemetry.trace.propagation import get_current_span

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Try optional exporters — each may or may not be installed
_JAEGER_AVAILABLE = False
_ZIPKIN_AVAILABLE = False
_OTLP_AVAILABLE = False

if OTEL_AVAILABLE:
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # noqa: F401
        _JAEGER_AVAILABLE = True
    except ImportError:  # optional Jaeger exporter not installed
        pass
    try:
        from opentelemetry.exporter.zipkin.json import ZipkinExporter  # noqa: F401
        _ZIPKIN_AVAILABLE = True
    except ImportError:  # optional Zipkin exporter not installed
        pass
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,  # noqa: F401
        )
        _OTLP_AVAILABLE = True
    except ImportError:  # optional OTLP exporter not installed
        pass


# ---------------------------------------------------------------------------
# No-op implementations (used when OTel is not installed)
# ---------------------------------------------------------------------------

class NoOpSpan:
    """No-op span when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """No-op."""

    def set_status(self, status: Any, description: str | None = None) -> None:
        """No-op."""

    def record_exception(self, exception: Exception) -> None:
        """No-op."""

    def update_name(self, name: str) -> None:
        """No-op."""

    @property
    def is_recording(self) -> bool:
        """Always ``False``: nothing is recorded."""
        return False

    def end(self) -> None:
        """No-op."""

    def __enter__(self) -> NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class NoOpTracer:
    """No-op tracer when OpenTelemetry is not installed."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> NoOpSpan:
        """Return a no-op span."""
        return NoOpSpan()

    @contextmanager
    def start_span(self, name: str, **kwargs: Any) -> Iterator[NoOpSpan]:
        """Yield a no-op span."""
        yield NoOpSpan()


class NoOpContext:
    """No-op context for trace propagation when OTel is not installed."""
    pass


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_tracer_provider: Any = None
_initialized: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tracer(name: str = "effgen") -> Any:
    """
    Get an OpenTelemetry tracer, or a no-op tracer if OTEL is not installed.

    Args:
        name: Tracer name (default: "effgen")

    Returns:
        Tracer instance (real or no-op)
    """
    if OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return NoOpTracer()


def setup_tracing(
    service_name: str | None = None,
    exporter_type: str | None = None,
    endpoint: str | None = None,
    export_to_console: bool = False,
) -> None:
    """
    Set up OpenTelemetry tracing for effGen.

    Configuration can be passed explicitly or via environment variables:
        OTEL_SERVICE_NAME, OTEL_EXPORTER_TYPE, OTEL_EXPORTER_ENDPOINT

    Args:
        service_name: Service name for traces (default: env or "effgen")
        exporter_type: "otlp", "jaeger", "zipkin", "console" (default: env or "otlp")
        endpoint: Exporter endpoint URL (default: env or exporter default)
        export_to_console: If True, also export spans to console
    """
    global _tracer_provider, _initialized

    if not OTEL_AVAILABLE:
        logger.info(
            "OpenTelemetry not installed. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk"
        )
        return

    if _initialized:
        logger.debug("Tracing already initialized, skipping")
        return

    service_name = service_name or os.environ.get("OTEL_SERVICE_NAME", "effgen")
    exporter_type = exporter_type or os.environ.get("OTEL_EXPORTER_TYPE", "otlp")
    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_ENDPOINT")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Add configured exporter
    _add_exporter(provider, exporter_type, endpoint)

    # Optionally also export to console
    if export_to_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    _initialized = True
    logger.info(
        f"OpenTelemetry tracing initialized: service={service_name}, "
        f"exporter={exporter_type}"
    )


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider."""
    global _tracer_provider, _initialized
    if _tracer_provider is not None and hasattr(_tracer_provider, "shutdown"):
        _tracer_provider.shutdown()
    _tracer_provider = None
    _initialized = False


# ---------------------------------------------------------------------------
# Span helpers — create child spans with standard effGen attributes
# ---------------------------------------------------------------------------

def trace_agent_run(agent_name: str, task: str, run_id: str | None = None) -> Any:
    """Create a span for an agent run.

    Returns a context-manager span.

    Args:
        agent_name: The agent the span is opened for.
        task: The task the run is working on.
        run_id: Id correlating the span with the run, generated when absent.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        "effgen.agent.name": agent_name,
        "effgen.agent.task": task[:500],
    }
    if run_id:
        attrs["effgen.run_id"] = run_id
    return tracer.start_as_current_span("agent.run", attributes=attrs)


def trace_agent_iterate(agent_name: str, iteration: int) -> Any:
    """Create a child span for a ReAct iteration."""
    tracer = get_tracer()
    return tracer.start_as_current_span(
        "agent.iterate",
        attributes={
            "effgen.agent.name": agent_name,
            "effgen.iteration": iteration,
        },
    )


def trace_tool_execute(tool_name: str, tool_input: str) -> Any:
    """Create a child span for a tool execution."""
    tracer = get_tracer()
    return tracer.start_as_current_span(
        "tool.execute",
        attributes={
            "effgen.tool.name": tool_name,
            "effgen.tool.input": tool_input[:500],
        },
    )


def trace_model_generate(model_name: str, prompt_tokens: int = 0) -> Any:
    """Create a child span for model generation."""
    tracer = get_tracer()
    attrs: dict[str, Any] = {"effgen.model.name": model_name}
    if prompt_tokens:
        attrs["effgen.model.prompt_tokens"] = prompt_tokens
    return tracer.start_as_current_span("model.generate", attributes=attrs)


# ---------------------------------------------------------------------------
# Cross-agent trace propagation
# ---------------------------------------------------------------------------

def get_trace_context() -> Any:
    """
    Get the current trace context for propagation to sub-agents.

    Returns an opaque context object (or NoOpContext if OTel is not installed).
    """
    if OTEL_AVAILABLE:
        return otel_context.get_current()
    return NoOpContext()


def attach_trace_context(ctx: Any) -> Any:
    """Attach a parent trace context so child spans link to the parent trace.

    Args:
        ctx: Context object from get_trace_context()

    Returns:
        A token that can be used to detach the context later.
    """
    if OTEL_AVAILABLE and isinstance(ctx, Context):
        return otel_context.attach(ctx)
    return None


def detach_trace_context(token: Any) -> None:
    """Detach a previously attached trace context."""
    if OTEL_AVAILABLE and token is not None:
        otel_context.detach(token)


def set_span_error(exception: Exception) -> None:
    """Mark the current span as errored with the given exception."""
    if OTEL_AVAILABLE:
        span = get_current_span()
        if span and span.is_recording():
            span.set_status(StatusCode.ERROR, str(exception))
            span.record_exception(exception)


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current active span."""
    if OTEL_AVAILABLE:
        span = get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, value)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_exporter(
    provider: Any,
    exporter_type: str,
    endpoint: str | None,
) -> None:
    """Add the appropriate span exporter to the provider."""
    exporter_type = exporter_type.lower()

    if exporter_type == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        return

    if exporter_type == "jaeger":
        if not _JAEGER_AVAILABLE:
            logger.warning(
                "Jaeger exporter not installed. "
                "Install with: pip install opentelemetry-exporter-jaeger"
            )
            return
        kwargs = {}
        if endpoint:
            # Parse host:port from endpoint
            parts = endpoint.replace("http://", "").replace("https://", "").split(":")
            kwargs["agent_host_name"] = parts[0]
            if len(parts) > 1:
                kwargs["agent_port"] = int(parts[1].split("/")[0])
        exporter = JaegerExporter(**kwargs)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        return

    if exporter_type == "zipkin":
        if not _ZIPKIN_AVAILABLE:
            logger.warning(
                "Zipkin exporter not installed. "
                "Install with: pip install opentelemetry-exporter-zipkin"
            )
            return
        kwargs = {}
        if endpoint:
            kwargs["endpoint"] = endpoint
        exporter = ZipkinExporter(**kwargs)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        return

    # Default: OTLP
    if not _OTLP_AVAILABLE:
        logger.warning(
            "OTLP exporter not installed. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-grpc. "
            "Falling back to console exporter."
        )
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        return
    kwargs = {}
    if endpoint:
        kwargs["endpoint"] = endpoint
    exporter = OTLPSpanExporter(**kwargs)
    provider.add_span_processor(BatchSpanProcessor(exporter))
