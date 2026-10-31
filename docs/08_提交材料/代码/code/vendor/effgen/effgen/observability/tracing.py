"""
OpenTelemetry tracing for effGen.

Samplers
--------
- ``AlwaysOnSampler``   — sample every span
- ``AlwaysOffSampler``  — drop every span
- ``TraceIdRatioSampler(p)``  — probabilistic, 0 ≤ p ≤ 1
- ``RateLimitedSampler(per_second)``  — token-bucket rate limiter
- ``ParentBasedSampler(root)``  — honour parent sampling decision; delegate
  root spans to *root* sampler (usually ``TraceIdRatioSampler`` or
  ``RateLimitedSampler``).

All are valid ``Sampler`` objects understood by the OTel SDK.

Configuration
-------------
Samplers are declared in ``ObservabilityConfig.tracing.sampler``.  Pass a
pre-built sampler instance::

    from effgen.observability.tracing import (
        ParentBasedSampler, TraceIdRatioSampler, setup_tracing
    )
    setup_tracing(sampler=ParentBasedSampler(TraceIdRatioSampler(0.1)))

Context managers
----------------
Every hot path has a dedicated ``start_*`` helper that returns an OTel span
and sets the correct span name + seed attributes from ``spans.py`` constants::

    from effgen.observability.tracing import (
        start_agent_run, start_agent_iteration,
        start_model_call, start_tool_call, start_router_decision
    )

    with start_agent_run(preset="default", task="...", run_id="...") as span:
        span.set_attribute(AgentAttrs.ITERATION, 1)
        with start_model_call(provider="openai", model="gpt-4o-mini") as mspan:
            ...

Retry events
------------
Call ``record_retry_attempt(span, attempt, reason, delay_s)`` to attach a
retry event to an existing span (usually the model.call span).

Telemetry is non-blocking.  Every helper silently catches and swallows its
own exceptions so a tracing failure never propagates to inference.
"""

from __future__ import annotations

import contextvars  # noqa: F401  re-exported
import logging
import os  # noqa: F401  re-exported
import threading  # noqa: F401  re-exported
import time  # noqa: F401  re-exported
import uuid  # noqa: F401  re-exported
from collections import deque  # noqa: F401  re-exported
from collections.abc import Generator  # noqa: F401  re-exported
from contextlib import contextmanager  # noqa: F401  re-exported
from typing import Any

from effgen.observability.tracing_buffer import (  # noqa: F401  re-exported
    _EXECUTION_CONTEXT,
    _RUN_CONTEXT,
    _SPAN_BUFFER,
    _SPAN_BUFFER_LOCK,
    _SPAN_OUTCOMES,
    _buffer_span,
    _outcome_scope,
    current_execution,
    execution_scope,
    get_recent_spans,
    mark_run_error,
    mark_span_error,
    new_execution_id,
    record_skipped_step,
)
from effgen.observability.tracing_otel import (  # noqa: F401  re-exported
    _OTEL_AVAILABLE,
    Context,
    Decision,
    ParentBased,
    Resource,
    Sampler,
    SamplingResult,
    SimpleSpanProcessor,
    SpanKind,
    StatusCode,
    TracerProvider,
    _get_current_span,
    _otel_trace,
)
from effgen.observability.tracing_provider import (  # noqa: F401  re-exported
    _NoOpSpan,
    _NoOpTracer,
    _provider_lock,
    get_tracer,
    reset_tracing,
    setup_tracing,
    shutdown_tracing,
)
from effgen.observability.tracing_samplers import (  # noqa: F401  re-exported
    AlwaysOffSampler,
    AlwaysOnSampler,
    ParentBasedSampler,
    RateLimitedSampler,
    TraceIdRatioSampler,
)
from effgen.observability.tracing_spans import (  # noqa: F401  re-exported
    _has_outcome,
    _has_tool_status,
    _mark_error,
    _mark_error_message,
    _mark_ok,
    _set_safe,
    record_retry_attempt,
    set_span_attribute,
    set_span_error,
    set_span_ok,
    stamp_call_cost,
    start_agent_iteration,
    start_agent_run,
    start_model_call,
    start_router_decision,
    start_tool_call,
)

from .spans import (  # noqa: F401  re-exported
    AgentAttrs,
    ExecutionAttrs,
    ModelAttrs,
    RetryAttrs,
    RouterAttrs,
    SpanName,
    ToolAttrs,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Samplers
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "RateLimitedSampler",
    "ParentBasedSampler",
    # Setup / teardown
    "setup_tracing",
    "get_tracer",
    "shutdown_tracing",
    # Span context managers
    "start_agent_run",
    "start_agent_iteration",
    "start_model_call",
    "start_tool_call",
    "start_router_decision",
    # Span helpers
    "record_retry_attempt",
    "set_span_ok",
    "set_span_error",
    "set_span_attribute",
    "stamp_call_cost",
    "mark_span_error",
    "mark_run_error",
    "record_skipped_step",
    # Multi-agent execution correlation
    "execution_scope",
    "current_execution",
    "new_execution_id",
    # Legacy compat shims (for existing agent.py / utils/tracing.py callers)
    "trace_agent_run",
    "trace_agent_iterate",
    "trace_model_generate",
    "trace_tool_execute",
]


def __getattr__(name: str) -> Any:
    """Resolve the two provider globals against the module that rebinds them.

    ``setup_tracing``, ``shutdown_tracing`` and ``reset_tracing`` rebind
    ``_provider`` and ``_initialized`` under ``global``. Binding a copy of
    either here would freeze this module's view at its import-time value while
    the real provider moved on, so both are looked up on demand instead.
    Module ``__getattr__`` runs only when normal lookup fails, which means
    neither name may be bound in this module.
    """
    if name in ("_provider", "_initialized"):
        from effgen.observability import tracing_provider

        return getattr(tracing_provider, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the forwarded provider globals alongside this module's own names."""
    return sorted([*globals(), "_provider", "_initialized"])


# ---------------------------------------------------------------------------
# Legacy compatibility shims
# (effgen.utils.tracing callers keep working without changes)
# ---------------------------------------------------------------------------


def trace_agent_run(agent_name: str, task: str, run_id: str | None = None) -> Any:
    """Legacy shim → ``start_agent_run``.  Returns a context manager.

    Args:
        agent_name: The agent the span is opened for.
        task: The task the run is working on.
        run_id: Id correlating the span with the run, generated when absent.
    """
    return start_agent_run(preset=agent_name, task=task, run_id=run_id)


def trace_agent_iterate(agent_name: str, iteration: int) -> Any:
    """Legacy shim → ``start_agent_iteration``."""
    return start_agent_iteration(preset=agent_name, iteration=iteration)


def trace_model_generate(model_name: str, prompt_tokens: int = 0) -> Any:
    """Legacy shim → ``start_model_call`` with provider inferred from model name."""
    provider = _infer_provider(model_name)
    return start_model_call(provider=provider, model=model_name)


def trace_tool_execute(tool_name: str, tool_input: str) -> Any:
    """Legacy shim → ``start_tool_call``."""
    return start_tool_call(tool_name=tool_name, tool_input=tool_input)


def _infer_provider(model_name: str) -> str:
    """Best-effort provider inference from a model name string.

    Kept in step with ``agent_runtime._infer_provider_from_model``: an explicit
    local-engine prefix settles the question before any family-name guess, so a
    run on this machine's own GPU is not attributed to a cloud provider.
    """
    m = model_name.lower()
    for engine in ("transformers", "vllm", "gguf", "mlx"):
        if m.startswith(f"{engine}:"):
            return engine
    if m.startswith(("gpt-", "o1", "o3", "o4", "text-")):
        return "openai"
    if m.startswith(("gemini", "models/gemini")):
        return "google"
    if m.startswith(("claude", "anthropic")):
        return "anthropic"
    if m.startswith(("llama", "qwen", "cerebras")):
        return "cerebras"
    if m.startswith(("mixtral", "mistral")):
        return "groq"
    return "unknown"
