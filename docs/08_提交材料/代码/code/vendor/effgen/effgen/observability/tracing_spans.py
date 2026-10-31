"""Span construction for effGen tracing.

The ``start_*`` context managers every hot path opens — an agent run, a ReAct
iteration, a model call, a tool call and a routing decision — plus the helpers
that stamp an outcome onto a span (``set_span_ok``, ``set_span_error``,
``set_span_attribute``, ``record_retry_attempt``).

Each helper opens the OTel span, sets the seed attributes named in
:mod:`effgen.observability.spans`, and — for the three that time work — appends
a record to the span buffer on the way out. Every one of them swallows its own
exceptions: telemetry never propagates a failure to inference.

Not to be confused with :mod:`effgen.observability.spans`, which holds the
attribute-name and span-name constants; this module holds the context managers
that set them.

Import the public names from :mod:`effgen.observability.tracing`; this module is
the implementation.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from effgen.observability.spans import (
    AgentAttrs,
    ExecutionAttrs,
    ModelAttrs,
    RetryAttrs,
    RouterAttrs,
    SpanName,
    ToolAttrs,
)
from effgen.observability.tracing_buffer import (
    _RUN_CONTEXT,
    _buffer_span,
    _outcome_scope,
    current_execution,
)
from effgen.observability.tracing_otel import _OTEL_AVAILABLE, StatusCode, _get_current_span
from effgen.observability.tracing_provider import get_tracer

# Pinned rather than derived from ``__name__``: the tracing layer logs under one
# name whichever module the log site lives in.
logger = logging.getLogger("effgen.observability.tracing")


# ---------------------------------------------------------------------------
# Span helper context managers
# ---------------------------------------------------------------------------


@contextmanager
def start_agent_run(
    preset: str,
    task: str = "",
    run_id: str | None = None,
) -> Generator[Any]:
    """
    Context manager that opens an ``effgen.agent.run`` span.

    Args:
        preset:  Agent name / preset identifier.
        task:    Task text (truncated to 500 chars).
        run_id:  Optional run UUID.

    Yields:
        The active span (real or no-op).
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        AgentAttrs.PRESET: preset,
        AgentAttrs.TASK: task[:500],
    }
    if run_id:
        attrs[AgentAttrs.RUN_ID] = run_id
    execution = current_execution()
    if execution and execution.get("execution_id"):
        attrs[ExecutionAttrs.ID] = execution["execution_id"]
        for attr_key, ctx_key in (
            (ExecutionAttrs.KIND, "execution_kind"),
            (ExecutionAttrs.NAME, "execution_name"),
            (ExecutionAttrs.PARENT_AGENT, "parent_agent"),
            (ExecutionAttrs.ROLE, "role"),
        ):
            if execution.get(ctx_key):
                attrs[attr_key] = execution[ctx_key]
    start = time.monotonic()
    # Correlate every span nested in this run so the dashboard can group them.
    _ctx_token = _RUN_CONTEXT.set({"run_id": run_id or uuid.uuid4().hex[:12], "start": start})
    try:
        with tracer.start_as_current_span(SpanName.AGENT_RUN, attributes=attrs) as span, \
                _outcome_scope("agent") as outcome:
            try:
                yield span
                if outcome["error"]:
                    _set_safe(span, AgentAttrs.ERROR, outcome["error"])
                _buffer_span(
                    f"{SpanName.AGENT_RUN} {preset}",
                    (time.monotonic() - start) * 1000,
                    error=outcome["error"],
                    start_monotonic=start,
                    kind="agent",
                    agent=preset,
                )
            except Exception as exc:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
                _mark_error(span, exc)
                _buffer_span(
                    f"{SpanName.AGENT_RUN} {preset}",
                    (time.monotonic() - start) * 1000,
                    error=str(exc),
                    start_monotonic=start,
                    kind="agent",
                    agent=preset,
                )
                raise
    except Exception:
        raise
    finally:
        _RUN_CONTEXT.reset(_ctx_token)


@contextmanager
def start_agent_iteration(
    preset: str,
    iteration: int,
) -> Generator[Any]:
    """
    Context manager for a single ReAct iteration span.

    Args:
        preset:    Agent name / preset.
        iteration: 1-based iteration counter.

    Yields:
        The active span.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        AgentAttrs.PRESET: preset,
        AgentAttrs.ITERATION: iteration,
    }
    try:
        with tracer.start_as_current_span(SpanName.AGENT_ITERATION, attributes=attrs) as span:
            try:
                yield span
            except Exception as exc:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
                _mark_error(span, exc)
                raise
    except Exception:
        raise


@contextmanager
def start_model_call(
    provider: str,
    model: str,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    parts_count: int | None = None,
) -> Generator[Any]:
    """
    Context manager for a model inference span.

    Seed attributes (tokens, cost, outcome, latency) should be set on the
    span after the call completes::

        with start_model_call("openai", "gpt-4o-mini") as span:
            result = client.generate(...)
            span.set_attribute(ModelAttrs.INPUT_TOKENS, result.prompt_tokens)
            span.set_attribute(ModelAttrs.OUTPUT_TOKENS, result.completion_tokens)
            span.set_attribute(ModelAttrs.OUTCOME, "ok")

    Args:
        provider:         Provider name.
        model:            Model identifier.
        reasoning_effort: Optional effort level for reasoning models.
        thinking_budget:  Optional token budget for reasoning.
        parts_count:      Number of multimodal parts (if applicable).

    Yields:
        The active span.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        ModelAttrs.PROVIDER: provider,
        ModelAttrs.NAME: model,
    }
    if reasoning_effort is not None:
        attrs[ModelAttrs.REASONING_EFFORT] = reasoning_effort
    if thinking_budget is not None:
        attrs[ModelAttrs.THINKING_BUDGET] = thinking_budget
    if parts_count is not None:
        attrs[ModelAttrs.PARTS_COUNT] = parts_count

    # Model ids often already carry a ``provider:`` prefix (e.g. the caller
    # passes provider="cerebras", model="cerebras:llama3.1-8b"). Avoid a
    # doubled "cerebras:cerebras:..." label in the dashboard span stream.
    _model_label = model if model.startswith(f"{provider}:") else f"{provider}:{model}"

    start = time.monotonic()
    try:
        with tracer.start_as_current_span(SpanName.MODEL_CALL, attributes=attrs) as span, \
                _outcome_scope("model") as outcome:
            try:
                yield span
                # Set latency on the way out (only if span is still recording)
                _set_safe(span, ModelAttrs.LATENCY_MS, round((time.monotonic() - start) * 1000, 1))
                if outcome["error"]:
                    _set_safe(span, ModelAttrs.OUTCOME, "error")
                    _mark_error_message(span, outcome["error"])
                else:
                    if not _has_outcome(span):
                        _set_safe(span, ModelAttrs.OUTCOME, "ok")
                    _mark_ok(span)
                _buffer_span(
                    f"{SpanName.MODEL_CALL} {_model_label}",
                    (time.monotonic() - start) * 1000,
                    error=outcome["error"],
                    start_monotonic=start,
                    kind="model",
                    model=_model_label,
                )
            except Exception as exc:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
                _set_safe(span, ModelAttrs.OUTCOME, "error")
                _set_safe(span, ModelAttrs.LATENCY_MS, round((time.monotonic() - start) * 1000, 1))
                _mark_error(span, exc)
                _buffer_span(
                    f"{SpanName.MODEL_CALL} {_model_label}",
                    (time.monotonic() - start) * 1000,
                    error=str(exc),
                    start_monotonic=start,
                    kind="model",
                    model=_model_label,
                )
                raise
    except Exception:
        raise


@contextmanager
def start_tool_call(
    tool_name: str,
    tool_input: str = "",
) -> Generator[Any]:
    """
    Context manager for a tool execution span.

    Args:
        tool_name:  Tool identifier (e.g. ``"calculator"``).
        tool_input: Serialised input arguments (truncated to 500 chars).

    Yields:
        The active span.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        ToolAttrs.NAME: tool_name,
        ToolAttrs.INPUT: str(tool_input)[:500],
    }
    start = time.monotonic()
    try:
        with tracer.start_as_current_span(SpanName.TOOL_CALL, attributes=attrs) as span, \
                _outcome_scope("tool") as outcome:
            try:
                yield span
                _set_safe(span, ToolAttrs.LATENCY_MS, round((time.monotonic() - start) * 1000, 1))
                if outcome["error"]:
                    _set_safe(span, ToolAttrs.STATUS, "error")
                    _mark_error_message(span, outcome["error"])
                else:
                    if not _has_tool_status(span):
                        _set_safe(span, ToolAttrs.STATUS, "ok")
                    _mark_ok(span)
                _buffer_span(
                    f"{SpanName.TOOL_CALL} {tool_name}",
                    (time.monotonic() - start) * 1000,
                    error=outcome["error"],
                    start_monotonic=start,
                    kind="tool",
                    tool=tool_name,
                )
            except Exception as exc:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
                _set_safe(span, ToolAttrs.STATUS, "error")
                _set_safe(span, ToolAttrs.LATENCY_MS, round((time.monotonic() - start) * 1000, 1))
                _mark_error(span, exc)
                _buffer_span(
                    f"{SpanName.TOOL_CALL} {tool_name}",
                    (time.monotonic() - start) * 1000,
                    error=str(exc),
                    start_monotonic=start,
                    kind="tool",
                    tool=tool_name,
                )
                raise
    except Exception:
        raise


@contextmanager
def start_router_decision(
    policy: str,
    candidates: list[Any] | None = None,
) -> Generator[Any]:
    """
    Context manager for a routing decision span.

    Args:
        policy:     Routing policy name.
        candidates: List of candidate (provider, model) pairs being considered.

    Yields:
        The active span.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        RouterAttrs.POLICY: policy,
    }
    if candidates:
        attrs[RouterAttrs.CONSIDERED] = ", ".join(
            f"{c[0]}/{c[1]}" if isinstance(c, list | tuple) else str(c)
            for c in candidates
        )
    try:
        with tracer.start_as_current_span(SpanName.ROUTER_DECISION, attributes=attrs) as span:
            try:
                yield span
                _mark_ok(span)
            except Exception as exc:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
                _mark_error(span, exc)
                raise
    except Exception:
        raise


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def record_retry_attempt(
    span: Any,
    attempt: int,
    reason: str,
    delay_s: float = 0.0,
) -> None:
    """
    Add a ``effgen.retry.attempt`` event to *span*.

    Non-blocking: exceptions are swallowed.

    Args:
        span:     The span to attach the event to (may be a no-op span).
        attempt:  1-based attempt number.
        reason:   Short description of why retry was triggered.
        delay_s:  Delay before this attempt in seconds.
    """
    try:
        if span is not None and hasattr(span, "add_event"):
            span.add_event(
                SpanName.RETRY_ATTEMPT,
                attributes={
                    RetryAttrs.ATTEMPT: attempt,
                    RetryAttrs.REASON: reason,
                    RetryAttrs.DELAY_S: float(delay_s),
                },
            )
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass


def set_span_ok(span: Any) -> None:
    """Mark *span* as OK.  Non-blocking."""
    _mark_ok(span)


def set_span_error(span_or_exc: Any, exc: Exception | None = None) -> None:
    """
    Mark the given span (or the current active span) as errored.

    Can be called as:
    - ``set_span_error(span, exc)``
    - ``set_span_error(exc)``  ← marks current active span
    """
    if not _OTEL_AVAILABLE:
        return
    try:
        if exc is None:
            # Called as set_span_error(exc)
            actual_exc: Exception = span_or_exc  # type: ignore[assignment]
            span = _get_current_span()
        else:
            span = span_or_exc
            actual_exc = exc
        if span is not None and hasattr(span, "is_recording") and span.is_recording():
            span.set_status(StatusCode.ERROR, str(actual_exc))
            span.record_exception(actual_exc)
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass


def stamp_call_cost(span: Any, metadata: dict[str, Any] | None) -> None:
    """Put one model call's cost on *span*, from the result's own metadata.

    The adapters stamp this too, but through :func:`set_span_attribute`, which
    writes to whatever span is *currently active* — so the attribute landed only
    when the adapter's call ran inside the span the agent had opened. On the
    other paths the span carried the provider, the model and both token counts
    and no money, which reads as a pricing gap rather than a telemetry one.

    Called with the span in hand, this does not depend on context propagation.
    Writing the same value twice is harmless; an unpriced model reports ``None``
    and no attribute is written, keeping "free" distinct from "no price".

    Args:
        span: The model-call span to annotate.
        metadata: The metadata the call returned; ``cost_usd`` is read from it.
    """
    cost = (metadata or {}).get("cost_usd")
    if cost is None:
        return
    try:
        span.set_attribute(ModelAttrs.COST_USD, float(cost))
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort
        pass


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the currently active span.  Non-blocking."""
    if not _OTEL_AVAILABLE:
        return
    try:
        span = _get_current_span()
        if span is not None and hasattr(span, "is_recording") and span.is_recording():
            span.set_attribute(key, value)
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _set_safe(span: Any, key: str, value: Any) -> None:
    """Set span attribute without raising."""
    try:
        if span is not None and hasattr(span, "is_recording") and span.is_recording():
            span.set_attribute(key, value)
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass


def _mark_ok(span: Any) -> None:
    """Set span status to OK without raising."""
    try:
        if _OTEL_AVAILABLE and span is not None and hasattr(span, "set_status"):
            span.set_status(StatusCode.OK)
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass


def _mark_error_message(span: Any, message: str) -> None:
    """Set span status to ERROR from a message (no exception object)."""
    try:
        if _OTEL_AVAILABLE and span is not None and hasattr(span, "set_status"):
            span.set_status(StatusCode.ERROR, str(message))
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass


def _mark_error(span: Any, exc: Exception) -> None:
    """Set span status to ERROR and record exception without raising."""
    try:
        if _OTEL_AVAILABLE and span is not None and hasattr(span, "set_status"):
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass


def _has_outcome(span: Any) -> bool:
    """Return True if ModelAttrs.OUTCOME is already set on *span*."""
    try:
        if hasattr(span, "attributes") and span.attributes:
            return ModelAttrs.OUTCOME in span.attributes
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass
    return False


def _has_tool_status(span: Any) -> bool:
    """Return True if ToolAttrs.STATUS is already set on *span*."""
    try:
        if hasattr(span, "attributes") and span.attributes:
            return ToolAttrs.STATUS in span.attributes
    except Exception:  # noqa: BLE001 - OTel telemetry is best-effort; never break the caller
        pass
    return False
