"""The in-memory span stream and the execution context behind it.

Two pieces of state live here, and both are read by more than one module:

- the bounded ring buffer of span records the dashboard reads
  (``_SPAN_BUFFER`` / ``get_recent_spans``), and
- the context variables that group those records — the current agent run, the
  current team/workflow execution, and the stack of in-flight span outcomes.

Every object in this module is shared by identity: the tracing facade
re-exports these names, so a caller clearing ``tracing._SPAN_BUFFER`` clears the
one buffer that exists, not a copy of it.

Import the public names from :mod:`effgen.observability.tracing`; this module is
the implementation.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from effgen.observability.spans import SpanName

# Pinned rather than derived from ``__name__``: the tracing layer logs under one
# name whichever module the log site lives in.
logger = logging.getLogger("effgen.observability.tracing")

# The run a buffered span belongs to. Set for the duration of an agent run so
# nested model-call/tool-call spans can be grouped into a per-run timeline.
_RUN_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "effgen_span_run_context", default=None
)

# The team/workflow execution the current work belongs to. Set by the
# orchestrator and the workflow runner around each sub-agent call so every span
# and every stored run record of one execution shares an id.
_EXECUTION_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "effgen_execution_context", default=None
)

# Stack of in-flight buffered spans, innermost last. A span's outcome can be
# recorded on it while it is open (for work that reports failure by returning a
# result rather than raising).
_SPAN_OUTCOMES: contextvars.ContextVar[tuple[dict[str, Any], ...]] = contextvars.ContextVar(
    "effgen_span_outcomes", default=()
)


# ---------------------------------------------------------------------------
# In-memory span ring buffer (used by the dashboard)
# ---------------------------------------------------------------------------

_SPAN_BUFFER_LOCK: threading.Lock = threading.Lock()
_SPAN_BUFFER: deque[dict] = deque(maxlen=500)  # type: ignore[type-arg]


def new_execution_id() -> str:
    """Return a fresh identifier for one team or workflow execution."""
    return uuid.uuid4().hex[:12]


def current_execution() -> dict[str, Any] | None:
    """Return the team/workflow execution the caller is running inside.

    The dict carries ``execution_id``, ``execution_kind`` (``"team"`` or
    ``"workflow"``), ``execution_name``, ``parent_agent`` and ``role``.
    Returns ``None`` outside any execution.
    """
    ctx = _EXECUTION_CONTEXT.get()
    return dict(ctx) if ctx else None


@contextmanager
def execution_scope(
    *,
    kind: str | None = None,
    name: str | None = None,
    execution_id: str | None = None,
    parent_agent: str | None = None,
    role: str | None = None,
) -> Generator[dict[str, Any]]:
    """Tag everything run inside the block as part of one multi-agent execution.

    Spans buffered inside the block, and run records written from it, carry the
    execution id, kind, name, delegating agent and role. Nesting inherits: an
    inner scope that omits ``execution_id`` keeps the outer id (and its kind and
    name), so a per-agent scope adds ``role``/``parent_agent`` without starting a
    new execution.

    Args:
        kind: What issued the execution — ``"team"`` or ``"workflow"``. Applied
            only when this scope starts the execution.
        name: Team or workflow name. Applied only when this scope starts the
            execution.
        execution_id: Reuse an existing id; a new one is issued when the scope
            is outermost and none is given.
        parent_agent: The agent that delegated this work, when one did.
        role: The role played inside the execution (``"manager"``, ``"worker"``,
            ``"node"``, …).

    Yields:
        The execution context dict that is in force inside the block.
    """
    outer = _EXECUTION_CONTEXT.get() or {}
    # ``kind`` and ``name`` identify the execution as a whole, so they are taken
    # only from the scope that starts one. A nested scope joining an execution
    # already in flight contributes its ``role``/``parent_agent`` and leaves the
    # execution's own identity alone.
    starts_execution = execution_id is not None or not outer.get("execution_id")
    ctx: dict[str, Any] = {
        "execution_id": execution_id or outer.get("execution_id") or new_execution_id(),
        "execution_kind": (kind or outer.get("execution_kind")) if starts_execution
        else outer.get("execution_kind"),
        "execution_name": (name or outer.get("execution_name")) if starts_execution
        else outer.get("execution_name"),
        "parent_agent": parent_agent if parent_agent is not None else outer.get("parent_agent"),
        "role": role if role is not None else outer.get("role"),
    }
    token = _EXECUTION_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _EXECUTION_CONTEXT.reset(token)


@contextmanager
def _outcome_scope(kind: str) -> Generator[dict[str, Any]]:
    """Push a record for the span being timed so its outcome can be set."""
    scope: dict[str, Any] = {"span_kind": kind, "error": None}
    token = _SPAN_OUTCOMES.set((*_SPAN_OUTCOMES.get(), scope))
    try:
        yield scope
    finally:
        _SPAN_OUTCOMES.reset(token)


def mark_span_error(error: str) -> None:
    """Record *error* as the outcome of the innermost span being timed.

    Use this when work reports failure by returning a result instead of raising,
    so the buffered span still reflects what happened. No-op outside a span.
    """
    stack = _SPAN_OUTCOMES.get()
    if stack:
        stack[-1]["error"] = str(error)[:300]


def mark_run_error(error: str) -> None:
    """Record *error* as the outcome of the enclosing agent-run span.

    No-op outside an agent run.
    """
    for scope in _SPAN_OUTCOMES.get():
        if scope.get("span_kind") == "agent":
            scope["error"] = str(error)[:300]
            return


def record_skipped_step(name: str, *, reason: str) -> None:
    """Record a step that was not run, with the reason it was skipped.

    A skipped step times nothing, so it has no span of its own; recording it
    keeps it visible to a consumer reading the buffer, which would otherwise
    show the step as absent rather than deliberately skipped.
    """
    _buffer_span(
        f"{SpanName.AGENT_RUN} {name}",
        0.0,
        kind="agent",
        agent=name,
        status="skipped",
        note=reason,
    )


def _buffer_span(
    name: str,
    duration_ms: float,
    error: str | None = None,
    *,
    start_monotonic: float | None = None,
    kind: str | None = None,
    agent: str | None = None,
    tool: str | None = None,
    model: str | None = None,
    status: str | None = None,
    note: str | None = None,
) -> None:
    """Append a span dict to the in-memory ring buffer (best-effort, never raises).

    Alongside the display ``name``, the record carries the span's ``kind``
    (``agent`` / ``model`` / ``tool`` / ``router``) and the identity of what it
    timed (``agent``, ``tool``, ``model``) as fields, so a consumer reads a field
    instead of parsing the name.

    When called within an agent run the record also carries the run's id and the
    span's start offset (ms from the run's start), so a consumer can group spans
    by run and lay them out on a timeline. Both default to ``None``/``0`` for a
    span recorded outside any run. Inside a team or workflow execution the record
    additionally carries the execution id, kind, name, delegating agent and role.
    """
    try:
        ctx = _RUN_CONTEXT.get()
        run_id = ctx.get("run_id") if ctx else None
        offset_ms = 0.0
        if ctx and start_monotonic is not None:
            offset_ms = round(max(0.0, (start_monotonic - ctx["start"]) * 1000.0), 1)
        record = {
            "ts": time.strftime("%H:%M:%S", time.gmtime()),
            "name": name,
            "kind": kind,
            "agent": agent,
            "tool": tool,
            "model": model,
            "duration_ms": round(duration_ms, 1),
            "status": status or ("error" if error else "ok"),
            "error": error,
            "note": note,
            "run_id": run_id,
            "offset_ms": offset_ms,
        }
        record.update(current_execution() or {
            "execution_id": None,
            "execution_kind": None,
            "execution_name": None,
            "parent_agent": None,
            "role": None,
        })
        with _SPAN_BUFFER_LOCK:
            _SPAN_BUFFER.append(record)
    except Exception:  # noqa: BLE001 - telemetry must never break inference
        pass


def get_recent_spans(*, limit: int = 100) -> list[dict]:  # type: ignore[type-arg]
    """Return the most-recent *limit* span records (newest first).

    These are populated when effGen agent / model-call context managers exit.
    Returns an empty list when no spans have been buffered.
    """
    with _SPAN_BUFFER_LOCK:
        spans = list(_SPAN_BUFFER)
    return list(reversed(spans))[:limit]
