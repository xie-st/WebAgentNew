"""
Structured logging for effGen framework.

.. note::
   ``effgen.observability.logs`` (``get_logger`` / ``configure_logging`` /
   ``EffGenLogger``) is the canonical structured-logging home and shares this
   module's thread-local run context. This module stays available for backwards
   compatibility; prefer ``effgen.observability`` in new code.

Provides JSON-formatted log output with consistent fields for agent execution,
tool calls, and model inference. Each Agent.run() gets a unique run_id;
multi-agent runs share a workflow_id.

Configuration:
    - JSON mode: machine-readable JSON lines (for log aggregation)
    - Human mode: coloured, human-readable output (default)

Usage:
    from effgen.utils.structured_logging import StructuredLogger, get_structured_logger

    slog = get_structured_logger("my_module")
    slog.agent_event("math_agent", "task_start", task="What is 2+2?")
    slog.tool_event("calculator", "call", input={"expression": "2+2"})

    # With run context (automatically set by Agent.run):
    from effgen.utils.structured_logging import LogRunContext
    with LogRunContext(run_id="abc-123", agent_name="math_agent"):
        slog.info("Processing task")  # run_id included automatically
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Thread-local storage for run context
_run_context = threading.local()


# ---------------------------------------------------------------------------
# Run / Workflow context management
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Generate a unique run ID."""
    return uuid.uuid4().hex[:12]


def generate_workflow_id() -> str:
    """Generate a unique workflow ID."""
    return uuid.uuid4().hex[:12]


def get_current_run_id() -> str | None:
    """Get the current run_id from thread-local context."""
    return getattr(_run_context, "run_id", None)


def get_current_workflow_id() -> str | None:
    """Get the current workflow_id from thread-local context."""
    return getattr(_run_context, "workflow_id", None)


def get_current_agent_name() -> str | None:
    """Get the current agent_name from thread-local context."""
    return getattr(_run_context, "agent_name", None)


def get_current_session_id() -> str | None:
    """Get the current session_id from thread-local context."""
    return getattr(_run_context, "session_id", None)


@contextmanager
def LogRunContext(
    run_id: str | None = None,
    workflow_id: str | None = None,
    agent_name: str | None = None,
    session_id: str | None = None,
) -> Iterator[None]:
    """Set run-level identifiers on every log entry within the scope.

    Nesting is supported — inner contexts override outer.

    Args:
        run_id: Unique ID for this Agent.run() invocation
        workflow_id: Shared ID across multi-agent workflow
        agent_name: Name of the agent
        session_id: Session identifier

    Returns:
        A context manager that carries the identifiers for its scope.
    """
    prev_run = getattr(_run_context, "run_id", None)
    prev_wf = getattr(_run_context, "workflow_id", None)
    prev_agent = getattr(_run_context, "agent_name", None)
    prev_session = getattr(_run_context, "session_id", None)

    _run_context.run_id = run_id or prev_run
    _run_context.workflow_id = workflow_id or prev_wf
    _run_context.agent_name = agent_name or prev_agent
    _run_context.session_id = session_id or prev_session

    try:
        yield
    finally:
        _run_context.run_id = prev_run
        _run_context.workflow_id = prev_wf
        _run_context.agent_name = prev_agent
        _run_context.session_id = prev_session


# ---------------------------------------------------------------------------
# JSON Formatter for stdlib logging
# ---------------------------------------------------------------------------

class EffGenJSONFormatter(logging.Formatter):
    """JSON formatter that injects effGen context fields into every log line.

    The injected fields are ``run_id``, ``workflow_id``, ``agent_name`` and
    ``session_id``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render *record* as one JSON line with the active run-context fields."""
        data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject run context
        run_id = getattr(record, "run_id", None) or get_current_run_id()
        if run_id:
            data["run_id"] = run_id
        wf_id = getattr(record, "workflow_id", None) or get_current_workflow_id()
        if wf_id:
            data["workflow_id"] = wf_id
        agent = getattr(record, "agent_name", None) or get_current_agent_name()
        if agent:
            data["agent_name"] = agent
        session = getattr(record, "session_id", None) or get_current_session_id()
        if session:
            data["session_id"] = session

        # Iteration number if present
        iteration = getattr(record, "iteration", None)
        if iteration is not None:
            data["iteration"] = iteration

        # Source location
        data["module"] = record.module
        data["function"] = record.funcName
        data["line"] = record.lineno

        # Exception info
        if record.exc_info and record.exc_info[0] is not None:
            data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # Extra fields passed via `extra=` parameter
        skip_keys = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "pathname",
            "process", "processName", "relativeCreated", "thread", "threadName",
            "exc_info", "exc_text", "stack_info", "message", "asctime",
            "run_id", "workflow_id", "agent_name", "session_id", "iteration",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip_keys and not key.startswith("_"):
                try:
                    json.dumps(value)  # only include JSON-serializable values
                    data[key] = value
                except (TypeError, ValueError):
                    data[key] = str(value)

        return json.dumps(data, default=str)


# ---------------------------------------------------------------------------
# StructuredLogger — high-level API for effGen events
# ---------------------------------------------------------------------------

class StructuredLogger:
    """
    High-level structured logger for effGen components.

    Wraps a stdlib logger and provides convenience methods for common events.
    Context fields (run_id, agent_name, etc.) are automatically injected.
    """

    def __init__(self, name: str, json_mode: bool = False) -> None:
        self._logger = logging.getLogger(name)
        self._json_mode = json_mode

    # -- Standard log levels (delegates to stdlib) --

    def debug(self, msg: str, **extra: Any) -> None:
        """Log at DEBUG with *extra* as structured fields."""
        self._log(logging.DEBUG, msg, **extra)

    def info(self, msg: str, **extra: Any) -> None:
        """Log at INFO with *extra* as structured fields."""
        self._log(logging.INFO, msg, **extra)

    def warning(self, msg: str, **extra: Any) -> None:
        """Log at WARNING with *extra* as structured fields."""
        self._log(logging.WARNING, msg, **extra)

    def error(self, msg: str, **extra: Any) -> None:
        """Log at ERROR with *extra* as structured fields."""
        self._log(logging.ERROR, msg, **extra)

    # -- Domain-specific convenience methods --

    def agent_event(
        self,
        agent_name: str,
        event: str,
        *,
        level: int = logging.INFO,
        **data: Any,
    ) -> None:
        """Log an agent-level event (task_start, task_complete, error, etc.).

        Args:
            agent_name: The agent the event belongs to.
            event: What happened, such as ``task_start`` or ``error``.
            level: Level the line is logged at.
            **data: Extra fields recorded with the entry.
        """
        self._log(
            level,
            f"[{agent_name}] {event}",
            event_type="agent",
            event=event,
            agent_name=agent_name,
            **data,
        )

    def tool_event(
        self,
        tool_name: str,
        event: str,
        *,
        level: int = logging.DEBUG,
        **data: Any,
    ) -> None:
        """Log a tool-level event (call, result, error).

        Args:
            tool_name: The tool the event belongs to.
            event: What happened, such as ``call``, ``result`` or ``error``.
            level: Level the line is logged at.
            **data: Extra fields recorded with the entry.
        """
        self._log(
            level,
            f"[tool:{tool_name}] {event}",
            event_type="tool",
            event=event,
            tool_name=tool_name,
            **data,
        )

    def model_event(
        self,
        model_name: str,
        event: str,
        *,
        level: int = logging.DEBUG,
        **data: Any,
    ) -> None:
        """Log a model-level event (generate, stream, error).

        Args:
            model_name: The model the event belongs to.
            event: What happened, such as ``generate``, ``stream`` or ``error``.
            level: Level the line is logged at.
            **data: Extra fields recorded with the entry.
        """
        self._log(
            level,
            f"[model:{model_name}] {event}",
            event_type="model",
            event=event,
            model_name=model_name,
            **data,
        )

    def iteration_event(
        self,
        iteration: int,
        event: str,
        *,
        level: int = logging.DEBUG,
        **data: Any,
    ) -> None:
        """Log a ReAct iteration event.

        Args:
            iteration: The reasoning iteration the event belongs to.
            event: What happened in that iteration.
            level: Level the line is logged at.
            **data: Extra fields recorded with the entry.
        """
        self._log(
            level,
            f"[iter:{iteration}] {event}",
            event_type="iteration",
            event=event,
            iteration=iteration,
            **data,
        )

    # -- Internal --

    def _log(self, level: int, msg: str, **extra: Any) -> None:
        """Emit a log record with extra context fields."""
        # Inject current run context into extra
        extra.setdefault("run_id", get_current_run_id())
        extra.setdefault("workflow_id", get_current_workflow_id())
        extra.setdefault("agent_name", get_current_agent_name())
        extra.setdefault("session_id", get_current_session_id())
        # Remove None values to keep logs clean
        extra = {k: v for k, v in extra.items() if v is not None}
        self._logger.log(level, msg, extra=extra)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_structured_loggers: dict[str, StructuredLogger] = {}


def get_structured_logger(name: str, json_mode: bool = False) -> StructuredLogger:
    """
    Get or create a StructuredLogger for the given module name.

    Args:
        name: Logger name (typically __name__)
        json_mode: Whether to use JSON formatting

    Returns:
        StructuredLogger instance
    """
    key = f"{name}:{json_mode}"
    if key not in _structured_loggers:
        _structured_loggers[key] = StructuredLogger(name, json_mode=json_mode)
    return _structured_loggers[key]


def setup_json_logging(
    level: str | int = "INFO",
    logger_name: str | None = None,
) -> None:
    """Configure a logger (or the root logger) to emit JSON lines.

    Installs an ``EffGenJSONFormatter`` stream handler.

    Args:
        level: Log level
        logger_name: Specific logger name, or None for root
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    target = logging.getLogger(logger_name)
    target.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(EffGenJSONFormatter())

    target.handlers.clear()
    target.addHandler(handler)
    target.propagate = False
