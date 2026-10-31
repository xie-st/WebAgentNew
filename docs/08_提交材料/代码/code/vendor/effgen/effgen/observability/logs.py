"""
Structured JSON logging for effGen.

Every log line emitted by ``StructuredFormatter`` is a valid JSON object with
the following guaranteed fields::

    {
      "ts":         "<ISO-8601 UTC timestamp>",
      "level":      "INFO",
      "module":     "effgen.core.agent",
      "event":      "agent.run.started",        # set via log.event()
      "attributes": {"model": "llama3.1-8b"},   # caller-supplied kwargs
      "trace_id":   "0af7651916cd43dd8448eb211c80319c",  # from OTel span ctx
      "span_id":    "b9c7c989f97918e1"
    }

All values in ``attributes`` are passed through :class:`~effgen.observability.redact.Redactor`
before serialisation so no raw API key or bearer token survives the encoder.

Quick start
-----------
    from effgen.observability import get_logger

    log = get_logger(__name__)
    log.event("model.call.started", model="llama3.1-8b", cached_tokens=0)
    log.event("model.call.done", model="llama3.1-8b", latency_ms=340)

Level helpers are also available::

    log.debug("low-level detail")
    log.info("something happened")
    log.warning("soft fault")
    log.error("hard fault")

All calls accept ``**kwargs`` which are stored in the ``attributes`` dict
after redaction.

Setup
-----
Call ``configure_logging(level, json=True)`` once at application startup to
attach a ``StructuredFormatter``-based handler to the root logger.  If you
never call it, a NullHandler is used so library usage stays silent.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from typing import Any

from .redact import get_redactor

# ---------------------------------------------------------------------------
# Optional OTel import — the span helpers no-op when it is absent
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def _get_otel_ids() -> tuple[str, str]:
    """
    Return (trace_id_hex, span_id_hex) from the active OTel span, or empty
    strings when OTel is unavailable or no span is active.
    """
    if not _OTEL_AVAILABLE:
        return "", ""
    try:
        span = _otel_trace.get_current_span()
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return "", ""
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")
        return trace_id, span_id
    except Exception:  # noqa: BLE001 — never break logging
        return "", ""


# ---------------------------------------------------------------------------
# Thread-local run context (shared with structured_logging.py)
# ---------------------------------------------------------------------------

_run_ctx = threading.local()


def _run_context() -> dict[str, Any]:
    """Return current thread-local run context as a flat dict (no Nones)."""
    out: dict[str, Any] = {}
    for key in ("run_id", "workflow_id", "agent_name", "session_id", "iteration"):
        v = getattr(_run_ctx, key, None)
        if v is not None:
            out[key] = v
    return out


def set_run_context(
    *,
    run_id: str | None = None,
    workflow_id: str | None = None,
    agent_name: str | None = None,
    session_id: str | None = None,
    iteration: int | None = None,
) -> None:
    """Set thread-local run-context fields.

    Called by :class:`~effgen.utils.structured_logging.LogRunContext` so that
    both the legacy and the new logging APIs share the same context.

    Args:
        run_id: Id of the run every later log line is stamped with.
        workflow_id: Id shared across a multi-agent workflow.
        agent_name: Name of the agent producing the lines.
        session_id: Session the run belongs to.
        iteration: Reasoning iteration reached so far.
    """
    if run_id is not None:
        _run_ctx.run_id = run_id
    if workflow_id is not None:
        _run_ctx.workflow_id = workflow_id
    if agent_name is not None:
        _run_ctx.agent_name = agent_name
    if session_id is not None:
        _run_ctx.session_id = session_id
    if iteration is not None:
        _run_ctx.iteration = iteration


def clear_run_context() -> None:
    """Clear all thread-local run-context fields."""
    for key in ("run_id", "workflow_id", "agent_name", "session_id", "iteration"):
        try:
            delattr(_run_ctx, key)
        except AttributeError:  # field already unset for this run
            pass


# ---------------------------------------------------------------------------
# StructuredFormatter
# ---------------------------------------------------------------------------

# Keys present on every LogRecord that should NOT be lifted into attributes
_STDLIB_KEYS: frozenset[str] = frozenset(
    [
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "thread",
        "threadName",
        "exc_info",
        "exc_text",
        "stack_info",
        "message",
        "asctime",
        "taskName",
        # effGen-injected keys — handled explicitly
        "run_id",
        "workflow_id",
        "agent_name",
        "session_id",
        "iteration",
        # Internal bookkeeping
        "_effgen_event",
        "_effgen_attrs",
    ]
)


class StructuredFormatter(logging.Formatter):
    """
    JSON-lines formatter that emits one JSON object per log record.

    Schema
    ------
    Required fields (always present):

    * ``ts``      — ISO-8601 UTC timestamp with microsecond precision
    * ``level``   — ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR`` / ``CRITICAL``
    * ``module``  — dotted module name (``record.name``)
    * ``event``   — free-text event label (message or ``log.event()`` name)

    Optional fields (present when non-empty):

    * ``attributes`` — caller-supplied kwargs, redacted
    * ``trace_id`` / ``span_id`` — from active OTel span
    * ``run_id``, ``workflow_id``, ``agent_name``, ``session_id``, ``iteration``
      — from thread-local run context
    * ``exception`` — ``{type, message, file, line}`` when ``exc_info`` is set
    * ``src`` — ``{file, line, func}`` source location

    All attribute values are scrubbed by :func:`~effgen.observability.redact.get_redactor`
    before serialisation.

    Args:
        include_src:    Include ``src`` source-location block (default ``True``).
        redact:         Apply secret redaction (default ``True``).  Set
                        ``False`` only in tests that inspect raw attribute values.
    """

    def __init__(
        self,
        include_src: bool = True,
        redact: bool = True,
    ) -> None:
        super().__init__()
        self._include_src = include_src
        self._redact = redact
        self._redactor = get_redactor() if redact else None

    # ------------------------------------------------------------------
    # logging.Formatter contract
    # ------------------------------------------------------------------

    def format(self, record: logging.LogRecord) -> str:
        """Serialise *record* to a single JSON line."""
        redact = self._redactor.scrub_value if self._redactor else (lambda x: x)

        # Core fields
        record_dict: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="microseconds"
            ),
            "level": record.levelname,
            "module": record.name,
            "event": getattr(record, "_effgen_event", record.getMessage()),
        }

        # OTel trace context
        trace_id, span_id = _get_otel_ids()
        if trace_id:
            record_dict["trace_id"] = trace_id
        if span_id:
            record_dict["span_id"] = span_id

        # Thread-local run context
        record_dict.update(_run_context())
        # Also check for context fields injected directly into the record
        for key in ("run_id", "workflow_id", "agent_name", "session_id", "iteration"):
            v = getattr(record, key, None)
            if v is not None and key not in record_dict:
                record_dict[key] = v

        # Caller-supplied attributes (from log.event() / extra=)
        attrs: dict[str, Any] = {}
        # Attributes explicitly stored by EffGenLogger.event()
        if hasattr(record, "_effgen_attrs"):
            attrs.update(record._effgen_attrs)
        # Additional ad-hoc extra= fields from stdlib .info/.debug/…
        for key, value in record.__dict__.items():
            if key not in _STDLIB_KEYS and not key.startswith("_"):
                try:
                    json.dumps(value)
                    attrs[key] = value
                except (TypeError, ValueError):
                    attrs[key] = str(value)

        if attrs:
            record_dict["attributes"] = redact(attrs)  # type: ignore[arg-type]

        # Exception
        if record.exc_info and record.exc_info[0] is not None:
            exc_type = record.exc_info[0]
            exc_val = record.exc_info[1]
            exc_tb = record.exc_info[2]
            exc_block: dict[str, Any] = {
                "type": exc_type.__name__,
                "message": redact(str(exc_val)),
            }
            if exc_tb is not None:
                exc_block["file"] = exc_tb.tb_frame.f_code.co_filename
                exc_block["line"] = exc_tb.tb_lineno
            record_dict["exception"] = exc_block

        # Source location
        if self._include_src:
            record_dict["src"] = {
                "file": record.pathname,
                "line": record.lineno,
                "func": record.funcName,
            }

        return json.dumps(record_dict, default=str)


# ---------------------------------------------------------------------------
# EffGenLogger — structured event logger
# ---------------------------------------------------------------------------


class EffGenLogger:
    """
    Structured logger for effGen components.

    Wraps a stdlib :class:`logging.Logger` and adds an :meth:`event` method
    for emitting canonical, redacted JSON log lines.

    The preferred way to get one::

        from effgen.observability import get_logger
        log = get_logger(__name__)

    All keyword arguments passed to :meth:`event` (or the level helpers)
    land in the ``attributes`` dict of the emitted JSON line after redaction.

    Args:
        name: Logger name — pass ``__name__`` from the calling module.
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        self.name = name

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def event(
        self,
        event_name: str,
        level: int = logging.INFO,
        **attributes: Any,
    ) -> None:
        """
        Emit a structured log line.

        Args:
            event_name:  A short dotted name, e.g. ``"model.call.started"``.
            level:       stdlib logging level (default: ``logging.INFO``).
            **attributes: Arbitrary key-value pairs stored under ``attributes``
                          in the JSON output.  Values are redacted before
                          serialisation.

        Example::

            log.event("model.call.done", model="llama3.1-8b", latency_ms=340)
        """
        if not self._logger.isEnabledFor(level):
            return
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=event_name,
            args=(),
            exc_info=None,
        )
        record._effgen_event = event_name  # type: ignore[attr-defined]
        record._effgen_attrs = attributes  # type: ignore[attr-defined]
        self._logger.handle(record)

    # ------------------------------------------------------------------
    # Level helpers — mirror stdlib API, kwargs → attributes
    # ------------------------------------------------------------------

    def debug(self, msg: str, **attributes: Any) -> None:
        """Log at DEBUG level; kwargs go into ``attributes``."""
        self._emit(logging.DEBUG, msg, **attributes)

    def info(self, msg: str, **attributes: Any) -> None:
        """Log at INFO level; kwargs go into ``attributes``."""
        self._emit(logging.INFO, msg, **attributes)

    def warning(self, msg: str, **attributes: Any) -> None:
        """Log at WARNING level; kwargs go into ``attributes``."""
        self._emit(logging.WARNING, msg, **attributes)

    def error(self, msg: str, **attributes: Any) -> None:
        """Log at ERROR level; kwargs go into ``attributes``."""
        self._emit(logging.ERROR, msg, **attributes)

    def critical(self, msg: str, **attributes: Any) -> None:
        """Log at CRITICAL level; kwargs go into ``attributes``."""
        self._emit(logging.CRITICAL, msg, **attributes)

    def exception(self, msg: str, exc_info: bool = True, **attributes: Any) -> None:
        """Log at ERROR level with exception traceback.

        Args:
            msg: The message to log.
            exc_info: Attach the active exception's traceback.
            **attributes: Extra fields recorded with the entry.
        """
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=logging.ERROR,
            fn="",
            lno=0,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )
        record._effgen_event = msg  # type: ignore[attr-defined]
        record._effgen_attrs = attributes  # type: ignore[attr-defined]
        self._logger.handle(record)

    # ------------------------------------------------------------------
    # Domain-specific convenience
    # ------------------------------------------------------------------

    def model_event(self, event: str, **attrs: Any) -> None:
        """Shorthand for ``log.event("model.<event>", ...)``.

        Example::

            log.model_event("call.started", model="llama3.1-8b", provider="cerebras")
        """
        self.event(f"model.{event}", **attrs)

    def tool_event(self, event: str, **attrs: Any) -> None:
        """Shorthand for ``log.event("tool.<event>", ...)``.

        Example::

            log.tool_event("call.started", tool="web_search", query="python news")
        """
        self.event(f"tool.{event}", **attrs)

    def agent_event(self, event: str, **attrs: Any) -> None:
        """Shorthand for ``log.event("agent.<event>", ...)``.

        Example::

            log.agent_event("run.started", name="my_agent", task="hello")
        """
        self.event(f"agent.{event}", **attrs)

    def router_event(self, event: str, **attrs: Any) -> None:
        """Shorthand for ``log.event("router.<event>", ...)``.

        Example::

            log.router_event("decision", policy="cost", selected="cerebras/llama3.1-8b")
        """
        self.event(f"router.{event}", **attrs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, level: int, msg: str, **attributes: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        record._effgen_event = msg  # type: ignore[attr-defined]
        record._effgen_attrs = attributes  # type: ignore[attr-defined]
        self._logger.handle(record)

    # Allow stdlib-style extra= dicts on the underlying logger for callers
    # that migrate gradually
    def getLogger(self) -> logging.Logger:
        """Return the underlying stdlib logger."""
        return self._logger


# ---------------------------------------------------------------------------
# Logger cache + factory
# ---------------------------------------------------------------------------

_logger_cache: dict[str, EffGenLogger] = {}
_cache_lock = threading.Lock()


def get_effgen_logger(name: str) -> EffGenLogger:
    """
    Get or create an :class:`EffGenLogger` for *name*.

    This is the low-level factory.  Prefer the top-level
    :func:`~effgen.observability.get_logger` helper.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.

    Returns:
        Cached :class:`EffGenLogger` instance.
    """
    with _cache_lock:
        if name not in _logger_cache:
            _logger_cache[name] = EffGenLogger(name)
        return _logger_cache[name]


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

_logging_configured = False


def configure_logging(
    level: str | int = "INFO",
    json: bool = True,
    stream: Any = None,
    include_src: bool = True,
    redact: bool = True,
) -> None:
    """
    Configure the root ``effgen.*`` logger to emit structured JSON lines.

    Call once at application startup.  Subsequent calls are idempotent unless
    *force* is set (not exposed — just call :func:`logging.getLogger` directly
    for advanced setups).

    Args:
        level:       Minimum log level string or int (default ``"INFO"``).
        json:        ``True`` → JSON formatter; ``False`` → plain text.
        stream:      Output stream (default: ``sys.stderr``).
        include_src: Include ``src`` block in JSON lines (default ``True``).
        redact:      Apply secret redaction (default ``True``).
    """
    global _logging_configured
    if _logging_configured:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    if json:
        handler.setFormatter(StructuredFormatter(include_src=include_src, redact=redact))

    root = logging.getLogger("effgen")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    _logging_configured = True


# Ensure effgen.* has at least a NullHandler so library consumers don't get
# "No handlers could be found" warnings when they don't call configure_logging.
logging.getLogger("effgen").addHandler(logging.NullHandler())


__all__ = [
    "StructuredFormatter",
    "EffGenLogger",
    "get_effgen_logger",
    "configure_logging",
    "set_run_context",
    "clear_run_context",
]
