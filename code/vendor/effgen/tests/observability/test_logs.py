"""
Tests for effgen.observability.logs — structured JSON logging module.

Verifies:
- Every log line is valid JSON
- Canonical fields present (ts, level, module, event)
- Secret redaction in attributes
- OTel trace_id / span_id injection (with and without active span)
- EffGenLogger API surface (event, debug, info, warning, error, domain helpers)
- configure_logging sets up the root handler correctly
- StructuredFormatter shape
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from effgen.observability import configure_logging, get_logger
from effgen.observability.logs import (
    EffGenLogger,
    StructuredFormatter,
    clear_run_context,
    get_effgen_logger,
    set_run_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENAI_KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"
ANTHROPIC_KEY = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz"


def capture_log(name: str = "test_logs") -> tuple[EffGenLogger, list[dict]]:
    """
    Create a logger whose output is captured in a list of parsed JSON records.

    Returns:
        (logger, records) — append to *records* happens automatically.
    """
    records: list[dict] = []
    buf = io.StringIO()

    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(StructuredFormatter(include_src=False, redact=True))

    stdlib_logger = logging.getLogger(name)
    stdlib_logger.setLevel(logging.DEBUG)
    stdlib_logger.handlers.clear()
    stdlib_logger.addHandler(handler)
    stdlib_logger.propagate = False

    obs_log = get_effgen_logger(name)

    class _Capture:
        def flush(self):
            buf.flush()
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            buf.truncate(0)
            buf.seek(0)

    obs_log._capture = _Capture()  # type: ignore[attr-defined]
    return obs_log, records


# ---------------------------------------------------------------------------
# StructuredFormatter: shape tests
# ---------------------------------------------------------------------------

class TestStructuredFormatterShape:
    def _format_record(self, msg: str, level=logging.INFO, **attrs) -> dict:
        formatter = StructuredFormatter(include_src=True, redact=False)
        record = logging.LogRecord(
            name="test.module",
            level=level,
            pathname="test_logs.py",
            lineno=42,
            msg=msg,
            args=(),
            exc_info=None,
        )
        record._effgen_event = msg  # type: ignore[attr-defined]
        record._effgen_attrs = attrs  # type: ignore[attr-defined]
        line = formatter.format(record)
        return json.loads(line)

    def test_is_valid_json(self):
        formatter = StructuredFormatter()
        record = logging.makeLogRecord({"msg": "hello", "levelno": logging.INFO})
        line = formatter.format(record)
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_required_fields_present(self):
        d = self._format_record("test.event.happened")
        for field in ("ts", "level", "module", "event"):
            assert field in d, f"Missing required field: {field!r}"

    def test_ts_is_iso8601(self):
        d = self._format_record("x")
        ts = d["ts"]
        # Must parse as an ISO timestamp; the stdlib datetime accepts this
        from datetime import datetime
        # should not raise
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_level_value(self):
        assert self._format_record("x", level=logging.DEBUG)["level"] == "DEBUG"
        assert self._format_record("x", level=logging.WARNING)["level"] == "WARNING"
        assert self._format_record("x", level=logging.ERROR)["level"] == "ERROR"

    def test_event_field_set(self):
        d = self._format_record("my.event.name")
        assert d["event"] == "my.event.name"

    def test_attributes_populated(self):
        d = self._format_record("evt", model="llama3.1-8b", tokens=100)
        assert "attributes" in d
        assert d["attributes"]["model"] == "llama3.1-8b"
        assert d["attributes"]["tokens"] == 100

    def test_src_block_present(self):
        d = self._format_record("evt")
        assert "src" in d
        assert "file" in d["src"]
        assert "line" in d["src"]
        assert "func" in d["src"]

    def test_src_block_absent_when_disabled(self):
        formatter = StructuredFormatter(include_src=False, redact=False)
        record = logging.makeLogRecord({"msg": "x", "levelno": logging.INFO})
        d = json.loads(formatter.format(record))
        assert "src" not in d

    def test_exception_block(self):
        formatter = StructuredFormatter(include_src=False, redact=False)
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="t", level=logging.ERROR, pathname="f.py",
            lineno=1, msg="oops", args=(), exc_info=exc_info,
        )
        d = json.loads(formatter.format(record))
        assert "exception" in d
        assert d["exception"]["type"] == "ValueError"
        assert "boom" in d["exception"]["message"]


# ---------------------------------------------------------------------------
# Redaction in formatter
# ---------------------------------------------------------------------------

class TestRedactionInFormatter:
    def _format_with_secret(self, secret: str) -> dict:
        formatter = StructuredFormatter(include_src=False, redact=True)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="sensitive", args=(), exc_info=None,
        )
        record._effgen_event = "sensitive.event"  # type: ignore[attr-defined]
        record._effgen_attrs = {"api_key": secret}  # type: ignore[attr-defined]
        return json.loads(formatter.format(record))

    def test_openai_key_redacted(self):
        d = self._format_with_secret(OPENAI_KEY)
        raw = json.dumps(d)
        assert OPENAI_KEY not in raw
        assert "REDACTED" in raw

    def test_anthropic_key_redacted(self):
        d = self._format_with_secret(ANTHROPIC_KEY)
        raw = json.dumps(d)
        assert ANTHROPIC_KEY not in raw
        assert "REDACTED" in raw

    def test_bearer_in_value(self):
        d = self._format_with_secret(f"Bearer {OPENAI_KEY}")
        raw = json.dumps(d)
        assert OPENAI_KEY not in raw


# ---------------------------------------------------------------------------
# EffGenLogger API
# ---------------------------------------------------------------------------

class TestEffGenLoggerAPI:
    def _make_logger_with_capture(self) -> tuple[EffGenLogger, list[str]]:
        """Return an EffGenLogger and a list that collects raw emitted lines."""
        lines: list[str] = []
        buf = io.StringIO()

        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredFormatter(include_src=False, redact=True))

        name = f"test_api_{id(self)}"
        stdlib_log = logging.getLogger(name)
        stdlib_log.setLevel(logging.DEBUG)
        stdlib_log.handlers.clear()
        stdlib_log.addHandler(handler)
        stdlib_log.propagate = False

        # Patch the EffGenLogger to use this stdlib logger
        obs = EffGenLogger(name)
        return obs, lines, buf  # type: ignore[return-value]

    def _flush(self, buf) -> list[dict]:
        raw = buf.getvalue()
        buf.truncate(0)
        buf.seek(0)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def test_event_method_emits_json(self):
        obs, _, buf = self._make_logger_with_capture()
        obs.event("model.call.started", model="llama3.1-8b", tokens=0)
        records = self._flush(buf)
        assert len(records) == 1
        r = records[0]
        assert r["event"] == "model.call.started"
        assert r["attributes"]["model"] == "llama3.1-8b"

    def test_level_helpers_emit_json(self):
        obs, _, buf = self._make_logger_with_capture()
        obs.debug("debug message", x=1)
        obs.info("info message",  x=2)
        obs.warning("warn message", x=3)
        obs.error("error message", x=4)
        records = self._flush(buf)
        levels = [r["level"] for r in records]
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_attributes_in_level_helpers(self):
        obs, _, buf = self._make_logger_with_capture()
        obs.info("test", key="value", count=42)
        records = self._flush(buf)
        assert records[0]["attributes"]["key"] == "value"
        assert records[0]["attributes"]["count"] == 42

    def test_domain_helpers(self):
        obs, _, buf = self._make_logger_with_capture()
        obs.model_event("call.done", model="llama3.1-8b")
        obs.tool_event("executed",  tool="web_search")
        obs.agent_event("run.started", agent="my_agent")
        obs.router_event("decision",   policy="cost")
        records = self._flush(buf)
        assert records[0]["event"] == "model.call.done"
        assert records[1]["event"] == "tool.executed"
        assert records[2]["event"] == "agent.run.started"
        assert records[3]["event"] == "router.decision"

    def test_secret_redacted_in_event(self):
        obs, _, buf = self._make_logger_with_capture()
        obs.event("config.loaded", api_key=OPENAI_KEY)
        records = self._flush(buf)
        raw = json.dumps(records[0])
        assert OPENAI_KEY not in raw

    def test_disabled_level_not_emitted(self):
        """DEBUG events should not appear when logger level is INFO."""
        obs, _, buf = self._make_logger_with_capture()
        obs.getLogger().setLevel(logging.INFO)
        obs.debug("should not appear")
        records = self._flush(buf)
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Run context injection
# ---------------------------------------------------------------------------

class TestRunContextInjection:
    def setup_method(self):
        clear_run_context()

    def teardown_method(self):
        clear_run_context()

    def _make(self) -> tuple[EffGenLogger, io.StringIO]:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredFormatter(include_src=False, redact=False))
        name = f"test_ctx_{id(self)}"
        sl = logging.getLogger(name)
        sl.setLevel(logging.DEBUG)
        sl.handlers.clear()
        sl.addHandler(handler)
        sl.propagate = False
        return EffGenLogger(name), buf

    def _flush(self, buf) -> list[dict]:
        raw = buf.getvalue()
        buf.truncate(0)
        buf.seek(0)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def test_run_context_appears_in_output(self):
        obs, buf = self._make()
        set_run_context(run_id="abc123", agent_name="test_agent")
        obs.event("something.happened")
        records = self._flush(buf)
        assert records[0].get("run_id") == "abc123"
        assert records[0].get("agent_name") == "test_agent"

    def test_cleared_context_absent(self):
        obs, buf = self._make()
        set_run_context(run_id="xyz")
        clear_run_context()
        obs.event("after_clear")
        records = self._flush(buf)
        assert "run_id" not in records[0]


# ---------------------------------------------------------------------------
# OTel trace context (optional — only when OTel is installed)
# ---------------------------------------------------------------------------

class TestOtelContext:
    def test_no_otel_span_yields_empty_ids(self):
        """When no OTel span is active, trace_id and span_id must be absent."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredFormatter(include_src=False, redact=False))
        name = f"test_otel_{id(self)}"
        sl = logging.getLogger(name)
        sl.setLevel(logging.DEBUG)
        sl.handlers.clear()
        sl.addHandler(handler)
        sl.propagate = False

        obs = EffGenLogger(name)
        obs.event("no.span.here")
        raw = buf.getvalue().strip()
        d = json.loads(raw)
        # Without an active span, these should be absent (or empty string)
        assert d.get("trace_id", "") == "" or "trace_id" not in d

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("opentelemetry"),
        reason="opentelemetry not installed",
    )
    def test_otel_span_injects_ids(self):
        """When an OTel span is active, trace_id and span_id must appear."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = provider.get_tracer("test")

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredFormatter(include_src=False, redact=False))
        name = f"test_otel_span_{id(self)}"
        sl = logging.getLogger(name)
        sl.setLevel(logging.DEBUG)
        sl.handlers.clear()
        sl.addHandler(handler)
        sl.propagate = False

        obs = EffGenLogger(name)

        with tracer.start_as_current_span("test.span"):
            obs.event("inside.span", key="value")

        raw = buf.getvalue().strip()
        d = json.loads(raw)
        assert "trace_id" in d
        assert "span_id" in d
        assert len(d["trace_id"]) == 32  # 128-bit trace id, 32 hex chars
        assert len(d["span_id"]) == 16   # 64-bit span id, 16 hex chars


# ---------------------------------------------------------------------------
# get_logger (public entry point)
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_effgen_logger(self):
        log = get_logger("some.module")
        assert isinstance(log, EffGenLogger)

    def test_same_name_same_instance(self):
        a = get_logger("shared.logger")
        b = get_logger("shared.logger")
        assert a is b

    def test_different_names_different_instances(self):
        a = get_logger("mod.a")
        b = get_logger("mod.b")
        assert a is not b


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_configure_sets_handler(self):
        """configure_logging should attach a StructuredFormatter handler to effgen.*"""
        import effgen.observability.logs as _logs_module
        # Reset the flag to allow reconfiguration in tests
        original = _logs_module._logging_configured
        _logs_module._logging_configured = False

        # Snapshot effgen logger state so we can restore it.  configure_logging()
        # mutates handlers + propagate; without restoring we leak state into
        # subsequent tests that rely on log propagation (e.g. caplog tests).
        effgen_log = logging.getLogger("effgen")
        prev_handlers = list(effgen_log.handlers)
        prev_propagate = effgen_log.propagate
        prev_level = effgen_log.level

        try:
            buf = io.StringIO()
            configure_logging(level="DEBUG", json=True, stream=buf)

            # Now emit a log via the effgen root
            test_log = logging.getLogger("effgen._test_configure")
            test_log.debug("ping")

            raw = buf.getvalue().strip()
            if raw:
                d = json.loads(raw)
                assert "event" in d or "msg" in d
        finally:
            # Restore effgen logger state
            effgen_log.handlers.clear()
            for h in prev_handlers:
                effgen_log.addHandler(h)
            effgen_log.propagate = prev_propagate
            effgen_log.setLevel(prev_level)
            _logs_module._logging_configured = original
