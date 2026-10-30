# Structured Logging

effGen emits structured JSON log lines on every critical path — agent runs, model calls, tool executions, and router decisions.  All log lines are redacted so no API key, bearer token, or webhook URL ever appears in a log file.

---

## Quick start

```python
from effgen.observability import get_logger, configure_logging

# Configure once at application startup
configure_logging(level="INFO")

# In any module
log = get_logger(__name__)
log.event("model.call.started", model="gpt-oss-120b", cached_tokens=0)
log.event("model.call.done",    model="gpt-oss-120b", latency_ms=340)
```

---

## Log line schema

Every line is a single JSON object on stdout / stderr:

| Field | Type | Always present | Description |
|---|---|---|---|
| `ts` | ISO-8601 UTC | ✅ | Microsecond-precision timestamp |
| `level` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` | ✅ | Log level |
| `module` | dotted name | ✅ | Logger name (typically `__name__`) |
| `event` | string | ✅ | Short dotted event label, e.g. `"model.call.done"` |
| `attributes` | object | when non-empty | Caller-supplied kwargs, after redaction |
| `trace_id` | 32-char hex | when OTel span active | OpenTelemetry trace ID |
| `span_id` | 16-char hex | when OTel span active | OpenTelemetry span ID |
| `run_id` | string | when set | Unique ID for this `Agent.run()` invocation |
| `workflow_id` | string | when set | Shared ID across multi-agent workflow |
| `agent_name` | string | when set | Name of the active agent |
| `session_id` | string | when set | Session identifier |
| `iteration` | int | when set | ReAct loop iteration number |
| `exception` | object | on errors | `{type, message, file, line}` |
| `src` | object | optional | `{file, line, func}` source location |

### Example lines

```jsonl
{"ts":"2026-05-22T05:53:39.025+00:00","level":"INFO","module":"effgen.core.agent","event":"agent.run.started","attributes":{"agent":"my_agent","task":"What is 2+2?","mode":"auto","run_id":"abc123"}}
{"ts":"2026-05-22T05:53:39.436+00:00","level":"INFO","module":"effgen.models.cerebras_adapter","event":"model.call.done","attributes":{"provider":"cerebras","model":"gpt-oss-120b","prompt_tokens":60,"completion_tokens":2,"cost_usd":0.0}}
{"ts":"2026-05-22T05:53:39.437+00:00","level":"INFO","module":"effgen.core.agent","event":"agent.run.completed","attributes":{"agent":"my_agent","run_id":"abc123","latency_ms":252.4,"tokens":2,"tool_calls":0,"success":true}}
```

---

## API reference

### `get_logger(name) → EffGenLogger`

Return a cached `EffGenLogger` for *name*.  Pass `__name__` from the calling module.

### `EffGenLogger.event(event_name, level=INFO, **kwargs)`

Emit a structured log line.

```python
log.event("model.call.started", model="gpt-oss-120b", cached_tokens=0)
```

### Level helpers

```python
log.debug("detail", key="value")
log.info("something happened", key="value")
log.warning("soft fault", key="value")
log.error("hard fault", key="value")
```

### Domain helpers

```python
log.model_event("call.done", provider="cerebras", model="gpt-oss-120b")
log.tool_event("executed",   tool="web_search", latency_ms=123)
log.agent_event("run.started", agent="my_agent", task="hello")
log.router_event("decision",   policy="cost", selected_provider="cerebras")
```

---

## Secret redaction

Redaction is applied at the log encoder level — every attribute value passes through `Redactor.scrub_value()` before JSON serialisation.  No secret can bypass redaction by being set as an attribute.

### Built-in patterns

| Pattern name | Regex | Replacement |
|---|---|---|
| `anthropic_key` | `sk-ant-[a-zA-Z0-9_\-]{20,}` | `<REDACTED:anthropic_key>` |
| `cerebras_key` | `csk-[a-zA-Z0-9_\-]{20,}` | `<REDACTED:cerebras_key>` |
| `google_key` | `AIza[0-9A-Za-z_\-]{35}` | `<REDACTED:google_key>` |
| `hf_key` | `hf_[a-zA-Z0-9]{20,}` | `<REDACTED:hf_key>` |
| `groq_key` | `gsk_[a-zA-Z0-9_\-]{20,}` | `<REDACTED:groq_key>` |
| `openai_key` | `sk-[a-zA-Z0-9_\-]{20,}` | `<REDACTED:openai_key>` |
| `bearer_token` | `Bearer [^\s]{6,}` | `<REDACTED:bearer_token>` |
| `slack_webhook` | Slack incoming webhook URL | `<REDACTED:slack_webhook>` |
| `discord_webhook` | Discord webhook URL | `<REDACTED:discord_webhook>` |

### Adding custom patterns

```python
from effgen.observability import get_redactor
r = get_redactor()
r.add_pattern("custom_token", r"tok-[A-Za-z0-9]{32}")
```

---

## Configuration

```python
from effgen.observability import configure_logging

configure_logging(
    level="INFO",    # or logging.INFO
    json=True,       # False → plain text (useful during development)
    stream=None,     # defaults to sys.stderr
    include_src=True,  # include src block (file/line/func)
    redact=True,     # apply secret redaction
)
```

Call once at application startup.  Configures the `effgen.*` logger namespace.

---

## OTel trace context

When an OpenTelemetry span is active, the formatter automatically reads the current `trace_id` and `span_id` and includes them in the log line.  No extra code is needed — just ensure OTel is set up before calling `configure_logging`.

```python
from effgen.utils.tracing import setup_tracing
setup_tracing(service_name="my-service", exporter_type="console")

from effgen.observability import configure_logging, get_logger
configure_logging()

log = get_logger(__name__)

from effgen.utils.tracing import get_tracer
with get_tracer().start_as_current_span("my_span"):
    log.event("inside.span")  # → includes trace_id, span_id
```
