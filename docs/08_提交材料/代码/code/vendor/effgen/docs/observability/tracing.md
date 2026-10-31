# effGen Distributed Tracing

effGen instruments every hot path with [OpenTelemetry](https://opentelemetry.io/) spans so you can
visualize the complete execution tree of any agent run in Jaeger, Zipkin, Grafana Tempo, or any
OTLP-compatible backend.

---

## Span hierarchy

```
effgen.agent.run                          # entire agent.run() call
  effgen.agent.iteration                  # one ReAct iteration
    effgen.model.call                     # model inference
    effgen.tool.call  [calculator]        # tool 1
    effgen.tool.call  [web_search]        # tool 2
    effgen.tool.call  [wikipedia]         # tool 3
  effgen.router.decision                  # policy-based routing (if enabled)
```

All spans share the same `trace_id`. Each span carries typed attributes; see
[Span attribute reference](#span-attribute-reference) below.

---

## Quick start

```python
from effgen.observability import (
    setup_tracing,
    ParentBasedSampler,
    TraceIdRatioSampler,
)

# Configure once at startup — 10 % sampling, honouring parent context
setup_tracing(
    service_name="my-service",
    sampler=ParentBasedSampler(TraceIdRatioSampler(0.10)),
)
```

No other changes are needed. The agent loop, model adapters, tools, and router
already emit spans automatically.

---

## Samplers

| Class | Description |
|---|---|
| `AlwaysOnSampler` | Sample every span. Good for development. |
| `AlwaysOffSampler` | Drop every span. Useful to disable tracing without removing config. |
| `TraceIdRatioSampler(p)` | Sample a fraction `p` of traces. Deterministic — same trace ID always gives the same decision. |
| `RateLimitedSampler(per_second)` | Sample at most `per_second` traces per second (token bucket). Good for rate-controlling high-traffic services. |
| `ParentBasedSampler(root)` | Honour the parent span's sampling decision; delegate root spans to `root`. **Recommended for production.** |

### Examples

```python
from effgen.observability import (
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    RateLimitedSampler,
    ParentBasedSampler,
    setup_tracing,
)

# Sample 5 % of traces
setup_tracing(sampler=ParentBasedSampler(TraceIdRatioSampler(0.05)))

# Rate-limit to 20 sampled traces/second
setup_tracing(sampler=ParentBasedSampler(RateLimitedSampler(20.0)))

# Full sampling (dev/test)
setup_tracing(sampler=AlwaysOnSampler())
```

---

## Span context managers

For custom instrumentation, import the context managers directly:

```python
from effgen.observability import (
    start_agent_run,
    start_agent_iteration,
    start_model_call,
    start_tool_call,
    start_router_decision,
    record_retry_attempt,
)
from effgen.observability import AgentAttrs, ModelAttrs, ToolAttrs

with start_agent_run(preset="my_agent", task="hello", run_id="abc") as span:
    span.set_attribute(AgentAttrs.RUN_ID, "abc")

    with start_agent_iteration(preset="my_agent", iteration=1):
        with start_model_call(provider="cerebras", model="gpt-oss-120b") as mspan:
            # ... call model ...
            mspan.set_attribute(ModelAttrs.INPUT_TOKENS, 400)
            mspan.set_attribute(ModelAttrs.OUTPUT_TOKENS, 60)
            mspan.set_attribute(ModelAttrs.OUTCOME, "ok")

        with start_tool_call("calculator", "sqrt(16)") as tspan:
            tspan.set_attribute(ToolAttrs.STATUS, "ok")
```

All context managers:

- **Automatically set** `latency_ms` and a default `outcome`/`status` attribute on exit.
- **Mark spans as ERROR** and record the exception if an exception propagates out.
- **Never raise** their own errors — tracing failures are logged at DEBUG level and swallowed.

---

## Retry events

Attach a retry event to the enclosing `model.call` span:

```python
from effgen.observability import record_retry_attempt

with start_model_call(provider="openai", model="gpt-4o-mini") as span:
    for attempt in range(3):
        try:
            result = client.generate(prompt)
            break
        except RateLimitError:
            record_retry_attempt(span, attempt=attempt + 1, reason="http_429", delay_s=2.0)
```

The event is visible in the span detail view in Jaeger/Grafana as `effgen.retry.attempt`.

---

## Span attribute reference

### `effgen.agent.run` / `effgen.agent.iteration`

| Attribute | Type | Description |
|---|---|---|
| `effgen.agent.preset` | string | Agent name / preset |
| `effgen.agent.iteration` | int | ReAct iteration counter (iteration spans only) |
| `effgen.agent.run_id` | string | Run UUID (run span only) |
| `effgen.agent.task` | string | Task text (truncated to 500 chars, run span only) |

### `effgen.model.call`

| Attribute | Type | Description |
|---|---|---|
| `effgen.model.provider` | string | Provider name (`openai`, `cerebras`, etc.) |
| `effgen.model.name` | string | Model identifier |
| `effgen.model.input_tokens` | int | Prompt token count |
| `effgen.model.output_tokens` | int | Completion token count |
| `effgen.model.cached_tokens` | int | Cached input tokens (prompt caching) |
| `effgen.model.cost_usd` | float | Per-call cost; absent when the model publishes no rate |
| `effgen.model.outcome` | string | `ok` \| `error` \| `timeout` |
| `effgen.model.latency_ms` | float | End-to-end latency |
| `effgen.model.reasoning_effort` | string | `low` / `medium` / `high` (reasoning models) |
| `effgen.model.thinking_budget` | int | Thinking token budget (Gemini thinking models) |
| `effgen.model.parts_count` | int | Multimodal content parts count |

### `effgen.tool.call`

| Attribute | Type | Description |
|---|---|---|
| `effgen.tool.name` | string | Tool name |
| `effgen.tool.input` | string | Serialised input (≤ 500 chars) |
| `effgen.tool.status` | string | `ok` \| `error` \| `timeout` |
| `effgen.tool.latency_ms` | float | Tool execution latency |

### `effgen.router.decision`

| Attribute | Type | Description |
|---|---|---|
| `effgen.router.policy` | string | Policy that made the selection |
| `effgen.router.selected_provider` | string | Chosen provider |
| `effgen.router.considered` | string | Comma-separated list of candidate pairs |
| `effgen.router.eliminated_count` | int | Number of eliminated candidates |

### `effgen.retry.attempt` (span event)

| Attribute | Type | Description |
|---|---|---|
| `effgen.retry.attempt` | int | Attempt number (1-based) |
| `effgen.retry.reason` | string | Reason for retry (e.g. `http_429`) |
| `effgen.retry.delay_s` | float | Delay before this attempt |

---

## Exporter configuration

`setup_tracing` accepts any OTel `SpanExporter`:

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from effgen.observability import setup_tracing, ParentBasedSampler, TraceIdRatioSampler

setup_tracing(
    service_name="my-effgen-service",
    sampler=ParentBasedSampler(TraceIdRatioSampler(0.1)),
    exporter=OTLPSpanExporter(endpoint="http://localhost:4317"),
)
```

For console output during development:

```python
setup_tracing(export_to_console=True)
```

---

## Testing with in-memory exporter

```python
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from effgen.observability import setup_tracing, reset_tracing, AlwaysOnSampler

def test_agent_emits_spans():
    reset_tracing()
    exporter = InMemorySpanExporter()
    setup_tracing(sampler=AlwaysOnSampler(), exporter=exporter)

    # run your agent ...

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "effgen.agent.run" in names
    assert "effgen.model.call" in names
```

Always call `reset_tracing()` in your test fixture's teardown so each test
starts with a clean OTel provider.

---

## Non-blocking guarantee

Tracing is **non-blocking**. A failed span export never surfaces to inference
code. If the OTel SDK is not installed, all helpers silently fall back to no-ops.
