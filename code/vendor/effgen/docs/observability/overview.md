# Observability & Reliability Overview

effGen ships a complete observability and reliability layer so you can run agents
confidently in production. Everything in this section is **async / non-blocking** —
a failed telemetry export never fails inference.

This page is the entry point. Each subsystem has its own reference:

| Topic | Doc | What it covers |
|---|---|---|
| Structured logging + secret redaction | [logging.md](logging.md) | JSON log lines, trace-context injection, the `Redactor` |
| Metrics | [metrics.md](metrics.md) | Prometheus histograms + counters, `/metrics` scrape |
| SLO tracking | [slos.md](slos.md) | Rolling-window error budgets, burn-rate, `/slo` endpoint |
| Tracing | [tracing.md](tracing.md) | OTel samplers, canonical span-attribute spec |
| Reliability primitives | [reliability.md](reliability.md) | Timeouts, retries, circuit breakers, bulkheads |
| Chaos harness | [chaos.md](chaos.md) | Deterministic fault injection for testing |
| Fuzz harness | [fuzz.md](fuzz.md) | Hypothesis-based input fuzzing |
| Load testing | [loadtest.md](loadtest.md) | `effgen loadtest` CLI + load-gen library |
| Alerting | [alerting.md](alerting.md) | Alertmanager rule pack + `AlertWebhook` |
| Live terminal monitor | [../cli/top.md](../cli/top.md) | `effgen top` — runs, traffic, spend and GPU on one screen |

---

## Quick start

### Structured logging

```python
from effgen.observability import get_logger, configure_logging

configure_logging(level="INFO")  # call once at startup
log = get_logger(__name__)
log.event("model.call.started", provider="cerebras", model="gpt-oss-120b", cached_tokens=0)
# → {"ts": "...", "level": "INFO", "event": "model.call.started", "attributes": {...}, ...}
```

Secrets are redacted at the encoder, so an API key passed in an attribute never
reaches the log file. See [logging.md](logging.md).

### Metrics

```python
from effgen.observability import record_model_call, export_metrics

# Histograms auto-record on agent/model/tool calls. You can also record directly:
record_model_call(provider="cerebras", model="gpt-oss-120b", outcome="ok", latency=0.42)
print(export_metrics())  # Prometheus text format
```

The FastAPI server exposes the same data at `GET /metrics`. See [metrics.md](metrics.md).

### SLO tracking

```python
from effgen.observability.slo import SLOTracker, SLO

tracker = SLOTracker()
tracker.register(SLO("model_call_success", target_pct=99.0, window_seconds=3600))
tracker.record("model_call_success", ok=True)
print(tracker.burn_rate("model_call_success"))  # 1.0 = on budget, >1.0 = alert
```

Exposed at `GET /slo`. See [slos.md](slos.md).

### Tracing

```python
from effgen.observability import (
    setup_tracing, ParentBasedSampler, TraceIdRatioSampler, start_agent_run,
)

setup_tracing(sampler=ParentBasedSampler(TraceIdRatioSampler(0.1)))  # sample 10%
with start_agent_run(preset="my_agent", task="hello"):
    ...
```

Every span attribute name lives in `effgen/observability/spans.py`. See [tracing.md](tracing.md).

### Reliability primitives

```python
from effgen.reliability import Retry, retryable, CircuitBreaker, Bulkhead

@retryable(Retry(max_attempts=3, base_delay=0.5, jitter=True))
def call_model(prompt):
    ...

breaker = CircuitBreaker("cerebras", failure_threshold=5, recovery_timeout=30)
if breaker.is_call_permitted():
    try:
        result = call_model("hello")
        breaker.on_success()
    except Exception as exc:
        breaker.on_failure(exc)
        raise

bulkhead = Bulkhead("cerebras", max_concurrency=10, queue_size=50)
with bulkhead.acquire():
    call_model("hello")
```

See [reliability.md](reliability.md).

---

## Design principles

1. **Never block agent work on observability.** All telemetry is async / non-blocking.
   A failed export is not a failed inference.
2. **Redaction is code, not configuration.** Secret patterns (API keys, bearer
   tokens, webhook URLs) are matched and stripped at the log encoder, so every path
   is covered.
3. **Explicit sampling.** The tracing sampler is declared explicitly — there is no
   implicit `head=1.0` shipped to production.
4. **Bound every wait.** Timeouts are declared in config and propagated into adapter
   `httpx` clients and tool executions. No default `timeout=None`.

---

## Configuration reference

Reliability defaults live in `effgen.reliability.ReliabilityConfig`:

| Timeout | Default | Applies to |
|---|---|---|
| `model_call` | 60 s | Model provider calls |
| `tool_call` | 30 s | Tool executions |
| `http` | 20 s | Raw HTTP requests |

Logging level is set with `configure_logging(level=...)`. Tracing samplers
(`AlwaysOnSampler`, `AlwaysOffSampler`, `TraceIdRatioSampler`, `RateLimitedSampler`,
`ParentBasedSampler`) are passed to `setup_tracing(sampler=...)`.

---

## See also

- [Load testing](loadtest.md) — benchmark a deployment with `effgen loadtest`.
- [Alerting](alerting.md) — wire Alertmanager rules and Slack/Discord webhooks.
- [Chaos harness](chaos.md) and [Fuzz harness](fuzz.md) — test the reliability stack.
