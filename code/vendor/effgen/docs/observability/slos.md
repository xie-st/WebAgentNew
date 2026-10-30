# effGen SLO Tracking

effGen ships an in-process SLO (Service Level Objective) tracker that maintains rolling windows of success/failure events and computes error-budget burn rates.

---

## Concepts

| Term | Meaning |
|---|---|
| **SLO** | A named objective: "99% of model calls succeed over 1 hour" |
| **Error budget** | The fraction of events that may fail: `(100 - target_pct) / 100` |
| **Burn rate** | How fast the error budget is being consumed: `bad_ratio / error_budget_fraction`. A rate of `1.0` is exactly on budget; `> 1.0` means over-budget. |
| **Rolling window** | Only events within the last `window_seconds` contribute to the current stats. Old events are lazily evicted on the next read. |

### Burn-rate examples (99% SLO, 1-hour window)

| Bad events | Total events | bad_ratio | burn_rate |
|---|---|---|---|
| 0 | 100 | 0.00 | 0.00 |
| 1 | 100 | 0.01 | **1.0** (on budget) |
| 2 | 100 | 0.02 | 2.0 (over budget) |
| 14 | 100 | 0.14 | **14.4** (fast-burn alert threshold) |

---

## Quick Start

```python
from effgen.observability.slo import SLO, get_tracker

tracker = get_tracker()

# Register once at startup:
tracker.register(SLO(
    name="model_call_success",
    target_pct=99.0,
    window_seconds=3600,       # 1-hour rolling window
    query='outcome="ok"',      # documentation-only
))

# After each model call:
try:
    response = model.call(...)
    tracker.record("model_call_success", ok=True)
except Exception:
    tracker.record("model_call_success", ok=False)
    raise

# Check burn rate at any time:
rate = tracker.burn_rate("model_call_success")
if rate > 14.4:
    alert("FAST BURN: model_call_success burn rate = {:.1f}x".format(rate))
```

---

## SLO Definition

```python
from effgen.observability.slo import SLO

slo = SLO(
    name="model_call_success",   # unique identifier
    target_pct=99.0,             # 99% success rate
    window_seconds=3600,         # 1-hour rolling window
    query='outcome="ok"',        # informal label (not evaluated)
)

# Computed property:
slo.error_budget_fraction  # → 0.01
```

---

## SLOTracker API

### `register(slo: SLO)`

Register an SLO. Idempotent — safe to call at startup on every import.

### `record(name: str, *, ok: bool, ts: float | None = None)`

Record one event. Pass `ok=True` for good events, `ok=False` for bad. Optionally pass `ts` for replay/testing (uses `time.monotonic()` by default).

### `burn_rate(name: str) -> float`

Error-budget burn rate over the current rolling window.

- Returns `0.0` when there are zero bad events.
- Returns `1.0` when consuming budget exactly at pace.
- Returns `float("inf")` for a 100%-target SLO that has any bad events.

### `status(name: str) -> dict`

Full status as a JSON-serialisable dict:

```json
{
  "name": "model_call_success",
  "target_pct": 99.0,
  "window_seconds": 3600,
  "query": "outcome=\"ok\"",
  "total_events": 1000,
  "good_events": 993,
  "bad_events": 7,
  "good_ratio": 0.993,
  "bad_ratio": 0.007,
  "burn_rate": 0.7,
  "within_budget": true
}
```

### `all_statuses() -> list[dict]`

Returns `status()` for all registered SLOs, sorted by name.

---

## HTTP Endpoint

The `/slo` endpoint (no auth required) returns JSON:

```
GET /slo HTTP/1.1
Host: localhost:8000

HTTP/1.1 200 OK
Content-Type: application/json

{
  "slos": [
    {
      "name": "model_call_success",
      "target_pct": 99.0,
      "burn_rate": 0.7,
      "within_budget": true,
      ...
    }
  ]
}
```

`/slo` reports the objectives *this server process registered* with the
tracker. It does not derive an SLO from request metrics, so it returns an empty
list — with a `detail` note — on a server that has served traffic but
registered no objective:

```json
{
  "slos": [],
  "detail": "No SLO objectives are registered in this process. ..."
}
```

Measured latency percentiles (p50/p95/p99), error-rate burn and availability
for the traffic a server has actually served are in the `slo` block of
`GET /dashboard/data.json`, and are rendered by the dashboard and by
`effgen top`.

---

## Alert Thresholds

Google SRE-style fast-burn / slow-burn thresholds for a 99% SLO:

| Window | Burn rate threshold | Meaning |
|---|---|---|
| 1 hour | > 14.4× | 5% error budget consumed in 1 hour (fast burn) |
| 6 hours | > 6× | 5% error budget consumed in 6 hours |
| 3 days | > 1× | Budget consumed before end of month |

These thresholds map to the `AlertWebhook` rules in `docs/observability/alerting.md`.

---

## Multiple SLOs

```python
tracker.register(SLO("model_call_success", 99.0, 3600))
tracker.register(SLO("tool_call_success",  99.5, 3600))
tracker.register(SLO("agent_run_success",  99.0, 86400))  # 24h window

# List all:
tracker.list_slos()
# → ['agent_run_success', 'model_call_success', 'tool_call_success']

# Status all:
tracker.all_statuses()
# → [{"name": "agent_run_success", ...}, ...]
```
