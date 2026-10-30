# Chaos Harness

The effGen chaos harness lets you inject deterministic, reproducible faults into
provider calls so you can validate that your reliability stack (retries, circuit
breakers, fallback routing) behaves correctly under adverse conditions.

## Core concepts

| Concept | What it does |
|---------|-------------|
| `Chaos(seed)` | Deterministic fault injector. Same seed → same sequence of faults every run. |
| `ChaosRule` | A rule binding a fault type to a provider with a trigger condition (`every_nth` or `fault_rate`). |
| `ChaosMiddleware` | Thin wrapper that intercepts adapter calls and runs fault injection before forwarding. |
| `AllProvidersFailed` | Exception raised by agent-level harness when every provider is exhausted. |

## Fault types

| Class | Exception raised | Attributes |
|-------|-----------------|------------|
| `NetworkTimeout` | `ChaosNetworkTimeout` | `provider`, `limit` |
| `Http5xx` | `ChaosHttp5xxError` | `provider`, `status_code` |
| `Http429` | `ChaosHttp429Error` | `provider`, `retry_after`, `status_code=429` |
| `SlowResponse` | *(no exception — just delays)* | `delay_s` |
| `PartialResponse` | `ChaosPartialResponseError` | `provider`, `partial_content` |
| `MalformedJSON` | `ChaosMalformedJSONError` | `provider`, `raw` |

### Exception integration

* `ChaosNetworkTimeout` is a subclass of `TimeoutError` → `is_transient_error()`
  returns `True`, so the standard retry policy handles it automatically.
* `ChaosHttp5xxError` and `ChaosHttp429Error` carry `status_code` → same path.
* `ChaosHttp429Error.retry_after` is honoured by `Retry.compute_delay()`.

## Quick start

```python
from effgen.reliability.chaos import (
    Chaos, Http5xx, Http429, NetworkTimeout, AllProvidersFailed
)
from effgen.models.registry import ProviderRegistry

# --- Set up chaos with a fixed seed ---
chaos = Chaos(seed=42)
chaos.add_rule("primary", Http5xx, every_nth=3)        # fail every 3rd call
chaos.add_rule("primary", Http429, every_nth=6,        # rate-limit every 6th call
               retry_after=2.0)

# --- Attach to the registry ---
registry = ProviderRegistry.with_chaos(chaos)

# --- Use in a call loop ---
try:
    result = registry.call("primary", adapter.generate, prompt="Hello")
except ChaosHttp5xxError:
    result = fallback_adapter.generate(prompt="Hello")
```

## Scenarios

Four canonical scenarios are provided in `tests/reliability/test_chaos.py`.

### Scenario A — Primary 5xx every 3rd call, fallback 200

```
chaos.add_rule("primary", Http5xx, every_nth=3)
```

* Every 3rd call to *primary* raises `ChaosHttp5xxError`.
* The router / agent loop catches it and falls over to *fallback*.
* All 6 calls ultimately return a successful response.
* The circuit breaker's `total_failures` reflects injected faults.

### Scenario B — 429 with Retry-After, retry honours the wait

```
chaos.add_rule("primary", Http429, every_nth=1, max_fires=1, retry_after=2.0)
```

* First call to *primary* raises `ChaosHttp429Error(retry_after=2.0)`.
* The retry loop reads `exc.retry_after` and waits 2 seconds before retrying.
* Second attempt succeeds (rule has `max_fires=1` so it fires only once).

### Scenario C — Slow response ≥ timeout

```python
# Instant stall: the call raises a TimeoutError immediately.
chaos.add_rule("primary", NetworkTimeout, every_nth=1, limit=60.0)

# Real wall-clock stall: the call sleeps past the enforced timeout.
from effgen.reliability.timeouts import apply_timeout
chaos.add_rule("primary", SlowResponse, every_nth=1, delay_s=90.0)
guarded = apply_timeout(call_primary, 60.0, operation="model_call")
```

* `NetworkTimeout` raises `ChaosNetworkTimeout` (a subclass of `TimeoutError`).
* A `SlowResponse` longer than the timeout, wrapped by `apply_timeout`, raises
  effGen's `TimeoutError` — the guarded call returns within
  `timeout + 1 s`, so a stalled provider never hangs the agent.
* `is_transient_error()` returns `True` for both → retry / circuit breaker
  handle them.
* After `failure_threshold` consecutive failures the circuit opens.
* After `recovery_timeout`, the circuit advances to `HALF_OPEN`.

### Scenario D — All providers fail → AllProvidersFailed

```python
for provider in ["primary", "secondary", "tertiary"]:
    chaos.add_rule(provider, Http5xx, every_nth=1)
```

* Every provider is faulted on every call.
* The agent harness collects failures and raises `AllProvidersFailed`.
* `AllProvidersFailed` is **never** a silent empty string — it always has a
  non-empty message and a `failures` dict mapping provider → last exception.

### Inspecting the trace

Scenarios A and B don't just check return values — they assert the emitted
span tree, the way you'd read a trace in your backend:

* **Scenario A** runs `agent.run → agent.iteration → router.decision / model.call`
  and the trace shows the switch: two `model.call` spans
  (`provider=primary`, `outcome=error`, then `provider=fallback`, `outcome=ok`)
  and two `router.decision` spans whose `effgen.router.selected_provider`
  flips from `primary` to `fallback`.
* **Scenario B** drives the real `@retryable` policy inside a `model.call` span;
  the span carries an `effgen.retry.attempt` event whose `effgen.retry.delay_s`
  equals the honoured `Retry-After` value.

You can reproduce this with the in-memory exporter:

```python
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from effgen.observability import setup_tracing, reset_tracing
from effgen.observability.tracing import AlwaysOnSampler

reset_tracing()
exporter = InMemorySpanExporter()
setup_tracing(sampler=AlwaysOnSampler(), exporter=exporter)
# ... run a chaos scenario ...
spans = exporter.get_finished_spans()
```

## Rule parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | required | Provider to target. Use `"*"` for all providers. |
| `fault_type` | `type[FaultBase]` | required | Which fault to inject. |
| `every_nth` | `int \| None` | `None` | Fire on every *nth* call (deterministic). |
| `fault_rate` | `float` | `0.0` | Probability to fire per call (probabilistic). |
| `max_fires` | `int \| None` | `None` | Cap on total fires. `None` = unlimited. |
| `**params` | `Any` | — | Extra kwargs forwarded to the fault constructor. |

### `every_nth` examples

| Value | Fires on calls |
|-------|---------------|
| `1` | 1, 2, 3, 4, … (every call) |
| `3` | 3, 6, 9, 12, … |
| `10` | 10, 20, 30, … |

### `max_fires` examples

```python
# Fire exactly once — ideal for retry tests
chaos.add_rule("p", Http429, every_nth=1, max_fires=1, retry_after=2.0)

# Fire at most 5 times
chaos.add_rule("p", Http5xx, fault_rate=0.5, max_fires=5)
```

## Determinism

The PRNG is seeded once at `Chaos(seed=...)` construction time.  Because
`every_nth` rules are purely counter-based they are seed-independent.
`fault_rate` rules consume PRNG draws and therefore produce a different
sequence per seed — but the **same seed always produces the same sequence**.

```python
@pytest.mark.parametrize("seed", range(10))
def test_deterministic(seed: int) -> None:
    chaos1 = Chaos(seed=seed)
    chaos1.add_rule("p", Http5xx, fault_rate=0.5)

    chaos2 = Chaos(seed=seed)
    chaos2.add_rule("p", Http5xx, fault_rate=0.5)

    results1 = [...]  # collect 20 fault outcomes
    results2 = [...]
    assert results1 == results2  # always True
```

Use `chaos.reset()` to replay the same sequence from the start:

```python
chaos.reset()  # re-seeds PRNG, resets all rule counters
```

## Running the chaos tests

```bash
# Run only chaos tests
pytest tests/reliability/test_chaos.py -v -m reliability

# Run all reliability tests
pytest tests/reliability/ -v -m reliability

# Run with specific seed (via parametrize)
pytest tests/reliability/test_chaos.py -k "seed0" -v
```

## Integration with ProviderRegistry

```python
from effgen.models.registry import ProviderRegistry
from effgen.reliability.chaos import Chaos, Http5xx

chaos = Chaos(seed=99)
chaos.add_rule("cerebras", Http5xx, every_nth=5)

registry = ProviderRegistry.with_chaos(chaos)

# Synchronous call
result = registry.call("cerebras", adapter.generate, prompt="Hello")

# Asynchronous call
result = await registry.async_call("cerebras", adapter.agenerate, prompt="Hello")
```

## Writing new chaos tests

```python
import pytest
from effgen.reliability.chaos import Chaos, Http5xx, ChaosHttp5xxError

pytestmark = pytest.mark.reliability

@pytest.mark.parametrize("seed", range(10))
def test_my_scenario(seed: int) -> None:
    chaos = Chaos(seed=seed)
    chaos.add_rule("my_provider", Http5xx, every_nth=3)

    failures = 0
    for _ in range(9):
        try:
            chaos.maybe_inject("my_provider")
        except ChaosHttp5xxError:
            failures += 1

    assert failures == 3  # calls 3, 6, 9
```
