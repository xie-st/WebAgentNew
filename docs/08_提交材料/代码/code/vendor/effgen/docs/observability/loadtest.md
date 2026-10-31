# Load Testing

effGen includes a built-in load-testing harness (`effgen/tools/loadgen.py`)
and a `effgen loadtest` CLI command for benchmarking your deployment.

---

## Quick start

```bash
# 30-second mock run, concurrency=10 (default)
effgen loadtest

# Custom duration and concurrency
effgen loadtest --duration 60 --concurrency 20

# Live run against Cerebras
effgen loadtest --provider cerebras --model gpt-oss-120b --concurrency 5 --duration 60

# Synthetic scenario, save report to custom path
effgen loadtest --scenario synthetic --output /tmp/report.json
```

---

## CLI reference

```
effgen loadtest [OPTIONS]

Options:
  -c, --concurrency N      Number of concurrent virtual users (default: 10)
  -d, --duration SECONDS   Test duration in seconds (default: 30)
  -s, --scenario SCENARIO  Workload scenario: fixed | synthetic | multi_tool
                           (default: fixed)
      --ramp-up SECONDS    Linear ramp-up period — VUs start gradually (default: 0)
      --think-time SECONDS Think-time between requests per VU (default: 0)
      --request-timeout N  Per-request timeout in seconds (default: 60)
      --provider NAME      Provider for live runs (e.g. cerebras, openai)
      --model MODEL_ID     Model id for live runs (e.g. gpt-oss-120b)
  -o, --output PATH        Write the report to PATH; a .html path renders the
                           self-contained HTML report, anything else writes
                           JSON (default: stdout only)
      --report PATH.html   Write a self-contained HTML report — latency
                           percentiles, throughput, error rate, and the error
                           breakdown
```

The HTML report opens with no network access; every style, script, and chart is
inline. See [CLI developer-experience surfaces](../dx/cli.md) for the shared
report surface across `compare`, `eval`, `cost`, and `loadtest`.

---

## Scenarios

| Scenario | Description |
|---|---|
| `fixed` | Every virtual user sends the same short prompt |
| `synthetic` | Prompts cycle through a library of 10 varied questions |
| `multi_tool` | VUs send calculator expressions (varies by VU ID + request index) |

---

## Report format

The JSON report written to disk (and printed to stdout) contains:

```json
{
  "scenario": "fixed",
  "concurrency": 10,
  "duration_s": 30.012,
  "total_requests": 4821,
  "successful_requests": 4821,
  "failed_requests": 0,
  "error_rate": 0.0,
  "throughput_rps": 160.63,
  "latency": {
    "p50": 0.0018,
    "p95": 0.0042,
    "p99": 0.0058,
    "min": 0.001,
    "max": 0.009,
    "mean": 0.0021,
    "stdev": 0.0009
  },
  "provider": null,
  "model": null
}
```

All latency values are in **seconds**.

---

## Library API

```python
from pathlib import Path
from effgen.tools.loadgen import LoadConfig, LoadGenerator, LoadScenario

cfg = LoadConfig(
    concurrency=10,
    duration=30.0,
    scenario=LoadScenario.SYNTHETIC,
    request_timeout=60.0,          # always explicit — never None
    output_path=Path("/tmp/report.json"),
)

gen = LoadGenerator(cfg)
report = gen.run()

print(f"p95 latency: {report.p95_latency * 1000:.1f} ms")
print(f"throughput : {report.throughput:.1f} req/s")
print(f"error rate : {report.error_rate * 100:.2f}%")

# Access raw request results
for r in report.raw_results[:5]:
    print(r.latency, r.success, r.error)
```

### Custom target callable

To run against your own backend, pass an `async` callable:

```python
import asyncio

async def my_target(prompt: str) -> str:
    # Call your model adapter, API, etc.
    await asyncio.sleep(0.1)   # simulate 100 ms latency
    return f"Response to: {prompt}"

gen = LoadGenerator(cfg, target=my_target)
report = gen.run()
```

---

## Writing the report to a file

By default the report is only printed to stdout. Pass `--output PATH` to also
write the JSON report to a file:

```bash
effgen loadtest --output report.json
```

Parent directories are created automatically.

---

## Integration with alerting

After a load test, evaluate latency thresholds and fire alerts if needed:

```python
from effgen.observability.alerting import AlertWebhook, Alert, AlertSeverity

if report.p95_latency > 10.0:
    hook = AlertWebhook("https://hooks.slack.com/...")
    hook.fire(Alert(
        name="LoadTestP95High",
        severity=AlertSeverity.WARNING,
        summary=f"p95 latency {report.p95_latency:.1f}s exceeds 10s threshold",
        value=report.p95_latency,
        threshold=10.0,
    ))
```
