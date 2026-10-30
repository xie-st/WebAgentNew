# Alerting

effGen ships a complete Alertmanager-compatible alert rule pack and an
`AlertWebhook` class that dispatches alert payloads to **Slack** or **Discord**
webhooks.

---

## Alert rule pack

The bundled rule file lives at:

```
docs/observability/alert_rules.yaml
```

It defines six alerts across five rule groups:

| Alert | Severity | Condition | Window |
|---|---|---|---|
| `HighErrorRate` | critical | Error rate > 5% | 10 min |
| `HighP95Latency` | warning | p95 latency > 10 s | 5 min |
| `CostBurnHigh` | warning | Est. cost > $10/day | instant |
| `SLOFastBurn` | critical | Burn rate > 14.4× | instant |
| `SLOSlowBurn` | warning | Burn rate > 3× | 60 min |
| `CircuitBreakerOpen` | warning | Breaker OPEN > 1 min | 1 min |

### Load into Prometheus

```yaml
# prometheus.yml
rule_files:
  - "docs/observability/alert_rules.yaml"
```

### Validate

If [promtool](https://prometheus.io/docs/prometheus/latest/command-line/promtool/)
is installed:

```bash
promtool check rules docs/observability/alert_rules.yaml
```

Or validate programmatically from Python:

```python
from pathlib import Path
from effgen.observability.alerting import validate_alert_rules_yaml

ok, errors = validate_alert_rules_yaml(Path("docs/observability/alert_rules.yaml"))
assert ok, errors
```

---

## AlertWebhook

`AlertWebhook` posts an `Alert` payload to a Slack or Discord webhook URL.
The URL type is auto-detected.

### Setup

```python
from effgen.observability.alerting import AlertWebhook, Alert, AlertSeverity

hook = AlertWebhook("https://hooks.slack.com/services/T.../B.../TOKEN")
# or
hook = AlertWebhook("https://discord.com/api/webhooks/WEBHOOK_ID/WEBHOOK_TOKEN")
```

### Fire an alert

```python
alert = Alert(
    name="HighErrorRate",
    severity=AlertSeverity.CRITICAL,
    summary="Error rate exceeded 5% for 10 minutes",
    value=0.08,
    threshold=0.05,
    labels={"provider": "cerebras", "model": "gpt-oss-120b"},
)
result = hook.fire(alert)
# {"ok": True, "webhook": "https://hooks.slack.com/***", "status": 200, ...}
```

### Fire multiple alerts

```python
results = hook.fire_many([alert1, alert2, alert3])
```

### Non-raising design

`AlertWebhook.fire()` **never raises**.  Network errors, HTTP errors, and
payload issues are caught, logged at `ERROR` level, and returned as:

```python
{"ok": False, "error": "<description>", "webhook": "<redacted>"}
```

This guarantees that failed alert delivery never blocks the agent loop.

---

## URL redaction

Webhook URLs contain secret tokens in the path.  effGen **never logs the full
URL**.  All log lines show only the origin (scheme + host):

```
Input  : https://hooks.slack.com/services/T.../B.../TOKEN
Logged : https://hooks.slack.com/***
```

This is enforced at the `AlertWebhook` layer (before the URL is passed to any
logger) and also covered by the global `Redactor` in structured log lines.

---

## Integration with SLO tracker

`check_slo_and_alert` reads one SLO's current status from the tracker and
fires through the webhook when its burn rate is over threshold — the
"read the live meter, decide, fire" loop, as a single reusable call instead
of hand-written glue at every call site:

```python
from effgen.observability import get_slo_tracker
from effgen.observability.alerting import AlertWebhook, check_slo_and_alert

tracker = get_slo_tracker()
webhook = AlertWebhook("https://hooks.slack.com/services/...")

# Returns the AlertWebhook.fire() result when the SLO is over budget,
# or None when it is within budget (nothing sent).
check_slo_and_alert(tracker, "model_call_success", webhook, burn_rate_threshold=14.4)
```

Call it on a schedule — a cron job, a background task, or after each batch of
requests — for every SLO you want paged on. It is also importable directly
from ``effgen`` (``from effgen import check_slo_and_alert``).

The server exposes the same tracker at `GET /slo` (public, no auth — see
[SLO tracking](slos.md#http-endpoint)), so an external process can poll
burn rates and drive this same bridge without importing effGen.

---

## Configuring Alertmanager

Add a webhook receiver to your Alertmanager configuration:

```yaml
# alertmanager.yml
receivers:
  - name: effgen-slack
    webhook_configs:
      - url: "https://hooks.slack.com/services/..."
        send_resolved: true
```

For programmatic firing from effGen code, use `AlertWebhook.fire()` directly —
you do not need a running Alertmanager instance.
