# effGen API Server — Audit Log

Every request/response pair is appended as a JSON line (JSONL) to a daily
log file.  The audit log is designed for compliance, forensics, and debugging.

---

## Location

```
~/.effgen/audit/<YYYY-MM-DD>.jsonl
```

Override the directory:

```bash
export EFFGEN_AUDIT_DIR=/var/log/effgen/audit
```

---

## Record format

Each line is a JSON object with these fields:

| Field | Type | Description |
|---|---|---|
| `ts` | string | ISO-8601 UTC timestamp |
| `principal` | string | JWT `sub` claim (or `"anonymous"`) |
| `roles` | list[str] | JWT roles at time of request |
| `endpoint` | string | `"METHOD /path"` |
| `request_summary` | string | Method, path, scrubbed query string |
| `response_summary` | string | `"HTTP <status> (<content-type>)"` |
| `outcome` | string | `"ok"`, `"error"`, or `"denied"` |
| `request_id` | string | `X-Request-ID` header value |
| `duration_ms` | float | Handler wall-clock time in milliseconds |
| `extra` | object | Reserved for future extension |

**Fields intentionally absent:**
- Request or response body content
- `Authorization` header value
- API keys or secrets in query strings (scrubbed to `[REDACTED]`)

---

## Sample record

```jsonl
{"ts": "2026-05-26T14:32:01.123456+00:00", "principal": "alice@example.com", "roles": ["researcher"], "endpoint": "POST /v1/chat/completions", "request_summary": "POST /v1/chat/completions", "response_summary": "HTTP 200 (application/json)", "outcome": "ok", "request_id": "req-a1b2c3", "duration_ms": 312.45, "extra": {}}
{"ts": "2026-05-26T14:32:05.654321+00:00", "principal": "anonymous", "roles": [], "endpoint": "GET /admin", "request_summary": "GET /admin", "response_summary": "HTTP 401 (application/json)", "outcome": "denied", "request_id": "", "duration_ms": 1.12, "extra": {}}
```

---

## Outcomes

| Value | Meaning |
|---|---|
| `ok` | HTTP 1xx–3xx |
| `denied` | HTTP 401 (unauthenticated), 403 (RBAC), or 429 (budget cap) |
| `error` | other HTTP 4xx or 5xx |

---

## Sensitive field scrubbing

Query string parameters that look like secrets are automatically scrubbed:

```
?api_key=sk-secret123  →  ?api_key=[REDACTED]
?token=xyz             →  ?token=[REDACTED]
```

Patterns matched: `key`, `token`, `secret`, `password`, `auth`, `api_key`,
`api-key`, `bearer`.

---

## Python API

### Read today's records

```python
from effgen.server.audit import read_audit_records, AuditRecord

records: list[AuditRecord] = read_audit_records()  # today
records_yesterday = read_audit_records("2026-05-25")
```

### Write a record manually

```python
from effgen.server.audit import AuditRecord, write_audit_record_sync
import asyncio

record = AuditRecord(
    ts="2026-05-26T12:00:00+00:00",
    principal="bot@example.com",
    roles=["researcher"],
    endpoint="POST /custom",
    request_summary="POST /custom",
    response_summary="HTTP 200",
    outcome="ok",
)

# Async (preferred in async handlers)
await write_audit_record(record)

# Sync (for CLI or scripts)
write_audit_record_sync(record)
```

---

## AuditMiddleware

The middleware is auto-installed by `create_app()`. To add it manually:

```python
from effgen.server.audit import AuditMiddleware

# `add_middleware` is LIFO: add AuditMiddleware *after* AuthMiddleware so that
# Audit wraps Auth (Audit runs outermost). The principal is read in Audit's
# response phase — after Auth has populated request.state.user — so every
# record carries the resolved principal, including for rejected requests.
app.add_middleware(AuditMiddleware)
```

---

## Rotating / shipping logs

The daily JSONL files can be shipped to any log aggregator:

```bash
# Tail today's log
tail -f ~/.effgen/audit/$(date +%Y-%m-%d).jsonl | jq .

# Send to Elasticsearch
filebeat -e -c filebeat-effgen.yml
```

---

## Security considerations

- The audit directory should be writable only by the effGen server process.
- The log files contain principal identifiers — treat them as sensitive.
- Body content is never logged, preventing credential or PII leakage.
- `Authorization` header values are never logged.
