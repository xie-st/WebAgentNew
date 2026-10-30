"""Audit log for the effGen API server.

Every request/response pair is appended as a JSON line to
``~/.effgen/audit/<YYYY-MM-DD>.jsonl``.

Record fields
-------------
ts              ISO-8601 UTC timestamp
principal       JWT sub claim (or "anonymous" / "dev-user")
roles           list of role names from the JWT
endpoint        HTTP method + path
request_summary brief description (method, path, query)
response_summary status code + content-type (no body content)
outcome         "ok" | "error" | "denied"
request_id      value of X-Request-ID header if present
duration_ms     wall-clock ms for the handler

Sensitive fields intentionally omitted:
  - Authorization header value
  - Request/response body content
  - API keys, passwords, tokens in query strings
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive-field scrubbing
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = re.compile(
    r"(key|token|secret|password|auth|api[-_]?key|bearer)[=: ]+[^\s&\"']+",
    re.IGNORECASE,
)


def _scrub(text: str) -> str:
    """Replace secret-looking values in *text* with [REDACTED]."""
    return _SECRET_PATTERNS.sub(r"\1=[REDACTED]", text)


def _safe_query(query: str) -> str:
    """Scrub sensitive params from a URL query string."""
    return _scrub(query) if query else ""


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------


@dataclass
class AuditRecord:
    """One audit-log entry: who called which endpoint, and the outcome."""

    ts: str
    principal: str
    roles: list[str]
    endpoint: str
    request_summary: str
    response_summary: str
    outcome: str
    request_id: str = ""
    duration_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Return the record as a single JSON line."""
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

_AUDIT_DIR: Path | None = None


def _get_audit_dir() -> Path:
    global _AUDIT_DIR
    if _AUDIT_DIR is not None:
        return _AUDIT_DIR
    base = Path(os.getenv("EFFGEN_AUDIT_DIR", Path.home() / ".effgen" / "audit"))
    base.mkdir(parents=True, exist_ok=True)
    _AUDIT_DIR = base
    return _AUDIT_DIR


def _audit_file() -> Path:
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    return _get_audit_dir() / f"{today}.jsonl"


_write_lock = asyncio.Lock()


async def write_audit_record(record: AuditRecord) -> None:
    """Append *record* to today's audit log (async, file-locked)."""
    line = record.to_jsonl() + "\n"
    try:
        async with _write_lock:
            with _audit_file().open("a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write audit record: %s", exc)


def write_audit_record_sync(record: AuditRecord) -> None:
    """Synchronous variant for non-async contexts."""
    line = record.to_jsonl() + "\n"
    try:
        with _audit_file().open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write audit record (sync): %s", exc)


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------


class AuditMiddleware:
    """ASGI middleware that appends an audit record for every HTTP request.

    Attach after :class:`~effgen.server.auth.AuthMiddleware` so that
    ``scope["state"]["user"]`` is already populated.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        # NOTE: do not read the principal here. AuthMiddleware runs *inside*
        # this middleware and only populates scope["state"]["user"] once the
        # inner app has been invoked. The principal is resolved in the
        # `finally` block below, after self.app(...) has run.
        state = scope.setdefault("state", {})

        method = scope.get("method", "GET")
        path: str = scope.get("path", "/")
        query = scope.get("query_string", b"").decode("latin-1", errors="replace")
        request_id = ""
        for name, value in scope.get("headers", []):
            if name.lower() == b"x-request-id":
                request_id = value.decode("latin-1", errors="replace")
                break

        request_summary = f"{method} {path}"
        if query:
            request_summary += f"?{_safe_query(query)}"

        response_status = 200
        response_ct = ""

        async def _send_wrapper(event: dict[str, Any]) -> None:
            nonlocal response_status, response_ct
            if event["type"] == "http.response.start":
                response_status = event.get("status", 200)
                for hname, hval in event.get("headers", []):
                    if hname.lower() == b"content-type":
                        response_ct = hval.decode("latin-1", errors="replace")
                        break
            await send(event)

        outcome = "ok"
        try:
            await self.app(scope, receive, _send_wrapper)
        except Exception:
            outcome = "error"
            raise
        finally:
            if response_status in (401, 403, 429):
                outcome = "denied"
            elif response_status >= 400:
                outcome = "error"

            # Resolve the principal now that inner middleware (auth) has run.
            user = state.get("user")
            principal = getattr(user, "sub", "anonymous") if user else "anonymous"
            roles: list[str] = getattr(user, "roles", []) if user else []

            duration_ms = (time.perf_counter() - start) * 1000.0
            response_summary = f"HTTP {response_status}"
            if response_ct:
                response_summary += f" ({response_ct.split(';')[0].strip()})"

            record = AuditRecord(
                ts=datetime.now(tz=UTC).isoformat(),
                principal=principal,
                roles=roles,
                endpoint=f"{method} {path}",
                request_summary=request_summary,
                response_summary=response_summary,
                outcome=outcome,
                request_id=request_id,
                duration_ms=round(duration_ms, 2),
            )
            # Fire-and-forget; don't block the response
            asyncio.ensure_future(write_audit_record(record))


# ---------------------------------------------------------------------------
# Helper: read recent audit records (for dashboard / tests)
# ---------------------------------------------------------------------------


def read_audit_records(date_str: str | None = None) -> list[AuditRecord]:
    """Read all audit records for *date_str* (YYYY-MM-DD, defaults to today)."""
    if date_str is None:
        date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    path = _get_audit_dir() / f"{date_str}.jsonl"
    if not path.exists():
        return []
    records: list[AuditRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            records.append(AuditRecord(**data))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed audit line: %s", exc)
    return records
