"""Tests for effgen.server.audit — audit log correctness, no-secrets guarantee."""
from __future__ import annotations

import asyncio
import json

import pytest

from effgen.server.audit import (
    AuditMiddleware,
    AuditRecord,
    _get_audit_dir,
    _scrub,
    read_audit_records,
    write_audit_record,
    write_audit_record_sync,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_tmp_dir(tmp_path, monkeypatch):
    """Redirect audit output to a temp directory."""
    monkeypatch.setenv("EFFGEN_AUDIT_DIR", str(tmp_path))
    # Reset the cached dir
    import effgen.server.audit as _audit

    _audit._AUDIT_DIR = None
    yield tmp_path
    _audit._AUDIT_DIR = None


def _make_record(**kwargs) -> AuditRecord:
    defaults = {
        "ts": "2026-05-26T12:00:00+00:00",
        "principal": "test-user",
        "roles": ["researcher"],
        "endpoint": "GET /v1/chat/completions",
        "request_summary": "GET /v1/chat/completions",
        "response_summary": "HTTP 200 (application/json)",
        "outcome": "ok",
        "request_id": "req-abc-123",
        "duration_ms": 42.0,
    }
    defaults.update(kwargs)
    return AuditRecord(**defaults)


# ---------------------------------------------------------------------------
# AuditRecord — fields and serialisation
# ---------------------------------------------------------------------------


class TestAuditRecord:
    def test_required_fields_present(self):
        r = _make_record()
        assert r.ts
        assert r.principal
        assert isinstance(r.roles, list)
        assert r.endpoint
        assert r.request_summary
        assert r.response_summary
        assert r.outcome in ("ok", "error", "denied")

    def test_to_jsonl_is_valid_json(self):
        r = _make_record()
        line = r.to_jsonl()
        data = json.loads(line)
        assert data["principal"] == "test-user"
        assert data["outcome"] == "ok"

    def test_no_secrets_in_default_record(self):
        """The standard fields must not contain raw secret values."""
        r = _make_record(
            request_summary="GET /secure?token=sk-super-secret&other=value",
        )
        # The request_summary is stored as-is — scrubbing is middleware responsibility.
        # Verify our _scrub helper works.
        scrubbed = _scrub(r.request_summary)
        assert "sk-super-secret" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_jsonl_no_auth_header_field(self):
        """AuditRecord must NOT have an 'authorization' field."""
        r = _make_record()
        data = json.loads(r.to_jsonl())
        assert "authorization" not in data
        assert "Authorization" not in data


# ---------------------------------------------------------------------------
# Write / read round-trip
# ---------------------------------------------------------------------------


class TestAuditFileIO:
    @pytest.mark.asyncio
    async def test_write_and_read_async(self, audit_tmp_dir):
        r = _make_record()
        await write_audit_record(r)
        records = read_audit_records()
        assert len(records) == 1
        assert records[0].principal == "test-user"

    def test_write_and_read_sync(self, audit_tmp_dir):
        r = _make_record(principal="sync-user")
        write_audit_record_sync(r)
        records = read_audit_records()
        assert any(rec.principal == "sync-user" for rec in records)

    @pytest.mark.asyncio
    async def test_multiple_records_appended(self, audit_tmp_dir):
        for i in range(5):
            await write_audit_record(_make_record(request_id=f"req-{i}"))
        records = read_audit_records()
        assert len(records) == 5
        req_ids = {r.request_id for r in records}
        assert req_ids == {f"req-{i}" for i in range(5)}

    def test_read_nonexistent_date_returns_empty(self, audit_tmp_dir):
        records = read_audit_records("1999-01-01")
        assert records == []

    def test_audit_dir_created_automatically(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "nested" / "audit"
        monkeypatch.setenv("EFFGEN_AUDIT_DIR", str(new_dir))
        import effgen.server.audit as _audit

        _audit._AUDIT_DIR = None
        _get_audit_dir()
        assert new_dir.exists()
        _audit._AUDIT_DIR = None


# ---------------------------------------------------------------------------
# Scrubbing helper
# ---------------------------------------------------------------------------


class TestScrub:
    @pytest.mark.parametrize(
        "input_str, should_be_redacted",
        [
            ("Authorization: Bearer sk-secret-token", "sk-secret-token"),
            ("api_key=ABCDEFG12345", "ABCDEFG12345"),
            ("token=my-tok&other=val", "my-tok"),
            ("password: hunter2", "hunter2"),
            ("key=abc123", "abc123"),
        ],
    )
    def test_sensitive_patterns_redacted(self, input_str, should_be_redacted):
        scrubbed = _scrub(input_str)
        assert should_be_redacted not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_safe_string_unchanged(self):
        safe = "GET /health?format=json"
        assert _scrub(safe) == safe


# ---------------------------------------------------------------------------
# AuditMiddleware — ASGI integration
# ---------------------------------------------------------------------------


class TestAuditMiddleware:
    async def _invoke(
        self,
        path: str = "/v1/chat",
        method: str = "POST",
        status: int = 200,
        user_sub: str | None = "test-user",
        user_roles: list[str] | None = None,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> list[AuditRecord]:
        """Run AuditMiddleware for a single request and return written records."""
        from effgen.server.audit import write_audit_record

        written: list[AuditRecord] = []
        original_write = write_audit_record

        async def _capture(record: AuditRecord) -> None:
            written.append(record)

        import effgen.server.audit as _audit

        _audit.write_audit_record = _capture  # type: ignore[assignment]
        try:
            class _FakeUser:
                sub = user_sub or "anonymous"
                roles = user_roles or []

            scope: dict = {
                "type": "http",
                "method": method,
                "path": path,
                "query_string": b"",
                "headers": extra_headers or [],
                "state": {"user": _FakeUser() if user_sub else None},
            }

            async def receive():
                return {"type": "http.request", "body": b"{}"}

            async def send(event):
                pass

            async def inner(s, r, snd):
                await snd({"type": "http.response.start", "status": status, "headers": []})
                await snd({"type": "http.response.body", "body": b""})

            mw = AuditMiddleware(inner)
            await mw(scope, receive, send)
            # Give the fire-and-forget future a chance to run
            await asyncio.sleep(0)
        finally:
            _audit.write_audit_record = original_write  # type: ignore[assignment]

        return written

    @pytest.mark.asyncio
    async def test_record_written_on_ok_request(self):
        records = await self._invoke(path="/v1/chat", status=200)
        assert len(records) == 1
        r = records[0]
        assert r.outcome == "ok"
        assert r.principal == "test-user"
        assert "POST" in r.endpoint

    @pytest.mark.asyncio
    async def test_record_outcome_denied_on_401(self):
        records = await self._invoke(path="/secure", status=401, user_sub=None)
        assert len(records) == 1
        assert records[0].outcome == "denied"

    @pytest.mark.asyncio
    async def test_record_outcome_denied_on_403(self):
        records = await self._invoke(path="/admin", status=403)
        assert len(records) == 1
        assert records[0].outcome == "denied"

    @pytest.mark.asyncio
    async def test_roles_captured_from_user(self):
        records = await self._invoke(user_roles=["admin", "researcher"])
        assert records[0].roles == ["admin", "researcher"]

    @pytest.mark.asyncio
    async def test_request_id_captured(self):
        records = await self._invoke(
            extra_headers=[(b"x-request-id", b"req-xyz-999")]
        )
        assert records[0].request_id == "req-xyz-999"

    @pytest.mark.asyncio
    async def test_no_body_content_in_summary(self):
        """The response_summary must NOT contain body content."""
        records = await self._invoke(status=200)
        r = records[0]
        # response_summary should only be "HTTP 200" or "HTTP 200 (type)"
        assert "HTTP 200" in r.response_summary
        # must not contain body-level content
        assert "{" not in r.response_summary
        assert "}" not in r.response_summary

    @pytest.mark.asyncio
    async def test_no_auth_header_in_record(self):
        """Authorization header value must never appear in the audit record."""
        records = await self._invoke(
            extra_headers=[(b"authorization", b"Bearer sk-super-secret")]
        )
        assert len(records) == 1
        line = records[0].to_jsonl()
        assert "sk-super-secret" not in line

    @pytest.mark.asyncio
    async def test_principal_resolved_after_inner_sets_state(self):
        """Regression: the principal must be read *after* inner middleware runs.

        AuthMiddleware populates scope["state"]["user"] while it is being
        called by AuditMiddleware. The audit record must capture that
        principal, not "anonymous".
        """
        from effgen.server.audit import AuditMiddleware

        written: list[AuditRecord] = []

        async def _capture(record: AuditRecord) -> None:
            written.append(record)

        import effgen.server.audit as _audit

        original = _audit.write_audit_record
        _audit.write_audit_record = _capture  # type: ignore[assignment]
        try:
            class _User:
                sub = "late-bound-user"
                roles = ["researcher"]

            # state starts WITHOUT a user; inner sets it mid-request.
            scope: dict = {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "query_string": b"",
                "headers": [],
                "state": {},
            }

            async def receive():
                return {"type": "http.request", "body": b"{}"}

            async def send(event):
                pass

            async def inner(s, r, snd):
                # Simulate AuthMiddleware populating the user during the call.
                s.setdefault("state", {})["user"] = _User()
                await snd({"type": "http.response.start", "status": 200, "headers": []})
                await snd({"type": "http.response.body", "body": b""})

            mw = AuditMiddleware(inner)
            await mw(scope, receive, send)
            await asyncio.sleep(0)
        finally:
            _audit.write_audit_record = original  # type: ignore[assignment]

        assert len(written) == 1
        assert written[0].principal == "late-bound-user"
        assert written[0].roles == ["researcher"]

    @pytest.mark.asyncio
    async def test_outcome_denied_on_429_budget(self):
        records = await self._invoke(path="/v1/chat/completions", status=429)
        assert records[0].outcome == "denied"
