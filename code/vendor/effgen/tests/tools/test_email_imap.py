"""Tests for EmailIMAPTool — unit (mocked IMAP) + integration (skip without creds)."""

from __future__ import annotations

import os
from email.mime.text import MIMEText
from unittest.mock import patch

import pytest

from effgen.errors import MissingCredentialsError
from effgen.tools.builtin.email_imap import EmailIMAPTool, _decode_header_value, _extract_body

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmailIMAPToolUnit:
    def test_instantiation(self):
        t = EmailIMAPTool()
        assert t.metadata.name == "email_imap"

    def test_decode_header_plain(self):
        assert _decode_header_value("Hello World") == "Hello World"

    def test_decode_header_none(self):
        assert _decode_header_value(None) == ""

    def test_extract_body_plain(self):
        msg = MIMEText("Test body", "plain", "utf-8")
        assert _extract_body(msg) == "Test body"

    def test_extract_body_fallback_html(self):
        msg = MIMEText("<b>Bold</b>", "html", "utf-8")
        body = _extract_body(msg)
        assert "Bold" in body

    @pytest.mark.asyncio
    async def test_missing_imap_credentials_raise(self):
        t = EmailIMAPTool()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingCredentialsError) as exc_info:
                await t._execute(operation="list_folders")
        assert exc_info.value.missing_vars == ["IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD"]

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_error(self):
        t = EmailIMAPTool()
        with patch.dict(
            os.environ,
            {"IMAP_HOST": "imap.example.com", "IMAP_USER": "u", "IMAP_PASSWORD": "p"},
        ):
            result = await t._execute(operation="bad_op")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    @pytest.mark.asyncio
    async def test_get_without_uid_returns_error(self):
        t = EmailIMAPTool()
        with patch.dict(
            os.environ,
            {"IMAP_HOST": "imap.example.com", "IMAP_USER": "u", "IMAP_PASSWORD": "p"},
        ):
            result = await t._execute(operation="get")
        assert result["success"] is False
        assert "uid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_folders_mocked(self):
        t = EmailIMAPTool()
        env = {"IMAP_HOST": "imap.example.com", "IMAP_USER": "u", "IMAP_PASSWORD": "p"}
        with patch.dict(os.environ, env):
            with patch.object(t, "_list_folders_sync", return_value=["INBOX", "Sent"]):
                result = await t._execute(operation="list_folders")
        assert result["success"] is True
        assert "INBOX" in result["data"]["folders"]
        assert "Sent" in result["data"]["folders"]

    @pytest.mark.asyncio
    async def test_fetch_recent_mocked(self):
        t = EmailIMAPTool()
        env = {"IMAP_HOST": "imap.example.com", "IMAP_USER": "u", "IMAP_PASSWORD": "p"}
        fake_msgs = [
            {"uid": "5", "from": "a@b.com", "subject": "Hello", "date": "Mon, 1 Jan 2024"}
        ]
        with patch.dict(os.environ, env):
            with patch.object(t, "_fetch_recent_sync", return_value=fake_msgs):
                result = await t._execute(operation="fetch_recent", n=5)
        assert result["success"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["messages"][0]["uid"] == "5"

    @pytest.mark.asyncio
    async def test_search_mocked(self):
        t = EmailIMAPTool()
        env = {"IMAP_HOST": "imap.example.com", "IMAP_USER": "u", "IMAP_PASSWORD": "p"}
        fake_msgs = [{"uid": "3", "from": "x@y.com", "subject": "Invoice", "date": ""}]
        with patch.dict(os.environ, env):
            with patch.object(t, "_search_sync", return_value=fake_msgs):
                result = await t._execute(operation="search", query="SUBJECT Invoice")
        assert result["success"] is True
        assert result["data"]["query"] == "SUBJECT Invoice"
        assert result["data"]["count"] == 1

    @pytest.mark.asyncio
    async def test_get_mocked(self):
        t = EmailIMAPTool()
        env = {"IMAP_HOST": "imap.example.com", "IMAP_USER": "u", "IMAP_PASSWORD": "p"}
        fake_msg = {"uid": "7", "from": "a@b.com", "subject": "Test", "body": "Hello!"}
        with patch.dict(os.environ, env):
            with patch.object(t, "_get_message_sync", return_value=fake_msg):
                result = await t._execute(operation="get", uid="7")
        assert result["success"] is True
        assert result["data"]["uid"] == "7"
        assert result["data"]["body"] == "Hello!"


# ---------------------------------------------------------------------------
# Integration tests (live IMAP — skip without creds)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmailIMAPIntegration:
    @pytest.fixture(autouse=True)
    def require_imap(self):
        missing = [
            name
            for name in ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD")
            if not os.getenv(name)
        ]
        if missing:
            pytest.skip(f"IMAP credentials not configured ({', '.join(missing)})")

    @pytest.mark.asyncio
    async def test_list_folders(self):
        t = EmailIMAPTool()
        result = await t._execute(operation="list_folders")
        assert result["success"] is True, result.get("error")
        assert isinstance(result["data"]["folders"], list)
        assert len(result["data"]["folders"]) > 0

    @pytest.mark.asyncio
    async def test_fetch_recent_inbox(self):
        t = EmailIMAPTool()
        result = await t._execute(operation="fetch_recent", folder="INBOX", n=5)
        assert result["success"] is True, result.get("error")
        assert isinstance(result["data"]["messages"], list)
