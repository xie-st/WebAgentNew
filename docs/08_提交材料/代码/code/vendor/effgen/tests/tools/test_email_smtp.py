"""Tests for EmailSMTPTool — unit (mocked SMTP) + integration (skip without creds)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from effgen.errors import MissingCredentialsError
from effgen.tools.builtin.email_smtp import EmailSMTPTool

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmailSMTPToolUnit:
    def test_instantiation(self):
        t = EmailSMTPTool()
        assert t.metadata.name == "email_smtp"

    def test_to_list_string(self):
        assert EmailSMTPTool._to_list("a@b.com, c@d.com") == ["a@b.com", "c@d.com"]

    def test_to_list_list(self):
        assert EmailSMTPTool._to_list(["a@b.com"]) == ["a@b.com"]

    def test_to_list_none(self):
        assert EmailSMTPTool._to_list(None) == []

    def test_build_message_plain(self):
        msg = EmailSMTPTool._build_message(
            "from@x.com", ["to@y.com"], [], "Subj", "Hello", False, []
        )
        assert msg["Subject"] == "Subj"
        assert msg["From"] == "from@x.com"
        assert msg["To"] == "to@y.com"
        assert msg["Message-ID"].startswith("<")

    def test_build_message_with_cc(self):
        msg = EmailSMTPTool._build_message(
            "from@x.com", ["a@x.com"], ["cc@x.com"], "Subj", "Body", False, []
        )
        assert msg["Cc"] == "cc@x.com"

    @pytest.mark.asyncio
    async def test_missing_smtp_credentials_raise(self):
        t = EmailSMTPTool()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingCredentialsError) as exc_info:
                await t._execute(to="a@b.com", subject="S", body="B")
        assert exc_info.value.missing_vars == ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"]

    @pytest.mark.asyncio
    async def test_missing_to_returns_error(self):
        t = EmailSMTPTool()
        with patch.dict(
            os.environ,
            {
                "SMTP_HOST": "smtp.example.com",
                "SMTP_USER": "user@example.com",
                "SMTP_PASSWORD": "secret",
            },
        ):
            result = await t._execute(subject="S", body="B")
        assert result["success"] is False
        assert "to" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_subject_returns_error(self):
        t = EmailSMTPTool()
        with patch.dict(
            os.environ,
            {
                "SMTP_HOST": "smtp.example.com",
                "SMTP_USER": "user@example.com",
                "SMTP_PASSWORD": "secret",
            },
        ):
            result = await t._execute(to="a@b.com", body="B")
        assert result["success"] is False
        assert "subject" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_send_success_mocked(self):
        t = EmailSMTPTool()
        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user@example.com",
            "SMTP_PASSWORD": "secret",
            "SMTP_FROM": "user@example.com",
        }
        mock_send_result = {
            "message_id": "<abc@effgen>",
            "accepted": ["to@example.com"],
            "rejected": [],
            "server": "smtp.example.com:587",
        }
        with patch.dict(os.environ, env):
            with patch.object(t, "_send_sync", return_value=mock_send_result):
                result = await t._execute(
                    to="to@example.com", subject="Hello", body="Test body"
                )
        assert result["success"] is True
        assert result["data"]["accepted"] == ["to@example.com"]
        assert result["data"]["rejected"] == []

    @pytest.mark.asyncio
    async def test_send_smtp_error_handled(self):
        t = EmailSMTPTool()
        env = {"SMTP_HOST": "smtp.example.com", "SMTP_USER": "u", "SMTP_PASSWORD": "p"}
        with patch.dict(os.environ, env):
            with patch.object(t, "_send_sync", side_effect=RuntimeError("SMTP error: auth failed")):
                result = await t._execute(to="a@b.com", subject="S", body="B")
        assert result["success"] is False
        assert "auth failed" in result["error"]

    def test_missing_credentials_error_type(self):
        err = MissingCredentialsError("EmailSMTPTool", ["SMTP_HOST"])
        assert "SMTP_HOST" in str(err)
        assert err.tool_name == "EmailSMTPTool"
        assert err.missing_vars == ["SMTP_HOST"]


# ---------------------------------------------------------------------------
# Integration tests (live SMTP — skip without creds)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmailSMTPIntegration:
    """Live SMTP send. Skipped if SMTP_HOST is not configured."""

    @pytest.fixture(autouse=True)
    def require_smtp(self):
        missing = [
            name
            for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")
            if not os.getenv(name)
        ]
        if missing:
            pytest.skip(f"SMTP credentials not configured ({', '.join(missing)})")

    @pytest.mark.asyncio
    async def test_send_email_to_self(self):
        t = EmailSMTPTool()
        recipient = os.getenv("SMTP_USER", os.getenv("SMTP_FROM", ""))
        if not recipient:
            pytest.skip("No SMTP_USER to send to")
        result = await t._execute(
            to=recipient,
            subject="[effGen] SMTP integration test",
            body="This is an automated test email from effGen v0.2.6.",
        )
        assert result["success"] is True, result.get("error")
        assert recipient in result["data"]["accepted"]
