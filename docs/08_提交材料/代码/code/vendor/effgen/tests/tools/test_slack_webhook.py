"""Tests for SlackWebhookTool — unit (mocked HTTP) + integration (skip without URL)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from effgen.errors import MissingCredentialsError
from effgen.tools.builtin.slack_webhook import SlackWebhookTool, _redact_webhook_url

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlackWebhookToolUnit:
    def test_instantiation(self):
        t = SlackWebhookTool()
        assert t.metadata.name == "slack_webhook"

    def test_redact_webhook_url_hides_path(self):
        url = "https://hooks.slack.com/services/T000/B000/SECRETTOKEN"
        redacted = _redact_webhook_url(url)
        assert "SECRETTOKEN" not in redacted
        assert "hooks.slack.com" in redacted
        assert redacted == "https://hooks.slack.com/***"

    def test_redact_empty_url(self):
        assert _redact_webhook_url("") == "***"

    @pytest.mark.asyncio
    async def test_missing_webhook_url_raises(self):
        t = SlackWebhookTool()
        env = {k: v for k, v in os.environ.items() if k != "SLACK_WEBHOOK_URL"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(MissingCredentialsError) as exc_info:
                await t._execute(text="Hello")
        assert exc_info.value.missing_vars == ["SLACK_WEBHOOK_URL"]

    @pytest.mark.asyncio
    async def test_missing_text_returns_error(self):
        t = SlackWebhookTool()
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            result = await t._execute(text="")
        assert result["success"] is False
        assert "text" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_post_success_mocked(self):
        t = SlackWebhookTool()
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch.object(SlackWebhookTool, "_post_sync", return_value=(True, "ok")):
                result = await t._execute(text="Hello Slack!")
        assert result["success"] is True
        assert result["data"]["ok"] is True
        assert result["data"]["response_body"] == "ok"
        assert "***" in result["data"]["webhook"]

    @pytest.mark.asyncio
    async def test_post_with_blocks_mocked(self):
        t = SlackWebhookTool()
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "*Bold*"}}]
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch.object(SlackWebhookTool, "_post_sync", return_value=(True, "ok")):
                result = await t._execute(text="Fallback", blocks=blocks)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_post_with_per_call_webhook_url(self):
        t = SlackWebhookTool()
        env = {k: v for k, v in os.environ.items() if k != "SLACK_WEBHOOK_URL"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(SlackWebhookTool, "_post_sync", return_value=(True, "ok")):
                result = await t._execute(
                    text="Hi",
                    webhook_url="https://hooks.slack.com/services/T/B/SECRET",
                )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_http_error_handled(self):
        t = SlackWebhookTool()
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch.object(SlackWebhookTool, "_post_sync", return_value=(False, "HTTP 403: Forbidden")):
                result = await t._execute(text="Hello")
        assert result["success"] is False
        assert "403" in result["error"]

    @pytest.mark.asyncio
    async def test_thread_ts_included(self):
        t = SlackWebhookTool()
        captured = {}
        def fake_post(url, payload, *args):
            captured.update(payload)
            return True, "ok"
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch.object(SlackWebhookTool, "_post_sync", side_effect=fake_post):
                await t._execute(text="Reply", thread_ts="1234567890.000001")
        assert captured.get("thread_ts") == "1234567890.000001"


# ---------------------------------------------------------------------------
# Integration tests (live Slack — skip without URL)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSlackWebhookIntegration:
    @pytest.fixture(autouse=True)
    def require_webhook(self):
        if not os.getenv("SLACK_WEBHOOK_URL"):
            pytest.skip("SLACK_WEBHOOK_URL not configured — skipping live Slack test")

    @pytest.mark.asyncio
    async def test_post_message(self):
        t = SlackWebhookTool()
        result = await t._execute(
            text="effGen v0.2.6 integration test — please ignore.",
            username="effGen-bot",
            icon_emoji=":robot_face:",
        )
        assert result["success"] is True, result.get("error")
        assert result["data"]["ok"] is True
