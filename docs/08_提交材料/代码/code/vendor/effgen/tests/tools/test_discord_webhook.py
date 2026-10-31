"""Tests for DiscordWebhookTool — unit (mocked HTTP) + integration (skip without URL)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from effgen.errors import MissingCredentialsError
from effgen.tools.builtin.discord_webhook import DiscordWebhookTool, _redact_webhook_url

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiscordWebhookToolUnit:
    def test_instantiation(self):
        t = DiscordWebhookTool()
        assert t.metadata.name == "discord_webhook"

    def test_redact_webhook_url_hides_path(self):
        url = "https://discord.com/api/webhooks/123456789/SECRETTOKEN"
        redacted = _redact_webhook_url(url)
        assert "SECRETTOKEN" not in redacted
        assert "discord.com" in redacted
        assert redacted == "https://discord.com/***"

    def test_redact_empty_url(self):
        assert _redact_webhook_url("") == "***"

    @pytest.mark.asyncio
    async def test_missing_webhook_url_raises(self):
        t = DiscordWebhookTool()
        env = {k: v for k, v in os.environ.items() if k != "DISCORD_WEBHOOK_URL"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(MissingCredentialsError) as exc_info:
                await t._execute(content="Hello")
        assert exc_info.value.missing_vars == ["DISCORD_WEBHOOK_URL"]

    @pytest.mark.asyncio
    async def test_missing_content_and_embeds_returns_error(self):
        t = DiscordWebhookTool()
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test"}):
            result = await t._execute()
        assert result["success"] is False
        assert "content" in result["error"].lower() or "embeds" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_post_success_mocked(self):
        t = DiscordWebhookTool()
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test"}):
            with patch.object(DiscordWebhookTool, "_post_sync", return_value=(True, 200, '{"id":"1"}')):
                result = await t._execute(content="Hello Discord!")
        assert result["success"] is True
        assert result["data"]["ok"] is True
        assert result["data"]["response_code"] == 200
        assert "***" in result["data"]["webhook"]

    @pytest.mark.asyncio
    async def test_post_with_embeds_mocked(self):
        t = DiscordWebhookTool()
        embeds = [{"title": "Status", "description": "All good", "color": 3066993}]
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test"}):
            with patch.object(DiscordWebhookTool, "_post_sync", return_value=(True, 200, "{}")):
                result = await t._execute(embeds=embeds)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_content_truncated_at_2000(self):
        t = DiscordWebhookTool()
        long_content = "x" * 2500
        captured = {}
        def fake_post(url, payload, *args):
            captured.update(payload)
            return True, 200, "{}"
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test"}):
            with patch.object(DiscordWebhookTool, "_post_sync", side_effect=fake_post):
                await t._execute(content=long_content)
        assert len(captured.get("content", "")) == 2000

    @pytest.mark.asyncio
    async def test_post_with_per_call_webhook_url(self):
        t = DiscordWebhookTool()
        env = {k: v for k, v in os.environ.items() if k != "DISCORD_WEBHOOK_URL"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(DiscordWebhookTool, "_post_sync", return_value=(True, 200, "{}")):
                result = await t._execute(
                    content="Hi",
                    webhook_url="https://discord.com/api/webhooks/123/SECRET",
                )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_http_error_handled(self):
        t = DiscordWebhookTool()
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test"}):
            with patch.object(
                DiscordWebhookTool, "_post_sync", return_value=(False, 401, "Unauthorized")
            ):
                result = await t._execute(content="Hello")
        assert result["success"] is False
        assert "Unauthorized" in result["error"]

    @pytest.mark.asyncio
    async def test_custom_username_passed(self):
        t = DiscordWebhookTool()
        captured = {}
        def fake_post(url, payload, *args):
            captured.update(payload)
            return True, 200, "{}"
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test"}):
            with patch.object(DiscordWebhookTool, "_post_sync", side_effect=fake_post):
                await t._execute(content="Hello", username="effGen-bot")
        assert captured.get("username") == "effGen-bot"


# ---------------------------------------------------------------------------
# Integration tests (live Discord — skip without URL)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDiscordWebhookIntegration:
    @pytest.fixture(autouse=True)
    def require_webhook(self):
        if not os.getenv("DISCORD_WEBHOOK_URL"):
            pytest.skip("DISCORD_WEBHOOK_URL not configured — skipping live Discord test")

    @pytest.mark.asyncio
    async def test_post_message(self):
        t = DiscordWebhookTool()
        result = await t._execute(
            content="effGen v0.2.6 integration test — please ignore.",
            username="effGen-bot",
        )
        assert result["success"] is True, result.get("error")
        assert result["data"]["ok"] is True

    @pytest.mark.asyncio
    async def test_post_with_embed(self):
        t = DiscordWebhookTool()
        result = await t._execute(
            content="Status report",
            embeds=[
                {
                    "title": "effGen Test",
                    "description": "Integration test embed.",
                    "color": 3447003,
                }
            ],
        )
        assert result["success"] is True, result.get("error")
