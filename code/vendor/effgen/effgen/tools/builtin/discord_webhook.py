"""
DiscordWebhookTool — post messages to a Discord channel via Webhook.

No OAuth required — just a webhook URL from channel settings.

Config (env var):
    DISCORD_WEBHOOK_URL — default webhook URL (overridable per call)

Security: Webhook URL is a secret. Only the domain is logged; the path
(which contains the webhook ID and token) is replaced with ***.

Operations
----------
- post(content, embeds=None, username=None, avatar_url=None, tts=False,
        webhook_url=None)
      → {ok: bool, response_code: int, response_body: str}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from ...errors import MissingCredentialsError
from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

logger = logging.getLogger(__name__)

_TIMEOUT = 15  # seconds


def _redact_webhook_url(url: str) -> str:
    """Replace path/query after the domain with *** for safe logging."""
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return "***"
        return f"{parsed.scheme}://{parsed.netloc}/***"
    except Exception:
        return "***"


def _redact_secret_from_text(text: str, secret: str) -> str:
    """Remove a webhook URL if a lower-level exception included it."""
    if not secret:
        return text
    return text.replace(secret, _redact_webhook_url(secret))


class DiscordWebhookTool(BaseTool):
    """
    Post messages to Discord via a Webhook URL.

    Supports plain content, rich embeds, custom usernames, and avatar URLs.
    The webhook URL is treated as a secret and never logged in full.
    Set DISCORD_WEBHOOK_URL env var, or pass webhook_url per call.
    """

    def __init__(self, allow_private: bool = False) -> None:
        """Args:
            allow_private: Allow posting to private/loopback/link-local/metadata
                addresses. Default ``False`` blocks them (SSRF protection); the
                webhook URL is user/model-suppliable.
        """
        self.allow_private = allow_private
        super().__init__(
            metadata=ToolMetadata(
                name="discord_webhook",
                description=(
                    "Post a message to a Discord channel using a Webhook URL. "
                    "Supports plain text, rich embeds, custom usernames, and avatar URLs. "
                    "Requires DISCORD_WEBHOOK_URL env var (or pass webhook_url per call)."
                ),
                category=ToolCategory.COMMUNICATION,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform. Currently only 'post'.",
                        required=False,
                        default="post",
                        enum=["post"],
                    ),
                    ParameterSpec(
                        name="content",
                        type=ParameterType.STRING,
                        description="Plain text message content (max 2000 chars).",
                        required=False,
                    ),
                    ParameterSpec(
                        name="embeds",
                        type=ParameterType.ANY,
                        description=(
                            "List of embed objects (dicts). Each embed may have "
                            "title, description, color, fields, footer, etc."
                        ),
                        required=False,
                    ),
                    ParameterSpec(
                        name="username",
                        type=ParameterType.STRING,
                        description="Override the webhook display name.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="avatar_url",
                        type=ParameterType.STRING,
                        description="URL of an image to use as the webhook avatar.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="tts",
                        type=ParameterType.BOOLEAN,
                        description="Send as text-to-speech message. Default False.",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="webhook_url",
                        type=ParameterType.STRING,
                        description=(
                            "Explicit webhook URL (overrides DISCORD_WEBHOOK_URL env var). "
                            "Treated as a secret; never logged."
                        ),
                        required=False,
                    ),
                ],
                returns={
                    "success": "bool",
                    "data": {
                        "ok": "bool",
                        "response_code": "int",
                        "response_body": "str",
                        "webhook": "str (redacted)",
                    },
                    "error": "str | null",
                },
                examples=[
                    {
                        "description": "Post a simple message",
                        "input": {"content": "Hello from effGen!"},
                    },
                    {
                        "description": "Post a rich embed",
                        "input": {
                            "content": "Status update",
                            "embeds": [
                                {
                                    "title": "effGen Report",
                                    "description": "All systems operational.",
                                    "color": 3066993,
                                }
                            ],
                        },
                    },
                ],
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_url(override: str | None) -> str:
        url = override or os.getenv("DISCORD_WEBHOOK_URL", "")
        if not url:
            raise MissingCredentialsError(
                "DiscordWebhookTool",
                ["DISCORD_WEBHOOK_URL"],
                "Set DISCORD_WEBHOOK_URL or pass webhook_url per call.",
            )
        return url

    @staticmethod
    def _post_sync(url: str, payload: dict, allow_private: bool = False) -> tuple[bool, int, str]:
        body = json.dumps(payload).encode("utf-8")
        # Discord wants ?wait=true to get a 200 response with message data
        req_url = url if "?" in url else url + "?wait=true"
        try:
            with safe_urlopen(
                req_url,
                data=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
                timeout=_TIMEOUT,
                allow_private=allow_private,
            ) as resp:
                resp_body = resp.read().decode("utf-8", errors="replace")
                # 200 with wait=true means success; 204 without
                ok = resp.status in (200, 204)
                return ok, resp.status, resp_body
        except HTTPError as exc:
            resp_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return False, exc.code, f"HTTP {exc.code}: {resp_body}"
        except URLError as exc:
            return False, 0, f"Network error: {exc.reason}"

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        webhook_url_override = kwargs.get("webhook_url")
        url = self._get_url(webhook_url_override)

        content = kwargs.get("content")
        embeds = kwargs.get("embeds")

        if not content and not embeds:
            return {
                "success": False,
                "data": None,
                "error": "At least one of 'content' or 'embeds' is required.",
            }

        # Discord: content max 2000 chars
        if content and len(content) > 2000:
            content = content[:1997] + "..."

        payload: dict[str, Any] = {}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds

        username = kwargs.get("username")
        if username:
            payload["username"] = username

        avatar_url = kwargs.get("avatar_url")
        if avatar_url:
            payload["avatar_url"] = avatar_url

        if kwargs.get("tts"):
            payload["tts"] = True

        redacted = _redact_webhook_url(url)
        logger.debug("Posting to Discord webhook %s", redacted)

        try:
            ok, code, resp_body = await asyncio.get_event_loop().run_in_executor(
                None, self._post_sync, url, payload, self.allow_private
            )
            return {
                "success": ok,
                "data": {
                    "ok": ok,
                    "response_code": code,
                    "response_body": resp_body,
                    "webhook": redacted,
                },
                "error": None if ok else resp_body,
            }
        except Exception as exc:
            safe_error = _redact_secret_from_text(str(exc), url)
            logger.error("DiscordWebhookTool error: %s", safe_error)
            return {"success": False, "data": None, "error": safe_error}
