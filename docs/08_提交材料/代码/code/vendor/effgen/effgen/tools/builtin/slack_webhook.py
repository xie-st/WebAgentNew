"""
SlackWebhookTool — post messages to a Slack channel via Incoming Webhooks.

No OAuth required — just a webhook URL from Slack App settings.

Config (env var):
    SLACK_WEBHOOK_URL — default webhook URL (can be overridden per call)

Security: The webhook URL is treated as a secret. Only the domain is logged;
the path (which encodes the secret token) is replaced with ***.

Operations
----------
- post(text, blocks=None, username=None, icon_emoji=None, thread_ts=None,
        webhook_url=None)
      → {ok: bool, response_body: str}
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


class SlackWebhookTool(BaseTool):
    """
    Post messages to Slack via an Incoming Webhook URL.

    The webhook URL is treated as a secret and never logged in full.
    Set SLACK_WEBHOOK_URL env var, or pass webhook_url per call.
    """

    def __init__(self, allow_private: bool = False) -> None:
        """Args:
            allow_private: Allow posting to private/loopback/link-local/metadata
                addresses. Default ``False`` blocks them (SSRF protection); the
                webhook URL is user/model-suppliable, so this prevents using the
                tool to reach internal services.
        """
        self.allow_private = allow_private
        super().__init__(
            metadata=ToolMetadata(
                name="slack_webhook",
                description=(
                    "Post a message to a Slack channel using an Incoming Webhook. "
                    "Supports plain text, Block Kit JSON blocks, custom usernames, "
                    "emoji icons, and thread replies. "
                    "Requires SLACK_WEBHOOK_URL env var (or pass webhook_url per call)."
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
                        name="text",
                        type=ParameterType.STRING,
                        description="Message text (fallback text when blocks are used).",
                        required=True,
                    ),
                    ParameterSpec(
                        name="blocks",
                        type=ParameterType.ANY,
                        description="Block Kit blocks (list of dicts) for rich formatting.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="username",
                        type=ParameterType.STRING,
                        description="Override the bot display name.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="icon_emoji",
                        type=ParameterType.STRING,
                        description="Emoji to use as the bot icon, e.g. ':robot_face:'.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="thread_ts",
                        type=ParameterType.STRING,
                        description="Thread timestamp to reply into a thread.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="webhook_url",
                        type=ParameterType.STRING,
                        description=(
                            "Explicit webhook URL (overrides SLACK_WEBHOOK_URL env var). "
                            "This value is secret and is never logged."
                        ),
                        required=False,
                    ),
                ],
                returns={
                    "success": "bool",
                    "data": {"ok": "bool", "response_body": "str", "webhook": "str (redacted)"},
                    "error": "str | null",
                },
                examples=[
                    {
                        "description": "Post a simple message",
                        "input": {"text": "Hello from effGen!"},
                    }
                ],
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_url(override: str | None) -> str:
        url = override or os.getenv("SLACK_WEBHOOK_URL", "")
        if not url:
            raise MissingCredentialsError(
                "SlackWebhookTool",
                ["SLACK_WEBHOOK_URL"],
                "Set SLACK_WEBHOOK_URL or pass webhook_url per call.",
            )
        return url

    @staticmethod
    def _post_sync(url: str, payload: dict, allow_private: bool = False) -> tuple[bool, str]:
        body = json.dumps(payload).encode("utf-8")
        try:
            with safe_urlopen(
                url,
                data=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
                timeout=_TIMEOUT,
                allow_private=allow_private,
            ) as resp:
                resp_body = resp.read().decode("utf-8", errors="replace")
                ok = resp.status == 200 and resp_body.strip() == "ok"
                return ok, resp_body
        except HTTPError as exc:
            resp_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return False, f"HTTP {exc.code}: {resp_body}"
        except URLError as exc:
            return False, f"Network error: {exc.reason}"

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        webhook_url_override = kwargs.get("webhook_url")
        url = self._get_url(webhook_url_override)

        text = kwargs.get("text", "")
        if not text:
            return {"success": False, "data": None, "error": "'text' is required."}

        payload: dict[str, Any] = {"text": text}

        blocks = kwargs.get("blocks")
        if blocks:
            payload["blocks"] = blocks

        username = kwargs.get("username")
        if username:
            payload["username"] = username

        icon_emoji = kwargs.get("icon_emoji")
        if icon_emoji:
            payload["icon_emoji"] = icon_emoji

        thread_ts = kwargs.get("thread_ts")
        if thread_ts:
            payload["thread_ts"] = thread_ts

        redacted = _redact_webhook_url(url)
        logger.debug("Posting to Slack webhook %s", redacted)

        try:
            ok, resp_body = await asyncio.get_event_loop().run_in_executor(
                None, self._post_sync, url, payload, self.allow_private
            )
            return {
                "success": ok,
                "data": {"ok": ok, "response_body": resp_body, "webhook": redacted},
                "error": None if ok else resp_body,
            }
        except Exception as exc:
            safe_error = _redact_secret_from_text(str(exc), url)
            logger.error("SlackWebhookTool error: %s", safe_error)
            return {"success": False, "data": None, "error": safe_error}
