"""
Communication tools for the effGen framework.

These tools DRAFT messages — they never actually send anything. The
agent produces formatted text that a human can review and send.
NotificationTool is the only tool here that actually produces a local
side effect, and only as a non-blocking desktop notification via plyer.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)

SAFETY_NOTE = "DRAFT ONLY — this tool does not send anything."


class EmailDraftTool(BaseTool):
    """Draft an email. Does NOT send — returns formatted text for review."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="email_draft",
                description=(
                    "Draft an email with subject, recipients, and body. "
                    "Returns formatted text — does NOT send any email."
                ),
                category=ToolCategory.COMMUNICATION,
                parameters=[
                    ParameterSpec(
                        name="to",
                        type=ParameterType.ARRAY,
                        description="Recipient email addresses",
                        required=True,
                        items_type=ParameterType.STRING,
                    ),
                    ParameterSpec(
                        name="subject",
                        type=ParameterType.STRING,
                        description="Email subject line",
                        required=True,
                        min_length=1,
                        max_length=300,
                    ),
                    ParameterSpec(
                        name="body",
                        type=ParameterType.STRING,
                        description="Email body text",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="cc",
                        type=ParameterType.ARRAY,
                        description="CC recipients",
                        required=False,
                        items_type=ParameterType.STRING,
                    ),
                    ParameterSpec(
                        name="bcc",
                        type=ParameterType.ARRAY,
                        description="BCC recipients",
                        required=False,
                        items_type=ParameterType.STRING,
                    ),
                    ParameterSpec(
                        name="from_address",
                        type=ParameterType.STRING,
                        description="Sender address (display only)",
                        required=False,
                    ),
                ],
                timeout_seconds=5,
                tags=["communication", "email", "draft", "safe"],
                examples=[
                    {
                        "to": ["alice@example.com"],
                        "subject": "Meeting notes",
                        "body": "Hi Alice,\n\nHere are the notes...",
                    }
                ],
            )
        )

    async def _execute(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        from_address: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        lines = []
        if from_address:
            lines.append(f"From: {from_address}")
        lines.append(f"To: {', '.join(to)}")
        if cc:
            lines.append(f"Cc: {', '.join(cc)}")
        if bcc:
            lines.append(f"Bcc: {', '.join(bcc)}")
        lines.append(f"Subject: {subject}")
        lines.append("")
        lines.append(body)
        formatted = "\n".join(lines)
        return {
            "draft": formatted,
            "to": to,
            "cc": cc or [],
            "bcc": bcc or [],
            "subject": subject,
            "body": body,
            "sent": False,
            "notice": SAFETY_NOTE,
        }


class SlackDraftTool(BaseTool):
    """Draft a Slack message. Does NOT send."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="slack_draft",
                description=(
                    "Draft a Slack message for a channel or user. Returns formatted "
                    "text — does NOT send any message."
                ),
                category=ToolCategory.COMMUNICATION,
                parameters=[
                    ParameterSpec(
                        name="channel",
                        type=ParameterType.STRING,
                        description="Target channel (e.g., '#general') or user (e.g., '@alice')",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="text",
                        type=ParameterType.STRING,
                        description="Message text (supports Slack mrkdwn)",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="thread_ts",
                        type=ParameterType.STRING,
                        description="Thread timestamp to reply to (optional)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="mentions",
                        type=ParameterType.ARRAY,
                        description="Users to @-mention",
                        required=False,
                        items_type=ParameterType.STRING,
                    ),
                ],
                timeout_seconds=5,
                tags=["communication", "slack", "draft", "safe"],
                examples=[
                    {"channel": "#engineering", "text": "Deploy completed successfully."},
                ],
            )
        )

    async def _execute(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        mentions: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        mention_prefix = ""
        if mentions:
            mention_prefix = " ".join(f"<@{m.lstrip('@')}>" for m in mentions) + " "
        draft = f"[{channel}]"
        if thread_ts:
            draft += f" (thread {thread_ts})"
        draft += f": {mention_prefix}{text}"
        return {
            "draft": draft,
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
            "mentions": mentions or [],
            "sent": False,
            "notice": SAFETY_NOTE,
        }


class NotificationTool(BaseTool):
    """Show a local desktop notification via plyer (optional dependency)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="notification",
                description=(
                    "Show a local desktop notification with title and message. "
                    "Requires the optional 'plyer' library. Falls back to returning "
                    "the formatted notification if plyer is not available."
                ),
                category=ToolCategory.COMMUNICATION,
                parameters=[
                    ParameterSpec(
                        name="title",
                        type=ParameterType.STRING,
                        description="Notification title",
                        required=True,
                        min_length=1,
                        max_length=200,
                    ),
                    ParameterSpec(
                        name="message",
                        type=ParameterType.STRING,
                        description="Notification body",
                        required=True,
                        min_length=1,
                        max_length=1000,
                    ),
                    ParameterSpec(
                        name="timeout",
                        type=ParameterType.INTEGER,
                        description="Seconds to show the notification",
                        required=False,
                        default=5,
                        min_value=1,
                        max_value=60,
                    ),
                ],
                timeout_seconds=10,
                tags=["communication", "notification", "desktop"],
                examples=[{"title": "Done", "message": "Task complete"}],
            )
        )

    async def _execute(
        self,
        title: str,
        message: str,
        timeout: int = 5,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            from plyer import notification  # type: ignore
        except ImportError:
            logger.info("plyer not installed; returning notification as text only")
            return {
                "title": title,
                "message": message,
                "shown": False,
                "reason": "plyer not installed (pip install plyer)",
            }
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="effGen",
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(f"plyer notification failed: {e}")
            return {
                "title": title,
                "message": message,
                "shown": False,
                "reason": f"plyer error: {e}",
            }
        return {
            "title": title,
            "message": message,
            "shown": True,
            "timeout": timeout,
        }
