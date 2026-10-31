"""
Notify preset — communication agent for sending notifications.

Bundles EmailSMTPTool, EmailIMAPTool, SlackWebhookTool, and DiscordWebhookTool.
Tools that lack credentials raise MissingCredentialsError when called.

Exported symbol: NOTIFY_PRESET (a PresetConfig instance).
Registered in PRESETS dict on import.
"""

from __future__ import annotations

from .registry import PRESETS, PresetConfig

NOTIFY_PRESET = PresetConfig(
    name="notify",
    description=(
        "Notification agent that can send emails (SMTP), read email (IMAP), "
        "and post Slack or Discord messages. Configure credentials via env vars: "
        "SMTP_HOST/SMTP_USER/SMTP_PASSWORD, IMAP_HOST/IMAP_USER/IMAP_PASSWORD, "
        "SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL."
    ),
    tool_names=[
        "email_smtp",
        "email_imap",
        "slack_webhook",
        "discord_webhook",
    ],
    system_prompt=(
        "You are a notification agent. Your job is to send alerts and summaries "
        "via email, Slack, or Discord. "
        "Use email_smtp to send email messages. "
        "Use email_imap when you need to inspect recent email or confirm delivery. "
        "Use slack_webhook to post to Slack channels. "
        "Use discord_webhook to post to Discord channels. "
        "When given an event or status update, compose a clear, concise message "
        "and send it to the appropriate channel. "
        "If a channel is unavailable because credentials are missing, report that clearly."
    ),
    max_iterations=6,
    temperature=0.3,
    tags=["notify", "email", "slack", "discord", "communication", "alerts", "webhooks"],
)

PRESETS["notify"] = NOTIFY_PRESET
