# Webhook Tools

effGen provides **SlackWebhookTool** and **DiscordWebhookTool** for posting messages to Slack and Discord channels. Both use Incoming Webhook URLs — no OAuth or bot token required.

> **Security warning:** Webhook URLs are secrets. Anyone with a webhook URL can post to your channel. Never commit them to version control, never log them in full, and never expose them in user-facing output. effGen automatically redacts webhook URLs in all logs and return values, replacing the path (which contains the token) with `***`.

---

## SlackWebhookTool

Post messages to a Slack channel via an [Incoming Webhook](https://api.slack.com/messaging/webhooks).

### Setup

1. Go to **api.slack.com/apps** → create an app → enable **Incoming Webhooks**.
2. Add a webhook for a channel — copy the URL.
3. Set the env var:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/<TEAM_ID>/<WEBHOOK_ID>/<WEBHOOK_TOKEN>"
```

### Operations

**`post`** (default)

```python
import asyncio
from effgen.tools.builtin.slack_webhook import SlackWebhookTool

tool = SlackWebhookTool()
result = asyncio.run(tool._execute(text="Hello from effGen!"))
print(result)
# {'success': True, 'data': {'ok': True, 'response_body': 'ok', 'webhook': 'https://hooks.slack.com/***'}}
```

**Parameters:**

| Parameter     | Type              | Required | Description                                                    |
|---------------|-------------------|----------|----------------------------------------------------------------|
| `text`        | str               | ✓        | Message text (shown as fallback when blocks are used)          |
| `blocks`      | list[dict]        |          | [Block Kit](https://api.slack.com/block-kit) blocks            |
| `username`    | str               |          | Override the bot display name                                  |
| `icon_emoji`  | str               |          | Emoji for the bot icon, e.g. `:robot_face:`                   |
| `thread_ts`   | str               |          | Reply into a thread (timestamp from a previous message)        |
| `webhook_url` | str               |          | Per-call webhook URL override (skips `SLACK_WEBHOOK_URL`)      |

If neither `SLACK_WEBHOOK_URL` nor `webhook_url` is provided, `_execute()` raises
`MissingCredentialsError` before opening a network connection.

### Rich example with Block Kit

```python
blocks = [
    {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*effGen Alert*\nAll systems operational.",
        },
    },
    {"type": "divider"},
    {
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": "Sent by effGen v0.2.6"}
        ],
    },
]
result = asyncio.run(tool._execute(text="Alert", blocks=blocks))
```

---

## DiscordWebhookTool

Post messages to a Discord channel via a [Webhook](https://discord.com/developers/docs/resources/webhook).

### Setup

1. Open your Discord server → channel settings → **Integrations** → **Webhooks** → create one.
2. Copy the webhook URL.
3. Set the env var:

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/WEBHOOK_ID/WEBHOOK_TOKEN"
```

### Operations

**`post`** (default)

```python
from effgen.tools.builtin.discord_webhook import DiscordWebhookTool

tool = DiscordWebhookTool()
result = asyncio.run(tool._execute(content="Hello from effGen!"))
print(result)
# {'success': True, 'data': {'ok': True, 'response_code': 200, 'response_body': '...', 'webhook': 'https://discord.com/***'}}
```

**Parameters:**

| Parameter     | Type       | Required           | Description                                          |
|---------------|------------|--------------------|------------------------------------------------------|
| `content`     | str        | ✓ if no embeds     | Plain text (max 2 000 chars, auto-truncated)         |
| `embeds`      | list[dict] | ✓ if no content    | Rich [embed objects](https://discord.com/developers/docs/resources/channel#embed-object) |
| `username`    | str        |                    | Override webhook display name                        |
| `avatar_url`  | str        |                    | Override webhook avatar image URL                    |
| `tts`         | bool       |                    | Send as TTS message (default: False)                 |
| `webhook_url` | str        |                    | Per-call webhook URL (skips `DISCORD_WEBHOOK_URL`)   |

If neither `DISCORD_WEBHOOK_URL` nor `webhook_url` is provided, `_execute()` raises
`MissingCredentialsError` before opening a network connection.

### Rich embed example

```python
result = asyncio.run(tool._execute(
    content="Deployment complete!",
    embeds=[
        {
            "title": "effGen — Deployment Status",
            "description": "v0.2.6 deployed successfully.",
            "color": 3066993,          # green in decimal
            "fields": [
                {"name": "Environment", "value": "production", "inline": True},
                {"name": "Duration",    "value": "42s",        "inline": True},
            ],
            "footer": {"text": "effGen CI"},
        }
    ],
    username="effGen-bot",
))
```

---

## URL redaction guarantee

Both tools replace the path component of the webhook URL (which contains the secret token) with `***` before it appears anywhere in:

- Python log records (`logging.*`)
- The `webhook` field in the returned `data` dict

Example:
```
Input  : https://hooks.slack.com/services/<TEAM_ID>/<WEBHOOK_ID>/<WEBHOOK_TOKEN>
Logged : https://hooks.slack.com/***
```

The original URL is **never** stored in the tool instance; it is fetched from the env var or the `webhook_url` parameter at call time and used immediately.

---

## Preset integration

Both tools are included in the **`notify`** and **`general`** presets.

```python
from effgen.presets import create_agent

agent = create_agent("notify", model)
result = agent.run(
    "Post a Slack message saying 'Build succeeded for commit abc123'."
)
```
