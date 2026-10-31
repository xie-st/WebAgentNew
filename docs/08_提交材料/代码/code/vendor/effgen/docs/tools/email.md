# Email Tools

effGen provides two email tools: **EmailSMTPTool** for sending mail and **EmailIMAPTool** for reading mail. Both use Python's standard library (`smtplib` / `imaplib`) — no extra dependencies required.

---

## EmailSMTPTool

Send email via SMTP with TLS support.

### Configuration

| Env var        | Required | Default  | Description                         |
|----------------|----------|----------|-------------------------------------|
| `SMTP_HOST`    | ✓        | —        | SMTP server hostname                |
| `SMTP_PORT`    |          | `587`    | Port (587=STARTTLS, 465=SSL)        |
| `SMTP_USER`    | ✓        | —        | Login username                      |
| `SMTP_PASSWORD`| ✓        | —        | Login password                      |
| `SMTP_FROM`    |          | SMTP_USER| Default sender address              |

If any required SMTP credential is absent, `_execute()` raises `MissingCredentialsError`
before opening a network connection.

### Operations

**`send`** (default)

```python
import asyncio
from effgen.tools.builtin.email_smtp import EmailSMTPTool

tool = EmailSMTPTool()
result = asyncio.run(tool._execute(
    to="alice@example.com",
    subject="Hello from effGen",
    body="This is a test email.",
))
print(result)
# {'success': True, 'data': {'message_id': '...', 'accepted': ['alice@...'], 'rejected': [], 'server': 'smtp.example.com:587'}}
```

**Parameters:**

| Parameter     | Type              | Required | Description                                    |
|---------------|-------------------|----------|------------------------------------------------|
| `to`          | str or list[str]  | ✓        | Recipient(s)                                   |
| `subject`     | str               | ✓        | Subject line                                   |
| `body`        | str               | ✓        | Message body (plain text or HTML)              |
| `from_addr`   | str               |          | Sender address (overrides SMTP_FROM)           |
| `cc`          | str or list[str]  |          | CC recipients                                  |
| `bcc`         | str or list[str]  |          | BCC recipients                                 |
| `html`        | bool              |          | If True, body is HTML (default: False)         |
| `attachments` | list[str]         |          | File paths to attach                           |

### Security note

SMTP credentials (`SMTP_USER`, `SMTP_PASSWORD`) are read from environment variables and **never logged**. Only the `host:port` string appears in debug logs.

---

## EmailIMAPTool

Read email via IMAP (SSL).

### Configuration

| Env var         | Required | Default | Description               |
|-----------------|----------|---------|---------------------------|
| `IMAP_HOST`     | ✓        | —       | IMAP server hostname      |
| `IMAP_PORT`     |          | `993`   | Port (SSL)                |
| `IMAP_USER`     | ✓        | —       | Login username            |
| `IMAP_PASSWORD` | ✓        | —       | Login password            |

If any required IMAP credential is absent, `_execute()` raises
`MissingCredentialsError` before opening a network connection.

### Operations

**`list_folders`** — list all mailbox folders.

```python
result = asyncio.run(tool._execute(operation="list_folders"))
# {'success': True, 'data': {'folders': ['INBOX', 'Sent', ...]}}
```

**`fetch_recent`** — fetch the N most-recent messages (headers only by default).

```python
result = asyncio.run(tool._execute(
    operation="fetch_recent",
    folder="INBOX",
    n=10,
    include_body=False,   # set True to also fetch bodies (slower)
))
```

**`search`** — search using an IMAP criteria string.

```python
result = asyncio.run(tool._execute(
    operation="search",
    query="UNSEEN",          # or "FROM alice@example.com", "SUBJECT Invoice", etc.
    folder="INBOX",
))
```

**`get`** — retrieve a full message by UID.

```python
result = asyncio.run(tool._execute(
    operation="get",
    uid="42",
    folder="INBOX",
))
```

### Security note

IMAP credentials are read from environment variables and **never logged**. The connection always uses SSL (`IMAP4_SSL`).

---

## Common SMTP/IMAP providers

| Provider    | SMTP host             | SMTP port | IMAP host             | IMAP port |
|-------------|----------------------|-----------|----------------------|-----------|
| Gmail       | smtp.gmail.com       | 587       | imap.gmail.com       | 993       |
| Outlook/365 | smtp.office365.com   | 587       | outlook.office365.com| 993       |
| Yahoo       | smtp.mail.yahoo.com  | 587       | imap.mail.yahoo.com  | 993       |
| Fastmail    | smtp.fastmail.com    | 587       | imap.fastmail.com    | 993       |

> **Gmail / Yahoo note:** You may need to enable app passwords or OAuth-based access
> for programmatic mail clients. Using an app password is strongly recommended over
> your account password.

---

## Preset integration

`EmailSMTPTool` and `EmailIMAPTool` are included in the **`general`** and
**`notify`** presets.

```python
from effgen.presets import create_agent

agent = create_agent("notify", model)
```
