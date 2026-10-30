"""
EmailSMTPTool — send email via SMTP (stdlib smtplib).

Config (env vars):
    SMTP_HOST     — SMTP server hostname (required)
    SMTP_PORT     — port, default 587
    SMTP_USER     — login username
    SMTP_PASSWORD — login password
    SMTP_FROM     — default sender address (falls back to SMTP_USER)

TLS is enabled by default (STARTTLS on port 587; SSL on port 465).

Operations
----------
- send(to, subject, body, from_addr=None, cc=None, bcc=None, attachments=[])
      → {message_id, accepted, rejected}
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import smtplib
import ssl
import uuid
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from ...errors import MissingCredentialsError
from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


def _redact_smtp_url(host: str, port: int) -> str:
    """Return a safe loggable SMTP server string (no credentials)."""
    return f"{host}:{port}"


class EmailSMTPTool(BaseTool):
    """
    Send email messages via SMTP.

    Requires SMTP_HOST, SMTP_USER, and SMTP_PASSWORD env vars.
    Raises MissingCredentialsError if any required credential is absent at call time.

    Webhook/secret note: SMTP credentials are never logged.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="email_smtp",
                description=(
                    "Send email via SMTP. Supports plain-text and HTML bodies, "
                    "CC/BCC recipients, and file attachments. "
                    "Requires SMTP_HOST, SMTP_USER, SMTP_PASSWORD env vars."
                ),
                category=ToolCategory.COMMUNICATION,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform. Currently only 'send'.",
                        required=False,
                        default="send",
                        enum=["send"],
                    ),
                    ParameterSpec(
                        name="to",
                        type=ParameterType.ANY,
                        description="Recipient address (str) or list of addresses.",
                        required=True,
                    ),
                    ParameterSpec(
                        name="subject",
                        type=ParameterType.STRING,
                        description="Email subject line.",
                        required=True,
                    ),
                    ParameterSpec(
                        name="body",
                        type=ParameterType.STRING,
                        description="Email body text (plain text or HTML).",
                        required=True,
                    ),
                    ParameterSpec(
                        name="from_addr",
                        type=ParameterType.STRING,
                        description="Sender address. Defaults to SMTP_FROM or SMTP_USER.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="cc",
                        type=ParameterType.ANY,
                        description="CC recipient(s) — str or list.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="bcc",
                        type=ParameterType.ANY,
                        description="BCC recipient(s) — str or list.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="html",
                        type=ParameterType.BOOLEAN,
                        description="If True, body is treated as HTML. Default False.",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="attachments",
                        type=ParameterType.ARRAY,
                        description="List of file paths to attach.",
                        required=False,
                    ),
                ],
                returns={
                    "success": "bool",
                    "data": {
                        "message_id": "str",
                        "accepted": "list[str]",
                        "rejected": "list[str]",
                        "server": "str",
                    },
                    "error": "str | null",
                },
                examples=[
                    {
                        "description": "Send a plain-text email",
                        "input": {
                            "to": "alice@example.com",
                            "subject": "Hello",
                            "body": "Hi Alice!",
                        },
                    }
                ],
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_config(self) -> dict[str, Any]:
        host = os.getenv("SMTP_HOST", "")
        user = os.getenv("SMTP_USER", "")
        password = os.getenv("SMTP_PASSWORD", "")
        missing = [
            name
            for name, value in (
                ("SMTP_HOST", host),
                ("SMTP_USER", user),
                ("SMTP_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise MissingCredentialsError(
                "EmailSMTPTool",
                missing,
                "Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, and optionally SMTP_PORT/SMTP_FROM.",
            )
        port = int(os.getenv("SMTP_PORT", "587"))
        from_addr = os.getenv("SMTP_FROM", "") or user
        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "from_addr": from_addr,
        }

    @staticmethod
    def _to_list(value: str | list | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [a.strip() for a in value.split(",") if a.strip()]
        return list(value)

    @staticmethod
    def _build_message(
        from_addr: str,
        to_list: list[str],
        cc_list: list[str],
        subject: str,
        body: str,
        html: bool,
        attachments: list[str],
    ) -> MIMEMultipart:
        msg = MIMEMultipart("mixed")
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_list)
        if cc_list:
            msg["Cc"] = ", ".join(cc_list)
        msg["Subject"] = subject
        msg["Message-ID"] = f"<{uuid.uuid4()}@effgen>"

        subtype = "html" if html else "plain"
        msg.attach(MIMEText(body, subtype, "utf-8"))

        for path_str in attachments or []:
            path = Path(path_str)
            if not path.exists():
                logger.warning("Attachment not found, skipping: %s", path)
                continue
            mime_type, _ = mimetypes.guess_type(str(path))
            main_type, sub_type = (mime_type or "application/octet-stream").split("/", 1)
            part = MIMEBase(main_type, sub_type)
            part.set_payload(path.read_bytes())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=path.name)
            msg.attach(part)

        return msg

    def _send_sync(
        self,
        cfg: dict,
        from_addr: str,
        all_recipients: list[str],
        msg: MIMEMultipart,
    ) -> dict[str, Any]:
        host, port = cfg["host"], cfg["port"]
        user, password = cfg["user"], cfg["password"]
        server_str = _redact_smtp_url(host, port)
        logger.debug("Connecting to SMTP server %s", server_str)

        context = ssl.create_default_context()

        try:
            if port == 465:
                with smtplib.SMTP_SSL(host, port, context=context) as smtp:
                    if user:
                        smtp.login(user, password)
                    refused = smtp.sendmail(from_addr, all_recipients, msg.as_string())
            else:
                with smtplib.SMTP(host, port) as smtp:
                    smtp.ehlo()
                    smtp.starttls(context=context)
                    smtp.ehlo()
                    if user:
                        smtp.login(user, password)
                    refused = smtp.sendmail(from_addr, all_recipients, msg.as_string())
        except smtplib.SMTPAuthenticationError as exc:
            raise RuntimeError(f"SMTP authentication failed: {exc.smtp_error.decode(errors='replace')}") from exc
        except smtplib.SMTPException as exc:
            raise RuntimeError(f"SMTP error: {exc}") from exc

        rejected = list(refused.keys())
        accepted = [r for r in all_recipients if r not in rejected]
        return {
            "message_id": msg["Message-ID"],
            "accepted": accepted,
            "rejected": rejected,
            "server": server_str,
        }

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        cfg = self._get_config()

        to_raw = kwargs.get("to")
        subject = kwargs.get("subject", "")
        body = kwargs.get("body", "")
        from_addr = kwargs.get("from_addr") or cfg["from_addr"]
        cc_raw = kwargs.get("cc")
        bcc_raw = kwargs.get("bcc")
        html = bool(kwargs.get("html", False))
        attachments = kwargs.get("attachments") or []

        if not to_raw:
            return {"success": False, "data": None, "error": "'to' is required."}
        if not subject:
            return {"success": False, "data": None, "error": "'subject' is required."}
        if not body:
            return {"success": False, "data": None, "error": "'body' is required."}

        to_list = self._to_list(to_raw)
        cc_list = self._to_list(cc_raw)
        bcc_list = self._to_list(bcc_raw)
        all_recipients = to_list + cc_list + bcc_list

        try:
            msg = self._build_message(from_addr, to_list, cc_list, subject, body, html, attachments)
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._send_sync, cfg, from_addr, all_recipients, msg
            )
            return {"success": True, "data": result, "error": None}
        except Exception as exc:
            logger.error("EmailSMTPTool send failed: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}
