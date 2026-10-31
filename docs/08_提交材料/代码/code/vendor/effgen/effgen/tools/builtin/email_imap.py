"""
EmailIMAPTool — read email via IMAP (stdlib imaplib).

Config (env vars):
    IMAP_HOST     — IMAP server hostname (required)
    IMAP_PORT     — port, default 993
    IMAP_USER     — login username
    IMAP_PASSWORD — login password

SSL is always used (imaplib.IMAP4_SSL).

Operations
----------
- list_folders()                            → list of mailbox folder names
- fetch_recent(folder, n)                   → n most-recent message summaries
- search(query, folder)                     → messages matching IMAP search query
- get(uid, folder)                          → full message for a given UID
"""

from __future__ import annotations

import asyncio
import email
import imaplib
import logging
import os
from email.header import decode_header
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


def _decode_header_value(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    parts = decode_header(str(raw))
    decoded_parts = []
    for content, charset in parts:
        if isinstance(content, bytes):
            decoded_parts.append(content.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(content)
    return "".join(decoded_parts)


def _extract_body(msg: email.message.Message) -> str:
    """Return plain-text body; fallback to first text/html stripped of tags."""
    text_parts: list[str] = []
    html_parts: list[str] = []
    for part in msg.walk():
        ct = part.get_content_type()
        if part.get_content_disposition() == "attachment":
            continue
        charset = part.get_content_charset() or "utf-8"
        if ct == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                text_parts.append(payload.decode(charset, errors="replace"))
        elif ct == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                html_parts.append(payload.decode(charset, errors="replace"))
    if text_parts:
        return "\n".join(text_parts)
    if html_parts:
        import re
        raw = "\n".join(html_parts)
        return re.sub(r"<[^>]+>", "", raw)
    return ""


def _summarize(uid: str, msg: email.message.Message) -> dict[str, Any]:
    return {
        "uid": uid,
        "from": _decode_header_value(msg.get("From")),
        "to": _decode_header_value(msg.get("To")),
        "subject": _decode_header_value(msg.get("Subject")),
        "date": msg.get("Date", ""),
        "message_id": msg.get("Message-ID", ""),
        "has_attachments": any(
            p.get_content_disposition() == "attachment" for p in msg.walk()
        ),
    }


class EmailIMAPTool(BaseTool):
    """
    Read email via IMAP (SSL).

    Requires IMAP_HOST, IMAP_USER, and IMAP_PASSWORD env vars.
    Raises MissingCredentialsError if any required credential is absent.
    Credentials are never logged.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="email_imap",
                description=(
                    "Read email via IMAP. Supports listing folders, fetching recent messages, "
                    "searching by IMAP query, and retrieving full message content. "
                    "Requires IMAP_HOST, IMAP_USER, IMAP_PASSWORD env vars."
                ),
                category=ToolCategory.COMMUNICATION,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="One of: list_folders, fetch_recent, search, get.",
                        required=True,
                        enum=["list_folders", "fetch_recent", "search", "get"],
                    ),
                    ParameterSpec(
                        name="folder",
                        type=ParameterType.STRING,
                        description="Mailbox folder name (default: INBOX).",
                        required=False,
                        default="INBOX",
                    ),
                    ParameterSpec(
                        name="n",
                        type=ParameterType.INTEGER,
                        description="Number of recent messages to fetch (default: 10).",
                        required=False,
                        default=10,
                    ),
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description=(
                            "IMAP search criteria string, e.g. 'UNSEEN', "
                            "'FROM noreply@example.com', 'SUBJECT Invoice'."
                        ),
                        required=False,
                    ),
                    ParameterSpec(
                        name="uid",
                        type=ParameterType.STRING,
                        description="Message UID to retrieve (for 'get' operation).",
                        required=False,
                    ),
                    ParameterSpec(
                        name="include_body",
                        type=ParameterType.BOOLEAN,
                        description="Include message body in fetch_recent results (slower).",
                        required=False,
                        default=False,
                    ),
                ],
                returns={
                    "success": "bool",
                    "data": "dict | list",
                    "error": "str | null",
                },
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_config(self) -> dict[str, Any]:
        host = os.getenv("IMAP_HOST", "")
        user = os.getenv("IMAP_USER", "")
        password = os.getenv("IMAP_PASSWORD", "")
        missing = [
            name
            for name, value in (
                ("IMAP_HOST", host),
                ("IMAP_USER", user),
                ("IMAP_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise MissingCredentialsError(
                "EmailIMAPTool",
                missing,
                "Set IMAP_HOST, IMAP_USER, IMAP_PASSWORD, and optionally IMAP_PORT.",
            )
        port = int(os.getenv("IMAP_PORT", "993"))
        return {"host": host, "port": port, "user": user, "password": password}

    def _connect(self, cfg: dict) -> imaplib.IMAP4_SSL:
        imap = imaplib.IMAP4_SSL(cfg["host"], cfg["port"])
        imap.login(cfg["user"], cfg["password"])
        return imap

    # ------------------------------------------------------------------
    # Sync operations (run in executor)
    # ------------------------------------------------------------------

    def _list_folders_sync(self, cfg: dict) -> list[str]:
        imap = self._connect(cfg)
        try:
            status, folders = imap.list()
            names: list[str] = []
            for item in folders or []:
                if isinstance(item, bytes):
                    parts = item.decode().split('"')
                    names.append(parts[-1].strip().strip('"'))
            return names
        finally:
            try:
                imap.logout()
            except Exception:  # best-effort IMAP logout
                pass

    def _fetch_recent_sync(
        self, cfg: dict, folder: str, n: int, include_body: bool
    ) -> list[dict]:
        imap = self._connect(cfg)
        try:
            imap.select(folder, readonly=True)
            status, data = imap.uid("search", None, "ALL")
            uid_bytes = data[0].split() if data and data[0] else []
            uids = [u.decode() for u in uid_bytes[-n:]] if uid_bytes else []

            messages = []
            for uid in reversed(uids):
                fetch_parts = "(BODY.PEEK[HEADER])" if not include_body else "(RFC822)"
                status, raw = imap.uid("fetch", uid, fetch_parts)
                if status != "OK" or not raw or raw[0] is None:
                    continue
                raw_bytes = raw[0][1] if isinstance(raw[0], tuple) else b""
                msg = email.message_from_bytes(raw_bytes)
                summary = _summarize(uid, msg)
                if include_body:
                    summary["body"] = _extract_body(msg)
                messages.append(summary)
            return messages
        finally:
            try:
                imap.logout()
            except Exception:  # best-effort IMAP logout
                pass

    def _search_sync(self, cfg: dict, query: str, folder: str) -> list[dict]:
        imap = self._connect(cfg)
        try:
            imap.select(folder, readonly=True)
            # IMAP search needs ASCII criteria string; UID commands keep returned ids stable.
            status, data = imap.uid("search", None, query)
            uid_bytes = data[0].split() if data and data[0] else []
            uids = [u.decode() for u in uid_bytes]

            messages = []
            for uid in reversed(uids[:50]):  # cap at 50
                status, raw = imap.uid("fetch", uid, "(BODY.PEEK[HEADER])")
                if status != "OK" or not raw or raw[0] is None:
                    continue
                raw_bytes = raw[0][1] if isinstance(raw[0], tuple) else b""
                msg = email.message_from_bytes(raw_bytes)
                messages.append(_summarize(uid, msg))
            return messages
        finally:
            try:
                imap.logout()
            except Exception:  # best-effort IMAP logout
                pass

    def _get_message_sync(self, cfg: dict, uid: str, folder: str) -> dict:
        imap = self._connect(cfg)
        try:
            imap.select(folder, readonly=True)
            status, raw = imap.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not raw or raw[0] is None:
                raise RuntimeError(f"Message UID {uid} not found in {folder}.")
            raw_bytes = raw[0][1] if isinstance(raw[0], tuple) else b""
            msg = email.message_from_bytes(raw_bytes)
            summary = _summarize(uid, msg)
            summary["body"] = _extract_body(msg)
            return summary
        finally:
            try:
                imap.logout()
            except Exception:  # best-effort IMAP logout
                pass

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        cfg = self._get_config()
        operation = kwargs.get("operation", "fetch_recent")
        folder = kwargs.get("folder") or "INBOX"
        loop = asyncio.get_event_loop()

        try:
            if operation == "list_folders":
                folders = await loop.run_in_executor(None, self._list_folders_sync, cfg)
                return {"success": True, "data": {"folders": folders}, "error": None}

            elif operation == "fetch_recent":
                n = int(kwargs.get("n") or 10)
                include_body = bool(kwargs.get("include_body", False))
                msgs = await loop.run_in_executor(
                    None, self._fetch_recent_sync, cfg, folder, n, include_body
                )
                return {
                    "success": True,
                    "data": {"folder": folder, "count": len(msgs), "messages": msgs},
                    "error": None,
                }

            elif operation == "search":
                query = kwargs.get("query", "ALL")
                msgs = await loop.run_in_executor(
                    None, self._search_sync, cfg, query, folder
                )
                return {
                    "success": True,
                    "data": {
                        "folder": folder,
                        "query": query,
                        "count": len(msgs),
                        "messages": msgs,
                    },
                    "error": None,
                }

            elif operation == "get":
                uid = kwargs.get("uid")
                if not uid:
                    return {"success": False, "data": None, "error": "'uid' is required for 'get'."}
                msg = await loop.run_in_executor(None, self._get_message_sync, cfg, str(uid), folder)
                return {"success": True, "data": msg, "error": None}

            else:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Unknown operation '{operation}'. Valid: list_folders, fetch_recent, search, get.",
                }

        except imaplib.IMAP4.error as exc:
            return {"success": False, "data": None, "error": f"IMAP error: {exc}"}
        except Exception as exc:
            logger.error("EmailIMAPTool error: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}
