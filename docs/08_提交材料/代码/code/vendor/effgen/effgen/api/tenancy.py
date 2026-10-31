"""Multi-tenancy: tenant config, API key management, per-tenant limits.

Stores state in a local SQLite database so no external dependencies are
required. The ``TenantManager`` is the single entry point used by the API
server middleware.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@dataclass
class Tenant:
    """A tenant with its rate limit, token quota and model/tool allow-lists."""

    id: str
    name: str
    rate_limit_per_min: int = 60
    daily_token_quota: int = 1_000_000
    allowed_models: list[str] = field(default_factory=list)  # empty = all
    allowed_tools: list[str] = field(default_factory=list)  # empty = all
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def allows_model(self, model: str) -> bool:
        """True when *model* is allowed (an empty allow-list permits all)."""
        return not self.allowed_models or model in self.allowed_models

    def allows_tool(self, tool: str) -> bool:
        """True when *tool* is allowed (an empty allow-list permits all)."""
        return not self.allowed_tools or tool in self.allowed_tools


@dataclass
class APIKey:
    """A stored API-key record: only the SHA-256 hash of the key is kept."""

    id: str
    tenant_id: str
    key_hash: str
    prefix: str
    name: str = ""
    created_at: float = field(default_factory=time.time)
    revoked: bool = False


class TenantManager:
    """SQLite-backed tenant and API key store with in-memory rate limiting."""

    _SCHEMA = [
        """
        CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            rate_limit_per_min INTEGER NOT NULL DEFAULT 60,
            daily_token_quota INTEGER NOT NULL DEFAULT 1000000,
            allowed_models TEXT NOT NULL DEFAULT '[]',
            allowed_tools TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            key_hash TEXT NOT NULL UNIQUE,
            prefix TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            revoked INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(tenant_id) REFERENCES tenants(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS usage (
            tenant_id TEXT NOT NULL,
            day TEXT NOT NULL,
            tokens INTEGER NOT NULL DEFAULT 0,
            requests INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(tenant_id, day)
        )
        """,
    ]

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.path.expanduser("~/.effgen/tenancy.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = Lock()
        # Rolling window rate-limit tracker: tenant_id -> list[timestamps]
        self._rate_window: dict[str, list[float]] = {}
        self._init_db()

    # ------------------------------------------------------------------ db

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            for stmt in self._SCHEMA:
                conn.execute(stmt)
            conn.commit()

    # ------------------------------------------------------------------ tenants

    def create_tenant(
        self,
        name: str,
        *,
        tenant_id: str | None = None,
        rate_limit_per_min: int = 60,
        daily_token_quota: int = 1_000_000,
        allowed_models: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Tenant:
        """Create a tenant, persist it and return the record.

        Args:
            name: Display name for the tenant.
            tenant_id: The id to use, generated when absent.
            rate_limit_per_min: Requests the tenant may make each minute.
            daily_token_quota: Tokens the tenant may spend each day.
            allowed_models: Model ids the tenant may call, or ``None`` for all.
            allowed_tools: Tool names the tenant may use, or ``None`` for all.
            metadata: Extra fields stored with the record.
        """
        tenant = Tenant(
            id=tenant_id or secrets.token_hex(8),
            name=name,
            rate_limit_per_min=rate_limit_per_min,
            daily_token_quota=daily_token_quota,
            allowed_models=allowed_models or [],
            allowed_tools=allowed_tools or [],
            metadata=metadata or {},
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO tenants (id, name, rate_limit_per_min, daily_token_quota,"
                " allowed_models, allowed_tools, metadata, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tenant.id,
                    tenant.name,
                    tenant.rate_limit_per_min,
                    tenant.daily_token_quota,
                    json.dumps(tenant.allowed_models),
                    json.dumps(tenant.allowed_tools),
                    json.dumps(tenant.metadata),
                    tenant.created_at,
                ),
            )
            conn.commit()
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Return the tenant with *tenant_id*, or ``None`` when unknown."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone()
        if not row:
            return None
        return Tenant(
            id=row["id"],
            name=row["name"],
            rate_limit_per_min=row["rate_limit_per_min"],
            daily_token_quota=row["daily_token_quota"],
            allowed_models=json.loads(row["allowed_models"]),
            allowed_tools=json.loads(row["allowed_tools"]),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
        )

    def list_tenants(self) -> list[Tenant]:
        """Return every stored tenant."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM tenants").fetchall()
        return [t for t in (self.get_tenant(r["id"]) for r in rows) if t]

    def delete_tenant(self, tenant_id: str) -> None:
        """Delete a tenant along with its API keys and usage records."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM api_keys WHERE tenant_id = ?", (tenant_id,))
            conn.execute("DELETE FROM usage WHERE tenant_id = ?", (tenant_id,))
            conn.execute("DELETE FROM tenants WHERE id = ?", (tenant_id,))
            conn.commit()

    # ------------------------------------------------------------------ keys

    def create_api_key(
        self, tenant_id: str, *, name: str = ""
    ) -> tuple[APIKey, str]:
        """Generate a new API key. Returns ``(record, raw_key)``.

        The raw key is shown only once; only its hash is persisted.
        """
        if not self.get_tenant(tenant_id):
            raise ValueError(f"unknown tenant: {tenant_id}")
        raw_key = f"eg-{secrets.token_urlsafe(32)}"
        prefix = raw_key[:8]
        record = APIKey(
            id=secrets.token_hex(8),
            tenant_id=tenant_id,
            key_hash=_hash_key(raw_key),
            prefix=prefix,
            name=name,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO api_keys (id, tenant_id, key_hash, prefix, name, created_at, revoked)"
                " VALUES (?, ?, ?, ?, ?, ?, 0)",
                (
                    record.id,
                    record.tenant_id,
                    record.key_hash,
                    record.prefix,
                    record.name,
                    record.created_at,
                ),
            )
            conn.commit()
        return record, raw_key

    def revoke_api_key(self, key_id: str) -> None:
        """Mark the key *key_id* revoked; it no longer resolves to a tenant."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,)
            )
            conn.commit()

    def resolve_api_key(self, raw_key: str) -> Tenant | None:
        """Look up a tenant by raw API key. Returns None if invalid/revoked."""
        key_hash = _hash_key(raw_key)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT tenant_id, revoked FROM api_keys WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
        if not row or row["revoked"]:
            return None
        return self.get_tenant(row["tenant_id"])

    def list_keys(self, tenant_id: str) -> list[APIKey]:
        """Return every API-key record (hashes, not raw keys) for a tenant."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys WHERE tenant_id = ?", (tenant_id,)
            ).fetchall()
        return [
            APIKey(
                id=r["id"],
                tenant_id=r["tenant_id"],
                key_hash=r["key_hash"],
                prefix=r["prefix"],
                name=r["name"],
                created_at=r["created_at"],
                revoked=bool(r["revoked"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ limits

    def check_rate_limit(self, tenant: Tenant) -> bool:
        """Sliding-window (60s) per-tenant rate limit. Returns False if over."""
        now = time.time()
        window = self._rate_window.setdefault(tenant.id, [])
        cutoff = now - 60.0
        # Drop timestamps outside window.
        while window and window[0] < cutoff:
            window.pop(0)
        if len(window) >= tenant.rate_limit_per_min:
            return False
        window.append(now)
        return True

    # ------------------------------------------------------------------ usage

    def record_usage(self, tenant_id: str, tokens: int = 0) -> None:
        """Add one request and *tokens* to the tenant's usage for today (UTC)."""
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO usage (tenant_id, day, tokens, requests) VALUES (?, ?, ?, 1)"
                " ON CONFLICT(tenant_id, day) DO UPDATE SET"
                " tokens = tokens + excluded.tokens,"
                " requests = requests + 1",
                (tenant_id, day, tokens),
            )
            conn.commit()

    def get_usage(self, tenant_id: str, day: str | None = None) -> dict[str, int]:
        """Return ``{"tokens", "requests"}`` used by the tenant on *day* (default today)."""
        day = day or time.strftime("%Y-%m-%d", time.gmtime())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT tokens, requests FROM usage WHERE tenant_id = ? AND day = ?",
                (tenant_id, day),
            ).fetchone()
        if not row:
            return {"tokens": 0, "requests": 0}
        return {"tokens": row["tokens"], "requests": row["requests"]}

    def check_quota(self, tenant: Tenant) -> bool:
        """True while the tenant is under its daily token quota."""
        usage = self.get_usage(tenant.id)
        return usage["tokens"] < tenant.daily_token_quota
