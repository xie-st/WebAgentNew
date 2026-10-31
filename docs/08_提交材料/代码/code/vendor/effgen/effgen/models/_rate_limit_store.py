"""
SQLite-backed sliding-window persistence for RateLimitCoordinator.

Provides cross-process safe rate-limit state sharing so that multiple
workers (e.g. Celery workers, Gunicorn processes) all respect the same
rate-limit budget without racing each other.

Schema
------
    rate_events(
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        provider  TEXT    NOT NULL,
        model     TEXT    NOT NULL,
        kind      TEXT    NOT NULL,   -- 'request' | 'tokens'
        timestamp REAL    NOT NULL,   -- UNIX epoch (time.time())
        tokens    INTEGER NOT NULL DEFAULT 0
    )

Concurrency
-----------
WAL journal mode + BEGIN IMMEDIATE transactions give SQLite
multi-reader / single-writer semantics without external locking.

Usage::

    from effgen.models._rate_limit_store import SQLiteRateLimitStore
    store = SQLiteRateLimitStore()          # ~/.effgen/rate_limits.sqlite
    store = SQLiteRateLimitStore(":memory:") # in-process tests

    store.insert_event("cerebras", "llama3.1-8b", "request", time.time(), 0)
    events = store.query_window("cerebras", "llama3.1-8b", "request", since=...)
    store.cleanup(max_age_seconds=86400 * 1.1)
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_DB_PATH = Path.home() / ".effgen" / "rate_limits.sqlite"


def _default_db_path() -> Path:
    """Where rate-limit events are stored when no path is given.

    Follows ``EFFGEN_HOME`` when set, so the whole effGen state directory moves
    with one variable; otherwise ``~/.effgen/rate_limits.sqlite``.
    """
    home = os.environ.get("EFFGEN_HOME")
    if home:
        return Path(os.path.expanduser(home)).absolute() / "rate_limits.sqlite"
    return _DEFAULT_DB_PATH

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS rate_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    provider  TEXT    NOT NULL,
    model     TEXT    NOT NULL,
    kind      TEXT    NOT NULL,
    timestamp REAL    NOT NULL,
    tokens    INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_rate_events_lookup
    ON rate_events (provider, model, kind, timestamp);
"""

_INSERT = """
INSERT INTO rate_events (provider, model, kind, timestamp, tokens)
VALUES (?, ?, ?, ?, ?);
"""

_QUERY_WINDOW = """
SELECT timestamp, tokens
FROM rate_events
WHERE provider = ? AND model = ? AND kind = ? AND timestamp >= ?
ORDER BY timestamp ASC;
"""

_DELETE_OLD = """
DELETE FROM rate_events WHERE timestamp < ?;
"""

_UPDATE_TOKENS = """
UPDATE rate_events
SET tokens = ?
WHERE id = ? AND kind = 'tokens';
"""


@dataclass(frozen=True)
class _ReservationResult:
    """Result from an atomic SQLite rate-limit reservation attempt."""

    allowed: bool
    wait_seconds: float = 0.0
    daily_exceeded: str | None = None
    request_event_id: int | None = None
    token_event_id: int | None = None


class SQLiteRateLimitStore:
    """Thread-safe, cross-process SQLite store for rate-limit events.

    Args:
        db_path: Path to the SQLite file.  Use ``":memory:"`` for tests.
                 Defaults to ``$EFFGEN_HOME/rate_limits.sqlite`` when
                 ``EFFGEN_HOME`` is set, else ``~/.effgen/rate_limits.sqlite``.
    """

    def __init__(self, db_path: str | os.PathLike | None = None) -> None:
        if db_path is None:
            db_path = _default_db_path()
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # Each thread gets its own connection (SQLite connections are not
        # thread-safe across threads without check_same_thread=False + locking)
        self._local = threading.local()
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread connection, creating it on first use."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._conn()
        with conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_event(
        self,
        provider: str,
        model: str,
        kind: str,
        timestamp: float,
        tokens: int = 0,
    ) -> None:
        """Record one rate-limit event atomically (BEGIN IMMEDIATE).

        Args:
            provider: The provider the call went to.
            model: The model id the call used.
            kind: What is being counted, such as a request or a token spend.
            timestamp: Unix time of the event.
            tokens: Tokens the event consumed, for token-window accounting.
        """
        conn = self._conn()
        # BEGIN IMMEDIATE: prevents concurrent writers from interleaving
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(_INSERT, (provider, model, kind, timestamp, tokens))
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def reserve_capacity(
        self,
        provider: str,
        model: str,
        timestamp: float,
        *,
        rpm: int,
        rph: int,
        rpd: int,
        tpm: int,
        tph: int,
        tpd: int,
        tokens_estimate: int = 0,
    ) -> _ReservationResult:
        """Atomically check windows and reserve one request slot.

        The coordinator needs the read and write to happen under one SQLite
        write transaction.  Otherwise two processes can both observe available
        capacity and then both write after the API call has already started.

        Args:
            provider: The provider the call will go to.
            model: The model id the call will use.
            timestamp: Unix time the reservation is made at.
            rpm: Requests allowed per minute.
            rph: Requests allowed per hour.
            rpd: Requests allowed per day.
            tpm: Tokens allowed per minute.
            tph: Tokens allowed per hour.
            tpd: Tokens allowed per day.
            tokens_estimate: Tokens the call is expected to consume.

        Returns:
            The reservation result, saying whether the slot was granted and, when it
            was not, which window is full and how long until it frees.
        """
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            req_min_events = self._query_window_locked(
                conn, provider, model, "request", timestamp - 60.0
            )
            req_hour_events = self._query_window_locked(
                conn, provider, model, "request", timestamp - 3600.0
            )
            req_day_events = self._query_window_locked(
                conn, provider, model, "request", timestamp - 86400.0
            )

            result = self._reservation_result(
                conn,
                provider,
                model,
                timestamp,
                req_min_events=req_min_events,
                req_hour_events=req_hour_events,
                req_day_events=req_day_events,
                rpm=rpm,
                rph=rph,
                rpd=rpd,
                tpm=tpm,
                tph=tph,
                tpd=tpd,
                tokens_estimate=tokens_estimate,
            )
            conn.execute("COMMIT;")
            return result
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def query_window(
        self,
        provider: str,
        model: str,
        kind: str,
        since: float,
    ) -> Sequence[tuple[float, int]]:
        """Return all (timestamp, tokens) events in [since, now] for this key.

        Args:
            provider: The provider to count events for.
            model: The model id to count events for.
            kind: The event kind to count.
            since: Unix time the window opens at.
        """
        conn = self._conn()
        rows = conn.execute(_QUERY_WINDOW, (provider, model, kind, since)).fetchall()
        return rows  # list of (timestamp, tokens)

    def update_token_event(self, event_id: int, tokens: int) -> bool:
        """Update a reserved token event with actual usage."""
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cursor = conn.execute(_UPDATE_TOKENS, (tokens, event_id))
            updated = cursor.rowcount > 0
            conn.execute("COMMIT;")
            return updated
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def cleanup(self, max_age_seconds: float) -> int:
        """Delete events older than *max_age_seconds*.  Returns rows deleted."""
        cutoff = time.time() - max_age_seconds
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cursor = conn.execute(_DELETE_OLD, (cutoff,))
            count = cursor.rowcount
            conn.execute("COMMIT;")
            return count
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def close(self) -> None:
        """Close the per-thread connection (call from the owning thread)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_window_locked(
        self,
        conn: sqlite3.Connection,
        provider: str,
        model: str,
        kind: str,
        since: float,
    ) -> list[tuple[float, int]]:
        rows = conn.execute(_QUERY_WINDOW, (provider, model, kind, since)).fetchall()
        return rows

    def _reservation_result(
        self,
        conn: sqlite3.Connection,
        provider: str,
        model: str,
        timestamp: float,
        *,
        req_min_events: Sequence[tuple[float, int]],
        req_hour_events: Sequence[tuple[float, int]],
        req_day_events: Sequence[tuple[float, int]],
        rpm: int,
        rph: int,
        rpd: int,
        tpm: int,
        tph: int,
        tpd: int,
        tokens_estimate: int,
    ) -> _ReservationResult:
        if len(req_day_events) >= rpd:
            return _ReservationResult(allowed=False, daily_exceeded="request")

        wait = 0.0
        if len(req_min_events) >= rpm and req_min_events:
            wait = max(wait, req_min_events[0][0] + 60.0 - timestamp)
        if len(req_hour_events) >= rph and req_hour_events:
            wait = max(wait, req_hour_events[0][0] + 3600.0 - timestamp)

        if tokens_estimate > 0:
            token_result = self._token_reservation_wait(
                conn,
                provider,
                model,
                timestamp,
                tpm=tpm,
                tph=tph,
                tpd=tpd,
                tokens_estimate=tokens_estimate,
            )
            if token_result.daily_exceeded is not None:
                return token_result
            wait = max(wait, token_result.wait_seconds)

        if wait > 0:
            return _ReservationResult(allowed=False, wait_seconds=max(wait, 0.0))

        request_cursor = conn.execute(
            _INSERT, (provider, model, "request", timestamp, 0)
        )
        token_event_id = None
        if tokens_estimate > 0:
            token_cursor = conn.execute(
                _INSERT, (provider, model, "tokens", timestamp, tokens_estimate)
            )
            token_event_id = int(token_cursor.lastrowid)

        return _ReservationResult(
            allowed=True,
            request_event_id=int(request_cursor.lastrowid),
            token_event_id=token_event_id,
        )

    def _token_reservation_wait(
        self,
        conn: sqlite3.Connection,
        provider: str,
        model: str,
        timestamp: float,
        *,
        tpm: int,
        tph: int,
        tpd: int,
        tokens_estimate: int,
    ) -> _ReservationResult:
        tok_day_events = self._query_window_locked(
            conn, provider, model, "tokens", timestamp - 86400.0
        )
        tok_day_total = sum(tokens for _, tokens in tok_day_events)
        if tok_day_total + tokens_estimate > tpd:
            return _ReservationResult(allowed=False, daily_exceeded="tokens")

        wait = 0.0
        tok_min_events = self._query_window_locked(
            conn, provider, model, "tokens", timestamp - 60.0
        )
        wait = max(
            wait,
            self._token_wait_seconds(tok_min_events, tpm, tokens_estimate, 60.0, timestamp),
        )

        tok_hour_events = self._query_window_locked(
            conn, provider, model, "tokens", timestamp - 3600.0
        )
        wait = max(
            wait,
            self._token_wait_seconds(
                tok_hour_events, tph, tokens_estimate, 3600.0, timestamp
            ),
        )
        return _ReservationResult(allowed=False, wait_seconds=wait)

    @staticmethod
    def _token_wait_seconds(
        events: Sequence[tuple[float, int]],
        limit: int,
        tokens_estimate: int,
        duration: float,
        timestamp: float,
    ) -> float:
        total = sum(tokens for _, tokens in events)
        if total + tokens_estimate <= limit or not events:
            return 0.0

        needed = total + tokens_estimate - limit
        freed = 0
        for event_ts, tokens in events:
            freed += tokens
            if freed >= needed:
                return max(0.0, event_ts + duration - timestamp)
        return 0.0
