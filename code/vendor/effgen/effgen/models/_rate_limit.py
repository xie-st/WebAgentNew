"""
Sliding-window rate-limit coordinator for effGen model adapters.

Supports both in-memory (default, back-compat) and SQLite-backed persistence
for cross-process rate-limit coordination.

Usage::

    from effgen.models._rate_limit import RateLimitCoordinator

    # In-memory (back-compat, single-process)
    coordinator = RateLimitCoordinator(
        provider="cerebras",
        model="llama3.1-8b",
        rpm=30, rph=900, rpd=14_400,
        tpm=60_000, tph=1_000_000, tpd=1_000_000,
    )

    # SQLite-backed (cross-process, multi-worker)
    from effgen.models._rate_limit_store import SQLiteRateLimitStore
    store = SQLiteRateLimitStore()   # ~/.effgen/rate_limits.sqlite
    coordinator = RateLimitCoordinator(
        provider="cerebras",
        model="llama3.1-8b",
        rpm=30, rph=900, rpd=14_400,
        tpm=60_000, tph=1_000_000, tpd=1_000_000,
        storage=store,
    )

    async def call():
        await coordinator.acquire(tokens_estimate=100)
        result = await make_api_call()
        coordinator.record(actual_tokens=result.tokens_used)
        return result
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when the daily budget for a model is exhausted.

    Carries the same structured ``.error_context`` shape as
    :mod:`effgen.models.errors`'s typed provider errors (category
    ``"rate_limited"``) so ``is_transient_error()``/``classify_provider_error()``
    classify it consistently with every other rate-limit signal.
    """

    def __init__(self, message: str) -> None:
        from effgen.errors import quote_for_message, with_next_step
        from effgen.models.errors import error_context_dict

        self.error_context = error_context_dict("", "", "request", "rate_limited")
        super().__init__(
            with_next_step(
                quote_for_message(message), self.error_context["remediation"]
            )
        )


@dataclass
class _Window:
    """Sliding-window counter for a fixed duration (seconds)."""
    duration: float                   # window length in seconds
    limit: int                        # max events allowed in the window
    _timestamps: deque[float] = field(default_factory=deque)

    def _evict(self, now: float) -> None:
        cutoff = now - self.duration
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def count(self, now: float | None = None) -> int:
        self._evict(now or time.monotonic())
        return len(self._timestamps)

    def add(self, now: float | None = None) -> None:
        self._timestamps.append(now or time.monotonic())

    def remaining(self, now: float | None = None) -> int:
        return max(0, self.limit - self.count(now))

    def wait_seconds(self, now: float | None = None) -> float:
        """Seconds to wait until one slot opens in the window."""
        now = now or time.monotonic()
        self._evict(now)
        if len(self._timestamps) < self.limit:
            return 0.0
        # Oldest event will expire at oldest_ts + duration
        return max(0.0, self._timestamps[0] + self.duration - now)


@dataclass
class _TokenWindow:
    """Sliding-window counter that tracks total tokens (not just request count)."""
    duration: float
    limit: int
    # Each entry is (timestamp, token_count)
    _entries: deque[tuple[float, int]] = field(default_factory=deque)

    def _evict(self, now: float) -> None:
        cutoff = now - self.duration
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def total(self, now: float | None = None) -> int:
        self._evict(now or time.monotonic())
        return sum(t for _, t in self._entries)

    def add(self, tokens: int, now: float | None = None) -> None:
        self._entries.append((now or time.monotonic(), tokens))

    def remaining(self, now: float | None = None) -> int:
        return max(0, self.limit - self.total(now))

    def wait_seconds(self, tokens_needed: int, now: float | None = None) -> float:
        """Seconds to wait until *tokens_needed* fit within the window."""
        now = now or time.monotonic()
        self._evict(now)
        available = self.limit - self.total(now)
        if available >= tokens_needed:
            return 0.0
        # Wait until enough old entries expire
        needed_to_free = tokens_needed - available
        freed = 0
        for ts, tok in self._entries:
            freed += tok
            if freed >= needed_to_free:
                return max(0.0, ts + self.duration - now)
        # All entries together don't cover the need (tokens_needed > limit)
        return 0.0


class RateLimitCoordinator:
    """
    Sliding-window rate-limit coordinator for a single (provider, model) pair.

    Tracks RPM / RPH / RPD and TPM / TPH / TPD simultaneously.  Any request
    that would violate a limit is delayed with ``asyncio.sleep`` until capacity
    is available.

    When a ``storage`` backend is provided (SQLiteRateLimitStore), all events
    are persisted to SQLite, enabling cross-process coordination — multiple
    workers sharing the same database will collectively respect the same limits.

    Args:
        provider: Provider name (e.g. ``"cerebras"``).
        model: Model ID (e.g. ``"llama3.1-8b"``).
        rpm: Max requests per minute.
        rph: Max requests per hour.
        rpd: Max requests per day.
        tpm: Max tokens per minute.
        tph: Max tokens per hour.
        tpd: Max tokens per day.
        storage: Optional SQLiteRateLimitStore for cross-process coordination.
                 Defaults to ``None`` (in-memory, back-compat).

    Raises:
        RateLimitExceeded: When the *daily* budget (RPD or TPD) is fully
            consumed and the next request cannot be scheduled before the
            day window resets.
    """

    # Largest window in seconds — used for housekeeping cutoff calculation.
    _MAX_WINDOW: float = 86_400.0

    def __init__(
        self,
        provider: str,
        model: str,
        rpm: int,
        rph: int,
        rpd: int,
        tpm: int,
        tph: int,
        tpd: int,
        storage: "SQLiteRateLimitStore | None" = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._storage = storage

        # Request-count windows (in-memory mirrors; authoritative when no storage)
        self._req_minute = _Window(duration=60.0, limit=rpm)
        self._req_hour = _Window(duration=3_600.0, limit=rph)
        self._req_day = _Window(duration=86_400.0, limit=rpd)

        # Token-count windows
        self._tok_minute = _TokenWindow(duration=60.0, limit=tpm)
        self._tok_hour = _TokenWindow(duration=3_600.0, limit=tph)
        self._tok_day = _TokenWindow(duration=86_400.0, limit=tpd)

        # Lazy-initialized so construction doesn't require a running event loop
        # (Python 3.9 binds asyncio.Lock to the event loop at creation time)
        self._lock: asyncio.Lock | None = None
        # One adapter — and so one coordinator — commonly serves several agents
        # at once. This guards the windows and the lifetime counters, which are
        # updated from whichever thread the completed call returned on.
        self._record_lock = threading.Lock()

        # Total counters (lifetime, not windowed) for observability
        self.total_requests: int = 0
        self.total_tokens: int = 0
        self.total_throttled: int = 0
        self.total_throttle_seconds: float = 0.0
        self._pending_sqlite_reservations: int = 0
        self._pending_sqlite_token_event_ids: deque[int | None] = deque()

        logger.debug(
            "RateLimitCoordinator ready: %s/%s  rpm=%d rph=%d rpd=%d  "
            "tpm=%d tph=%d tpd=%d  storage=%s",
            provider, model, rpm, rph, rpd, tpm, tph, tpd,
            "sqlite" if storage else "memory",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_lock(self) -> asyncio.Lock:
        """Return the lock, creating it lazily inside the current event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self, tokens_estimate: int = 0) -> None:
        """Block until a request slot (and token budget) is available.

        Must be called *before* making the API request.  Pair with
        :meth:`record` after the call completes.

        Args:
            tokens_estimate: Expected token count (prompt + completion).
                Pass ``0`` if unknown; the coordinator will only enforce
                request-count limits in that case.

        Raises:
            RateLimitExceeded: If the daily request or token budget is
                exhausted (the day window has no capacity).
        """
        async with self._get_lock():
            if self._storage is not None:
                await self._wait_for_capacity_sqlite(tokens_estimate)
            else:
                await self._wait_for_capacity(tokens_estimate)

    def record(self, actual_tokens: int = 0) -> None:
        """Record a completed request.

        Must be called *after* the API call returns.

        Args:
            actual_tokens: Actual tokens used (from ``usage`` in the response).
        """
        with self._record_lock:
            if self._storage is not None:
                self._record_sqlite(actual_tokens)
            else:
                self._record_memory(actual_tokens)

            self.total_requests += 1
            self.total_tokens += actual_tokens

        logger.debug(
            "RLC record %s/%s: req=%d tokens=%d",
            self.provider, self.model, self.total_requests, self.total_tokens,
        )

    def status(self) -> dict:
        """Return a snapshot of current window usage (for debugging/logging)."""
        now = time.monotonic()
        data: dict = {
            "provider": self.provider,
            "model": self.model,
            "storage": "sqlite" if self._storage else "memory",
            "req_minute_used": self._req_minute.count(now),
            "req_minute_limit": self._req_minute.limit,
            "req_hour_used": self._req_hour.count(now),
            "req_hour_limit": self._req_hour.limit,
            "req_day_used": self._req_day.count(now),
            "req_day_limit": self._req_day.limit,
            "tok_minute_used": self._tok_minute.total(now),
            "tok_minute_limit": self._tok_minute.limit,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_throttled": self.total_throttled,
            "total_throttle_seconds": round(self.total_throttle_seconds, 3),
        }
        if self._storage is not None:
            # Also expose the cross-process counts from SQLite
            wall_now = time.time()
            req_min = len(self._storage.query_window(
                self.provider, self.model, "request", wall_now - 60.0
            ))
            req_hour = len(self._storage.query_window(
                self.provider, self.model, "request", wall_now - 3600.0
            ))
            tok_min = sum(
                tokens for _, tokens in self._storage.query_window(
                    self.provider, self.model, "tokens", wall_now - 60.0
                )
            )
            tok_hour = sum(
                tokens for _, tokens in self._storage.query_window(
                    self.provider, self.model, "tokens", wall_now - 3600.0
                )
            )
            data["sqlite_req_minute"] = req_min
            data["sqlite_req_hour"] = req_hour
            data["sqlite_tok_minute"] = tok_min
            data["sqlite_tok_hour"] = tok_hour
        return data

    def cleanup_storage(self) -> int:
        """Remove old events from SQLite (housekeeping).

        Deletes events older than the largest window (24 h) + 10% margin.
        No-op when using in-memory storage.

        Returns:
            Number of rows deleted (0 for in-memory mode).
        """
        if self._storage is None:
            return 0
        max_age = self._MAX_WINDOW * 1.1
        deleted = self._storage.cleanup(max_age_seconds=max_age)
        logger.debug("RLC housekeeping: removed %d old events", deleted)
        return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_memory(self, actual_tokens: int) -> None:
        now = time.monotonic()
        self._req_minute.add(now)
        self._req_hour.add(now)
        self._req_day.add(now)

        if actual_tokens > 0:
            self._tok_minute.add(actual_tokens, now)
            self._tok_hour.add(actual_tokens, now)
            self._tok_day.add(actual_tokens, now)

    def _record_sqlite(self, actual_tokens: int) -> None:
        assert self._storage is not None

        token_event_id: int | None = None
        if self._pending_sqlite_reservations > 0:
            self._pending_sqlite_reservations -= 1
            if self._pending_sqlite_token_event_ids:
                token_event_id = self._pending_sqlite_token_event_ids.popleft()
        else:
            # Defensive fallback for callers that invoke record() without a
            # preceding acquire(). The documented path reserves in acquire().
            now = time.time()
            self._storage.insert_event(self.provider, self.model, "request", now, 0)
            now_mono = time.monotonic()
            self._req_minute.add(now_mono)
            self._req_hour.add(now_mono)
            self._req_day.add(now_mono)

        if actual_tokens <= 0:
            return

        if token_event_id is not None:
            self._storage.update_token_event(token_event_id, actual_tokens)
        else:
            now = time.time()
            self._storage.insert_event(
                self.provider, self.model, "tokens", now, actual_tokens
            )
            self._tok_minute.add(actual_tokens)
            self._tok_hour.add(actual_tokens)
            self._tok_day.add(actual_tokens)

    async def _wait_for_capacity(self, tokens_estimate: int) -> None:
        """In-memory path — compute required sleep and block."""
        while True:
            now = time.monotonic()

            # Check daily limits first (raise immediately — daily waits are unusable)
            if self._req_day.remaining(now) == 0:
                raise RateLimitExceeded(
                    f"Daily request budget exhausted for {self.provider}/{self.model}. "
                    "Resets in 24 h."
                )

            if tokens_estimate > 0 and self._tok_day.remaining(now) < tokens_estimate:
                raise RateLimitExceeded(
                    f"Daily token budget exhausted for {self.provider}/{self.model}. "
                    "Resets in 24 h."
                )

            waits = [
                self._req_minute.wait_seconds(now),
                self._req_hour.wait_seconds(now),
            ]
            if tokens_estimate > 0:
                waits += [
                    self._tok_minute.wait_seconds(tokens_estimate, now),
                    self._tok_hour.wait_seconds(tokens_estimate, now),
                ]

            wait = max(waits)
            if wait <= 0:
                break

            logger.debug(
                "RLC throttling %s/%s for %.3f s (tokens_estimate=%d)",
                self.provider, self.model, wait, tokens_estimate,
            )
            self.total_throttled += 1
            self.total_throttle_seconds += wait
            lock = self._get_lock()
            lock.release()
            try:
                await asyncio.sleep(wait)
            finally:
                await lock.acquire()

    async def _wait_for_capacity_sqlite(self, tokens_estimate: int) -> None:
        """SQLite-backed path — atomically reserve cross-process capacity."""
        assert self._storage is not None
        while True:
            now_wall = time.time()
            result = self._storage.reserve_capacity(
                self.provider,
                self.model,
                now_wall,
                rpm=self._req_minute.limit,
                rph=self._req_hour.limit,
                rpd=self._req_day.limit,
                tpm=self._tok_minute.limit,
                tph=self._tok_hour.limit,
                tpd=self._tok_day.limit,
                tokens_estimate=tokens_estimate,
            )

            if result.daily_exceeded == "request":
                raise RateLimitExceeded(
                    f"Daily request budget exhausted for {self.provider}/{self.model}. "
                    "Resets in 24 h."
                )
            if result.daily_exceeded == "tokens":
                raise RateLimitExceeded(
                    f"Daily token budget exhausted for {self.provider}/{self.model}. "
                    "Resets in 24 h."
                )

            if result.allowed:
                now_mono = time.monotonic()
                self._req_minute.add(now_mono)
                self._req_hour.add(now_mono)
                self._req_day.add(now_mono)
                if tokens_estimate > 0:
                    self._tok_minute.add(tokens_estimate, now_mono)
                    self._tok_hour.add(tokens_estimate, now_mono)
                    self._tok_day.add(tokens_estimate, now_mono)
                self._pending_sqlite_reservations += 1
                self._pending_sqlite_token_event_ids.append(result.token_event_id)
                break

            wait = max(result.wait_seconds, 0.001)
            logger.debug(
                "RLC (sqlite) throttling %s/%s for %.3f s",
                self.provider, self.model, wait,
            )
            self.total_throttled += 1
            self.total_throttle_seconds += wait
            lock = self._get_lock()
            lock.release()
            try:
                await asyncio.sleep(wait)
            finally:
                await lock.acquire()
