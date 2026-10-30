"""Request queue with priority, fair scheduling, and backpressure.

A lightweight in-process request queue used by the API server to shape
incoming traffic. It supports:

- Multiple priority levels (high/normal/low)
- Fair scheduling across tenants (round-robin within a priority)
- Per-request deadlines (timeout) and queue-level backpressure
- Backpressure: when full, `enqueue()` raises `QueueFullError` which the
  caller should translate into HTTP 429.
"""
from __future__ import annotations

import asyncio
import enum
import itertools
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


class RequestPriority(enum.IntEnum):
    """Scheduling priority for a queued request (lower value serves first)."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


class QueueFullError(Exception):
    """Raised when the queue is full and cannot accept more requests."""


class RequestTimeoutError(Exception):
    """Raised when a queued request exceeds its deadline."""


@dataclass(order=True)
class QueuedRequest:
    """A request waiting in the queue, ordered by priority then arrival."""

    # Ordering: priority first, then enqueue counter (FIFO within priority).
    priority: int
    seq: int
    id: str = field(compare=False)
    tenant_id: str = field(compare=False, default="default")
    payload: Any = field(compare=False, default=None)
    enqueued_at: float = field(compare=False, default_factory=time.time)
    deadline: float | None = field(compare=False, default=None)
    future: "asyncio.Future[Any]" | None = field(compare=False, default=None)

    def expired(self, now: float | None = None) -> bool:
        """True when the request's deadline (if any) has passed."""
        if self.deadline is None:
            return False
        return (now or time.time()) >= self.deadline


class RequestQueue:
    """Priority queue with fair scheduling across tenants.

    Design notes
    ------------
    - We keep a min-heap of ``(priority, seq)`` entries, but the entries for a
      given priority are drawn fairly across tenants using a per-priority
      round-robin over tenant FIFOs. This prevents a single noisy tenant from
      starving others inside the same priority band.
    - The queue is async-safe: ``enqueue`` and ``dequeue`` can be called from
      coroutines. An ``asyncio.Condition`` is used to wake waiters.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_timeout: float | None = 30.0,
    ) -> None:
        self.max_size = max_size
        self.default_timeout = default_timeout
        self._counter = itertools.count()
        # Per-(priority, tenant) FIFO of pending requests.
        self._buckets: dict[int, dict[str, deque[QueuedRequest]]] = defaultdict(
            lambda: defaultdict(deque)
        )
        # Round-robin tenant order per priority.
        self._rr_order: dict[int, deque[str]] = defaultdict(deque)
        self._size = 0
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._closed = False
        # Stats
        self.stats = {
            "enqueued": 0,
            "dequeued": 0,
            "rejected": 0,
            "expired": 0,
        }

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        """True when the queue holds ``max_size`` requests."""
        return self._size >= self.max_size

    async def enqueue(
        self,
        payload: Any,
        *,
        tenant_id: str = "default",
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float | None = None,
    ) -> QueuedRequest:
        """Add a request to the queue. Raises ``QueueFullError`` on overflow.

        Args:
            payload: The request to run when the slot comes up.
            tenant_id: The tenant the request is accounted to.
            priority: Where the request sits relative to others waiting.
            timeout: Seconds the request may wait before it is dropped.

        Returns:
            The queued request, carrying its id and position.
        """
        async with self._cond:
            if self._closed:
                raise RuntimeError("Queue is closed")
            if self._size >= self.max_size:
                self.stats["rejected"] += 1
                raise QueueFullError(
                    f"request queue full ({self._size}/{self.max_size}). "
                    "Retry once in-flight requests drain, or raise max_size "
                    "if the server has capacity for a deeper queue."
                )
            now = time.time()
            effective_timeout = timeout if timeout is not None else self.default_timeout
            deadline = now + effective_timeout if effective_timeout else None
            req = QueuedRequest(
                priority=int(priority),
                seq=next(self._counter),
                id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                payload=payload,
                enqueued_at=now,
                deadline=deadline,
                future=asyncio.get_event_loop().create_future(),
            )
            bucket = self._buckets[int(priority)]
            if tenant_id not in bucket or not bucket[tenant_id]:
                self._rr_order[int(priority)].append(tenant_id)
            bucket[tenant_id].append(req)
            self._size += 1
            self.stats["enqueued"] += 1
            self._cond.notify()
            return req

    async def dequeue(self, timeout: float | None = None) -> QueuedRequest | None:
        """Pop the next request. Returns None if ``timeout`` elapses empty."""
        async with self._cond:
            end = time.time() + timeout if timeout is not None else None
            while True:
                req = self._pop_next_locked()
                if req is not None:
                    return req
                if self._closed:
                    return None
                remaining = None
                if end is not None:
                    remaining = end - time.time()
                    if remaining <= 0:
                        return None
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except TimeoutError:
                    return None

    def _pop_next_locked(self) -> QueuedRequest | None:
        now = time.time()
        for priority in sorted(self._buckets.keys()):
            order = self._rr_order[priority]
            bucket = self._buckets[priority]
            while order:
                tenant_id = order.popleft()
                q = bucket.get(tenant_id)
                if not q:
                    continue
                req = q.popleft()
                # Reinsert tenant at the back for fairness if they still have work.
                if q:
                    order.append(tenant_id)
                self._size -= 1
                if req.expired(now):
                    self.stats["expired"] += 1
                    if req.future and not req.future.done():
                        req.future.set_exception(RequestTimeoutError(req.id))
                    continue
                self.stats["dequeued"] += 1
                return req
        return None

    async def close(self) -> None:
        """Close the queue; waiting consumers wake and pending enqueues fail."""
        async with self._cond:
            self._closed = True
            self._cond.notify_all()

    def snapshot_stats(self) -> dict[str, Any]:
        """Return queue counters plus the current size and closed flag."""
        return {
            **self.stats,
            "size": self._size,
            "max_size": self.max_size,
            "closed": self._closed,
        }
