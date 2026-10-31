"""
Bulkhead isolation for effGen provider calls.

A bulkhead limits the maximum concurrency for a given provider so that a
slow or misbehaving provider cannot starve threads/coroutines destined for
other providers.

Thread-safe and works with both sync and async callers.

Usage::

    from effgen.reliability.bulkhead import Bulkhead, BulkheadFull

    bh = Bulkhead(
        name="cerebras",
        max_concurrency=10,
        queue_size=50,
        queue_timeout=5.0,   # seconds to wait for a permit
    )

    # Sync context manager
    with bh.acquire():
        result = call_cerebras(...)

    # Async context manager
    async with bh.async_acquire():
        result = await async_call_cerebras(...)

    # Decorator
    @bh.guard()
    def call_cerebras(...):
        ...

    @bh.async_guard()
    async def async_call_cerebras(...):
        ...
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "Bulkhead",
    "BulkheadFull",
    "BulkheadRegistry",
    "get_bulkhead",
]


class BulkheadFull(Exception):
    """Raised when both the concurrency cap and queue are exhausted.

    Attributes:
        name:            The bulkhead name.
        max_concurrency: Configured concurrency cap.
        queue_size:      Configured queue depth.
    """

    def __init__(self, name: str, max_concurrency: int, queue_size: int) -> None:
        super().__init__(
            f"Bulkhead '{name}' is at capacity — "
            f"{max_concurrency} active calls and {queue_size} queued. "
            "Consider increasing max_concurrency / queue_size or reducing call rate."
        )
        self.name = name
        self.max_concurrency = max_concurrency
        self.queue_size = queue_size


class Bulkhead:
    """Thread-safe concurrency limiter (bulkhead pattern).

    Args:
        name:           Human-readable label (e.g. provider name).
        max_concurrency:  Maximum simultaneous active calls.
        queue_size:     Maximum calls that may wait for a permit.  When the
                        queue is also full, :class:`BulkheadFull` is raised.
        queue_timeout:  Seconds to wait for a permit before giving up with
                        :class:`BulkheadFull`.  Must be > 0.
    """

    def __init__(
        self,
        name: str,
        max_concurrency: int = 20,
        queue_size: int = 100,
        queue_timeout: float = 5.0,
    ) -> None:
        if queue_timeout is None or queue_timeout <= 0:
            raise ValueError(
                f"queue_timeout must be > 0, got {queue_timeout!r}. "
                "effGen forbids timeout=None."
            )
        self.name = name
        self.max_concurrency = max_concurrency
        self.queue_size = queue_size
        self.queue_timeout = queue_timeout

        self._lock = threading.Lock()
        self._active: int = 0
        self._queued: int = 0
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._total_accepted: int = 0
        self._total_rejected: int = 0
        self._total_timeout: int = 0

        # Async semaphore (lazy-created inside the event loop)
        self._async_semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def acquire(self) -> Iterator[None]:
        """Synchronous context manager that acquires a bulkhead permit.

        Raises:
            :class:`BulkheadFull`: If concurrency + queue are exhausted or
                                   the queue_timeout elapses.
        """
        with self._lock:
            if self._active >= self.max_concurrency:
                if self._queued >= self.queue_size:
                    self._total_rejected += 1
                    raise BulkheadFull(self.name, self.max_concurrency, self.queue_size)
                self._queued += 1

        acquired = self._semaphore.acquire(timeout=self.queue_timeout)

        with self._lock:
            if self._queued > 0:
                self._queued -= 1
            if not acquired:
                self._total_timeout += 1
                raise BulkheadFull(self.name, self.max_concurrency, self.queue_size)
            self._active += 1
            self._total_accepted += 1

        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    def guard(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that wraps a *synchronous* function with this bulkhead."""

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.acquire():
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    def _get_async_semaphore(self) -> asyncio.Semaphore:
        """Return (or lazily create) the asyncio.Semaphore."""
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._async_semaphore

    @contextlib.asynccontextmanager
    async def async_acquire(self) -> AsyncIterator[None]:
        """Async context manager that acquires a bulkhead permit.

        Raises:
            :class:`BulkheadFull`: If the semaphore cannot be acquired within
                                   ``queue_timeout`` seconds.
        """
        sem = self._get_async_semaphore()

        with self._lock:
            if self._queued >= self.queue_size and sem.locked():
                self._total_rejected += 1
                raise BulkheadFull(self.name, self.max_concurrency, self.queue_size)
            self._queued += 1

        try:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=self.queue_timeout)
            except TimeoutError:
                with self._lock:
                    self._queued -= 1
                    self._total_timeout += 1
                raise BulkheadFull(self.name, self.max_concurrency, self.queue_size)

            with self._lock:
                self._queued -= 1
                self._active += 1
                self._total_accepted += 1

            try:
                yield
            finally:
                with self._lock:
                    self._active -= 1
                sem.release()
        except BulkheadFull:
            raise

    def async_guard(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that wraps an *async* function with this bulkhead."""

        def decorator(fn):
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self.async_acquire():
                    return await fn(*args, **kwargs)

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of current statistics."""
        with self._lock:
            return {
                "name": self.name,
                "max_concurrency": self.max_concurrency,
                "queue_size": self.queue_size,
                "active": self._active,
                "queued": self._queued,
                "total_accepted": self._total_accepted,
                "total_rejected": self._total_rejected,
                "total_timeout": self._total_timeout,
                "utilization_pct": round(
                    100.0 * self._active / self.max_concurrency, 1
                ) if self.max_concurrency > 0 else 0.0,
            }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"Bulkhead(name={self.name!r}, "
            f"active={s['active']}/{self.max_concurrency}, "
            f"queued={s['queued']}/{self.queue_size})"
        )


# ---------------------------------------------------------------------------
# Registry — one bulkhead per provider, shared globally
# ---------------------------------------------------------------------------


class BulkheadRegistry:
    """Thread-safe registry of :class:`Bulkhead` instances keyed by name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bulkheads: dict[str, Bulkhead] = {}

    def get_or_create(
        self,
        name: str,
        *,
        max_concurrency: int = 20,
        queue_size: int = 100,
        queue_timeout: float = 5.0,
    ) -> Bulkhead:
        """Return existing :class:`Bulkhead` or create one.

        Args:
            name: The bulkhead's name in the registry.
            max_concurrency: Calls that may run at once.
            queue_size: Calls that may wait for a slot.
            queue_timeout: Seconds a call may wait before it is rejected.
        """
        with self._lock:
            if name not in self._bulkheads:
                self._bulkheads[name] = Bulkhead(
                    name=name,
                    max_concurrency=max_concurrency,
                    queue_size=queue_size,
                    queue_timeout=queue_timeout,
                )
            return self._bulkheads[name]

    def get(self, name: str) -> Bulkhead | None:
        """Return the bulkhead for *name*, or None if not registered."""
        with self._lock:
            return self._bulkheads.get(name)

    def all_stats(self) -> list[dict[str, Any]]:
        """Stats snapshot for all bulkheads."""
        with self._lock:
            return [bh.stats() for bh in self._bulkheads.values()]

    def clear(self) -> None:
        """Remove all bulkheads (useful for testing)."""
        with self._lock:
            self._bulkheads.clear()


# Module-level default registry
_default_registry = BulkheadRegistry()


def get_bulkhead(
    name: str,
    *,
    max_concurrency: int = 20,
    queue_size: int = 100,
    queue_timeout: float = 5.0,
) -> Bulkhead:
    """Get or create a :class:`Bulkhead` from the default registry.

    Args:
        name: The bulkhead's name in the registry.
        max_concurrency: Calls that may run at once.
        queue_size: Calls that may wait for a slot.
        queue_timeout: Seconds a call may wait before it is rejected.

    Returns:
        The registered bulkhead.
    """
    return _default_registry.get_or_create(
        name,
        max_concurrency=max_concurrency,
        queue_size=queue_size,
        queue_timeout=queue_timeout,
    )
