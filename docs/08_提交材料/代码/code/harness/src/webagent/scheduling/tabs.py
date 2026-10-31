"""Tab/page ownership and the one-in-flight-action rule.

A branch owns exactly one Tab for its whole lifetime; the Tab is never handed to
another branch while that branch is active. Every page action on the Tab takes
the lease's single action slot, so an in-flight action on one Tab cannot overlap
another action on the same Tab, even if a caller starts a second thread.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from .errors import TabUnavailable, WaitCancelled


class TabLease:
    def __init__(self, page_id: str, branch_id: str) -> None:
        self.page_id = page_id
        self.branch_id = branch_id
        self._action_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.action_count = 0
        self.released = False

    @property
    def in_flight(self) -> int:
        with self._state_lock:
            return self._in_flight

    def _acquire_action(self, cancel_event: threading.Event | None) -> None:
        while not self._action_lock.acquire(timeout=0.02):
            if cancel_event is not None and cancel_event.is_set():
                raise WaitCancelled(
                    f"branch {self.branch_id} cancelled while waiting for Tab {self.page_id}"
                )

    @contextmanager
    def action_slot(
        self,
        *,
        cancel_event: threading.Event | None = None,
    ) -> Iterator["TabLease"]:
        """Hold the Tab's single in-flight action slot for the body."""

        self._acquire_action(cancel_event)
        with self._state_lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            yield self
        finally:
            with self._state_lock:
                self._in_flight -= 1
                self.action_count += 1
            self._action_lock.release()


@dataclass
class TabPool:
    """Fixed pool of Tab identities, owned exclusively by active branches."""

    capacity: int
    prefix: str = "tab"
    _available: list[str] = field(default_factory=list)
    _owners: dict[str, str] = field(default_factory=dict)
    _leases: dict[str, TabLease] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.capacity, int) or isinstance(self.capacity, bool) or self.capacity < 1:
            raise ValueError("TabPool capacity must be a positive integer")
        if not self._available:
            self._available = [f"{self.prefix}-{index}" for index in range(1, self.capacity + 1)]

    @property
    def owners(self) -> dict[str, str]:
        with self._lock:
            return dict(self._owners)

    def acquire(
        self,
        branch_id: str,
        *,
        requested_page_id: str | None = None,
    ) -> TabLease:
        with self._lock:
            if requested_page_id is not None:
                if requested_page_id in self._owners:
                    raise TabUnavailable(
                        f"Tab {requested_page_id!r} is owned by branch {self._owners[requested_page_id]!r}"
                    )
                if requested_page_id not in self._available:
                    raise TabUnavailable(
                        f"Tab {requested_page_id!r} is not part of this pool"
                    )
                page_id = requested_page_id
                self._available.remove(page_id)
            else:
                if not self._available:
                    raise TabUnavailable("no free Tab remains in the pool")
                page_id = self._available.pop(0)
            self._owners[page_id] = branch_id
            lease = TabLease(page_id, branch_id)
            self._leases[page_id] = lease
            return lease

    def release(self, lease: TabLease) -> None:
        with self._lock:
            if lease.released:
                return
            self._owners.pop(lease.page_id, None)
            self._leases.pop(lease.page_id, None)
            lease.released = True
            if lease.page_id not in self._available and len(self._available) < self.capacity:
                self._available.append(lease.page_id)
                self._available.sort()

    def active_page_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._owners))
