"""Rate-limit acquire must engage under a running event loop.

Adapters used to do ``loop = asyncio.get_event_loop(); if loop.is_running():
<skip acquire>`` which silently disabled throttling inside FastAPI/Jupyter/any
async caller. These tests reproduce the *sync-acquire-from-within-a-loop* path
that the adapters take and assert the coordinator is actually driven.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from effgen.models._rate_limit import RateLimitCoordinator
from effgen.utils import run_coroutine_sync


def _coordinator(rpm: int, *, window: float = 0.4) -> RateLimitCoordinator:
    """A coordinator with a deliberately tiny request window so throttling is
    observable in well under a second."""
    coord = RateLimitCoordinator(
        provider="test",
        model="m",
        rpm=rpm,
        rph=10**6,
        rpd=10**6,
        tpm=10**12,
        tph=10**12,
        tpd=10**12,
    )
    # Shrink the request windows so the test doesn't wait a real minute.
    coord._req_minute.duration = window
    coord._req_hour.duration = window
    coord._req_day.duration = window
    return coord


def _adapter_sync_acquire(coord: RateLimitCoordinator, est_tokens: int) -> None:
    """Mirror the (fixed) adapter sync-generate acquire block exactly."""
    if coord is not None:
        run_coroutine_sync(coord.acquire(est_tokens))


def test_sync_acquire_records_all_requests_under_running_loop():
    """No request is silently skipped when driven from inside a running loop."""
    coord = _coordinator(rpm=100)

    async def caller():
        for _ in range(5):
            _adapter_sync_acquire(coord, 10)
            coord.record(10)

    asyncio.run(caller())
    assert coord.total_requests == 5


def test_sync_acquire_throttles_under_running_loop():
    """With rpm=2 and a 0.4s window, the 3rd request inside a running loop must
    be delayed — proving the limiter is engaged, not bypassed."""
    coord = _coordinator(rpm=2, window=0.4)

    async def caller():
        t0 = time.monotonic()
        for _ in range(3):
            _adapter_sync_acquire(coord, 10)
            coord.record(10)
        return time.monotonic() - t0

    elapsed = asyncio.run(caller())
    assert coord.total_requests == 3
    assert coord.total_throttled >= 1, "third request should have been throttled"
    assert elapsed >= 0.3, f"throttling should add a delay, got {elapsed:.3f}s"


def test_sync_acquire_throttles_without_running_loop():
    """Same guarantee on the plain synchronous path (no loop in the thread)."""
    coord = _coordinator(rpm=2, window=0.4)
    t0 = time.monotonic()
    for _ in range(3):
        _adapter_sync_acquire(coord, 10)
        coord.record(10)
    elapsed = time.monotonic() - t0
    assert coord.total_requests == 3
    assert coord.total_throttled >= 1
    assert elapsed >= 0.3


@pytest.mark.asyncio
async def test_concurrent_async_acquire_spacing():
    """N concurrent requests on a single running loop (the FastAPI scenario)
    are spaced out by the limiter rather than all firing at once."""
    coord = _coordinator(rpm=3, window=0.5)

    t0 = time.monotonic()

    async def one_call():
        await coord.acquire(10)
        coord.record(10)
        return time.monotonic() - t0

    rel = sorted(await asyncio.gather(*[one_call() for _ in range(6)]))

    assert coord.total_requests == 6
    # First 3 fit the window; the next batch must wait for it to slide (~0.5s).
    assert max(rel) >= 0.4, f"later requests should be delayed, stamps={rel}"
    assert coord.total_throttled >= 1
