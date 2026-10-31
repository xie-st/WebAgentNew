"""Unit tests for RateLimitCoordinator — synthetic burst tests."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from effgen.models._rate_limit import RateLimitCoordinator, RateLimitExceeded


def _make_coordinator(rpm: int = 30, tpm: int = 60_000) -> RateLimitCoordinator:
    """Create a coordinator inside a running event loop so the Lock binds correctly."""
    return RateLimitCoordinator(
        provider="test",
        model="test-model",
        rpm=rpm,
        rph=900,
        rpd=14_400,
        tpm=tpm,
        tph=1_000_000,
        tpd=1_000_000,
    )


# ---------------------------------------------------------------------------
# Basic window mechanics (all async, use pytest-asyncio)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_acquire_does_not_block():
    coord = _make_coordinator(rpm=30)
    await coord.acquire(10)


@pytest.mark.asyncio
async def test_record_increments_totals():
    coord = _make_coordinator()
    await coord.acquire(0)
    coord.record(200)
    assert coord.total_requests == 1
    assert coord.total_tokens == 200


@pytest.mark.asyncio
async def test_status_contains_expected_keys():
    coord = _make_coordinator()
    s = coord.status()
    for key in (
        "provider", "model", "req_minute_used", "req_minute_limit",
        "tok_minute_used", "tok_minute_limit", "total_requests",
    ):
        assert key in s, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_status_reflects_records():
    coord = _make_coordinator()
    await coord.acquire(0)
    coord.record(500)
    s = coord.status()
    assert s["total_requests"] == 1
    assert s["total_tokens"] == 500


# ---------------------------------------------------------------------------
# Throttle test: burst against low rpm limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_throttle_increments_counter():
    """total_throttled increments when limits are hit."""
    coord = _make_coordinator(rpm=5)

    base = time.monotonic()
    for _ in range(5):
        coord._req_minute.add(base)

    # Patch asyncio.sleep to advance simulated time instead of actually waiting.
    # Coordinator uses time.monotonic() internally for window eviction.
    current = [base]

    def fake_monotonic():
        return current[0]

    async def fake_sleep(seconds):
        current[0] += seconds + 0.001  # advance past the window boundary

    with patch("effgen.models._rate_limit.time.monotonic", side_effect=fake_monotonic), \
         patch("effgen.models._rate_limit.asyncio.sleep", side_effect=fake_sleep):
        await coord.acquire(0)

    assert coord.total_throttled >= 1
    assert coord.total_throttle_seconds > 0


@pytest.mark.asyncio
async def test_40_req_burst_against_30_rpm_throttles():
    """30 reqs pre-filled; 10 more concurrent requests → ≥1 throttled."""
    coord = _make_coordinator(rpm=30)

    base = time.monotonic()
    for _ in range(30):
        coord._req_minute.add(base)
        coord._req_hour.add(base)
        coord._req_day.add(base)

    async def one_request() -> None:
        await coord.acquire(0)
        coord.record(0)

    # Advance simulated time each sleep so windows evict and throttling resolves
    current = [base]

    def fake_monotonic():
        return current[0]

    async def fake_sleep(seconds):
        current[0] += seconds + 0.001

    with patch("effgen.models._rate_limit.time.monotonic", side_effect=fake_monotonic), \
         patch("effgen.models._rate_limit.asyncio.sleep", side_effect=fake_sleep):
        await asyncio.gather(*[one_request() for _ in range(10)])

    assert coord.total_throttled >= 1


# ---------------------------------------------------------------------------
# Daily limit exhaustion raises RateLimitExceeded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rpd_exhausted_raises():
    coord = RateLimitCoordinator(
        provider="test",
        model="m",
        rpm=9999,
        rph=9999,
        rpd=2,
        tpm=9999,
        tph=9999,
        tpd=9999,
    )

    now = time.monotonic()
    coord._req_day.add(now)
    coord._req_day.add(now)

    with pytest.raises(RateLimitExceeded, match="[Dd]aily"):
        await coord.acquire(0)


# ---------------------------------------------------------------------------
# Multiple coordinators are independent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_coordinators_do_not_share_state():
    c1 = _make_coordinator(rpm=30)
    c2 = _make_coordinator(rpm=30)

    await c1.acquire(0)
    c1.record(100)

    assert c1.total_requests == 1
    assert c2.total_requests == 0
