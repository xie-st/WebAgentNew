"""
Integration test: two processes share a SQLite rate-limit store and the
combined burst correctly throttles under the shared rpm limit.

This test is marked `slow` and requires no API keys — it uses the
SQLiteRateLimitStore directly without making real API calls.
"""
from __future__ import annotations

import asyncio
import multiprocessing
import os
import tempfile
import time
from unittest.mock import patch

import pytest

from effgen.models._rate_limit import RateLimitCoordinator, RateLimitExceeded
from effgen.models._rate_limit_store import SQLiteRateLimitStore


def _worker_burst(
    db_path: str,
    rpm: int,
    n_requests: int,
    result_queue: multiprocessing.Queue,
    fake_start_time: float | None = None,
    start_event: multiprocessing.Event | None = None,
) -> None:
    """Worker: fires n_requests and reports how many were throttled vs completed."""
    if start_event is not None:
        start_event.wait(timeout=10)

    store = SQLiteRateLimitStore(db_path)
    coord = RateLimitCoordinator(
        provider="test",
        model="model",
        rpm=rpm, rph=rpm * 60, rpd=rpm * 1440,
        tpm=1_000_000, tph=10_000_000, tpd=100_000_000,
        storage=store,
    )

    completed = 0

    async def run():
        nonlocal completed
        for _ in range(n_requests):
            try:
                await coord.acquire()
                coord.record(actual_tokens=10)
                completed += 1
            except RateLimitExceeded:
                pass  # daily limit — not expected in this test

    if fake_start_time is None:
        asyncio.run(run())
    else:
        current = [fake_start_time]

        def fake_time():
            return current[0]

        async def fake_sleep(seconds):
            current[0] += seconds + 0.001

        with patch("effgen.models._rate_limit.time.time", side_effect=fake_time), \
             patch("effgen.models._rate_limit.asyncio.sleep", side_effect=fake_sleep):
            asyncio.run(run())

    store.close()
    result_queue.put({
        "completed": completed,
        "coordinator_throttled": coord.total_throttled,
        "throttle_seconds": round(coord.total_throttle_seconds, 2),
    })


@pytest.mark.slow
def test_multiprocess_throttle_detected():
    """
    Pre-fill the DB with rpm-1 events from process 1, then process 2 tries to
    fire 2 more. It should see the window is full and throttle.

    We verify cross-process throttling is detected without waiting for the full
    window to expire: the coordinator should increment total_throttled >= 1 and
    eventually complete all requests.
    """
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    try:
        rpm = 3
        store = SQLiteRateLimitStore(db_path)
        # Simulate process 1 having already used rpm slots in the last 30 seconds
        now = time.time()
        for i in range(rpm):
            store.insert_event("test", "model", "request", now - (30 - i * 2), 0)
        store.close()

        # Process 2 now tries to fire 2 more requests against the same DB
        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue()

        p = ctx.Process(
            target=_worker_burst,
            args=(db_path, rpm, 2, result_queue, now),
            daemon=False,
        )
        p.start()
        p.join(timeout=10)

        assert not p.is_alive(), "Worker process did not complete within timeout"

        try:
            result = result_queue.get(timeout=5)
        except Exception as e:
            pytest.fail(f"No result from worker: {e}")

        # Worker should have had to throttle (the window was already full)
        assert result["completed"] == 2, f"Expected 2 completed: {result}"
        assert result["coordinator_throttled"] >= 1, (
            f"Expected >= 1 throttle events from cross-process window: {result}"
        )
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


@pytest.mark.slow
def test_multiprocess_shared_db_event_persistence():
    """Events written by one process are visible to another via the DB."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    try:
        store = SQLiteRateLimitStore(db_path)
        now = time.time()
        store.insert_event("groq", "model", "request", now, 0)
        store.close()

        # Simulate second process opening the same DB
        store2 = SQLiteRateLimitStore(db_path)
        rows = store2.query_window("groq", "model", "request", now - 1)
        store2.close()

        assert len(rows) == 1
    finally:
        os.unlink(db_path)


@pytest.mark.slow
def test_multiprocess_two_workers_combined():
    """Two workers each fire rpm+1 requests; combined they exceed rpm, so at
    least one worker must throttle when it reads the other's events."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    try:
        rpm = 3
        n_per_worker = 2  # combined 4 > 3 rpm
        fake_start_time = time.time()

        # Initialize schema
        SQLiteRateLimitStore(db_path).close()

        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue()
        start_event = ctx.Event()

        p1 = ctx.Process(
            target=_worker_burst,
            args=(db_path, rpm, n_per_worker, result_queue, fake_start_time, start_event),
            daemon=False,
        )
        p2 = ctx.Process(
            target=_worker_burst,
            args=(db_path, rpm, n_per_worker, result_queue, fake_start_time, start_event),
            daemon=False,
        )

        p1.start()
        p2.start()
        start_event.set()
        p1.join(timeout=10)
        p2.join(timeout=10)

        assert not p1.is_alive(), "Worker 1 did not complete within timeout"
        assert not p2.is_alive(), "Worker 2 did not complete within timeout"

        results = []
        for _ in range(2):
            try:
                results.append(result_queue.get(timeout=5))
            except Exception:
                pass

        assert len(results) == 2, f"Both workers must report results: {results}"

        total_completed = sum(r["completed"] for r in results)
        total_throttled = sum(r["coordinator_throttled"] for r in results)

        # Both workers must eventually complete all requests
        assert total_completed == n_per_worker * 2, (
            f"All requests should complete: {results}"
        )
        # Combined > rpm means at least one worker must have seen throttling
        assert total_throttled >= 1, (
            f"Expected >= 1 throttle event from cross-process coordination: {results}"
        )

        # Verify events are present in the DB
        final_store = SQLiteRateLimitStore(db_path)
        all_events = final_store.query_window(
            "test", "model", "request", time.time() - 86400
        )
        final_store.close()
        assert len(all_events) == n_per_worker * 2, (
            f"Expected {n_per_worker * 2} events in DB, got {len(all_events)}"
        )
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass
