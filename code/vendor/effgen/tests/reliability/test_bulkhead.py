"""
Tests for Bulkhead isolation pattern.

Coverage:
- acquire() context manager — sync
- async_acquire() context manager — async
- BulkheadFull raised when at capacity
- queue_size limits total waiting callers
- queue_timeout enforced
- guard() / async_guard() decorators
- Stats snapshot
- BulkheadRegistry
- ProviderRegistry.get_bulkhead() integration
- Concurrent callers (thread-safety)
- Invalid queue_timeout raises ValueError
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from effgen.reliability.bulkhead import (
    Bulkhead,
    BulkheadFull,
    BulkheadRegistry,
)

pytestmark = pytest.mark.reliability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slow_work(seconds: float = 0.05) -> str:
    time.sleep(seconds)
    return "done"


async def _async_slow_work(seconds: float = 0.05) -> str:
    await asyncio.sleep(seconds)
    return "done"


# ---------------------------------------------------------------------------
# Sync acquire()
# ---------------------------------------------------------------------------


class TestBulkheadSyncAcquire:
    def test_basic_acquire_release(self):
        bh = Bulkhead("test", max_concurrency=2, queue_size=0, queue_timeout=1.0)
        with bh.acquire():
            s = bh.stats()
            assert s["active"] == 1
        s = bh.stats()
        assert s["active"] == 0

    def test_returns_result(self):
        bh = Bulkhead("test2", max_concurrency=2, queue_size=10, queue_timeout=1.0)
        result = None
        with bh.acquire():
            result = "value"
        assert result == "value"

    def test_released_on_exception(self):
        """Active count must return to zero even when the body raises."""
        bh = Bulkhead("test3", max_concurrency=2, queue_size=0, queue_timeout=1.0)
        with pytest.raises(RuntimeError):
            with bh.acquire():
                raise RuntimeError("body error")
        assert bh.stats()["active"] == 0

    def test_bulkhead_full_when_queue_exhausted(self):
        """When concurrency and queue are both full, BulkheadFull is raised."""
        bh = Bulkhead("full_test", max_concurrency=1, queue_size=0, queue_timeout=0.01)

        results: list[Exception | str] = []

        def _work():
            try:
                with bh.acquire():
                    _slow_work(0.2)
                    results.append("ok")
            except BulkheadFull as e:
                results.append(e)

        # Fill the one slot
        t1 = threading.Thread(target=_work)
        t1.start()
        time.sleep(0.02)  # Let t1 enter the bulkhead

        # Try to acquire while full → BulkheadFull
        with pytest.raises(BulkheadFull) as exc_info:
            with bh.acquire():
                pass

        t1.join()
        err = exc_info.value
        assert err.name == "full_test"

    def test_queue_allows_waiting(self):
        """A queued caller should succeed once a permit is released."""
        bh = Bulkhead("queue_test", max_concurrency=1, queue_size=5, queue_timeout=2.0)
        results: list[str] = []

        def _work(label: str):
            with bh.acquire():
                _slow_work(0.05)
                results.append(label)

        threads = [threading.Thread(target=_work, args=(f"t{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == ["t0", "t1", "t2"]

    def test_timeout_raises_bulkhead_full(self):
        """If a queued caller waits longer than queue_timeout, BulkheadFull is raised."""
        bh = Bulkhead("timeout_test", max_concurrency=1, queue_size=5, queue_timeout=0.05)

        holder_entered = threading.Event()
        holder_release = threading.Event()

        def _holder():
            with bh.acquire():
                holder_entered.set()
                holder_release.wait(timeout=2.0)

        t = threading.Thread(target=_holder)
        t.start()
        holder_entered.wait(timeout=2.0)

        # This caller should time out waiting for the permit
        with pytest.raises(BulkheadFull):
            with bh.acquire():
                pass

        holder_release.set()
        t.join()


# ---------------------------------------------------------------------------
# guard() decorator — sync
# ---------------------------------------------------------------------------


class TestBulkheadGuardDecorator:
    def test_guard_allows_call(self):
        bh = Bulkhead("guard_test", max_concurrency=5, queue_size=10, queue_timeout=1.0)

        @bh.guard()
        def fn() -> str:
            return "guarded"

        assert fn() == "guarded"

    def test_guard_tracks_active(self):
        bh = Bulkhead("guard_track", max_concurrency=5, queue_size=10, queue_timeout=1.0)
        active_during: list[int] = []

        @bh.guard()
        def fn():
            active_during.append(bh.stats()["active"])
            return "ok"

        fn()
        assert active_during == [1]
        assert bh.stats()["active"] == 0

    def test_guard_preserves_name(self):
        bh = Bulkhead("guard_name", max_concurrency=5, queue_size=10, queue_timeout=1.0)

        @bh.guard()
        def my_function():
            return "x"

        assert my_function.__name__ == "my_function"


# ---------------------------------------------------------------------------
# Async async_acquire()
# ---------------------------------------------------------------------------


class TestBulkheadAsyncAcquire:
    @pytest.mark.asyncio
    async def test_basic_async_acquire(self):
        bh = Bulkhead("async_test", max_concurrency=2, queue_size=5, queue_timeout=1.0)
        async with bh.async_acquire():
            s = bh.stats()
            assert s["active"] == 1
        assert bh.stats()["active"] == 0

    @pytest.mark.asyncio
    async def test_async_bulkhead_full_on_timeout(self):
        bh = Bulkhead("async_full", max_concurrency=1, queue_size=5, queue_timeout=0.05)

        results: list[str | Exception] = []

        async def _hold():
            async with bh.async_acquire():
                await asyncio.sleep(0.3)
                results.append("held")

        async def _try():
            try:
                async with bh.async_acquire():
                    results.append("got_permit")
            except BulkheadFull:
                results.append("full")

        holder = asyncio.create_task(_hold())
        await asyncio.sleep(0.01)  # Let holder acquire
        await _try()
        await holder

        assert "full" in results

    @pytest.mark.asyncio
    async def test_async_concurrent_tasks_within_limit(self):
        bh = Bulkhead("async_concurrent", max_concurrency=5, queue_size=20, queue_timeout=2.0)
        results: list[str] = []

        async def _worker(label: str):
            async with bh.async_acquire():
                await asyncio.sleep(0.02)
                results.append(label)

        await asyncio.gather(*[_worker(f"t{i}") for i in range(5)])
        assert len(results) == 5


# ---------------------------------------------------------------------------
# async_guard() decorator
# ---------------------------------------------------------------------------


class TestBulkheadAsyncGuardDecorator:
    @pytest.mark.asyncio
    async def test_async_guard_allows_call(self):
        bh = Bulkhead("async_guard", max_concurrency=5, queue_size=10, queue_timeout=1.0)

        @bh.async_guard()
        async def fn() -> str:
            return "async_guarded"

        assert await fn() == "async_guarded"

    @pytest.mark.asyncio
    async def test_async_guard_preserves_name(self):
        bh = Bulkhead("async_guard_name", max_concurrency=5, queue_size=10, queue_timeout=1.0)

        @bh.async_guard()
        async def my_async_fn():
            return "x"

        assert my_async_fn.__name__ == "my_async_fn"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestBulkheadStats:
    def test_initial_stats(self):
        bh = Bulkhead("stats", max_concurrency=10, queue_size=50, queue_timeout=5.0)
        s = bh.stats()
        assert s["name"] == "stats"
        assert s["max_concurrency"] == 10
        assert s["queue_size"] == 50
        assert s["active"] == 0
        assert s["queued"] == 0
        assert s["total_accepted"] == 0
        assert s["total_rejected"] == 0
        assert s["total_timeout"] == 0
        assert s["utilization_pct"] == 0.0

    def test_stats_after_calls(self):
        bh = Bulkhead("stats2", max_concurrency=5, queue_size=10, queue_timeout=1.0)
        with bh.acquire():
            pass
        with bh.acquire():
            pass
        s = bh.stats()
        assert s["total_accepted"] == 2

    def test_repr(self):
        bh = Bulkhead("repr_bh", max_concurrency=10, queue_size=20, queue_timeout=1.0)
        r = repr(bh)
        assert "repr_bh" in r
        assert "10" in r

    def test_utilization_pct(self):
        bh = Bulkhead("util_test", max_concurrency=4, queue_size=10, queue_timeout=2.0)

        done = threading.Event()
        active_pct: list[float] = []

        def _hold():
            with bh.acquire():
                active_pct.append(bh.stats()["utilization_pct"])
                done.wait(timeout=2.0)

        t = threading.Thread(target=_hold)
        t.start()
        time.sleep(0.02)
        done.set()
        t.join()

        # 1 active out of 4 = 25%
        assert active_pct[0] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------


class TestBulkheadInvalidConfig:
    def test_none_queue_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            Bulkhead("bad", queue_timeout=None)  # type: ignore[arg-type]

    def test_zero_queue_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            Bulkhead("bad", queue_timeout=0.0)

    def test_negative_queue_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            Bulkhead("bad", queue_timeout=-1.0)


# ---------------------------------------------------------------------------
# BulkheadRegistry
# ---------------------------------------------------------------------------


class TestBulkheadRegistry:
    def setup_method(self):
        self.registry = BulkheadRegistry()

    def test_get_or_create_new(self):
        bh = self.registry.get_or_create("provider_x")
        assert bh.name == "provider_x"

    def test_get_or_create_returns_same(self):
        bh1 = self.registry.get_or_create("provider_y")
        bh2 = self.registry.get_or_create("provider_y")
        assert bh1 is bh2

    def test_get_returns_none(self):
        assert self.registry.get("nonexistent") is None

    def test_all_stats(self):
        self.registry.get_or_create("p1")
        self.registry.get_or_create("p2")
        stats = self.registry.all_stats()
        names = {s["name"] for s in stats}
        assert "p1" in names and "p2" in names

    def test_clear(self):
        self.registry.get_or_create("p3")
        self.registry.clear()
        assert self.registry.get("p3") is None


# ---------------------------------------------------------------------------
# ProviderRegistry integration
# ---------------------------------------------------------------------------


class TestProviderRegistryBulkheadIntegration:
    def setup_method(self):
        from effgen.models.registry import ProviderRegistry

        ProviderRegistry.clear()

        class FakeAdapter:
            pass

        ProviderRegistry.register(
            "bulkhead_provider",
            FakeAdapter,
            {"model-1": {"name": "Model 1"}},
            env_keys=["FAKE_KEY"],
        )

    def teardown_method(self):
        from effgen.models.registry import ProviderRegistry

        ProviderRegistry.reset()

    def test_get_bulkhead_creates(self):
        from effgen.models.registry import ProviderRegistry

        bh = ProviderRegistry.get_bulkhead("bulkhead_provider", max_concurrency=5)
        assert bh.name == "bulkhead_provider"
        assert bh.max_concurrency == 5

    def test_get_bulkhead_same_instance(self):
        from effgen.models.registry import ProviderRegistry

        bh1 = ProviderRegistry.get_bulkhead("bulkhead_provider")
        bh2 = ProviderRegistry.get_bulkhead("bulkhead_provider")
        assert bh1 is bh2

    def test_get_bulkhead_unknown_provider(self):
        from effgen.models.registry import ProviderRegistry

        with pytest.raises(KeyError):
            ProviderRegistry.get_bulkhead("does_not_exist")


# ---------------------------------------------------------------------------
# Concurrent stress test
# ---------------------------------------------------------------------------


class TestBulkheadConcurrency:
    def test_concurrent_threads_respect_limit(self):
        """No more than max_concurrency threads should be active simultaneously."""
        max_c = 3
        bh = Bulkhead("stress", max_concurrency=max_c, queue_size=50, queue_timeout=5.0)
        high_water_mark: list[int] = []
        lock = threading.Lock()

        def _work():
            with bh.acquire():
                with lock:
                    high_water_mark.append(bh.stats()["active"])
                time.sleep(0.02)

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(_work) for _ in range(10)]
            for f in as_completed(futures):
                f.result()  # re-raise any exceptions

        assert max(high_water_mark) <= max_c, (
            f"Concurrency exceeded limit: {max(high_water_mark)} > {max_c}"
        )
