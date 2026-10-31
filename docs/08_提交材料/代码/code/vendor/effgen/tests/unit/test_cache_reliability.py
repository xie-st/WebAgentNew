"""Behavioral coverage for the ``cache/`` and ``reliability/`` subsystems.

These subsystems shipped with construction-only smoke coverage. This module
exercises their *behavior*: cache hit/miss, TTL expiry, invalidation, LRU
eviction, concurrent access, and the "no stale result after a config change"
property; and for reliability, that ``Retry``/``is_transient_error`` honor the
core error taxonomy (never retry auth/not-found), that ``Bulkhead`` caps
concurrency, that timeouts fire, and that chaos injects only when enabled.

Everything here is pure in-process logic — no network, no mocked live behavior.
"""

from __future__ import annotations

import threading
import time

import pytest

from effgen.cache import PromptCache, PromptCacheEntry, ResultCache
from effgen.models.errors import (
    ModelAuthError,
    ModelNotFoundError,
    ProviderTransientError,
)
from effgen.reliability.bulkhead import Bulkhead, BulkheadFull
from effgen.reliability.chaos import Chaos, Http5xx
from effgen.reliability.retry import (
    Retry,
    RetryExhausted,
    is_transient_error,
    retryable,
)
from effgen.reliability.timeouts import TimeoutError as EffGenTimeout
from effgen.reliability.timeouts import with_timeout

# --------------------------------------------------------------------------- #
# PromptCache
# --------------------------------------------------------------------------- #

def test_prompt_cache_hit_and_miss():
    cache = PromptCache(max_size=8)
    assert cache.get("k") is None  # miss
    cache.put("k", {"kv": 1})
    assert cache.get("k") == {"kv": 1}  # hit
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_prompt_cache_entry_ttl_expiry_deterministic():
    entry = PromptCacheEntry(key="k", payload=1, created_at=1000.0, ttl=10.0)
    assert entry.is_expired(now=1005.0) is False
    assert entry.is_expired(now=1011.0) is True
    # No TTL → never expires.
    assert PromptCacheEntry(key="k", payload=1, ttl=None).is_expired(now=1e12) is False


def test_prompt_cache_ttl_expiry_via_get():
    cache = PromptCache(max_size=8, default_ttl=0.05)
    cache.put("k", "v")
    assert cache.get("k") == "v"
    time.sleep(0.12)
    assert cache.get("k") is None  # expired → evicted on access


def test_prompt_cache_invalidate_and_clear():
    cache = PromptCache()
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.invalidate("a") is True
    assert cache.invalidate("a") is False  # already gone
    assert cache.get("a") is None
    cache.clear()
    assert len(cache) == 0 and cache.get("b") is None


def test_prompt_cache_lru_eviction():
    cache = PromptCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")           # 'a' is now most-recently-used
    cache.put("c", 3)        # evicts the LRU entry ('b')
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3
    assert len(cache) == 2


def test_prompt_cache_fingerprint_separates_by_discriminator():
    # The same prompt under a different model/config must not collide → a config
    # change yields a different key and therefore a clean miss (no stale reuse).
    k1 = PromptCache.fingerprint("hello", "model-a")
    k2 = PromptCache.fingerprint("hello", "model-b")
    assert k1 != k2


def test_prompt_cache_concurrent_access_is_consistent():
    cache = PromptCache(max_size=1000)
    errors: list[Exception] = []

    def worker(base: int):
        try:
            for i in range(200):
                key = f"k{base}-{i}"
                cache.put(key, base * 1000 + i)
                assert cache.get(key) == base * 1000 + i
        except Exception as exc:  # pragma: no cover - surfaced via assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# --------------------------------------------------------------------------- #
# ResultCache
# --------------------------------------------------------------------------- #

def test_result_cache_hit_miss_and_invalidate():
    cache = ResultCache(max_size=16, default_ttl=None)
    assert cache.get("q", tool="calc") is None
    cache.put("q", "42", tool="calc")
    assert cache.get("q", tool="calc") == "42"
    # A different tool discriminator is a separate entry.
    assert cache.get("q", tool="other") is None
    assert cache.invalidate("q", tool="calc") is True
    assert cache.get("q", tool="calc") is None


def test_result_cache_per_tool_ttl_and_invalidate_tool():
    cache = ResultCache(max_size=16, default_ttl=None)
    cache.set_tool_ttl("weather", 0.05)
    cache.put("today", "sunny", tool="weather")
    assert cache.get("today", tool="weather") == "sunny"
    time.sleep(0.12)
    assert cache.get("today", tool="weather") is None  # per-tool TTL expired

    # invalidate_tool drops every entry for a tool (e.g. after a config change),
    # so a stale result can't survive a reconfiguration.
    cache.put("a", 1, tool="db")
    cache.put("b", 2, tool="db")
    cache.put("c", 3, tool="keep")
    removed = cache.invalidate_tool("db")
    assert removed == 2
    assert cache.get("a", tool="db") is None and cache.get("b", tool="db") is None
    assert cache.get("c", tool="keep") == 3


def test_result_cache_lru_eviction():
    cache = ResultCache(max_size=2, default_ttl=None)
    cache.put("a", 1, tool="t")
    cache.put("b", 2, tool="t")
    cache.get("a", tool="t")        # 'a' most-recently-used
    cache.put("c", 3, tool="t")     # evicts 'b'
    assert cache.get("b", tool="t") is None
    assert cache.get("a", tool="t") == 1 and cache.get("c", tool="t") == 3


# --------------------------------------------------------------------------- #
# Retry / error taxonomy
# --------------------------------------------------------------------------- #

def test_is_transient_error_honors_core_taxonomy():
    # Auth / not-found are fail-fast in the core taxonomy → never retried.
    assert is_transient_error(ModelAuthError(provider="openai", message="bad key")) is False
    assert is_transient_error(
        ModelNotFoundError(provider="openai", model_name="ghost")
    ) is False
    # A provider 5xx is transient → retried.
    assert is_transient_error(
        ProviderTransientError(provider="openai", status_code=503)
    ) is True
    # Plain network errors remain retryable via the heuristic fallback.
    assert is_transient_error(ConnectionError("reset")) is True
    assert is_transient_error(ValueError("just wrong")) is False


def test_retryable_does_not_retry_auth_error():
    calls = {"n": 0}

    @retryable(Retry(max_attempts=4, base_delay=0.0))
    def call():
        calls["n"] += 1
        raise ModelAuthError(provider="openai", message="bad key")

    with pytest.raises(ModelAuthError):
        call()
    assert calls["n"] == 1  # failed fast, no retry storm


def test_retryable_retries_transient_then_exhausts():
    calls = {"n": 0}

    @retryable(Retry(max_attempts=3, base_delay=0.0))
    def call():
        calls["n"] += 1
        raise ProviderTransientError(provider="openai", status_code=500)

    with pytest.raises(RetryExhausted):
        call()
    assert calls["n"] == 3  # all attempts used


def test_retryable_succeeds_after_transient_failures():
    calls = {"n": 0}

    @retryable(Retry(max_attempts=5, base_delay=0.0))
    def call():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderTransientError(provider="openai", status_code=502)
        return "ok"

    assert call() == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retryable_async_does_not_retry_not_found():
    calls = {"n": 0}

    @retryable(Retry(max_attempts=4, base_delay=0.0))
    async def call():
        calls["n"] += 1
        raise ModelNotFoundError(provider="openai", model_name="ghost")

    with pytest.raises(ModelNotFoundError):
        await call()
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Bulkhead
# --------------------------------------------------------------------------- #

def test_bulkhead_caps_concurrency():
    bh = Bulkhead("test", max_concurrency=2, queue_size=8, queue_timeout=2.0)
    peak = {"v": 0}
    active = {"v": 0}
    lock = threading.Lock()
    start = threading.Barrier(4)

    def worker():
        start.wait()
        with bh.acquire():
            with lock:
                active["v"] += 1
                peak["v"] = max(peak["v"], active["v"])
            time.sleep(0.05)
            with lock:
                active["v"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak["v"] <= 2  # never more than max_concurrency in the section


def test_bulkhead_rejects_when_full():
    # max_concurrency=1, no queue: a second concurrent acquire is rejected fast.
    bh = Bulkhead("full", max_concurrency=1, queue_size=0, queue_timeout=0.2)
    held = threading.Event()
    release = threading.Event()
    rejected = {"v": False}

    def holder():
        with bh.acquire():
            held.set()
            release.wait()

    t = threading.Thread(target=holder)
    t.start()
    held.wait()
    try:
        with bh.acquire():
            pass
    except BulkheadFull:
        rejected["v"] = True
    finally:
        release.set()
        t.join()
    assert rejected["v"] is True


# --------------------------------------------------------------------------- #
# Timeouts
# --------------------------------------------------------------------------- #

def test_with_timeout_fires_on_overrun():
    with pytest.raises(EffGenTimeout):
        with with_timeout(0.1, operation="slow"):
            time.sleep(1.0)


def test_with_timeout_passes_under_budget():
    with with_timeout(1.0, operation="fast"):
        result = 1 + 1
    assert result == 2


# --------------------------------------------------------------------------- #
# Chaos — only injects when explicitly enabled
# --------------------------------------------------------------------------- #

def test_chaos_no_op_without_rules():
    chaos = Chaos(seed=1, enabled=True)
    # No rules configured → maybe_inject is a no-op regardless of enabled.
    for _ in range(50):
        chaos.maybe_inject("openai")  # must not raise


def test_chaos_disabled_never_injects():
    chaos = Chaos(seed=1, enabled=False)
    chaos.add_rule("openai", Http5xx, fault_rate=1.0)
    for _ in range(20):
        chaos.maybe_inject("openai")  # disabled → no fault


def test_chaos_injects_when_enabled_with_rule():
    chaos = Chaos(seed=1, enabled=True)
    chaos.add_rule("openai", Http5xx, fault_rate=1.0)
    from effgen.reliability.chaos import ChaosHttp5xxError
    with pytest.raises(ChaosHttp5xxError):
        chaos.maybe_inject("openai")


# --------------------------------------------------------------------------- #
# CircuitBreaker — one canonical home (single consolidated implementation)
# --------------------------------------------------------------------------- #

def test_single_circuit_breaker_home():
    # The per-tool manager in utils/ must wrap the canonical reliability core,
    # not ship a second divergent implementation.
    from effgen.reliability.circuit import CircuitBreaker as Core
    from effgen.utils.circuit_breaker import _CoreCircuitBreaker

    assert _CoreCircuitBreaker is Core

