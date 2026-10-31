"""Unit tests for effgen.cache.prompt_cache.PromptCache."""
from __future__ import annotations

import time

import pytest

from effgen.cache.prompt_cache import PromptCache, PromptCacheEntry


class TestPromptCacheEntry:
    def test_not_expired_without_ttl(self):
        e = PromptCacheEntry(key="k", payload="v")
        assert not e.is_expired()

    def test_expired_when_past_ttl(self):
        e = PromptCacheEntry(key="k", payload="v", ttl=0.01)
        time.sleep(0.02)
        assert e.is_expired()

    def test_not_expired_within_ttl(self):
        e = PromptCacheEntry(key="k", payload="v", ttl=10.0)
        assert not e.is_expired()


class TestPromptCacheBasics:
    def test_max_size_must_be_positive(self):
        with pytest.raises(ValueError):
            PromptCache(max_size=0)
        with pytest.raises(ValueError):
            PromptCache(max_size=-1)

    def test_put_then_get_round_trip(self):
        c = PromptCache(max_size=4)
        c.put("k1", {"x": 1})
        assert c.get("k1") == {"x": 1}

    def test_miss_returns_none(self):
        c = PromptCache(max_size=4)
        assert c.get("nope") is None

    def test_contains(self):
        c = PromptCache(max_size=4)
        c.put("k", "v")
        assert c.contains("k")
        assert not c.contains("missing")

    def test_invalidate_returns_true_when_present(self):
        c = PromptCache(max_size=4)
        c.put("k", "v")
        assert c.invalidate("k") is True
        assert c.get("k") is None
        assert c.invalidate("k") is False  # already gone

    def test_clear_resets_state(self):
        c = PromptCache(max_size=4)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.get("missing")
        assert len(c) == 2
        c.clear()
        assert len(c) == 0
        assert c.stats()["hits"] == 0
        assert c.stats()["misses"] == 0


class TestPromptCacheLRU:
    def test_eviction_drops_oldest(self):
        c = PromptCache(max_size=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_get_promotes_recency(self):
        c = PromptCache(max_size=2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")          # touch "a" → "b" is now LRU
        c.put("c", 3)        # should evict "b"
        assert c.get("b") is None
        assert c.get("a") == 1
        assert c.get("c") == 3


class TestPromptCacheTTL:
    def test_default_ttl_applies(self):
        c = PromptCache(max_size=4, default_ttl=0.01)
        c.put("k", "v")
        time.sleep(0.02)
        assert c.get("k") is None
        # Expired entry must be removed
        assert "k" not in c._entries  # noqa: SLF001 - inspecting test state

    def test_explicit_ttl_overrides_default(self):
        c = PromptCache(max_size=4, default_ttl=0.01)
        c.put("k", "v", ttl=10.0)
        time.sleep(0.02)
        assert c.get("k") == "v"

    def test_contains_returns_false_after_expiry(self):
        c = PromptCache(max_size=4)
        c.put("k", "v", ttl=0.01)
        time.sleep(0.02)
        assert not c.contains("k")


class TestPromptCacheStats:
    def test_hit_and_miss_counters(self):
        c = PromptCache(max_size=4)
        c.put("a", 1)
        c.get("a")            # hit
        c.get("a")            # hit
        c.get("nope")         # miss
        s = c.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["size"] == 1
        assert s["max_size"] == 4
        assert abs(s["hit_rate"] - (2 / 3)) < 1e-9

    def test_hit_rate_zero_when_no_calls(self):
        c = PromptCache(max_size=4)
        assert c.stats()["hit_rate"] == 0.0


class TestPromptCacheFingerprint:
    def test_deterministic(self):
        a = PromptCache.fingerprint("hello")
        b = PromptCache.fingerprint("hello")
        assert a == b

    def test_different_prompts_differ(self):
        assert PromptCache.fingerprint("x") != PromptCache.fingerprint("y")

    def test_extras_change_key(self):
        assert PromptCache.fingerprint("x", "modelA") != PromptCache.fingerprint("x", "modelB")

    def test_bytes_supported(self):
        a = PromptCache.fingerprint("hi")
        b = PromptCache.fingerprint(b"hi")
        # Both produce a hex digest, both encode "hi" as utf-8 bytes
        assert a == b
