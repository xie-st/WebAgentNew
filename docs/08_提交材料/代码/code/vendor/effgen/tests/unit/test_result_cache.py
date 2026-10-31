"""Unit tests for effgen.cache.result_cache.ResultCache."""
from __future__ import annotations

import time

import pytest

from effgen.cache.result_cache import ResultCache, ResultCacheEntry, _cosine


def _embed(text: str) -> list[float]:
    """Toy embedding: bag-of-letter-counts over a fixed alphabet."""
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    counts = [0] * len(alphabet)
    for ch in text.lower():
        if ch in alphabet:
            counts[alphabet.index(ch)] += 1
    norm = (sum(c * c for c in counts) ** 0.5) or 1.0
    return [c / norm for c in counts]


class TestCosine:
    def test_zero_for_empty_or_mismatched(self):
        assert _cosine([], [1, 2, 3]) == 0.0
        assert _cosine([1, 2], [1, 2, 3]) == 0.0
        assert _cosine([0, 0, 0], [1, 2, 3]) == 0.0

    def test_one_for_identical(self):
        v = [1.0, 0.0, 1.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_perpendicular(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)


class TestResultCacheEntry:
    def test_no_ttl_never_expires(self):
        e = ResultCacheEntry(key="k", query="q", value=1)
        assert not e.is_expired()

    def test_expires_after_ttl(self):
        e = ResultCacheEntry(key="k", query="q", value=1, ttl=0.01)
        time.sleep(0.02)
        assert e.is_expired()


class TestResultCacheBasics:
    def test_max_size_must_be_positive(self):
        with pytest.raises(ValueError):
            ResultCache(max_size=0)

    def test_make_key_is_query_normalized(self):
        a = ResultCache.make_key("  Hello World  ")
        b = ResultCache.make_key("hello world")
        assert a == b
        assert ResultCache.make_key("x", "tool1") != ResultCache.make_key("x", "tool2")

    def test_put_get_roundtrip(self):
        c = ResultCache(max_size=4, default_ttl=None)
        c.put("what is 2+2", 4, tool="calc")
        assert c.get("what is 2+2", tool="calc") == 4

    def test_miss_returns_none(self):
        c = ResultCache(max_size=4)
        assert c.get("nope") is None

    def test_returned_key_is_stable(self):
        c = ResultCache(max_size=4)
        k1 = c.put("query", 1)
        k2 = ResultCache.make_key("query", "")
        assert k1 == k2

    def test_invalidate(self):
        c = ResultCache(max_size=4, default_ttl=None)
        c.put("q", 1, tool="t")
        assert c.invalidate("q", tool="t") is True
        assert c.get("q", tool="t") is None
        assert c.invalidate("q", tool="t") is False

    def test_invalidate_tool(self):
        c = ResultCache(max_size=8, default_ttl=None)
        c.put("a", 1, tool="t1")
        c.put("b", 2, tool="t1")
        c.put("c", 3, tool="t2")
        n = c.invalidate_tool("t1")
        assert n == 2
        assert c.get("a", tool="t1") is None
        assert c.get("c", tool="t2") == 3

    def test_clear_resets(self):
        c = ResultCache(max_size=4, default_ttl=None)
        c.put("a", 1)
        c.get("a")
        c.get("missing")
        c.clear()
        assert len(c) == 0
        s = c.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["semantic_hits"] == 0


class TestResultCacheTTL:
    def test_default_ttl_expires_entries(self):
        c = ResultCache(max_size=4, default_ttl=0.01)
        c.put("q", 1)
        time.sleep(0.02)
        assert c.get("q") is None

    def test_explicit_ttl_overrides_default(self):
        c = ResultCache(max_size=4, default_ttl=0.01)
        c.put("q", 1, ttl=10.0)
        time.sleep(0.02)
        assert c.get("q") == 1

    def test_per_tool_ttl_override(self):
        c = ResultCache(max_size=4, default_ttl=10.0)
        c.set_tool_ttl("fast", 0.01)
        c.put("q", 1, tool="fast")
        time.sleep(0.02)
        assert c.get("q", tool="fast") is None


class TestResultCacheLRU:
    def test_eviction(self):
        c = ResultCache(max_size=2, default_ttl=None)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        # "a" should be evicted
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3


class TestResultCacheStats:
    def test_hit_rate(self):
        c = ResultCache(max_size=4, default_ttl=None)
        c.put("q", 1)
        c.get("q")
        c.get("q")
        c.get("missing")
        s = c.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["hit_rate"] == pytest.approx(2 / 3)


class TestResultCacheSemantic:
    def test_get_similar_returns_match_above_threshold(self):
        c = ResultCache(
            max_size=4,
            default_ttl=None,
            embed_fn=_embed,
            similarity_threshold=0.5,
        )
        c.put("the cat sat on the mat", "answer1", tool="t")
        # Different wording but high lexical overlap → should hit semantic
        v = c.get_similar("the cat sat on the matt", tool="t")
        assert v == "answer1"

    def test_get_similar_returns_none_when_below_threshold(self):
        c = ResultCache(
            max_size=4,
            default_ttl=None,
            embed_fn=_embed,
            similarity_threshold=0.99,
        )
        c.put("the cat sat on the mat", "answer1", tool="t")
        assert c.get_similar("zebras roam the savanna", tool="t") is None

    def test_get_falls_back_to_semantic(self):
        c = ResultCache(
            max_size=4,
            default_ttl=None,
            embed_fn=_embed,
            similarity_threshold=0.5,
        )
        c.put("the cat sat on the mat", 99)
        # Exact key miss; semantic should pick it up
        assert c.get("the cat sat on the matt") == 99
        assert c.stats()["semantic_hits"] == 1

    def test_get_similar_without_embed_returns_none(self):
        c = ResultCache(max_size=4, default_ttl=None)
        c.put("q", 1)
        assert c.get_similar("q") is None
