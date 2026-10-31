"""Result caching for tool outputs and agent responses.

Provides a thread-safe LRU cache with per-entry TTL, hash-based exact lookup,
and an *optional* semantic similarity lookup based on user-supplied embeddings.

The cache deliberately knows nothing about how to embed text — that keeps the
core dependency-free. Callers (or higher-level wrappers) may pass an
``embed_fn`` that returns a ``list[float]`` vector; if available, similarity
lookups become possible via :meth:`get_similar`.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

EmbedFn = Callable[[str], "list[float]"]


@dataclass
class ResultCacheEntry:
    """A cached result with its query, optional embedding and TTL."""

    key: str
    query: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: float | None = None
    hits: int = 0
    embedding: list[float] | None = None
    tool: str | None = None

    def is_expired(self, now: float | None = None) -> bool:
        """True when the entry has outlived its TTL (a ``None`` TTL never expires)."""
        if self.ttl is None:
            return False
        now = now if now is not None else time.time()
        return (now - self.created_at) > self.ttl


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class ResultCache:
    """LRU cache for tool / agent results.

    Parameters
    ----------
    max_size:
        Maximum number of entries.
    default_ttl:
        Default TTL in seconds. ``None`` disables expiry by default.
    embed_fn:
        Optional function ``str -> list[float]``. When provided, ``put`` will
        compute and store an embedding for each entry, and :meth:`get_similar`
        becomes available.
    similarity_threshold:
        Minimum cosine similarity to count as a semantic hit (default 0.92).
    """

    def __init__(
        self,
        max_size: int = 512,
        default_ttl: float | None = 3600.0,
        embed_fn: EmbedFn | None = None,
        similarity_threshold: float = 0.92,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self._tool_ttl: dict[str, float | None] = {}
        self._entries: OrderedDict[str, ResultCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._semantic_hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ keys
    @staticmethod
    def make_key(query: str, *extra: Any) -> str:
        """Return the cache key for *query* (case/whitespace-insensitive) plus *extra*."""
        h = hashlib.sha256()
        h.update(query.strip().lower().encode("utf-8"))
        for item in extra:
            h.update(b"\x1f")
            h.update(str(item).encode("utf-8"))
        return h.hexdigest()

    # ------------------------------------------------------------------ config
    def set_tool_ttl(self, tool_name: str, ttl: float | None) -> None:
        """Configure a per-tool TTL override."""
        self._tool_ttl[tool_name] = ttl

    # ------------------------------------------------------------------ ops
    def put(
        self,
        query: str,
        value: Any,
        *,
        ttl: float | None = None,
        tool: str | None = None,
        key: str | None = None,
    ) -> str:
        """Cache a result for *query*. Returns the storage key.

        Args:
            query: The query the result answers.
            value: The result to cache.
            ttl: Seconds the entry stays valid, defaulting to the cache's own.
            tool: The tool the result came from, which scopes the entry.
            key: The storage key to use instead of one derived from *query*.
        """
        if ttl is None:
            if tool is not None and tool in self._tool_ttl:
                ttl = self._tool_ttl[tool]
            else:
                ttl = self.default_ttl
        cache_key = key or self.make_key(query, tool or "")
        embedding: list[float] | None = None
        if self._embed_fn is not None:
            try:
                embedding = list(self._embed_fn(query))
            except Exception:
                embedding = None
        entry = ResultCacheEntry(
            key=cache_key,
            query=query,
            value=value,
            ttl=ttl,
            embedding=embedding,
            tool=tool,
        )
        with self._lock:
            self._entries[cache_key] = entry
            self._entries.move_to_end(cache_key)
            self._evict_if_needed()
        return cache_key

    def get(
        self,
        query: str,
        *,
        tool: str | None = None,
        key: str | None = None,
    ) -> Any | None:
        """Return the cached value for *query*, or ``None``.

        First tries an exact hash match; if ``embed_fn`` is configured, falls
        back to semantic-similarity lookup.

        Args:
            query: The query to look up.
            tool: The tool the result came from, which scopes the lookup.
            key: The storage key to read instead of one derived from *query*.
        """
        cache_key = key or self.make_key(query, tool or "")
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                if entry.is_expired():
                    self._entries.pop(cache_key, None)
                else:
                    entry.hits += 1
                    self._entries.move_to_end(cache_key)
                    self._hits += 1
                    return entry.value
        # Semantic fallback
        if self._embed_fn is not None:
            similar = self.get_similar(query, tool=tool)
            if similar is not None:
                return similar
        with self._lock:
            self._misses += 1
        return None

    def get_similar(
        self,
        query: str,
        *,
        tool: str | None = None,
        threshold: float | None = None,
    ) -> Any | None:
        """Look up by semantic similarity. Requires ``embed_fn``.

        Args:
            query: The query to look up.
            tool: The tool the result came from, which scopes the lookup.
            threshold: Least similarity a hit must reach, defaulting to the
                cache's own.

        Returns:
            The closest cached value above the threshold, or ``None``.
        """
        if self._embed_fn is None:
            return None
        try:
            query_vec = list(self._embed_fn(query))
        except Exception:
            return None
        threshold = threshold if threshold is not None else self.similarity_threshold
        best_key: str | None = None
        best_score = 0.0
        with self._lock:
            for key, entry in self._entries.items():
                if entry.is_expired():
                    continue
                if tool is not None and entry.tool != tool:
                    continue
                if not entry.embedding:
                    continue
                score = _cosine(query_vec, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_key = key
            if best_key is not None and best_score >= threshold:
                entry = self._entries[best_key]
                entry.hits += 1
                self._entries.move_to_end(best_key)
                self._semantic_hits += 1
                return entry.value
        return None

    def invalidate(self, query: str, *, tool: str | None = None) -> bool:
        """Drop the exact-match entry for *query*; returns whether one was removed."""
        cache_key = self.make_key(query, tool or "")
        with self._lock:
            return self._entries.pop(cache_key, None) is not None

    def invalidate_tool(self, tool: str) -> int:
        """Drop all entries belonging to *tool*. Returns count removed."""
        with self._lock:
            keys = [k for k, e in self._entries.items() if e.tool == tool]
            for k in keys:
                self._entries.pop(k, None)
            return len(keys)

    def clear(self) -> None:
        """Remove every entry and reset the hit/miss counters."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._semantic_hits = 0
            self._misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        """Return size, capacity and exact/semantic hit counters with the hit rate."""
        with self._lock:
            total = self._hits + self._semantic_hits + self._misses
            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "hits": self._hits,
                "semantic_hits": self._semantic_hits,
                "misses": self._misses,
                "hit_rate": ((self._hits + self._semantic_hits) / total) if total else 0.0,
            }

    # ------------------------------------------------------------------ internals
    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)
