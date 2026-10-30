"""Prompt prefix caching.

A lightweight in-process cache for system prompts and other long, repeated
prefixes. The cache is hash-keyed, supports TTL and a maximum entry count,
and stores arbitrary "payload" objects — typically a tokenized prompt or
a backend-specific KV-cache handle.

Backends that support prefix reuse (vLLM prefix caching, Transformers
``past_key_values``) can use :meth:`PromptCache.fingerprint` to derive a
stable key for a prompt and stash their native cache object as the value.
This module never holds GPU tensors itself; it only stores whatever the
caller hands it.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptCacheEntry:
    """A single cached prompt prefix."""

    key: str
    payload: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hits: int = 0
    ttl: float | None = None

    def is_expired(self, now: float | None = None) -> bool:
        """True when the entry has outlived its TTL (a ``None`` TTL never expires)."""
        if self.ttl is None:
            return False
        now = now if now is not None else time.time()
        return (now - self.created_at) > self.ttl


class PromptCache:
    """LRU cache for prompt prefixes / KV-cache hints.

    Parameters
    ----------
    max_size:
        Maximum number of entries kept. Oldest (LRU) entries are evicted.
    default_ttl:
        Default time-to-live in seconds. ``None`` disables expiry.
    """

    def __init__(self, max_size: int = 128, default_ttl: float | None = None) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._entries: OrderedDict[str, PromptCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ keys
    @staticmethod
    def fingerprint(prompt: str | bytes, *extra: Any) -> str:
        """Return a stable hash key for *prompt* and any *extra* discriminators.

        Extras (model name, tool list, etc.) are coerced to ``str`` so that two
        identical prompts under different models do not collide.
        """
        h = hashlib.sha256()
        if isinstance(prompt, str):
            h.update(prompt.encode("utf-8"))
        else:
            h.update(prompt)
        for item in extra:
            h.update(b"\x1f")
            h.update(str(item).encode("utf-8"))
        return h.hexdigest()

    # ------------------------------------------------------------------ ops
    def put(self, key: str, payload: Any, ttl: float | None = None) -> None:
        """Insert or refresh an entry.

        Args:
            key: The cache key to store under.
            payload: The value to store.
            ttl: Seconds the entry stays valid, defaulting to the cache's own.
        """
        ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            entry = PromptCacheEntry(key=key, payload=payload, ttl=ttl)
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._evict_if_needed()

    def get(self, key: str) -> Any | None:
        """Return cached payload, or ``None`` on miss/expiry."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                self._entries.pop(key, None)
                self._misses += 1
                return None
            entry.last_used = time.time()
            entry.hits += 1
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.payload

    def contains(self, key: str) -> bool:
        """True when *key* is cached and not expired."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.is_expired():
                return False
            return True

    def invalidate(self, key: str) -> bool:
        """Drop *key* from the cache; returns whether an entry was removed."""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Remove every entry and reset the hit/miss counters."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        """Return size, capacity and hit/miss counters with the hit rate."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
            }

    # ------------------------------------------------------------------ internals
    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)
