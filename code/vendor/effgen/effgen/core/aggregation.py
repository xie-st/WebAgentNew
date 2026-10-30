"""
Result Aggregation & Deduplication for effGen batch execution.

Combines, deduplicates, ranks, and merges results from batch/parallel runs.
Also provides a cross-query tool-result cache for knowledge sharing.

Usage:
    from effgen.core.aggregation import ResultAggregator

    aggregator = ResultAggregator()
    unique = aggregator.deduplicate(results)
    ranked = aggregator.rank(results, key="confidence")
    merged = aggregator.merge(results, strategy="best")
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategies for merging multiple results."""

    FIRST = "first"  # Keep the first result
    BEST = "best"  # Keep the best by score
    CONSENSUS = "consensus"  # Combine outputs that agree
    UNION = "union"  # Combine all unique outputs


@dataclass
class AggregatedResult:
    """A single aggregated result entry.

    Attributes:
        query: Original query string.
        output: Aggregated output text.
        score: Score used for ranking (higher = better).
        sources: Indices of original results that contributed.
        metadata: Extra metadata.
    """

    query: str
    output: str
    score: float = 0.0
    sources: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolResultCache:
    """Thread-safe cache for tool results shared across queries.

    Allows one query's tool output to be reused by another query
    that would invoke the same tool with the same (or similar) input.
    """

    def __init__(self, max_size: int = 1000, ttl: float = 300.0) -> None:
        """
        Args:
            max_size: Maximum number of cached entries.
            ttl: Time-to-live for each entry in seconds.
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self.max_size = max_size
        self.ttl = ttl
        self._hits = 0
        self._misses = 0

    def key(self, tool_name: str, tool_input: str) -> str:
        """Compute a deterministic cache key."""
        raw = f"{tool_name}::{tool_input}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, tool_name: str, tool_input: str) -> Any | None:
        """Return cached result or None."""
        k = self.key(tool_name, tool_input)
        with self._lock:
            entry = self._cache.get(k)
            if entry is None:
                self._misses += 1
                return None
            value, ts = entry
            if time.time() - ts > self.ttl:
                del self._cache[k]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def put(self, tool_name: str, tool_input: str, result: Any) -> None:
        """Store a tool result in the cache.

        Args:
            tool_name: The tool the result came from.
            tool_input: The input the result was produced for.
            result: The value to cache.
        """
        k = self.key(tool_name, tool_input)
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Evict oldest entry
                oldest_key = min(self._cache, key=lambda x: self._cache[x][1])
                del self._cache[oldest_key]
            self._cache[k] = (result, time.time())

    def clear(self) -> None:
        """Remove every cached tool result."""
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> dict[str, int]:
        """Hit/miss counters and the current cache size."""
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


class ResultAggregator:
    """Aggregate, deduplicate, rank, and merge batch results.

    Works with AgentResponse objects or any object that has
    ``.output`` (str) and ``.success`` (bool) attributes.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.85,
        tool_cache: ToolResultCache | None = None,
    ) -> None:
        """
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy deduplication
                (0.0–1.0). Pairs above this are considered duplicates.
            tool_cache: Optional shared ToolResultCache instance.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.tool_cache = tool_cache or ToolResultCache()

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        results: list[Any],
        queries: list[str] | None = None,
        fuzzy: bool = False,
    ) -> list[AggregatedResult]:
        """Remove duplicate results.

        Args:
            results: List of AgentResponse-like objects.
            queries: Optional parallel list of query strings.
            fuzzy: If True, use fuzzy similarity; otherwise exact hash.

        Returns:
            Deduplicated list of AggregatedResult.
        """
        seen_hashes: dict[str, int] = {}
        aggregated: list[AggregatedResult] = []

        for i, resp in enumerate(results):
            output = getattr(resp, "output", "") or ""
            query = queries[i] if queries and i < len(queries) else ""
            h = hashlib.md5(output.encode()).hexdigest()

            if fuzzy:
                is_dup = False
                for ar in aggregated:
                    if self._similarity(output, ar.output) >= self.fuzzy_threshold:
                        ar.sources.append(i)
                        is_dup = True
                        break
                if is_dup:
                    continue
            else:
                if h in seen_hashes:
                    aggregated[seen_hashes[h]].sources.append(i)
                    continue
                seen_hashes[h] = len(aggregated)

            score = 1.0 if getattr(resp, "success", False) else 0.0
            aggregated.append(
                AggregatedResult(
                    query=query,
                    output=output,
                    score=score,
                    sources=[i],
                )
            )

        logger.debug(
            "Deduplication: %d results -> %d unique (%s)",
            len(results),
            len(aggregated),
            "fuzzy" if fuzzy else "exact",
        )
        return aggregated

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(
        self,
        results: list[Any],
        key: str = "confidence",
        custom_scorer: Callable[[Any], float] | None = None,
        reverse: bool = True,
    ) -> list[Any]:
        """Sort results by a scoring criterion.

        Args:
            results: List of AgentResponse-like objects.
            key: Built-in ranking key — "confidence" (success + speed),
                "relevance" (output length heuristic), or "speed".
            custom_scorer: A callable that takes a result and returns a float.
            reverse: If True, highest score first.

        Returns:
            Sorted list (new list, original not mutated).
        """
        if custom_scorer:
            scorer = custom_scorer
        elif key == "confidence":
            def scorer(r: Any) -> float:
                s = 1.0 if getattr(r, "success", False) else 0.0
                # Faster is better — subtract normalised time
                t = getattr(r, "execution_time", 0.0)
                return s - min(t / 120.0, 0.5)
        elif key == "relevance":
            def scorer(r: Any) -> float:
                out = getattr(r, "output", "") or ""
                # Longer, successful outputs are heuristically more relevant
                s = 1.0 if getattr(r, "success", False) else 0.0
                return s + min(len(out) / 1000.0, 0.5)
        elif key == "speed":
            def scorer(r: Any) -> float:
                return -getattr(r, "execution_time", 0.0)
        else:
            raise ValueError(f"Unknown ranking key: {key}")

        return sorted(results, key=scorer, reverse=reverse)

    # ------------------------------------------------------------------
    # Merge strategies
    # ------------------------------------------------------------------

    def merge(
        self,
        results: list[Any],
        strategy: str | MergeStrategy = MergeStrategy.BEST,
        queries: list[str] | None = None,
    ) -> AggregatedResult:
        """Merge multiple results into one using the given strategy.

        Args:
            results: List of AgentResponse-like objects.
            strategy: One of "first", "best", "consensus", "union".
            queries: Optional parallel list of query strings.

        Returns:
            A single AggregatedResult.
        """
        if isinstance(strategy, str):
            strategy = MergeStrategy(strategy)

        if not results:
            return AggregatedResult(query="", output="")

        outputs = [getattr(r, "output", "") or "" for r in results]
        query = queries[0] if queries else ""

        if strategy == MergeStrategy.FIRST:
            return AggregatedResult(
                query=query, output=outputs[0], score=1.0, sources=[0],
            )

        elif strategy == MergeStrategy.BEST:
            ranked = self.rank(results)
            best_idx = results.index(ranked[0])
            return AggregatedResult(
                query=query,
                output=getattr(ranked[0], "output", ""),
                score=1.0,
                sources=[best_idx],
            )

        elif strategy == MergeStrategy.CONSENSUS:
            # Group by exact output, pick the most common
            from collections import Counter
            counts = Counter(outputs)
            most_common_output, count = counts.most_common(1)[0]
            sources = [i for i, o in enumerate(outputs) if o == most_common_output]
            return AggregatedResult(
                query=query,
                output=most_common_output,
                score=count / len(results),
                sources=sources,
            )

        elif strategy == MergeStrategy.UNION:
            # Combine all unique outputs with separator
            unique_outputs: list[str] = []
            seen: set[str] = set()
            sources = []
            for i, o in enumerate(outputs):
                h = hashlib.md5(o.encode()).hexdigest()
                if h not in seen and o.strip():
                    seen.add(h)
                    unique_outputs.append(o)
                    sources.append(i)
            return AggregatedResult(
                query=query,
                output="\n---\n".join(unique_outputs),
                score=1.0,
                sources=sources,
            )

        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Simple character-level Jaccard similarity."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0
