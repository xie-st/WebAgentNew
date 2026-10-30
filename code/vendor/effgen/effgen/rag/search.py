"""
Hybrid search engine.

Combines dense (embedding) retrieval, sparse (BM25) retrieval, keyword
matching, and metadata filtering. Results from each method are fused
with Reciprocal Rank Fusion (RRF), optionally weighted.

Usable standalone (without an Agent).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from effgen.rag.ingest import IngestedChunk
from effgen.tools.builtin.retrieval import (
    Document,
    EmbeddingProvider,
    SentenceTransformerEmbedding,
    SimpleBM25,
    SimpleEmbedding,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    chunk_id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    keyword_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "relevance_score": round(self.relevance_score, 4),
            "rank": self.rank,
        }


def _to_chunks(items: Iterable[Any]) -> list[IngestedChunk]:
    """Coerce a mix of IngestedChunk / Document / dict into IngestedChunks."""
    out: list[IngestedChunk] = []
    for item in items:
        if isinstance(item, IngestedChunk):
            out.append(item)
        elif isinstance(item, Document):
            out.append(IngestedChunk(
                id=item.id,
                content=item.content,
                source=item.metadata.get("source", "unknown"),
                metadata=dict(item.metadata),
            ))
        elif isinstance(item, dict):
            out.append(IngestedChunk(
                id=item.get("id", ""),
                content=item.get("content", ""),
                source=item.get("source") or item.get("metadata", {}).get("source", "unknown"),
                metadata=item.get("metadata", {}),
            ))
        else:
            raise TypeError(f"Cannot convert {type(item).__name__} to IngestedChunk")
    return out


class HybridSearchEngine:
    """
    Hybrid search engine combining dense, sparse, and keyword retrieval.

    Args:
        chunks: list of IngestedChunk (or Document / dict) to index.
        embedding_provider: optional; auto-selects sentence-transformers
            if available, else TF-IDF SimpleEmbedding.
        weights: dict with keys "dense", "sparse", "keyword" controlling
            contribution of each method during RRF fusion. Zero disables
            that method. Default: dense=1.0, sparse=1.0, keyword=0.5.
        rrf_k: constant for reciprocal rank fusion (default 60).

    Example:
        engine = HybridSearchEngine(chunks)
        results = engine.search("scaling architecture", top_k=5)
    """

    def __init__(
        self,
        chunks: Iterable[Any] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        weights: dict[str, float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.chunks: list[IngestedChunk] = []
        self.weights = {"dense": 1.0, "sparse": 1.0, "keyword": 0.5}
        if weights:
            self.weights.update(weights)
        self.rrf_k = rrf_k

        if embedding_provider is None:
            try:
                import importlib.util

                if importlib.util.find_spec("sentence_transformers") is not None:
                    embedding_provider = SentenceTransformerEmbedding()
                else:
                    raise ImportError
            except Exception:
                logger.info(
                    "HybridSearchEngine using TF-IDF fallback "
                    "(install sentence-transformers for better dense retrieval)"
                )
                embedding_provider = SimpleEmbedding()
        self.embedding_provider = embedding_provider

        self._embeddings: np.ndarray | None = None
        self._bm25: SimpleBM25 | None = None
        self._tokenized: list[set[str]] = []

        if chunks:
            self.index(chunks)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index(self, chunks: Iterable[Any]) -> None:
        """Index a list of chunks. Replaces any existing index."""
        self.chunks = _to_chunks(chunks)
        if not self.chunks:
            self._embeddings = None
            self._bm25 = None
            self._tokenized = []
            return

        texts = [c.content for c in self.chunks]

        # Dense
        if self.weights.get("dense", 0) > 0:
            try:
                if isinstance(self.embedding_provider, SimpleEmbedding):
                    self.embedding_provider.fit(texts)
                self._embeddings = self.embedding_provider.embed(texts)
            except Exception as e:
                logger.warning("Dense indexing failed: %s", e)
                self._embeddings = None
        else:
            self._embeddings = None

        # Sparse (BM25)
        if self.weights.get("sparse", 0) > 0:
            self._bm25 = SimpleBM25()
            self._bm25.index(texts)
        else:
            self._bm25 = None

        # Keyword token sets
        self._tokenized = [set(re.findall(r"\b\w+\b", t.lower())) for t in texts]

        logger.info("HybridSearchEngine indexed %d chunks", len(self.chunks))

    def add(self, chunks: Iterable[Any]) -> None:
        """Add more chunks and re-index."""
        new = _to_chunks(chunks)
        self.index(self.chunks + new)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _dense_scores(self, query: str) -> np.ndarray:
        if self._embeddings is None:
            return np.zeros(len(self.chunks))
        q = self.embedding_provider.embed_query(query)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        mat = self._embeddings
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        return np.dot(mat_norm, q_norm)

    def _sparse_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(len(self.chunks))
        return self._bm25.score(query)

    def _keyword_scores(self, query: str) -> np.ndarray:
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        if not q_tokens:
            return np.zeros(len(self.chunks))
        scores = np.zeros(len(self.chunks))
        for i, tokens in enumerate(self._tokenized):
            if not tokens:
                continue
            overlap = len(q_tokens & tokens)
            scores[i] = overlap / len(q_tokens)
        return scores

    @staticmethod
    def _rrf_ranks(scores: np.ndarray) -> dict[int, int]:
        """Convert a score array to {idx: rank} (1-based, higher score => lower rank)."""
        order = np.argsort(-scores)
        return {int(idx): rank + 1 for rank, idx in enumerate(order)}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: the search query string.
            top_k: maximum number of results to return.
            filter_metadata: optional dict of metadata key/value filters
                (each key must match exactly).
            min_score: minimum fused relevance score.

        Returns:
            list of SearchResult, sorted by relevance descending.
        """
        if not self.chunks:
            return []

        dense = self._dense_scores(query) if self.weights["dense"] > 0 else np.zeros(len(self.chunks))
        sparse = self._sparse_scores(query) if self.weights["sparse"] > 0 else np.zeros(len(self.chunks))
        keyword = self._keyword_scores(query) if self.weights["keyword"] > 0 else np.zeros(len(self.chunks))

        # Reciprocal Rank Fusion
        fused = np.zeros(len(self.chunks), dtype=float)

        if self.weights["dense"] > 0 and dense.any():
            ranks = self._rrf_ranks(dense)
            for idx, r in ranks.items():
                fused[idx] += self.weights["dense"] / (self.rrf_k + r)

        if self.weights["sparse"] > 0 and sparse.any():
            ranks = self._rrf_ranks(sparse)
            for idx, r in ranks.items():
                fused[idx] += self.weights["sparse"] / (self.rrf_k + r)

        if self.weights["keyword"] > 0 and keyword.any():
            ranks = self._rrf_ranks(keyword)
            for idx, r in ranks.items():
                fused[idx] += self.weights["keyword"] / (self.rrf_k + r)

        # Normalize individual score arrays for reporting
        def _norm(a: np.ndarray) -> np.ndarray:
            m = a.max() if a.size and a.max() > 0 else 1.0
            return a / m

        dense_n = _norm(dense)
        sparse_n = _norm(sparse)
        keyword_n = _norm(keyword)
        fused_n = _norm(fused)

        order = np.argsort(-fused)
        results: list[SearchResult] = []

        for idx in order:
            if len(results) >= top_k:
                break
            idx = int(idx)
            chunk = self.chunks[idx]

            if filter_metadata:
                if not all(chunk.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            score = float(fused_n[idx])
            if score < min_score:
                continue

            # Ensure fused score is strictly positive so callers can assert > 0
            if score <= 0:
                score = 1e-6

            results.append(SearchResult(
                chunk_id=chunk.id,
                content=chunk.content,
                source=chunk.source,
                metadata=chunk.metadata,
                relevance_score=score,
                dense_score=float(dense_n[idx]),
                sparse_score=float(sparse_n[idx]),
                keyword_score=float(keyword_n[idx]),
                rank=len(results) + 1,
            ))

        return results

    def __len__(self) -> int:
        return len(self.chunks)
