"""
Rerankers for RAG search results.

- CrossEncoderReranker: sentence-transformers cross-encoder (optional).
- LLMReranker: uses the agent's own model to score each result (free default).
- RuleBasedReranker: boosts by recency, source authority, keyword presence.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from effgen.rag.search import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Base reranker interface."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Reorder *results* for *query*, optionally truncating to *top_k*.

        Args:
            query: The query the results were retrieved for.
            results: The results to reorder.
            top_k: Keep only this many results after reordering.

        Returns:
            The results in their new order.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement rerank(). Use one of the "
            "concrete rerankers (CrossEncoderReranker, LLMReranker, "
            "RuleBasedReranker) or subclass Reranker and override rerank()."
        )


# ---------------------------------------------------------------------------
# Cross-encoder (optional)
# ---------------------------------------------------------------------------

class CrossEncoderReranker(Reranker):
    """
    Rerank with a sentence-transformers CrossEncoder.

    Requires `sentence-transformers`. If unavailable, acts as a pass-through.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._available: bool | None = None

    def _load(self):
        if self._available is False:
            return None
        if self._model is None:
            try:
                from effgen.utils.embedding_backend import import_sentence_transformers

                (cross_encoder,) = import_sentence_transformers("CrossEncoder")
                self._model = cross_encoder(self.model_name)
                self._available = True
            except ImportError as exc:
                logger.warning("%s CrossEncoderReranker is a pass-through.", exc)
                self._available = False
                return None
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rescore with the cross-encoder and reorder (pass-through when absent).

        Args:
            query: The query the results were retrieved for.
            results: The results to reorder.
            top_k: Keep only this many results after reordering.

        Returns:
            The results in their new order.
        """
        if not results:
            return results
        model = self._load()
        if model is None:
            return results[:top_k] if top_k else results

        pairs = [(query, r.content) for r in results]
        scores = model.predict(pairs)
        for r, s in zip(results, scores):
            r.relevance_score = float(s)
        ranked = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked[:top_k] if top_k else ranked


# ---------------------------------------------------------------------------
# LLM reranker (free default)
# ---------------------------------------------------------------------------

class LLMReranker(Reranker):
    """
    Rerank search results using a language model.

    The model is asked to rate the relevance of each result on a 0-10
    scale. Works with any effgen BaseModel that supports `generate`.
    """

    _PROMPT = (
        "You are a search relevance judge. On a scale of 0-10, rate how "
        "relevant the following passage is to the query. Respond with ONLY "
        "a single integer 0-10, nothing else.\n\n"
        "Query: {query}\n\n"
        "Passage: {passage}\n\n"
        "Rating (0-10):"
    )

    def __init__(self, model: Any, max_passage_chars: int = 1000) -> None:
        self.model = model
        self.max_passage_chars = max_passage_chars

    def _generate(self, prompt: str) -> str:
        """Call the model using whichever generate signature it supports."""
        # 1) Try effgen BaseModel-style: generate(prompt, config=GenerationConfig)
        try:
            from effgen.models.base import GenerationConfig  # type: ignore

            cfg = GenerationConfig(max_tokens=8, temperature=0.01)
            out = self.model.generate(prompt, config=cfg)
        except Exception:
            # 2) Try kwargs-style: generate(prompt, max_tokens=8, temperature=0.01)
            try:
                out = self.model.generate(prompt, max_tokens=8, temperature=0.01)
            except Exception:
                # 3) Last resort: positional only
                out = self.model.generate(prompt)

        if isinstance(out, str):
            return out
        for attr in ("text", "content", "output", "generated_text"):
            v = getattr(out, attr, None)
            if isinstance(v, str):
                return v
        return str(out)

    def _score_one(self, query: str, passage: str) -> float:
        prompt = self._PROMPT.format(
            query=query,
            passage=passage[: self.max_passage_chars],
        )
        try:
            text = self._generate(prompt)
            m = re.search(r"\d+", text)
            if m:
                score = float(m.group(0))
                return max(0.0, min(10.0, score)) / 10.0
        except Exception as e:
            logger.debug("LLMReranker scoring failed: %s", e)
        return 0.0

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Score each result with the judge model and reorder by that score.

        Args:
            query: The query the results were retrieved for.
            results: The results to reorder.
            top_k: Keep only this many results after reordering.

        Returns:
            The results in their new order.
        """
        if not results:
            return results
        for r in results:
            r.relevance_score = self._score_one(query, r.content)
        ranked = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked[:top_k] if top_k else ranked


# ---------------------------------------------------------------------------
# Rule-based reranker
# ---------------------------------------------------------------------------

@dataclass
class RuleBasedReranker(Reranker):
    """
    Rule-based reranker that boosts results by:

    - Recency (metadata["date"] or metadata["timestamp"]), newer => higher.
    - Source authority (explicit map source -> multiplier).
    - Query keyword presence (literal match in content).
    - Title matches.

    Each boost is additive on top of the base relevance score.
    """

    recency_weight: float = 0.2
    keyword_weight: float = 0.15
    authority_weight: float = 0.25
    title_weight: float = 0.1
    authority_map: dict[str, float] = field(default_factory=dict)
    now_ts: float | None = None

    def _recency_boost(self, meta: dict[str, Any]) -> float:
        ts = meta.get("timestamp") or meta.get("date")
        if ts is None:
            return 0.0
        now = self.now_ts if self.now_ts is not None else time.time()
        try:
            if isinstance(ts, int | float):
                t = float(ts)
            else:
                # Try ISO 8601
                from datetime import datetime

                t = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0
        age_days = max(0.0, (now - t) / 86400.0)
        # Exponential decay with 30-day half-life
        import math

        return math.exp(-age_days / 30.0)

    def _authority_boost(self, source: str) -> float:
        for key, mult in self.authority_map.items():
            if key in source:
                return mult
        return 0.0

    def _keyword_boost(self, query: str, content: str) -> float:
        q_tokens = [t for t in re.findall(r"\b\w+\b", query.lower()) if len(t) > 2]
        if not q_tokens:
            return 0.0
        c_lower = content.lower()
        hits = sum(1 for t in q_tokens if t in c_lower)
        return hits / len(q_tokens)

    def _title_boost(self, query: str, meta: dict[str, Any]) -> float:
        title = meta.get("title") or meta.get("breadcrumb") or ""
        if not title:
            return 0.0
        q_tokens = [t for t in re.findall(r"\b\w+\b", query.lower()) if len(t) > 2]
        if not q_tokens:
            return 0.0
        t_lower = str(title).lower()
        hits = sum(1 for t in q_tokens if t in t_lower)
        return hits / len(q_tokens)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Add recency/authority/keyword/title boosts to each score and reorder.

        Args:
            query: The query the results were retrieved for.
            results: The results to reorder.
            top_k: Keep only this many results after reordering.

        Returns:
            The results in their new order.
        """
        for r in results:
            boost = (
                self.recency_weight * self._recency_boost(r.metadata)
                + self.authority_weight * self._authority_boost(r.source)
                + self.keyword_weight * self._keyword_boost(query, r.content)
                + self.title_weight * self._title_boost(query, r.metadata)
            )
            r.relevance_score = r.relevance_score + boost

        ranked = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked[:top_k] if top_k else ranked
