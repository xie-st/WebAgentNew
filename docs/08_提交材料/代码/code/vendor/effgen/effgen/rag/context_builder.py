"""
Context builder — assemble retrieved chunks into an LLM-ready context.

Features:
- Token budget management (approximates tokens as chars / 4 by default).
- Source deduplication: keep only the top-ranked chunk per source by default,
  or allow multiple per source with a cap.
- Relevance-first ordering (default) vs. chronological ordering.
- Inline numeric citation markers `[1]`, `[2]`, ... matching a
  returned citation list.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from effgen.rag.attribution import Citation
from effgen.rag.search import SearchResult


@dataclass
class BuiltContext:
    """Result of ContextBuilder.build()."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    used_chunks: list[SearchResult] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False


def _default_token_counter(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


class ContextBuilder:
    """
    Build an LLM-ready context block from a list of SearchResults.

    Args:
        max_tokens: maximum tokens for the full context block.
        token_counter: function(text) -> int. Default: chars / 4.
        per_source_limit: max chunks per source (0 = unlimited).
        order: "relevance" or "chronological".
        include_citations: if True, prepends `[N]` citation markers and
            returns a parallel Citation list.
        separator: string used to join chunks.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        token_counter: Callable[[str], int] | None = None,
        per_source_limit: int = 1,
        order: str = "relevance",
        include_citations: bool = True,
        separator: str = "\n\n---\n\n",
    ) -> None:
        self.max_tokens = max_tokens
        self.token_counter = token_counter or _default_token_counter
        self.per_source_limit = per_source_limit
        self.order = order
        self.include_citations = include_citations
        self.separator = separator

    def build(self, results: list[SearchResult]) -> BuiltContext:
        """Assemble search results into a token-budgeted context with citations."""
        if not results:
            return BuiltContext(text="", citations=[], used_chunks=[], total_tokens=0)

        # Dedupe by source if requested
        selected: list[SearchResult] = []
        source_counts: dict[str, int] = {}
        for r in results:
            if self.per_source_limit > 0:
                if source_counts.get(r.source, 0) >= self.per_source_limit:
                    continue
            selected.append(r)
            source_counts[r.source] = source_counts.get(r.source, 0) + 1

        # Budget selection (greedy by relevance)
        budget = self.max_tokens
        used: list[SearchResult] = []
        truncated = False
        for r in selected:
            tokens = self.token_counter(r.content)
            if tokens > budget:
                truncated = True
                continue
            used.append(r)
            budget -= tokens

        if not used and selected:
            # At least include a truncated version of the top result
            top = selected[0]
            approx_chars = self.max_tokens * 4
            truncated_content = top.content[:approx_chars]
            truncated_result = SearchResult(
                chunk_id=top.chunk_id,
                content=truncated_content,
                source=top.source,
                metadata=top.metadata,
                relevance_score=top.relevance_score,
                rank=top.rank,
            )
            used = [truncated_result]
            truncated = True

        if self.order == "chronological":
            def _ts(r: SearchResult) -> float:
                t = r.metadata.get("timestamp") or r.metadata.get("date") or 0
                try:
                    return float(t)
                except (TypeError, ValueError):
                    return 0.0
            used = sorted(used, key=_ts)

        # Build text + citations
        parts: list[str] = []
        citations: list[Citation] = []
        for i, r in enumerate(used, start=1):
            if self.include_citations:
                citation = Citation(
                    index=i,
                    source=r.source,
                    chunk_id=r.chunk_id,
                    relevance_score=r.relevance_score,
                    quote=r.content[:200],
                    page=r.metadata.get("page"),
                    section=r.metadata.get("breadcrumb") or r.metadata.get("section"),
                )
                citations.append(citation)
                parts.append(f"[{i}] Source: {r.source}\n{r.content}")
            else:
                parts.append(r.content)

        text = self.separator.join(parts)
        return BuiltContext(
            text=text,
            citations=citations,
            used_chunks=used,
            total_tokens=self.token_counter(text),
            truncated=truncated,
        )
