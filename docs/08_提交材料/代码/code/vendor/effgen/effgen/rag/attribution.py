"""
Source attribution and citation tracking for RAG answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    """
    A single citation pointing to a chunk that contributed to an answer.

    Attributes:
        index: 1-based citation number (for inline `[1]` markers).
        source: source identifier (file path, URL, etc.).
        chunk_id: unique chunk ID in the index.
        relevance_score: relevance score of the chunk. A web-search-derived
            citation has no reranker score to fill this from, so it carries
            the dataclass default of ``0.0`` — read that as "unscored", not
            as "scored zero relevance".
        quote: short quote from the chunk (up to ~200 chars).
        page: optional page number (PDF).
        section: optional section / breadcrumb.
    """

    index: int
    source: str
    chunk_id: str = ""
    relevance_score: float = 0.0
    quote: str = ""
    page: int | None = None
    section: str | None = None

    def format(self, style: str = "numeric") -> str:
        """Format the citation as a string."""
        if style == "numeric":
            return f"[{self.index}]"
        if style == "inline":
            parts = [self.source]
            if self.page is not None:
                parts.append(f"p.{self.page}")
            if self.section:
                parts.append(self.section)
            return f"[Source: {', '.join(parts)}]"
        if style == "full":
            return (
                f"[{self.index}] {self.source}"
                + (f" (p.{self.page})" if self.page else "")
                + (f" — {self.section}" if self.section else "")
            )
        return f"[{self.index}]"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "index": self.index,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "relevance_score": round(self.relevance_score, 4),
            "quote": self.quote,
            "page": self.page,
            "section": self.section,
        }


@dataclass
class CitationTracker:
    """
    Tracks citations used in an answer and provides verification helpers.
    """

    citations: list[Citation] = field(default_factory=list)

    def add(self, citation: Citation) -> None:
        """Record a citation."""
        self.citations.append(citation)

    def sources(self) -> list[str]:
        """Return the deduplicated list of sources."""
        seen: set[str] = set()
        out: list[str] = []
        for c in self.citations:
            if c.source not in seen:
                seen.add(c.source)
                out.append(c.source)
        return out

    def extract_used_indices(self, answer: str) -> list[int]:
        """Parse `[N]` markers out of an answer text."""
        return sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})

    def filter_used(self, answer: str) -> list[Citation]:
        """Return only the citations actually referenced in the answer."""
        used = set(self.extract_used_indices(answer))
        if not used:
            return list(self.citations)
        return [c for c in self.citations if c.index in used]

    def verify(self, claim: str, min_overlap: float = 0.3) -> list[Citation]:
        """Return citations whose ``quote`` shares token overlap with *claim* (heuristic).

        A citation counts as supporting when at least *min_overlap* of the
        claim's tokens appear in its quote.
        """
        claim_tokens = set(re.findall(r"\b\w+\b", claim.lower()))
        if not claim_tokens:
            return []
        supporting: list[Citation] = []
        for c in self.citations:
            q_tokens = set(re.findall(r"\b\w+\b", c.quote.lower()))
            if not q_tokens:
                continue
            overlap = len(claim_tokens & q_tokens) / len(claim_tokens)
            if overlap >= min_overlap:
                supporting.append(c)
        return supporting

    def to_list(self) -> list[dict[str, Any]]:
        """Return every citation as a JSON-serializable dict."""
        return [c.to_dict() for c in self.citations]
