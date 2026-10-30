"""
effgen.rag — Advanced Retrieval-Augmented Generation pipeline.

Provides a RAG pipeline built on top of the existing
`effgen.tools.builtin.retrieval` primitives:

- DocumentIngester: load documents from many formats with metadata + dedup.
- Advanced chunkers: Semantic, Code, Table, Hierarchical.
- HybridSearchEngine: dense + sparse + keyword + metadata filtering with RRF.
- Rerankers: CrossEncoder (optional), LLM-based, rule-based.
- ContextBuilder: token-budgeted context assembly with source dedup.
- Citation / attribution tracking.

All external document-loading libraries are OPTIONAL. Core pipeline works
with just TXT, MD, JSON, JSONL, CSV, and HTML (stdlib html.parser).
"""

from __future__ import annotations

from effgen.rag.attribution import Citation, CitationTracker
from effgen.rag.chunking import (
    CodeChunker,
    HierarchicalChunker,
    SemanticChunker,
    TableChunker,
    TextChunker,
)
from effgen.rag.context_builder import ContextBuilder
from effgen.rag.ingest import DocumentIngester, IngestedChunk, IngestSummary
from effgen.rag.reranker import (
    CrossEncoderReranker,
    LLMReranker,
    RuleBasedReranker,
)
from effgen.rag.search import HybridSearchEngine, SearchResult

__all__ = [
    "DocumentIngester",
    "IngestedChunk",
    "IngestSummary",
    "TextChunker",
    "SemanticChunker",
    "CodeChunker",
    "TableChunker",
    "HierarchicalChunker",
    "HybridSearchEngine",
    "SearchResult",
    "CrossEncoderReranker",
    "LLMReranker",
    "RuleBasedReranker",
    "ContextBuilder",
    "Citation",
    "CitationTracker",
]
