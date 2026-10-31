"""
Standard Retrieval (RAG) tool with embedding-based search.

This module provides a retrieval tool that uses embeddings to search
through a knowledge base and return relevant documents/chunks.

Features:
- Multiple embedding providers (Sentence Transformers, TF-IDF fallback)
- Document loaders: txt, md, pdf, csv, tsv, json, jsonl
- Chunking strategies: fixed-size, sentence-based, paragraph-based, recursive
- Hybrid search: vector similarity + BM25 keyword matching
- Persistent index storage
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document in the knowledge base."""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class RetrievalResult:
    """A retrieval result with score."""
    document: Document
    score: float
    rank: int


# ---------------------------------------------------------------------------
# Embedding Providers
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """Base class for embedding providers."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""
        raise NotImplementedError("Subclasses must implement embed()")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query text."""
        return self.embed([query])[0]


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Sentence Transformers embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from effgen.utils.embedding_backend import load_sentence_transformer

            try:
                self._model = load_sentence_transformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError as exc:
                raise RuntimeError(str(exc)) from exc
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one sentence-transformer vector per input text."""
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings


class SimpleEmbedding(EmbeddingProvider):
    """Simple TF-IDF based embedding for environments without sentence-transformers."""

    def __init__(self) -> None:
        self._vectorizer = None
        self._fitted = False

    def _get_vectorizer(self):
        if self._vectorizer is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._vectorizer = TfidfVectorizer(
                    max_features=512,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            except ImportError:
                raise RuntimeError(
                    "scikit-learn not installed. "
                    "Install with: pip install scikit-learn"
                )
        return self._vectorizer

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF vocabulary on *texts*."""
        vectorizer = self._get_vectorizer()
        vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one TF-IDF vector per input text (fits on first call)."""
        vectorizer = self._get_vectorizer()
        if not self._fitted:
            embeddings = vectorizer.fit_transform(texts).toarray()
            self._fitted = True
        else:
            embeddings = vectorizer.transform(texts).toarray()
        return embeddings


# ---------------------------------------------------------------------------
# Chunking Strategies
# ---------------------------------------------------------------------------

class ChunkingStrategy:
    """Base chunking strategy."""

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Split *text* into :class:`Document` chunks."""
        raise NotImplementedError("Subclasses must implement chunk()")

    def split_text(self, text: str) -> list[str]:
        """Return the chunk texts for ``text`` as plain strings.

        A convenience over :meth:`chunk` for callers that only want the pieces
        and not the :class:`Document` wrappers or an id. Available on every
        chunking strategy.
        """
        return [doc.content for doc in self.chunk(text)]


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunks with overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Split into fixed-size character windows with the configured overlap."""
        chunks = []
        start = 0
        chunk_num = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                for sep in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        chunk_text = chunk_text[:last_sep + len(sep)]
                        end = start + len(chunk_text)
                        break

            chunks.append(Document(
                id=f"{doc_id}_chunk_{chunk_num}",
                content=chunk_text.strip(),
                metadata={
                    "parent_doc_id": doc_id,
                    "chunk_index": chunk_num,
                    "start_char": start,
                    "end_char": end,
                    "chunking": "fixed_size",
                }
            ))

            chunk_num += 1
            start = end - self.overlap
            if start >= end:
                break

        return chunks


class SentenceChunker(ChunkingStrategy):
    """Sentence-based chunking: groups sentences up to max_chunk_size."""

    def __init__(self, max_chunk_size: int = 500, overlap_sentences: int = 1) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        return re.split(r'(?<=[.!?])\s+', text.strip())

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Group whole sentences into chunks up to the size limit."""
        sentences = self._split_sentences(text)
        if not sentences:
            return [Document(id=f"{doc_id}_chunk_0", content=text, metadata={"parent_doc_id": doc_id, "chunking": "sentence"})]

        chunks = []
        current_sents: list[str] = []
        current_len = 0
        chunk_num = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            if current_len + len(sent) > self.max_chunk_size and current_sents:
                chunk_text = " ".join(current_sents)
                chunks.append(Document(
                    id=f"{doc_id}_chunk_{chunk_num}",
                    content=chunk_text,
                    metadata={"parent_doc_id": doc_id, "chunk_index": chunk_num, "chunking": "sentence"},
                ))
                chunk_num += 1
                # Keep overlap
                overlap = current_sents[-self.overlap_sentences:] if self.overlap_sentences else []
                current_sents = overlap
                current_len = sum(len(s) for s in current_sents)

            current_sents.append(sent)
            current_len += len(sent)

        if current_sents:
            chunks.append(Document(
                id=f"{doc_id}_chunk_{chunk_num}",
                content=" ".join(current_sents),
                metadata={"parent_doc_id": doc_id, "chunk_index": chunk_num, "chunking": "sentence"},
            ))

        return chunks


class ParagraphChunker(ChunkingStrategy):
    """Paragraph-based chunking: splits on double newlines."""

    def __init__(self, max_chunk_size: int = 1000) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Group whole paragraphs into chunks up to the size limit."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return [Document(id=f"{doc_id}_chunk_0", content=text, metadata={"parent_doc_id": doc_id, "chunking": "paragraph"})]

        chunks = []
        current_paras: list[str] = []
        current_len = 0
        chunk_num = 0

        for para in paragraphs:
            if current_len + len(para) > self.max_chunk_size and current_paras:
                chunks.append(Document(
                    id=f"{doc_id}_chunk_{chunk_num}",
                    content="\n\n".join(current_paras),
                    metadata={"parent_doc_id": doc_id, "chunk_index": chunk_num, "chunking": "paragraph"},
                ))
                chunk_num += 1
                current_paras = []
                current_len = 0

            current_paras.append(para)
            current_len += len(para)

        if current_paras:
            chunks.append(Document(
                id=f"{doc_id}_chunk_{chunk_num}",
                content="\n\n".join(current_paras),
                metadata={"parent_doc_id": doc_id, "chunk_index": chunk_num, "chunking": "paragraph"},
            ))

        return chunks


class RecursiveChunker(ChunkingStrategy):
    """Recursive character text splitter: tries multiple separators in order."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100,
                 separators: list[str] | None = None) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        if not separators:
            return [text]

        sep = separators[0]
        rest = separators[1:]

        if not sep:
            # Character-level split
            parts = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]
            return parts

        splits = text.split(sep)
        result = []
        current = ""

        for piece in splits:
            candidate = (current + sep + piece) if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(piece) > self.chunk_size:
                    result.extend(self._split(piece, rest))
                    current = ""
                else:
                    current = piece

        if current:
            result.append(current)

        return result

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Split by trying each separator in order until pieces fit the limit."""
        pieces = self._split(text, self.separators)
        chunks = []
        for i, piece in enumerate(pieces):
            chunks.append(Document(
                id=f"{doc_id}_chunk_{i}",
                content=piece.strip(),
                metadata={"parent_doc_id": doc_id, "chunk_index": i, "chunking": "recursive"},
            ))
        return [c for c in chunks if c.content]


# ---------------------------------------------------------------------------
# Document Loaders
# ---------------------------------------------------------------------------

def load_txt(path: Path) -> list[dict[str, Any]]:
    """Load plain text file."""
    content = path.read_text(encoding="utf-8")
    return [{"content": content, "id": path.stem, "metadata": {"source": str(path), "type": "txt"}}]


def load_markdown(path: Path) -> list[dict[str, Any]]:
    """Load markdown file, preserving structure."""
    content = path.read_text(encoding="utf-8")
    return [{"content": content, "id": path.stem, "metadata": {"source": str(path), "type": "markdown"}}]


def load_csv(path: Path, text_columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Load CSV/TSV file, treating each row as a document."""
    delimiter = '\t' if path.suffix.lower() in ('.tsv',) else ','
    documents = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if text_columns:
                content = " ".join(str(row.get(col, "")) for col in text_columns if row.get(col))
            else:
                content = " ".join(str(v) for v in row.values() if v)
            if content.strip():
                documents.append({
                    "content": content,
                    "id": f"{path.stem}_row_{i}",
                    "metadata": {"source": str(path), "type": "csv", "row": i, **row},
                })
    return documents


def load_json_file(path: Path) -> list[dict[str, Any]]:
    """Load JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or json.dumps(item)
                doc = {
                    "content": content,
                    "id": item.get("id") or f"{path.stem}_{i}",
                    "metadata": {k: v for k, v in item.items() if k not in ("content", "text", "id")},
                }
            else:
                doc = {"content": str(item), "id": f"{path.stem}_{i}", "metadata": {}}
            doc["metadata"]["source"] = str(path)
            doc["metadata"]["type"] = "json"
            documents.append(doc)
    elif isinstance(data, dict):
        documents.append({
            "content": data.get("content") or data.get("text") or json.dumps(data),
            "id": data.get("id") or path.stem,
            "metadata": {"source": str(path), "type": "json"},
        })
    return documents


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    documents = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            content = item.get("content") or item.get("text") or item.get("question") or json.dumps(item)
            doc = {
                "content": content,
                "id": item.get("id") or f"{path.stem}_{i}",
                "metadata": {k: v for k, v in item.items() if k not in ("content", "text", "id")},
            }
            doc["metadata"]["source"] = str(path)
            doc["metadata"]["type"] = "jsonl"
            documents.append(doc)
    return documents


def load_pdf(path: Path) -> list[dict[str, Any]]:
    """
    Load PDF file. Tries pymupdf (fitz), then pdfplumber, then falls back to the next backend.
    """
    text_pages: list[str] = []

    # Try pymupdf first
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        for page in doc:
            text_pages.append(page.get_text())
        doc.close()
    except ImportError:
        # Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_pages.append(t)
        except ImportError:
            raise ImportError(
                "No PDF library available. Install one of:\n"
                "  pip install pymupdf    (recommended)\n"
                "  pip install pdfplumber"
            )

    content = "\n\n".join(text_pages)
    if not content.strip():
        return []

    return [{
        "content": content,
        "id": path.stem,
        "metadata": {"source": str(path), "type": "pdf", "pages": len(text_pages)},
    }]


# Map file extensions to loaders
DOCUMENT_LOADERS = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".csv": load_csv,
    ".tsv": load_csv,
    ".json": load_json_file,
    ".jsonl": load_jsonl,
    ".pdf": load_pdf,
}


# ---------------------------------------------------------------------------
# BM25 (keyword-based scoring)
# ---------------------------------------------------------------------------

class SimpleBM25:
    """Simple BM25 scoring for hybrid search. No external dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: dict[str, int] = Counter()
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0
        self._n_docs: int = 0
        self._tokenized_docs: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def index(self, documents: list[str]) -> None:
        """Index a list of document strings."""
        self._tokenized_docs = [self._tokenize(d) for d in documents]
        self._n_docs = len(documents)
        self._doc_lens = [len(d) for d in self._tokenized_docs]
        self._avg_dl = sum(self._doc_lens) / max(self._n_docs, 1)

        self._doc_freqs = Counter()
        for tokens in self._tokenized_docs:
            seen = set(tokens)
            for t in seen:
                self._doc_freqs[t] += 1

    def score(self, query: str) -> np.ndarray:
        """Score all documents against a query. Returns array of scores."""
        query_tokens = self._tokenize(query)
        scores = np.zeros(self._n_docs)

        for token in query_tokens:
            df = self._doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for i, doc_tokens in enumerate(self._tokenized_docs):
                tf = doc_tokens.count(token)
                dl = self._doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                scores[i] += idf * numerator / denominator

        return scores


# ---------------------------------------------------------------------------
# Retrieval Tool
# ---------------------------------------------------------------------------

class Retrieval(BaseTool):
    """
    Standard Retrieval (RAG) tool with embedding-based search.

    Features:
    - Embedding-based semantic search
    - Multiple embedding providers (Sentence Transformers, TF-IDF fallback)
    - Document loaders: txt, md, pdf, csv, tsv, json, jsonl
    - Chunking strategies: fixed-size, sentence, paragraph, recursive
    - Hybrid search: vector similarity + BM25 keyword matching
    - Persistent index storage
    - Metadata filtering
    - Score thresholds

    Usage:
        retrieval = Retrieval()
        retrieval.add_documents([{"content": "...", "id": "doc1"}])
        result = await retrieval.execute(query="your question", top_k=5)
    """

    CHUNKING_STRATEGIES = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "recursive": RecursiveChunker,
    }

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        chunking_strategy: str = "fixed",
        index_path: str | None = None,
        knowledge_base_path: str | None = None,
        enable_hybrid_search: bool = True,
        hybrid_alpha: float = 0.7,
        allow_pickle: bool = False,
        default_top_k: int = 5,
        diversity: float = 0.0,
    ) -> None:
        """
        Initialize the retrieval tool.

        Args:
            embedding_provider: Provider for text embeddings
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between chunks in characters
            chunking_strategy: One of 'fixed', 'sentence', 'paragraph', 'recursive'
            index_path: Path to persist the index
            knowledge_base_path: Path to a directory to auto-load documents from
            enable_hybrid_search: Combine vector + BM25 search (default True)
            hybrid_alpha: Weight for vector score in hybrid (0=BM25 only, 1=vector only)
            allow_pickle: Permit loading a *legacy* pickle index. Default
                ``False``: indexes are stored/loaded as JSON, and an untrusted
                pickle file is refused (unpickling arbitrary data is remote code
                execution). Only set ``True`` for index files you created and
                trust.
            default_top_k: Number of results a query returns when ``top_k`` is
                not passed explicitly (default 5).
            diversity: Diversity weight in [0, 1] for result ranking. ``0.0``
                (default) ranks purely by relevance. A positive value applies
                maximal-marginal-relevance re-ranking so near-duplicate chunks
                do not fill every slot, which broadens coverage for multi-topic
                queries. Callers can override per query via the ``diversity``
                argument.
        """
        super().__init__(
            metadata=ToolMetadata(
                name="retrieval",
                description=(
                    "Search a knowledge base using semantic similarity. "
                    "Returns the most relevant documents/passages for a given query. "
                    "Use this tool when you need to find information from the indexed knowledge base."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="The search query to find relevant documents",
                        required=True,
                        min_length=1,
                        max_length=1000,
                    ),
                    ParameterSpec(
                        name="top_k",
                        type=ParameterType.INTEGER,
                        description="Number of top results to return",
                        required=False,
                        default=default_top_k,
                        min_value=1,
                        max_value=50,
                    ),
                    ParameterSpec(
                        name="score_threshold",
                        type=ParameterType.FLOAT,
                        description="Minimum similarity score threshold (0-1)",
                        required=False,
                        default=0.0,
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    ParameterSpec(
                        name="diversity",
                        type=ParameterType.FLOAT,
                        description=(
                            "Diversity weight in 0-1. 0 ranks purely by "
                            "relevance; higher values spread results across "
                            "distinct passages so near-duplicate chunks do not "
                            "fill every slot (useful for multi-topic queries)."
                        ),
                        required=False,
                        default=diversity,
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    ParameterSpec(
                        name="filter_metadata",
                        type=ParameterType.OBJECT,
                        description="Filter results by metadata fields",
                        required=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "score": {"type": "number"},
                                    "metadata": {"type": "object"},
                                },
                            },
                        },
                        "total_found": {"type": "integer"},
                    },
                },
                timeout_seconds=30,
                tags=["retrieval", "rag", "search", "knowledge-base", "semantic"],
                examples=[
                    {
                        "query": "What is machine learning?",
                        "top_k": 3,
                        "output": {
                            "results": [
                                {
                                    "content": "Machine learning is a subset of AI...",
                                    "score": 0.92,
                                    "metadata": {"source": "ml_intro.txt"},
                                }
                            ],
                            "total_found": 3,
                        },
                    }
                ],
            )
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path
        self.enable_hybrid_search = enable_hybrid_search
        self.hybrid_alpha = hybrid_alpha
        self.allow_pickle = allow_pickle
        self.default_top_k = default_top_k
        self.diversity = max(0.0, min(1.0, diversity))

        # Initialize chunker
        if chunking_strategy in self.CHUNKING_STRATEGIES:
            if chunking_strategy == "fixed":
                self._chunker = FixedSizeChunker(chunk_size, chunk_overlap)
            elif chunking_strategy == "sentence":
                self._chunker = SentenceChunker(chunk_size)
            elif chunking_strategy == "paragraph":
                self._chunker = ParagraphChunker(chunk_size)
            elif chunking_strategy == "recursive":
                self._chunker = RecursiveChunker(chunk_size, chunk_overlap)
        else:
            self._chunker = FixedSizeChunker(chunk_size, chunk_overlap)

        # Initialize embedding provider
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            try:
                import importlib.util
                if importlib.util.find_spec("sentence_transformers") is not None:
                    self.embedding_provider = SentenceTransformerEmbedding()
                else:
                    raise ImportError("sentence_transformers not found")
            except (ImportError, Exception):
                logger.warning(
                    "\u26a0\ufe0f  sentence-transformers not installed. Using simple TF-IDF embeddings.\n"
                    "    For better results: pip install sentence-transformers"
                )
                self.embedding_provider = SimpleEmbedding()

        # Document storage
        self.documents: dict[str, Document] = {}
        self.embeddings_matrix: np.ndarray | None = None
        self.doc_ids: list[str] = []

        # BM25 index for hybrid search
        self._bm25: SimpleBM25 | None = None

        # Load existing index if available
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)

        # Auto-load knowledge base directory
        if knowledge_base_path and os.path.isdir(knowledge_base_path):
            self._load_directory(knowledge_base_path)

    def configure(
        self,
        default_top_k: int | None = None,
        diversity: float | None = None,
    ) -> None:
        """Update the default ``top_k`` / ``diversity`` after construction.

        Sets both the runtime defaults used when a query omits the argument and
        the tool-schema defaults a model reads, so the two stay in step.
        """
        specs = {p.name: p for p in self.metadata.parameters}
        if default_top_k is not None:
            self.default_top_k = int(default_top_k)
            if "top_k" in specs:
                specs["top_k"].default = int(default_top_k)
        if diversity is not None:
            self.diversity = max(0.0, min(1.0, float(diversity)))
            if "diversity" in specs:
                specs["diversity"].default = self.diversity

    def _load_directory(self, dir_path: str):
        """Load all supported files from a directory."""
        loaded = 0
        p = Path(dir_path)
        for ext, loader_fn in DOCUMENT_LOADERS.items():
            for file_path in p.glob(f"*{ext}"):
                try:
                    docs = loader_fn(file_path)
                    if docs:
                        loaded += self.add_documents(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        if loaded:
            logger.info(f"Loaded {loaded} documents/chunks from {dir_path}")

    async def initialize(self) -> None:
        """Initialize the retrieval tool."""
        await super().initialize()
        logger.info("Retrieval tool initialized")

    def _chunk_text(self, text: str, doc_id: str) -> list[Document]:
        """Split text into chunks using the configured strategy."""
        return self._chunker.chunk(text, doc_id)

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        chunk: bool = True,
    ) -> int:
        """
        Add documents to the index.

        Args:
            documents: List of documents with 'content', optional 'id' and 'metadata'
            chunk: Whether to chunk documents

        Returns:
            Number of documents/chunks added
        """
        all_chunks = []

        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue

            doc_id = doc.get("id") or hashlib.md5(content.encode()).hexdigest()[:12]
            metadata = doc.get("metadata", {})

            if chunk and len(content) > self.chunk_size:
                chunks = self._chunk_text(content, doc_id)
                for c in chunks:
                    c.metadata.update(metadata)
                all_chunks.extend(chunks)
            else:
                all_chunks.append(Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                ))

        if not all_chunks:
            return 0

        # Generate embeddings
        texts = [c.content for c in all_chunks]

        if isinstance(self.embedding_provider, SimpleEmbedding):
            all_texts = [d.content for d in self.documents.values()] + texts
            self.embedding_provider.fit(all_texts)

        embeddings = self.embedding_provider.embed(texts)

        for i, chunk in enumerate(all_chunks):
            chunk.embedding = embeddings[i]
            self.documents[chunk.id] = chunk
            self.doc_ids.append(chunk.id)

        self._rebuild_embedding_matrix()
        self._rebuild_bm25_index()

        logger.info(f"Added {len(all_chunks)} documents/chunks to index")
        return len(all_chunks)

    def add_from_file(
        self,
        file_path: str,
        file_type: str = "auto",
        chunk: bool = True,
    ) -> int:
        """
        Add documents from a file.

        Supports: txt, md, pdf, csv, tsv, json, jsonl

        Args:
            file_path: Path to the file
            file_type: File type (auto-detected from extension by default)
            chunk: Whether to chunk documents

        Returns:
            Number of documents added
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect file type
        ext = path.suffix.lower()
        if file_type != "auto":
            ext = f".{file_type}" if not file_type.startswith(".") else file_type

        loader_fn = DOCUMENT_LOADERS.get(ext)
        if loader_fn is None:
            # Fallback to plain text
            loader_fn = load_txt

        documents = loader_fn(path)
        return self.add_documents(documents, chunk=chunk)

    def _rebuild_embedding_matrix(self):
        """Rebuild the embedding matrix from stored documents."""
        if not self.documents:
            self.embeddings_matrix = None
            self.doc_ids = []
            return

        self.doc_ids = list(self.documents.keys())
        embeddings = [self.documents[doc_id].embedding for doc_id in self.doc_ids]
        self.embeddings_matrix = np.vstack(embeddings)

    def _rebuild_bm25_index(self):
        """Rebuild the BM25 index for hybrid search."""
        if not self.enable_hybrid_search or not self.documents:
            self._bm25 = None
            return
        self._bm25 = SimpleBM25()
        texts = [self.documents[doc_id].content for doc_id in self.doc_ids]
        self._bm25.index(texts)

    def _cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all documents."""
        if self.embeddings_matrix is None:
            return np.array([])

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.embeddings_matrix / (
            np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-8
        )
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    def _mmr_order(
        self,
        candidate_indices: list[int],
        relevance: np.ndarray,
        top_k: int,
        diversity: float,
    ) -> list[int]:
        """Re-order candidates by maximal marginal relevance.

        Each pick maximizes ``(1 - diversity) * relevance - diversity * max
        similarity to an already-picked chunk``, so near-identical chunks do
        not consume every slot and a multi-topic query keeps distinct passages.
        Falls back to the relevance order when embeddings are unavailable.
        """
        if self.embeddings_matrix is None or diversity <= 0.0 or not candidate_indices:
            return candidate_indices[:top_k]

        mat = self.embeddings_matrix
        norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        cand = list(candidate_indices)
        # Min-max normalize relevance over the candidate pool so the two MMR
        # terms are on the same 0-1 scale regardless of the score distribution.
        rel_vals = relevance[cand]
        rmin, rmax = float(rel_vals.min()), float(rel_vals.max())
        span = (rmax - rmin) or 1.0
        rel_norm = {i: (float(relevance[i]) - rmin) / span for i in cand}

        selected: list[int] = []
        remaining = set(cand)
        while remaining and len(selected) < top_k:
            best_idx = None
            best_score = None
            for i in cand:
                if i not in remaining:
                    continue
                if selected:
                    sims = norms[selected] @ norms[i]
                    redundancy = float(np.max(sims))
                else:
                    redundancy = 0.0
                mmr = (1.0 - diversity) * rel_norm[i] - diversity * redundancy
                if best_score is None or mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(best_idx)
            remaining.discard(best_idx)
        return selected

    async def _execute(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
        diversity: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute retrieval query.

        Uses hybrid search (vector + BM25) by default if enabled. When
        ``diversity`` is positive (per call or via the tool's configured
        default), results are re-ranked with maximal marginal relevance so
        near-duplicate chunks do not crowd out other relevant passages.
        """
        if top_k is None:
            top_k = self.default_top_k
        if diversity is None:
            diversity = self.diversity
        diversity = max(0.0, min(1.0, float(diversity)))
        if not self.documents:
            return {
                "results": [],
                "total_found": 0,
                "message": "No documents in index. Add documents first using add_documents() or add_from_file().",
            }

        # Get query embedding
        query_embedding = self.embedding_provider.embed_query(query)

        # Vector similarity scores
        vector_scores = self._cosine_similarity(query_embedding)

        # Hybrid: combine with BM25
        if self.enable_hybrid_search and self._bm25 is not None:
            bm25_scores = self._bm25.score(query)
            # Normalize BM25 scores to [0, 1]
            bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
            bm25_norm = bm25_scores / bm25_max
            # Normalize vector scores to [0, 1]
            vec_max = vector_scores.max() if vector_scores.max() > 0 else 1.0
            vec_norm = vector_scores / vec_max
            combined_scores = self.hybrid_alpha * vec_norm + (1 - self.hybrid_alpha) * bm25_norm
        else:
            combined_scores = vector_scores

        ranked = np.argsort(combined_scores)[::-1]

        # Apply score threshold + metadata filter first, then order. Diversity
        # re-ranking (when on) considers a candidate pool wider than top_k so it
        # can trade a near-duplicate for a distinct passage further down.
        candidates: list[int] = []
        for idx in ranked:
            i = int(idx)
            if float(combined_scores[i]) < score_threshold:
                continue
            if filter_metadata:
                doc = self.documents[self.doc_ids[i]]
                if any(doc.metadata.get(k) != v for k, v in filter_metadata.items()):
                    continue
            candidates.append(i)

        if diversity > 0.0:
            pool = candidates[: max(top_k * 4, top_k)]
            ordered = self._mmr_order(pool, combined_scores, top_k, diversity)
        else:
            ordered = candidates[:top_k]

        results = []
        for i in ordered:
            doc = self.documents[self.doc_ids[i]]
            results.append({
                "content": doc.content,
                "score": round(float(combined_scores[i]), 4),
                "id": doc.id,
                "metadata": doc.metadata,
                "rank": len(results) + 1,
            })

        return {
            "results": results,
            "total_found": len(results),
            "query": query,
        }

    # Magic header so a loader can recognise an effGen JSON index regardless of
    # file extension.
    _JSON_INDEX_MARKER = "effgen-retrieval-index-v1"

    def save_index(self, path: str | None = None) -> None:
        """Save the index to disk as JSON (safe to load anywhere).

        The previous format was a Python pickle, which is unsafe to load from an
        untrusted source. JSON carries the same data (embeddings are stored as
        plain number lists) without the arbitrary-code-execution risk.
        """
        path = path or self.index_path
        if not path:
            raise ValueError("No index path specified")

        data = {
            "format": self._JSON_INDEX_MARKER,
            "documents": {
                doc_id: {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding.tolist() if doc.embedding is not None else None,
                }
                for doc_id, doc in self.documents.items()
            },
            "doc_ids": self.doc_ids,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        logger.info(f"Saved index to {path}")

    def _load_index(self, path: str):
        """Load the index from disk.

        Prefers the safe JSON format. A legacy pickle file is only loaded when
        ``allow_pickle=True`` was passed, and then with a loud warning — loading
        an untrusted pickle is arbitrary code execution.
        """
        data = None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            data = None  # not JSON — likely a legacy pickle

        if data is None:
            if not self.allow_pickle:
                raise ValueError(
                    f"Index file '{path}' is not a JSON index and may be a legacy "
                    "pickle. Loading a pickle can execute arbitrary code, so it is "
                    "refused by default. If you created this file and trust it, "
                    "re-create it with save_index() (JSON), or pass "
                    "allow_pickle=True to load it once."
                )
            logger.warning(
                "⚠️  Loading a LEGACY PICKLE index from '%s'. Unpickling "
                "untrusted data can execute arbitrary code; only do this for files "
                "you created and trust. Re-save with save_index() to migrate to the "
                "safe JSON format.",
                path,
            )
            with open(path, "rb") as f:
                data = pickle.load(f)  # noqa: S301 - explicitly gated by allow_pickle

        self.documents = {}
        for doc_id, doc_data in data["documents"].items():
            embedding = doc_data.get("embedding")
            self.documents[doc_id] = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                embedding=np.array(embedding) if embedding is not None else None,
            )

        self.doc_ids = data["doc_ids"]
        self.chunk_size = data.get("chunk_size", self.chunk_size)
        self.chunk_overlap = data.get("chunk_overlap", self.chunk_overlap)

        self._rebuild_embedding_matrix()
        self._rebuild_bm25_index()
        logger.info(f"Loaded index from {path} with {len(self.documents)} documents")

    def clear(self) -> None:
        """Clear all documents from the index."""
        self.documents = {}
        self.embeddings_matrix = None
        self.doc_ids = []
        self._bm25 = None
        logger.info("Cleared retrieval index")

    @property
    def num_documents(self) -> int:
        """Return the number of documents in the index."""
        return len(self.documents)
