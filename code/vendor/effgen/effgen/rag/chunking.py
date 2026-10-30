"""
Advanced chunking strategies for RAG.

Extends the basic chunkers from `effgen.tools.builtin.retrieval` with:

- SemanticChunker: splits on embedding-similarity drops (optional deps)
- CodeChunker: language-aware splitting on function/class boundaries
- TableChunker: detects and preserves markdown / pipe tables
- HierarchicalChunker: preserves heading hierarchy (markdown-style)
"""

from __future__ import annotations

import logging
import re

from effgen.tools.builtin.retrieval import (
    ChunkingStrategy,
    Document,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Friendly default chunker
# ---------------------------------------------------------------------------

class TextChunker(RecursiveChunker):
    """
    General-purpose plain-text chunker — the friendly default.

    Splits text recursively on paragraph, line, sentence, and word boundaries
    so chunks stay close to ``chunk_size`` characters while respecting natural
    breaks. This is the obvious chunker to reach for when you just have text and
    don't need code/table/semantic awareness.

    Example:
        >>> long_document = "A paragraph about chunking. " * 200
        >>> chunker = TextChunker(chunk_size=500, overlap=50)
        >>> pieces = chunker.split_text(long_document)        # list[str]
        >>> docs = chunker.chunk(long_document, doc_id="doc1")  # list[Document]
    """

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` and return the chunk contents as plain strings."""
        return [doc.content for doc in self.chunk(text, doc_id="text")]


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------

class SemanticChunker(ChunkingStrategy):
    """
    Split text on semantic boundaries using sentence embeddings.

    Groups consecutive sentences while their embedding cosine similarity
    stays above `similarity_threshold`. When similarity drops (indicating a
    topic shift), starts a new chunk.

    Requires `sentence-transformers`. Falls back to SentenceChunker if
    unavailable.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        max_chunk_size: int = 1000,
        min_sentences: int = 2,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_sentences = min_sentences
        self.model_name = model_name
        self._model = None
        self._fallback = SentenceChunker(max_chunk_size=max_chunk_size)

    def _get_model(self):
        if self._model is None:
            from effgen.utils.embedding_backend import load_sentence_transformer

            try:
                self._model = load_sentence_transformer(self.model_name)
            except (ImportError, RuntimeError) as exc:
                logger.warning(
                    "SemanticChunker falling back to SentenceChunker: %s", exc
                )
                self._model = False
        return self._model

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Split on sentence-embedding similarity drops (fixed-size fallback)."""
        model = self._get_model()
        if not model:
            return self._fallback.chunk(text, doc_id)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if len(sentences) < self.min_sentences:
            return [Document(
                id=f"{doc_id}_chunk_0",
                content=text,
                metadata={"parent_doc_id": doc_id, "chunking": "semantic"},
            )]

        import numpy as np

        embeddings = model.encode(sentences, convert_to_numpy=True)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        chunks: list[Document] = []
        current_sents: list[str] = [sentences[0]]
        current_len = len(sentences[0])
        chunk_num = 0

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
            sent = sentences[i]

            would_exceed = current_len + len(sent) > self.max_chunk_size
            semantic_break = sim < self.similarity_threshold and len(current_sents) >= self.min_sentences

            if would_exceed or semantic_break:
                chunks.append(Document(
                    id=f"{doc_id}_chunk_{chunk_num}",
                    content=" ".join(current_sents),
                    metadata={
                        "parent_doc_id": doc_id,
                        "chunk_index": chunk_num,
                        "chunking": "semantic",
                    },
                ))
                chunk_num += 1
                current_sents = [sent]
                current_len = len(sent)
            else:
                current_sents.append(sent)
                current_len += len(sent)

        if current_sents:
            chunks.append(Document(
                id=f"{doc_id}_chunk_{chunk_num}",
                content=" ".join(current_sents),
                metadata={
                    "parent_doc_id": doc_id,
                    "chunk_index": chunk_num,
                    "chunking": "semantic",
                },
            ))

        return chunks


# ---------------------------------------------------------------------------
# Code chunker
# ---------------------------------------------------------------------------

_LANG_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [
        re.compile(r"^(?:async\s+)?def\s+\w+", re.MULTILINE),
        re.compile(r"^class\s+\w+", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+", re.MULTILINE),
        re.compile(r"^(?:export\s+)?class\s+\w+", re.MULTILINE),
        re.compile(r"^const\s+\w+\s*=\s*(?:async\s*)?\(", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+", re.MULTILINE),
        re.compile(r"^(?:export\s+)?class\s+\w+", re.MULTILINE),
        re.compile(r"^(?:export\s+)?interface\s+\w+", re.MULTILINE),
    ],
    "go": [
        re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s*)?\w+", re.MULTILINE),
        re.compile(r"^type\s+\w+\s+struct", re.MULTILINE),
    ],
    "rust": [
        re.compile(r"^(?:pub\s+)?fn\s+\w+", re.MULTILINE),
        re.compile(r"^(?:pub\s+)?struct\s+\w+", re.MULTILINE),
        re.compile(r"^(?:pub\s+)?impl\s+", re.MULTILINE),
    ],
    "java": [
        re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\(", re.MULTILINE),
        re.compile(r"^\s*(?:public|private)?\s*class\s+\w+", re.MULTILINE),
    ],
}

_EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}


class CodeChunker(ChunkingStrategy):
    """
    Language-aware chunker that splits code on function/class boundaries.

    Supported languages: python, javascript, typescript, go, rust, java.
    Falls back to fixed-size chunking for unrecognized languages.
    """

    def __init__(
        self,
        language: str | None = None,
        max_chunk_size: int = 1500,
        fallback_chunk_size: int = 1000,
    ) -> None:
        self.language = language
        self.max_chunk_size = max_chunk_size
        self._fallback = FixedSizeChunker(
            chunk_size=fallback_chunk_size, overlap=100
        )

    @classmethod
    def detect_language(cls, filename_or_ext: str) -> str | None:
        """Map a filename or extension to a known language id, or ``None``."""
        ext = filename_or_ext if filename_or_ext.startswith(".") else ""
        if not ext and "." in filename_or_ext:
            ext = "." + filename_or_ext.rsplit(".", 1)[-1]
        return _EXT_LANG.get(ext.lower())

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Split source code on function/class boundaries (fixed-size fallback)."""
        lang = self.language
        patterns = _LANG_PATTERNS.get(lang) if lang else None
        if not patterns:
            return self._fallback.chunk(text, doc_id)

        # Find all boundary offsets
        boundaries: list[int] = [0]
        for pat in patterns:
            for m in pat.finditer(text):
                boundaries.append(m.start())
        boundaries = sorted(set(boundaries))
        boundaries.append(len(text))

        chunks: list[Document] = []
        chunk_num = 0
        buf_start = boundaries[0]
        for i in range(1, len(boundaries)):
            seg_end = boundaries[i]
            seg = text[buf_start:seg_end]
            if len(seg) >= self.max_chunk_size or i == len(boundaries) - 1:
                if seg.strip():
                    chunks.append(Document(
                        id=f"{doc_id}_chunk_{chunk_num}",
                        content=seg.strip(),
                        metadata={
                            "parent_doc_id": doc_id,
                            "chunk_index": chunk_num,
                            "chunking": "code",
                            "language": lang,
                        },
                    ))
                    chunk_num += 1
                buf_start = seg_end

        if not chunks:
            return self._fallback.chunk(text, doc_id)
        return chunks


# ---------------------------------------------------------------------------
# Table chunker
# ---------------------------------------------------------------------------

class TableChunker(ChunkingStrategy):
    """Keeps each markdown/pipe table intact in a single chunk.

    Non-table prose is chunked with FixedSizeChunker.
    """

    # A very permissive markdown table detector
    _TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
    _SEP_LINE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

    def __init__(self, max_chunk_size: int = 1000) -> None:
        self.max_chunk_size = max_chunk_size
        self._prose_chunker = FixedSizeChunker(chunk_size=max_chunk_size, overlap=100)

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Split prose normally while keeping each markdown table in one chunk."""
        lines = text.splitlines()
        blocks: list[tuple[str, str]] = []  # (kind, content)
        i = 0
        prose: list[str] = []

        while i < len(lines):
            line = lines[i]
            # Table must be at least 2 consecutive pipe lines with a separator
            if self._TABLE_LINE.match(line):
                j = i + 1
                while j < len(lines) and self._TABLE_LINE.match(lines[j]):
                    j += 1
                table_lines = lines[i:j]
                if len(table_lines) >= 2 and any(self._SEP_LINE.match(tl) for tl in table_lines):
                    if prose:
                        blocks.append(("prose", "\n".join(prose)))
                        prose = []
                    blocks.append(("table", "\n".join(table_lines)))
                    i = j
                    continue
            prose.append(line)
            i += 1

        if prose:
            blocks.append(("prose", "\n".join(prose)))

        chunks: list[Document] = []
        chunk_num = 0
        for kind, content in blocks:
            if not content.strip():
                continue
            if kind == "table":
                chunks.append(Document(
                    id=f"{doc_id}_chunk_{chunk_num}",
                    content=content,
                    metadata={
                        "parent_doc_id": doc_id,
                        "chunk_index": chunk_num,
                        "chunking": "table",
                        "is_table": True,
                    },
                ))
                chunk_num += 1
            else:
                sub = self._prose_chunker.chunk(content, doc_id)
                for s in sub:
                    s.id = f"{doc_id}_chunk_{chunk_num}"
                    s.metadata["chunk_index"] = chunk_num
                    s.metadata["chunking"] = "table"
                    s.metadata["is_table"] = False
                    chunks.append(s)
                    chunk_num += 1

        if not chunks:
            return [Document(
                id=f"{doc_id}_chunk_0",
                content=text,
                metadata={"parent_doc_id": doc_id, "chunking": "table"},
            )]
        return chunks


# ---------------------------------------------------------------------------
# Hierarchical chunker
# ---------------------------------------------------------------------------

class HierarchicalChunker(ChunkingStrategy):
    """
    Markdown-style hierarchical chunker.

    Splits on headings (`#`, `##`, `###`, ...) and attaches the heading
    hierarchy path (breadcrumb) to each chunk's metadata.
    """

    _HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def __init__(self, max_chunk_size: int = 1500) -> None:
        self.max_chunk_size = max_chunk_size
        self._fallback = FixedSizeChunker(chunk_size=max_chunk_size, overlap=100)

    def chunk(self, text: str, doc_id: str = "doc") -> list[Document]:
        """Split by heading hierarchy, stamping each chunk with its section path."""
        lines = text.splitlines()
        sections: list[tuple[list[str], list[str]]] = []  # (heading_path, body_lines)
        path: list[str] = []
        body: list[str] = []

        def flush():
            if body and any(line.strip() for line in body):
                sections.append((list(path), body.copy()))

        for line in lines:
            m = self._HEADING.match(line)
            if m:
                flush()
                body.clear()
                level = len(m.group(1))
                heading = m.group(2).strip()
                path = path[: level - 1]
                while len(path) < level - 1:
                    path.append("")
                path.append(heading)
            else:
                body.append(line)
        flush()

        if not sections:
            return self._fallback.chunk(text, doc_id)

        chunks: list[Document] = []
        chunk_num = 0
        for heading_path, body_lines in sections:
            content = "\n".join(body_lines).strip()
            if not content:
                continue
            breadcrumb = " > ".join(h for h in heading_path if h)
            header = f"{breadcrumb}\n\n" if breadcrumb else ""
            full = header + content

            if len(full) <= self.max_chunk_size:
                chunks.append(Document(
                    id=f"{doc_id}_chunk_{chunk_num}",
                    content=full,
                    metadata={
                        "parent_doc_id": doc_id,
                        "chunk_index": chunk_num,
                        "chunking": "hierarchical",
                        "hierarchy": heading_path,
                        "breadcrumb": breadcrumb,
                        "section": heading_path[-1] if heading_path else None,
                    },
                ))
                chunk_num += 1
            else:
                sub = self._fallback.chunk(full, doc_id)
                for s in sub:
                    s.id = f"{doc_id}_chunk_{chunk_num}"
                    s.metadata["chunk_index"] = chunk_num
                    s.metadata["chunking"] = "hierarchical"
                    s.metadata["hierarchy"] = heading_path
                    s.metadata["breadcrumb"] = breadcrumb
                    s.metadata["section"] = heading_path[-1] if heading_path else None
                    chunks.append(s)
                    chunk_num += 1

        return chunks or self._fallback.chunk(text, doc_id)
