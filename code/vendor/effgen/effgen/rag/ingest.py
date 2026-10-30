"""
Document ingestion pipeline.

Loads documents from many file formats, extracts metadata, deduplicates
content, and produces a list of `IngestedChunk` objects ready for indexing.

All third-party loaders (pymupdf, python-docx, bs4, ebooklib, openpyxl) are OPTIONAL.
Core formats (txt, md, json, jsonl, csv, html via stdlib) work with no
external dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from effgen.tools.builtin.retrieval import (
    Document,
    RecursiveChunker,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestedChunk:
    """A chunk of an ingested document, ready for indexing."""

    id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()[:16]

    def to_document(self) -> Document:
        """Convert the chunk to a :class:`Document` carrying source metadata."""
        md = dict(self.metadata)
        md.setdefault("source", self.source)
        md.setdefault("content_hash", self.content_hash)
        return Document(id=self.id, content=self.content, metadata=md)


@dataclass
class IngestSummary:
    """Coverage report for a single :meth:`DocumentIngester.ingest` call.

    Available on :attr:`DocumentIngester.last_summary` after ``ingest()``
    returns, so a pipeline can log coverage without scraping log lines: how
    many chunks were indexed, how many files were seen, which files were
    skipped and why (mirrors :attr:`DocumentIngester.last_skipped`), and the
    elapsed time / throughput.
    """

    indexed_chunks: int = 0
    files_seen: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def skipped_count(self) -> int:
        """Number of files skipped."""
        return len(self.skipped)

    @property
    def indexed_files(self) -> int:
        """Files that contributed at least one indexed chunk."""
        return self.files_seen - len(self.skipped)

    @property
    def chunks_per_s(self) -> float:
        """Indexing throughput in chunks per second (0.0 for an instant run)."""
        return self.indexed_chunks / self.elapsed_s if self.elapsed_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Loaders — always-available (stdlib only)
# ---------------------------------------------------------------------------

def _load_txt(path: Path) -> list[dict[str, Any]]:
    return [{
        "content": path.read_text(encoding="utf-8", errors="replace"),
        "metadata": {"type": "txt"},
    }]


def _load_code(path: Path) -> list[dict[str, Any]]:
    """Read a source-code or plain-text config file as UTF-8 text.

    Source files carry no separate structure to extract — the text itself is the
    content — so they are read verbatim, with the extension recorded as the
    ``language`` so a downstream reader can label the snippet.
    """
    return [{
        "content": path.read_text(encoding="utf-8", errors="replace"),
        "metadata": {"type": "code", "language": path.suffix.lower().lstrip(".")},
    }]


# Source-code and plain-text config extensions read verbatim as UTF-8 text. The
# most natural input to a coding assistant is a source file, so these are
# accepted alongside documents rather than rejected as unknown types.
_CODE_EXTENSIONS: tuple[str, ...] = (
    ".py", ".pyw", ".pyi",
    ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
    ".cs", ".go", ".rs", ".java", ".kt", ".kts", ".scala",
    ".swift", ".m", ".mm", ".rb", ".php", ".pl", ".pm",
    ".lua", ".r", ".jl", ".dart", ".hs", ".clj", ".cljs",
    ".ex", ".exs", ".erl", ".vb",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql", ".proto",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties",
    ".xml", ".css", ".scss", ".sass", ".less",
    ".rst", ".tex", ".text",
)


def _load_markdown(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Attempt title extraction from first H1
    title = None
    for line in text.splitlines():
        m = re.match(r"^#\s+(.+)$", line.strip())
        if m:
            title = m.group(1).strip()
            break
    return [{
        "content": text,
        "metadata": {"type": "markdown", "title": title},
    }]


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        out = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or json.dumps(item)
                meta = {k: v for k, v in item.items() if k not in ("content", "text")}
            else:
                content = str(item)
                meta = {}
            meta["type"] = "json"
            meta["index"] = i
            out.append({"content": content, "metadata": meta})
        return out
    if isinstance(data, dict):
        content = data.get("content") or data.get("text") or json.dumps(data)
        return [{"content": content, "metadata": {"type": "json"}}]
    return [{"content": str(data), "metadata": {"type": "json"}}]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or json.dumps(item)
                meta = {k: v for k, v in item.items() if k not in ("content", "text")}
            else:
                content = str(item)
                meta = {}
            meta["type"] = "jsonl"
            meta["line"] = i
            out.append({"content": content, "metadata": meta})
    return out


def _load_csv(path: Path) -> list[dict[str, Any]]:
    out = []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            content = " ".join(str(v) for v in row.values() if v)
            if content.strip():
                out.append({
                    "content": content,
                    "metadata": {"type": "csv", "row": i, **row},
                })
    return out


def _load_html(path: Path) -> list[dict[str, Any]]:
    """Load HTML. Prefers bs4 if available, falls back to stdlib html.parser."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    title: str | None = None
    text: str

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        from html.parser import HTMLParser

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: list[str] = []
                self._skip = 0
                self._in_title = False
                self.title: str | None = None

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip += 1
                if tag == "title":
                    self._in_title = True

            def handle_endtag(self, tag):
                if tag in ("script", "style") and self._skip:
                    self._skip -= 1
                if tag == "title":
                    self._in_title = False

            def handle_data(self, data):
                if self._skip:
                    return
                if self._in_title:
                    self.title = (self.title or "") + data
                self.parts.append(data)

        s = _Stripper()
        s.feed(raw)
        title = (s.title or "").strip() or None
        text = "\n".join(p for p in s.parts if p.strip())

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return [{"content": text, "metadata": {"type": "html", "title": title}}]


# ---------------------------------------------------------------------------
# Optional loaders
# ---------------------------------------------------------------------------

_PDF_DATE_RE = re.compile(
    r"^D:(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"(?P<hour>\d{2})?(?P<minute>\d{2})?(?P<second>\d{2})?"
    r"(?P<tz>Z|[+-]\d{2}'?\d{2}'?)?"
)


def _normalize_pdf_date(raw: str) -> str:
    """Convert a PDF ``D:YYYYMMDDHHmmSS±HH'mm'`` date to ISO-8601.

    Returns the input unchanged if it does not match the PDF date format.
    """
    if not raw:
        return raw
    m = _PDF_DATE_RE.match(raw.strip())
    if not m:
        return raw
    year, month, day = m.group("year"), m.group("month"), m.group("day")
    hour = m.group("hour") or "00"
    minute = m.group("minute") or "00"
    second = m.group("second") or "00"
    iso = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    tz = m.group("tz")
    if tz == "Z":
        iso += "+00:00"
    elif tz:
        tz_digits = tz.replace("'", "")
        sign, tz_hour, tz_minute = tz_digits[0], tz_digits[1:3], tz_digits[3:5] or "00"
        iso += f"{sign}{tz_hour}:{tz_minute}"
    return iso


def _load_pdf_pymupdf(path: Path) -> list[dict[str, Any]]:
    import fitz  # type: ignore  # pymupdf

    doc = fitz.open(str(path))
    doc_metadata: dict[str, Any] = {"type": "pdf", "pages": doc.page_count}
    try:
        info = doc.metadata or {}
        for k in ("title", "author", "subject", "creationDate"):
            v = info.get(k)
            if v:
                dest = "date" if k == "creationDate" else k.lower()
                doc_metadata[dest] = _normalize_pdf_date(v) if dest == "date" else v
    except Exception:  # best-effort metadata; the document still ingests without it
        pass
    # One row per page (not one row for the whole document) so each
    # resulting chunk's metadata carries the page it came from — a
    # Citation for a PDF-sourced chunk can then point at a page number
    # instead of always leaving it None.
    results = [
        {"content": page.get_text(), "metadata": {**doc_metadata, "page": page_num}}
        for page_num, page in enumerate(doc, start=1)
    ]
    doc.close()
    return results


def _load_pdf_pypdf(path: Path) -> list[dict[str, Any]]:
    import pypdf  # type: ignore

    reader = pypdf.PdfReader(str(path))
    doc_metadata: dict[str, Any] = {"type": "pdf", "pages": len(reader.pages)}
    try:
        info = reader.metadata or {}
        for raw_key, dest in (("/Title", "title"), ("/Author", "author"),
                              ("/Subject", "subject"), ("/CreationDate", "date")):
            v = info.get(raw_key)
            if v:
                doc_metadata[dest] = _normalize_pdf_date(str(v)) if dest == "date" else str(v)
    except Exception:  # best-effort metadata; the document still ingests without it
        pass
    # One row per page — see _load_pdf_pymupdf for why.
    return [
        {"content": p.extract_text() or "", "metadata": {**doc_metadata, "page": page_num}}
        for page_num, p in enumerate(reader.pages, start=1)
    ]


def _load_pdf_pdfplumber(path: Path) -> list[dict[str, Any]]:
    import pdfplumber  # type: ignore

    with pdfplumber.open(str(path)) as pdf:
        doc_metadata: dict[str, Any] = {"type": "pdf", "pages": len(pdf.pages)}
        # One row per page — see _load_pdf_pymupdf for why.
        return [
            {"content": page.extract_text() or "", "metadata": {**doc_metadata, "page": page_num}}
            for page_num, page in enumerate(pdf.pages, start=1)
        ]


def _load_pdf(path: Path) -> list[dict[str, Any]]:
    """Load a PDF, preferring pymupdf but falling back to pypdf / pdfplumber.

    The standalone ``pdf`` tool extracts text with pypdf/pdfplumber, so a PDF
    that works there should ingest here too — we no longer hard-require pymupdf
    and silently skip the file when only the other backends are installed.
    """
    backends = (
        ("pymupdf", _load_pdf_pymupdf),
        ("pypdf", _load_pdf_pypdf),
        ("pdfplumber", _load_pdf_pdfplumber),
    )
    last_runtime_error: Exception | None = None
    for _name, loader in backends:
        try:
            return loader(path)
        except ImportError:
            continue  # backend not installed — try the next one
        except Exception as e:  # backend installed but failed on this file
            last_runtime_error = e
            continue
    if last_runtime_error is not None:
        raise last_runtime_error
    raise ImportError(
        "PDF support requires a PDF backend. Install one with: "
        "pip install pypdf  (or 'effgen[documents]', or pymupdf)."
    )


def _load_docx(path: Path) -> list[dict[str, Any]]:
    try:
        import docx  # type: ignore  # python-docx
    except ImportError as e:
        raise ImportError(
            "DOCX support requires python-docx. Install with: pip install python-docx"
        ) from e

    d = docx.Document(str(path))
    paragraphs = [p.text for p in d.paragraphs if p.text]
    meta: dict[str, Any] = {"type": "docx"}
    try:
        core = d.core_properties
        if core.title:
            meta["title"] = core.title
        if core.author:
            meta["author"] = core.author
        if core.created:
            meta["date"] = (
                core.created.isoformat()
                if hasattr(core.created, "isoformat")
                else str(core.created)
            )
    except Exception:  # best-effort metadata; the document still ingests without it
        pass
    return [{"content": "\n\n".join(paragraphs), "metadata": meta}]


def _load_xlsx(path: Path) -> list[dict[str, Any]]:
    try:
        import openpyxl  # type: ignore
    except ImportError as e:
        raise ImportError(
            "XLSX support requires openpyxl. Install with: pip install openpyxl "
            "(or 'effgen[documents]')."
        ) from e

    wb = openpyxl.load_workbook(str(path), data_only=True)
    workbook_meta: dict[str, Any] = {"type": "xlsx", "sheet_count": len(wb.sheetnames)}
    try:
        props = wb.properties
        if props.title:
            workbook_meta["title"] = props.title
        if props.creator:
            workbook_meta["author"] = props.creator
        if props.created:
            workbook_meta["date"] = (
                props.created.isoformat()
                if hasattr(props.created, "isoformat")
                else str(props.created)
            )
    except Exception:  # best-effort metadata; the document still ingests without it
        pass

    docs: list[dict[str, Any]] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = [list(row) for row in ws.iter_rows(values_only=True)]
        if not rows or all(all(cell is None for cell in row) for row in rows):
            continue
        lines = [" | ".join("" if cell is None else str(cell) for cell in row) for row in rows]
        sheet_meta = {**workbook_meta, "sheet": sheet_name}
        docs.append({
            "content": f"Sheet: {sheet_name}\n" + "\n".join(lines),
            "metadata": sheet_meta,
        })
    wb.close()
    return docs


def _load_epub(path: Path) -> list[dict[str, Any]]:
    try:
        import ebooklib  # type: ignore
        from ebooklib import epub  # type: ignore
    except ImportError as e:
        raise ImportError(
            "EPUB support requires ebooklib. Install with: pip install ebooklib"
        ) from e

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as e:
        raise ImportError(
            "EPUB support also requires beautifulsoup4. Install with: pip install beautifulsoup4"
        ) from e

    book = epub.read_epub(str(path))
    texts = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text(separator="\n"))
    meta: dict[str, Any] = {"type": "epub"}
    try:
        titles = book.get_metadata("DC", "title")
        if titles:
            meta["title"] = titles[0][0]
        authors = book.get_metadata("DC", "creator")
        if authors:
            meta["author"] = authors[0][0]
    except Exception:  # best-effort metadata; the document still ingests without it
        pass
    return [{"content": "\n\n".join(texts), "metadata": meta}]


# Extension → loader map
LOADERS: dict[str, Callable[[Path], list[dict[str, Any]]]] = {
    ".txt": _load_txt,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".json": _load_json,
    ".jsonl": _load_jsonl,
    ".csv": _load_csv,
    ".tsv": _load_csv,
    ".html": _load_html,
    ".htm": _load_html,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".epub": _load_epub,
    ".xlsx": _load_xlsx,
}

# Source-code / plain-text config extensions all read verbatim as UTF-8 text.
for _ext in _CODE_EXTENSIONS:
    LOADERS.setdefault(_ext, _load_code)


# ---------------------------------------------------------------------------
# Document Ingester
# ---------------------------------------------------------------------------

class DocumentIngester:
    """Load, chunk and deduplicate documents into indexable ``IngestedChunk`` objects.

    Accepts a directory or a list of paths.

    Features:
    - Multi-format loaders (txt, md, json, jsonl, csv, html, pdf*, docx*, epub*, xlsx*)
    - Metadata extraction (title, author, date, source)
    - Content deduplication via SHA-256 hash of chunk text
    - Progress tracking via tqdm (optional)

    Optional loaders (*) raise a helpful ImportError at load time.
    """

    def __init__(
        self,
        chunker: Any = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        dedupe: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.chunker = chunker or RecursiveChunker(
            chunk_size=chunk_size, overlap=chunk_overlap
        )
        self.dedupe = dedupe
        self.show_progress = show_progress
        self._seen_hashes: set[str] = set()
        # Files skipped during the most recent ingest(), as (path, reason). Lets
        # callers report *why* nothing was indexed instead of a bare "0 docs".
        self.last_skipped: list[tuple[str, str]] = []
        # Coverage report for the most recent ingest() (counts + throughput).
        self.last_summary: IngestSummary | None = None
        # Per-file bookkeeping set by _ingest_file so ingest() can tell a
        # fully-deduplicated file apart from a genuinely empty one.
        self._last_file_had_content = False
        self._last_file_dedup_dropped = 0

    def _iter_with_progress(self, items: list[Any], desc: str) -> Iterable[Any]:
        if not self.show_progress:
            return items
        try:
            from tqdm import tqdm  # type: ignore

            return tqdm(items, desc=desc)
        except ImportError:
            return items

    def supported_extensions(self) -> list[str]:
        """Return the file extensions the ingester can load."""
        return sorted(LOADERS.keys())

    def ingest(
        self,
        source: str | Path | list[str | Path],
        recursive: bool = True,
    ) -> list[IngestedChunk]:
        """
        Ingest documents from a file, directory, or list of paths.

        Args:
            source: file path, directory path, or list of paths.
            recursive: whether to recurse into subdirectories.

        Returns:
            A list of `IngestedChunk` objects.

        After the call, two attributes carry the coverage report for the run:
        :attr:`last_skipped` is a list of ``(path, reason)`` for every file that
        was not indexed (unsupported extension, corrupt file, no extractable
        text, or all-duplicate content), and :attr:`last_summary` is an
        :class:`IngestSummary` with the indexed/skipped counts and throughput.
        """
        import time as _time

        _start = _time.perf_counter()
        self.last_skipped = []
        paths: list[Path] = []
        if isinstance(source, list | tuple):
            for s in source:
                paths.extend(self._expand(Path(s), recursive))
        else:
            paths.extend(self._expand(Path(source), recursive))

        chunks: list[IngestedChunk] = []
        for path in self._iter_with_progress(paths, "Ingesting"):
            ext = path.suffix.lower()
            if ext not in LOADERS:
                # Every path handed to ingest() is accounted for: an
                # extension outside LOADERS is reported rather than dropped,
                # whether it came from a single path or a directory scan.
                self.last_skipped.append((str(path), self._unsupported_reason(ext)))
                continue
            try:
                file_chunks = self._ingest_file(path)
                if file_chunks:
                    chunks.extend(file_chunks)
                elif (
                    self.dedupe
                    and self._last_file_had_content
                    and self._last_file_dedup_dropped > 0
                ):
                    # The file had extractable text, but every chunk it produced
                    # was an exact duplicate of content already indexed in this
                    # run — distinct from a genuinely empty file, so a corpus
                    # audit isn't misled into chasing a phantom empty file.
                    self.last_skipped.append((
                        str(path),
                        ("all content duplicates an already-ingested file "
                        "(dedupe=True)"),
                    ))
                elif ext == ".pdf":
                    # A PDF loads but yields no text most often because it is
                    # a scanned/image-only PDF: the pages are images with no
                    # text layer, and ingestion does not run OCR on them.
                    self.last_skipped.append((
                        str(path),
                        ("PDF has no extractable text layer (likely a scanned "
                        "or image-only PDF); OCR is not applied during ingestion"),
                    ))
                else:
                    # A supported extension that loaded but produced no
                    # usable content (empty file, blank sheet, ...).
                    self.last_skipped.append((
                        str(path),
                        "file contained no extractable text (empty or whitespace-only)",
                    ))
            except ImportError as e:
                logger.warning("Skipping %s: %s", path, e)
                self.last_skipped.append((str(path), str(e)))
            except Exception as e:
                # The raw parser exception (e.g. pypdf's "No /Root object! -
                # Is this really a PDF?") is not plain English next to the
                # other skip reasons above; keep it in the log for debugging
                # but report a normalized reason as the primary message.
                logger.warning("Failed to ingest %s: %s", path, e)
                self.last_skipped.append((str(path), self._corrupt_file_reason(ext)))

        self.last_summary = IngestSummary(
            indexed_chunks=len(chunks),
            files_seen=len(paths),
            skipped=list(self.last_skipped),
            elapsed_s=_time.perf_counter() - _start,
        )
        logger.info(
            "Ingested %d chunks from %d file(s) (%d skipped)",
            len(chunks), len(paths), len(self.last_skipped),
        )
        return chunks

    #: Image extensions that ingestion does not index as text, but which the
    #: multimodal path can describe/OCR. Named so the skip reason can point a
    #: user at the right tool instead of a bare "unsupported".
    _IMAGE_EXTENSIONS: frozenset[str] = frozenset({
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp",
    })

    def _unsupported_reason(self, ext: str) -> str:
        label = ext if ext else "(no extension)"
        if ext in self._IMAGE_EXTENSIONS:
            return (
                f"image file '{label}' is not indexed as text during ingestion; "
                "read it with a vision model instead — e.g. pass the image to "
                "agent.run(..., inputs=['picture.png']) or the multimodal preset "
                "(create_agent('multimodal', model=...))"
            )
        return (
            f"unsupported extension '{label}'; supported: "
            f"{', '.join(self.supported_extensions())}"
        )

    _CORRUPT_FILE_LABELS: dict[str, str] = {
        ".pdf": "PDF",
        ".docx": "DOCX",
        ".xlsx": "XLSX",
        ".epub": "EPUB",
        ".json": "JSON",
        ".jsonl": "JSONL",
    }

    def _corrupt_file_reason(self, ext: str) -> str:
        label = self._CORRUPT_FILE_LABELS.get(ext, ext.lstrip(".").upper() or "file")
        return f"file is not a valid {label} (corrupt or truncated)"

    def _expand(self, path: Path, recursive: bool) -> list[Path]:
        if path.is_file():
            return [path]
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            return [p for p in path.glob(pattern) if p.is_file()]
        logger.warning("Path does not exist: %s", path)
        return []

    def _ingest_file(self, path: Path) -> list[IngestedChunk]:
        # Reset per-file bookkeeping so ingest() can classify an empty return
        # (genuinely empty vs. fully deduplicated).
        self._last_file_had_content = False
        self._last_file_dedup_dropped = 0
        ext = path.suffix.lower()
        loader = LOADERS.get(ext)
        if loader is None:
            return []

        docs = loader(path)
        if not docs:
            return []

        out: list[IngestedChunk] = []
        source = str(path)
        base_id = path.stem

        for doc_idx, doc in enumerate(docs):
            content = doc.get("content", "") or ""
            if not content.strip():
                continue
            doc_meta = doc.get("metadata", {}) or {}

            # Chunk the content
            doc_id = f"{base_id}_{doc_idx}" if len(docs) > 1 else base_id
            chunk_docs = self.chunker.chunk(content, doc_id)

            for c in chunk_docs:
                if not c.content.strip():
                    continue
                self._last_file_had_content = True
                content_hash = hashlib.sha256(
                    c.content.encode("utf-8")
                ).hexdigest()[:16]

                if self.dedupe and content_hash in self._seen_hashes:
                    self._last_file_dedup_dropped += 1
                    continue
                self._seen_hashes.add(content_hash)

                merged_meta = {**doc_meta, **c.metadata}
                merged_meta["source"] = source
                out.append(IngestedChunk(
                    id=c.id,
                    content=c.content,
                    source=source,
                    metadata=merged_meta,
                    content_hash=content_hash,
                ))

        return out

    def ingest_text(
        self,
        text: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[IngestedChunk]:
        """Ingest a raw text string directly (no file on disk required).

        Chunks ``text`` the same way file ingestion does and returns the
        resulting :class:`IngestedChunk` objects. Useful when documents live in
        memory rather than on disk (e.g. ``create_agent("rag",
        knowledge_base=["...raw text..."])``).

        Args:
            text: The document body to ingest.
            doc_id: Optional base id for the produced chunks (defaults to a
                hash-derived ``inline-<hash>`` id so repeated text is stable).
            metadata: Optional metadata merged into every chunk.

        Returns:
            A list of `IngestedChunk` objects (empty if ``text`` is blank).
        """
        if text is None or not str(text).strip():
            return []
        content = str(text)
        base_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        base_id = doc_id or f"inline-{base_hash}"
        source = doc_id or f"inline:{base_hash}"
        doc_meta = dict(metadata or {})

        out: list[IngestedChunk] = []
        for c in self.chunker.chunk(content, base_id):
            if not c.content.strip():
                continue
            content_hash = hashlib.sha256(
                c.content.encode("utf-8")
            ).hexdigest()[:16]
            if self.dedupe and content_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(content_hash)
            merged_meta = {**doc_meta, **c.metadata}
            merged_meta["source"] = source
            out.append(IngestedChunk(
                id=c.id,
                content=c.content,
                source=source,
                metadata=merged_meta,
                content_hash=content_hash,
            ))
        return out

    def reset_dedupe(self) -> None:
        """Clear the deduplication cache."""
        self._seen_hashes.clear()
