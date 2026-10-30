"""
Unit tests for the effgen.rag pipeline.

**IMPORTANT: These are MOCK-MODEL tests.**

All LLM calls in this file use hand-written mock classes (or the
tests.fixtures.mock_models.MockModel) — NO real model is loaded.
These tests exercise the RAG plumbing (chunkers, search, rerankers,
context builder, citations, preset wiring) quickly and deterministically
without GPU requirements.

For REAL-MODEL end-to-end tests of the same pipeline, see
`tests/integration/test_rag_real_model.py`, which loads Qwen2.5-3B
via the `real_model` session fixture and verifies grounded answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effgen.rag import (
    Citation,
    CitationTracker,
    CodeChunker,
    ContextBuilder,
    DocumentIngester,
    HierarchicalChunker,
    HybridSearchEngine,
    LLMReranker,
    RuleBasedReranker,
    SearchResult,
    SemanticChunker,
    TableChunker,
    TextChunker,
)
from effgen.rag.ingest import IngestedChunk

# ---------------------------------------------------------------------------
# DocumentIngester
# ---------------------------------------------------------------------------

class TestDocumentIngester:
    def test_ingest_txt(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("Hello world. This is a test document.")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        assert len(chunks) >= 1
        assert all(isinstance(c, IngestedChunk) for c in chunks)
        assert "Hello world" in chunks[0].content
        assert chunks[0].source.endswith("a.txt")
        assert chunks[0].content_hash  # auto-populated

    def test_ingest_markdown_extracts_title(self, tmp_path: Path):
        (tmp_path / "doc.md").write_text("# My Title\n\nSome body text here.")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("title") == "My Title"
        assert chunks[0].metadata.get("type") == "markdown"

    def test_ingest_json_list(self, tmp_path: Path):
        data = [
            {"content": "first doc", "tag": "a"},
            {"content": "second doc", "tag": "b"},
        ]
        (tmp_path / "d.json").write_text(json.dumps(data))
        chunks = DocumentIngester(show_progress=False).ingest(tmp_path)
        contents = [c.content for c in chunks]
        assert any("first" in c for c in contents)
        assert any("second" in c for c in contents)

    def test_ingest_jsonl(self, tmp_path: Path):
        lines = [
            json.dumps({"text": "line one content"}),
            json.dumps({"text": "line two content"}),
            "",  # empty line should be skipped
            "not json",  # invalid line should be skipped
        ]
        (tmp_path / "d.jsonl").write_text("\n".join(lines))
        chunks = DocumentIngester(show_progress=False).ingest(tmp_path)
        assert len(chunks) == 2

    def test_ingest_csv(self, tmp_path: Path):
        (tmp_path / "d.csv").write_text("name,description\nalice,engineer\nbob,artist\n")
        chunks = DocumentIngester(show_progress=False).ingest(tmp_path)
        assert len(chunks) >= 2

    def test_ingest_html_stdlib_fallback(self, tmp_path: Path):
        html = "<html><head><title>T</title></head><body><script>alert(1)</script><p>Real content</p></body></html>"
        (tmp_path / "d.html").write_text(html)
        chunks = DocumentIngester(show_progress=False).ingest(tmp_path)
        assert len(chunks) >= 1
        assert "Real content" in chunks[0].content
        # Script content must be stripped
        assert "alert(1)" not in chunks[0].content

    def test_ingest_deduplication(self, tmp_path: Path):
        same = "Exactly the same content across two files."
        (tmp_path / "a.txt").write_text(same)
        (tmp_path / "b.txt").write_text(same)
        chunks = DocumentIngester(show_progress=False, dedupe=True).ingest(tmp_path)
        # Only one copy kept
        assert len(chunks) == 1

    def test_ingest_dedup_disabled(self, tmp_path: Path):
        same = "Exactly the same content across two files."
        (tmp_path / "a.txt").write_text(same)
        (tmp_path / "b.txt").write_text(same)
        chunks = DocumentIngester(show_progress=False, dedupe=False).ingest(tmp_path)
        assert len(chunks) == 2

    def test_ingest_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")
        (tmp_path / "top.txt").write_text("top content")
        chunks = DocumentIngester(show_progress=False).ingest(tmp_path, recursive=True)
        assert len(chunks) == 2

    def test_ingest_non_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")
        (tmp_path / "top.txt").write_text("top content")
        chunks = DocumentIngester(show_progress=False).ingest(tmp_path, recursive=False)
        assert len(chunks) == 1
        assert "top" in chunks[0].content

    def test_ingest_single_file(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("single file content")
        chunks = DocumentIngester(show_progress=False).ingest(f)
        assert len(chunks) == 1

    def test_ingest_list_of_paths(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("content a")
        (tmp_path / "b.txt").write_text("content b")
        chunks = DocumentIngester(show_progress=False).ingest(
            [tmp_path / "a.txt", tmp_path / "b.txt"]
        )
        assert len(chunks) == 2

    def test_ingest_missing_path_warns(self, tmp_path: Path):
        chunks = DocumentIngester(show_progress=False).ingest(tmp_path / "nope")
        assert chunks == []

    def test_ingest_unsupported_extension_skipped(self, tmp_path: Path):
        (tmp_path / "ignore.xyz").write_text("should be ignored")
        (tmp_path / "keep.txt").write_text("should be kept")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        assert len(chunks) == 1
        assert "kept" in chunks[0].content
        # The unsupported file is reported, not silently dropped.
        assert len(ingester.last_skipped) == 1
        skipped_path, reason = ingester.last_skipped[0]
        assert "ignore.xyz" in skipped_path
        assert "unsupported extension '.xyz'" in reason
        assert ".txt" in reason  # names the supported set

    def test_ingest_directory_reports_every_unsupported_file(self, tmp_path: Path):
        # A real-world folder mixing recognized and unrecognized formats: every
        # file is accounted for, either indexed or named in last_skipped —
        # never dropped without a trace.
        (tmp_path / "report.txt").write_text("indexable content")
        (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        (tmp_path / "deck.pptx").write_bytes(b"PK\x03\x04")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        indexed = {Path(c.source).name for c in chunks}
        skipped = {Path(p).name for p, _ in ingester.last_skipped}
        assert indexed == {"report.txt"}
        assert skipped == {"photo.png", "deck.pptx"}

    def test_ingest_single_unsupported_file_names_extension(self, tmp_path: Path):
        f = tmp_path / "slides.pptx"
        f.write_bytes(b"PK\x03\x04")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(f)
        assert chunks == []
        assert len(ingester.last_skipped) == 1
        _, reason = ingester.last_skipped[0]
        assert "unsupported extension '.pptx'" in reason

    def test_ingest_empty_file(self, tmp_path: Path):
        (tmp_path / "empty.txt").write_text("")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        assert chunks == []
        # Distinct from an unsupported-extension skip reason.
        _, reason = ingester.last_skipped[0]
        assert "unsupported extension" not in reason
        assert "no extractable text" in reason

    def test_ingest_large_file_chunks(self, tmp_path: Path):
        # Force chunking by using small chunk_size
        big = "sentence one. " * 500
        (tmp_path / "big.txt").write_text(big)
        ingester = DocumentIngester(show_progress=False, chunk_size=200, chunk_overlap=20)
        chunks = ingester.ingest(tmp_path)
        assert len(chunks) > 1

    def test_reset_dedupe(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("dup content")
        ingester = DocumentIngester(show_progress=False)
        chunks1 = ingester.ingest(tmp_path)
        assert len(chunks1) == 1
        # Second call would dedupe to zero
        chunks2 = ingester.ingest(tmp_path)
        assert chunks2 == []
        ingester.reset_dedupe()
        chunks3 = ingester.ingest(tmp_path)
        assert len(chunks3) == 1

    def test_ingest_xlsx(self, tmp_path: Path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Budget"
        ws.append(["Department", "Q1", "Q2"])
        ws.append(["Engineering", 120000, 125000])
        ws.append(["Marketing", 60000, 72000])
        wb.properties.title = "FY26 Budget"
        wb.properties.creator = "Finance Team"
        f = tmp_path / "budget.xlsx"
        wb.save(f)

        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(f)
        assert len(chunks) == 1
        assert ingester.last_skipped == []
        content = chunks[0].content
        assert "Budget" in content
        assert "Engineering" in content
        assert "125000" in content
        assert chunks[0].metadata["title"] == "FY26 Budget"
        assert chunks[0].metadata["author"] == "Finance Team"
        assert chunks[0].metadata["type"] == "xlsx"

    def test_ingest_xlsx_multi_sheet(self, tmp_path: Path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Sales"
        ws1.append(["region", "total"])
        ws1.append(["West", 100])
        ws2 = wb.create_sheet("Costs")
        ws2.append(["category", "amount"])
        ws2.append(["Shipping", 20])
        f = tmp_path / "book.xlsx"
        wb.save(f)

        chunks = DocumentIngester(show_progress=False).ingest(f)
        sheets = {c.metadata["sheet"] for c in chunks}
        assert sheets == {"Sales", "Costs"}

    def test_ingest_xlsx_empty_sheet_skipped(self, tmp_path: Path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        f = tmp_path / "blank.xlsx"
        wb.save(f)  # default sheet with no data rows

        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(f)
        assert chunks == []
        _, reason = ingester.last_skipped[0]
        assert "no extractable text" in reason

    def _write_pdf(self, path: Path, lines: list[str]) -> None:
        reportlab = pytest.importorskip("reportlab")  # noqa: F841
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(path), pagesize=letter)
        y = 720
        for ln in lines:
            c.drawString(72, y, ln)
            y -= 20
        c.save()

    def test_ingest_pdf_without_pymupdf(self, tmp_path: Path):
        # PDFs ingest via pypdf/pdfplumber when pymupdf is absent — the ingester
        # no longer hard-requires pymupdf the way the standalone pdf tool never
        # did. (Skips only if no PDF backend at all is installed.)
        pytest.importorskip("pypdf")
        pdf = tmp_path / "doc.pdf"
        self._write_pdf(pdf, ["Revenue was 1284.6 million dollars.",
                              "EPS was 1.47 dollars."])
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(pdf)
        assert len(chunks) >= 1
        assert "1284.6" in chunks[0].content
        assert ingester.last_skipped == []

    def test_ingest_records_skip_reason(self, tmp_path: Path):
        # A file that cannot be parsed is recorded with its reason so callers can
        # surface *why* nothing was indexed instead of a bare "0 documents".
        # The reason is a normalized, plain-English message — not the raw
        # parser exception text (e.g. pypdf's "No /Root object!").
        pytest.importorskip("pypdf")
        bad = tmp_path / "broken.pdf"
        bad.write_text("this is not a real pdf")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(bad)
        assert chunks == []
        assert len(ingester.last_skipped) == 1
        skipped_path, reason = ingester.last_skipped[0]
        assert "broken.pdf" in skipped_path
        assert reason == "file is not a valid PDF (corrupt or truncated)"

    def test_corrupt_docx_skip_reason_is_normalized(self, tmp_path: Path):
        pytest.importorskip("docx")
        bad = tmp_path / "broken.docx"
        bad.write_bytes(b"not a real docx zip")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(bad)
        assert chunks == []
        _, reason = ingester.last_skipped[0]
        assert reason == "file is not a valid DOCX (corrupt or truncated)"

    def test_corrupt_xlsx_skip_reason_is_normalized(self, tmp_path: Path):
        pytest.importorskip("openpyxl")
        bad = tmp_path / "broken.xlsx"
        bad.write_bytes(b"not a real xlsx zip")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(bad)
        assert chunks == []
        _, reason = ingester.last_skipped[0]
        assert reason == "file is not a valid XLSX (corrupt or truncated)"

    def test_image_only_pdf_skip_reason_names_scanned_pdf(self, tmp_path: Path):
        # A valid PDF whose pages are images with no text layer (a scanned
        # document) parses fine but yields zero text. The skip reason must
        # name that specific, common case rather than a generic message that
        # also covers unsupported file extensions.
        pytest.importorskip("pypdf")
        pytest.importorskip("PIL")
        pytest.importorskip("reportlab")
        from PIL import Image
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        img_path = tmp_path / "page.png"
        Image.new("RGB", (200, 100), color="white").save(img_path)

        pdf_path = tmp_path / "scanned.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawImage(str(img_path), 72, 600, width=200, height=100)
        c.save()

        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(pdf_path)
        assert chunks == []
        assert len(ingester.last_skipped) == 1
        skipped_path, reason = ingester.last_skipped[0]
        assert "scanned.pdf" in skipped_path
        assert "scanned" in reason.lower() or "image-only" in reason.lower()
        assert "OCR" in reason

    def test_pdf_creation_date_normalized_to_iso8601(self, tmp_path: Path):
        pytest.importorskip("pypdf")
        pdf = tmp_path / "dated.pdf"
        self._write_pdf(pdf, ["Some content."])
        chunks = DocumentIngester(show_progress=False).ingest(pdf)
        date = chunks[0].metadata.get("date")
        assert date is not None
        assert not date.startswith("D:")
        # ISO-8601: YYYY-MM-DDTHH:MM:SS with a numeric or Z offset.
        import re
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z)?$", date)

    def test_docx_creation_date_normalized_to_iso8601(self, tmp_path: Path):
        # The docx `date` uses the same ISO-8601 shape as the pdf and xlsx
        # loaders, so a consumer can parse `metadata["date"]` the same way for
        # every document type.
        docx = pytest.importorskip("docx")
        path = tmp_path / "dated.docx"
        doc = docx.Document()
        doc.add_paragraph("Some content.")
        doc.save(str(path))
        chunks = DocumentIngester(show_progress=False).ingest(path)
        date = chunks[0].metadata.get("date")
        assert date is not None
        import re
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?$", date
        ), date

    def test_duplicate_file_reported_as_duplicate_not_empty(self, tmp_path: Path):
        # A file whose content is an exact duplicate of an already-indexed file
        # is named as a duplicate, distinct from a genuinely empty file, so a
        # corpus audit is not misled into chasing a phantom empty file.
        same = "Exactly the same content across two distinct files here."
        (tmp_path / "a.txt").write_text(same)
        (tmp_path / "b.txt").write_text(same)
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        assert len(chunks) == 1
        assert len(ingester.last_skipped) == 1
        _, reason = ingester.last_skipped[0]
        assert "duplicate" in reason.lower()
        assert "empty" not in reason.lower()

    def test_dedup_reason_absent_when_dedupe_disabled(self, tmp_path: Path):
        same = "Exactly the same content across two distinct files here."
        (tmp_path / "a.txt").write_text(same)
        (tmp_path / "b.txt").write_text(same)
        ingester = DocumentIngester(show_progress=False, dedupe=False)
        chunks = ingester.ingest(tmp_path)
        assert len(chunks) == 2
        assert ingester.last_skipped == []

    def test_ingest_summary_reports_coverage(self, tmp_path: Path):
        (tmp_path / "keep.txt").write_text("indexable content that is real")
        (tmp_path / "skip.xyz").write_text("unsupported")
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        summary = ingester.last_summary
        assert summary is not None
        assert summary.indexed_chunks == len(chunks)
        assert summary.files_seen == 2
        assert summary.skipped_count == 1
        assert summary.indexed_files == 1
        assert summary.elapsed_s >= 0.0
        assert summary.chunks_per_s >= 0.0

    def test_image_skip_reason_points_at_multimodal(self, tmp_path: Path):
        (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        assert chunks == []
        _, reason = ingester.last_skipped[0]
        assert "image" in reason.lower()
        assert "vision" in reason.lower() or "multimodal" in reason.lower()


class TestNormalizePdfDate:
    def test_normalizes_with_timezone(self):
        from effgen.rag.ingest import _normalize_pdf_date

        assert _normalize_pdf_date("D:20260705162102-04'00'") == "2026-07-05T16:21:02-04:00"

    def test_normalizes_without_timezone(self):
        from effgen.rag.ingest import _normalize_pdf_date

        assert _normalize_pdf_date("D:20260705162102") == "2026-07-05T16:21:02"

    def test_leaves_non_matching_value_unchanged(self):
        from effgen.rag.ingest import _normalize_pdf_date

        assert _normalize_pdf_date("not a pdf date") == "not a pdf date"

    def test_empty_string_unchanged(self):
        from effgen.rag.ingest import _normalize_pdf_date

        assert _normalize_pdf_date("") == ""


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------

class TestCodeChunker:
    def test_python_splits_on_functions(self):
        code = (
            "def foo():\n    return 1\n\n"
            "def bar():\n    return 2\n\n"
            "class Baz:\n    def method(self):\n        pass\n"
        ) * 5  # make it big enough to exceed max_chunk_size
        chunker = CodeChunker(language="python", max_chunk_size=100)
        chunks = chunker.chunk(code, "file")
        assert len(chunks) > 1
        assert all(c.metadata.get("language") == "python" for c in chunks)

    def test_unknown_language_fallback(self):
        code = "some random text " * 200
        chunker = CodeChunker(language="cobol", max_chunk_size=100)
        chunks = chunker.chunk(code, "file")
        assert len(chunks) > 1  # fallback chunker still works

    def test_detect_language(self):
        assert CodeChunker.detect_language("foo.py") == "python"
        assert CodeChunker.detect_language("foo.ts") == "typescript"
        assert CodeChunker.detect_language("foo.unknown") is None

    def test_empty_code(self):
        chunker = CodeChunker(language="python")
        chunks = chunker.chunk("", "file")
        # Fallback returns no content chunks for empty input
        assert all(c.content for c in chunks)


class TestTableChunker:
    def test_preserves_table_intact(self):
        text = (
            "Intro paragraph.\n\n"
            "| col1 | col2 |\n"
            "|------|------|\n"
            "| a    | b    |\n"
            "| c    | d    |\n\n"
            "Trailing paragraph."
        )
        chunks = TableChunker(max_chunk_size=500).chunk(text, "doc")
        table_chunks = [c for c in chunks if c.metadata.get("is_table")]
        assert len(table_chunks) == 1
        assert "col1" in table_chunks[0].content
        assert "a    | b" in table_chunks[0].content

    def test_no_table_just_prose(self):
        text = "Just some prose without any tables at all."
        chunks = TableChunker().chunk(text, "doc")
        assert len(chunks) >= 1
        assert not any(c.metadata.get("is_table") for c in chunks)


class TestHierarchicalChunker:
    def test_preserves_heading_hierarchy(self):
        text = (
            "# Top\n\nintro\n\n"
            "## Section A\n\ncontent a\n\n"
            "### Sub A1\n\ncontent a1\n\n"
            "## Section B\n\ncontent b\n"
        )
        chunks = HierarchicalChunker().chunk(text, "doc")
        assert len(chunks) >= 3
        # All chunks should have hierarchy metadata
        for c in chunks:
            assert "hierarchy" in c.metadata
            assert "breadcrumb" in c.metadata
        # Sub A1 chunk should have the full breadcrumb
        sub = [c for c in chunks if c.metadata.get("section") == "Sub A1"]
        assert len(sub) == 1
        assert "Top" in sub[0].metadata["breadcrumb"]
        assert "Section A" in sub[0].metadata["breadcrumb"]

    def test_no_headings_fallback(self):
        text = "plain text with no headings at all " * 50
        chunks = HierarchicalChunker(max_chunk_size=200).chunk(text, "doc")
        assert len(chunks) >= 1


class TestSemanticChunker:
    def test_fallback_when_no_model(self):
        # sentence-transformers may or may not be installed; either way
        # semantic chunker must return chunks.
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = SemanticChunker(max_chunk_size=500).chunk(text, "doc")
        assert len(chunks) >= 1
        assert all(c.content for c in chunks)

    def test_very_short_text(self):
        chunks = SemanticChunker().chunk("One sentence.", "doc")
        assert len(chunks) == 1


class TestChunkerUniformApi:
    """Every chunker takes an optional doc_id and offers split_text()."""

    _TEXT = (
        "First paragraph with a couple of sentences. It carries content.\n\n"
        "Second paragraph with more content. Another sentence follows here."
    )

    @pytest.mark.parametrize(
        "cls",
        [TextChunker, SemanticChunker, CodeChunker, TableChunker, HierarchicalChunker],
    )
    def test_chunk_without_doc_id(self, cls):
        # The natural first call a newcomer makes must not raise TypeError.
        chunks = cls().chunk(self._TEXT)
        assert len(chunks) >= 1
        assert all(c.content for c in chunks)
        assert all(c.id for c in chunks)

    @pytest.mark.parametrize(
        "cls",
        [TextChunker, SemanticChunker, CodeChunker, TableChunker, HierarchicalChunker],
    )
    def test_split_text_returns_strings(self, cls):
        pieces = cls().split_text(self._TEXT)
        assert isinstance(pieces, list)
        assert pieces and all(isinstance(p, str) for p in pieces)


# ---------------------------------------------------------------------------
# HybridSearchEngine
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> list[IngestedChunk]:
    return [
        IngestedChunk(
            id="c1",
            content="Horizontal scaling architecture uses sharding and replication across nodes.",
            source="arch.md",
            metadata={"source": "arch.md", "type": "markdown", "title": "Architecture"},
        ),
        IngestedChunk(
            id="c2",
            content="The cafeteria serves lunch at noon every weekday.",
            source="misc.md",
            metadata={"source": "misc.md", "type": "markdown"},
        ),
        IngestedChunk(
            id="c3",
            content="Architecture decisions favor stateless services for horizontal scaling.",
            source="arch.md",
            metadata={"source": "arch.md", "type": "markdown"},
        ),
        IngestedChunk(
            id="c4",
            content="Python is a popular programming language for data science.",
            source="python.md",
            metadata={"source": "python.md", "type": "markdown"},
        ),
    ]


class TestHybridSearchEngine:
    def test_empty_engine_returns_no_results(self):
        eng = HybridSearchEngine([])
        assert eng.search("anything", top_k=5) == []
        assert len(eng) == 0

    def test_basic_search_returns_results(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks)
        results = eng.search("scaling architecture", top_k=3)
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.relevance_score > 0 for r in results)
        # Architecture chunks should rank above the cafeteria chunk
        top_ids = [r.chunk_id for r in results[:2]]
        assert "c1" in top_ids or "c3" in top_ids
        assert "c2" not in top_ids[:1]  # cafeteria shouldn't be #1

    def test_top_k_respected(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks)
        results = eng.search("anything", top_k=2)
        assert len(results) <= 2

    def test_results_ranked_descending(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks)
        results = eng.search("scaling", top_k=4)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_metadata_filter(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks)
        results = eng.search(
            "scaling architecture",
            top_k=5,
            filter_metadata={"source": "arch.md"},
        )
        assert len(results) >= 1
        assert all(r.source == "arch.md" for r in results)

    def test_metadata_filter_no_matches(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks)
        results = eng.search("scaling", filter_metadata={"source": "nonexistent.md"})
        assert results == []

    def test_min_score_filter(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks)
        # Low threshold: should return results
        lots = eng.search("scaling", top_k=5, min_score=0.01)
        assert len(lots) >= 1
        # Impossibly high threshold: nothing clears it
        none = eng.search("scaling", top_k=5, min_score=10.0)
        assert none == []
        # Threshold actually filters
        mid = eng.search("scaling", top_k=5, min_score=0.5)
        assert all(r.relevance_score >= 0.5 for r in mid)

    def test_add_more_chunks(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks[:2])
        assert len(eng) == 2
        eng.add(sample_chunks[2:])
        assert len(eng) == 4

    def test_reindex(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks)
        eng.index(sample_chunks[:1])
        assert len(eng) == 1

    def test_custom_weights_sparse_only(self, sample_chunks):
        eng = HybridSearchEngine(
            sample_chunks,
            weights={"dense": 0.0, "sparse": 1.0, "keyword": 0.0},
        )
        results = eng.search("scaling", top_k=2)
        assert len(results) >= 1

    def test_custom_weights_keyword_only(self, sample_chunks):
        eng = HybridSearchEngine(
            sample_chunks,
            weights={"dense": 0.0, "sparse": 0.0, "keyword": 1.0},
        )
        results = eng.search("scaling", top_k=2)
        assert len(results) >= 1
        # Keyword-only should find chunks containing "scaling"
        assert any("scaling" in r.content.lower() for r in results)

    def test_accepts_documents(self, sample_chunks):
        """HybridSearchEngine should accept Document objects too."""
        from effgen.tools.builtin.retrieval import Document

        docs = [
            Document(id="d1", content="test content", metadata={"source": "test.md"}),
        ]
        eng = HybridSearchEngine(docs)
        results = eng.search("test", top_k=1)
        assert len(results) == 1
        assert results[0].source == "test.md"

    def test_accepts_dicts(self):
        dicts = [
            {"id": "x1", "content": "alpha content", "source": "x.md", "metadata": {"k": "v"}},
        ]
        eng = HybridSearchEngine(dicts)
        results = eng.search("alpha", top_k=1)
        assert len(results) == 1
        assert results[0].metadata.get("k") == "v"

    def test_unicode_content(self):
        chunks = [
            IngestedChunk(id="u1", content="日本語のテスト文書です", source="jp.md"),
            IngestedChunk(id="u2", content="Résumé with café", source="fr.md"),
        ]
        eng = HybridSearchEngine(chunks)
        results = eng.search("café", top_k=2)
        assert len(results) >= 1

    def test_all_zero_weights_returns_empty_but_no_crash(self, sample_chunks):
        eng = HybridSearchEngine(
            sample_chunks,
            weights={"dense": 0.0, "sparse": 0.0, "keyword": 0.0},
        )
        # Should not crash; may return nothing useful
        results = eng.search("anything", top_k=3)
        assert isinstance(results, list)

    def test_search_result_to_dict(self, sample_chunks):
        eng = HybridSearchEngine(sample_chunks)
        results = eng.search("scaling", top_k=1)
        d = results[0].to_dict()
        assert "chunk_id" in d
        assert "relevance_score" in d
        assert "source" in d


# ---------------------------------------------------------------------------
# Rerankers
# ---------------------------------------------------------------------------

class TestLLMReranker:
    def test_rerank_with_mock_model(self):
        class MockModel:
            def generate(self, prompt, max_tokens=8, temperature=0.0):
                # Extract the Passage: section and score based on it
                passage = prompt.split("Passage:", 1)[-1].split("Rating", 1)[0]
                return "9" if "scaling" in passage.lower() else "2"

        results = [
            SearchResult(chunk_id="1", content="unrelated stuff", source="a"),
            SearchResult(chunk_id="2", content="discussion of scaling", source="b"),
        ]
        ranked = LLMReranker(MockModel()).rerank("scaling", results)
        assert ranked[0].chunk_id == "2"
        assert ranked[0].relevance_score > ranked[1].relevance_score

    def test_rerank_empty(self):
        class MockModel:
            def generate(self, *a, **k):
                return "5"
        assert LLMReranker(MockModel()).rerank("q", []) == []

    def test_rerank_handles_garbage_output(self):
        class BadModel:
            def generate(self, *a, **k):
                return "no number here"

        results = [SearchResult(chunk_id="1", content="x", source="a")]
        ranked = LLMReranker(BadModel()).rerank("q", results)
        assert ranked[0].relevance_score == 0.0

    def test_rerank_top_k(self):
        class MockModel:
            def generate(self, *a, **k):
                return "5"

        results = [
            SearchResult(chunk_id=str(i), content=f"c{i}", source="a")
            for i in range(5)
        ]
        ranked = LLMReranker(MockModel()).rerank("q", results, top_k=2)
        assert len(ranked) == 2


class TestRuleBasedReranker:
    def test_keyword_boost(self):
        results = [
            SearchResult(chunk_id="1", content="nothing relevant", source="a", relevance_score=0.5),
            SearchResult(chunk_id="2", content="scaling architecture here", source="b", relevance_score=0.5),
        ]
        ranked = RuleBasedReranker().rerank("scaling architecture", results)
        assert ranked[0].chunk_id == "2"

    def test_authority_boost(self):
        results = [
            SearchResult(chunk_id="1", content="same", source="random-blog.com", relevance_score=0.5),
            SearchResult(chunk_id="2", content="same", source="official-docs.com", relevance_score=0.5),
        ]
        rr = RuleBasedReranker(authority_map={"official-docs.com": 1.0})
        ranked = rr.rerank("q", results)
        assert ranked[0].chunk_id == "2"

    def test_title_boost(self):
        results = [
            SearchResult(
                chunk_id="1", content="body", source="a", relevance_score=0.5,
                metadata={"title": "Unrelated Post"},
            ),
            SearchResult(
                chunk_id="2", content="body", source="b", relevance_score=0.5,
                metadata={"title": "Scaling Architecture Guide"},
            ),
        ]
        ranked = RuleBasedReranker().rerank("scaling architecture", results)
        assert ranked[0].chunk_id == "2"

    def test_recency_boost(self):
        import time
        now = time.time()
        results = [
            SearchResult(
                chunk_id="old", content="c", source="a", relevance_score=0.5,
                metadata={"timestamp": now - 365 * 86400},  # 1 year old
            ),
            SearchResult(
                chunk_id="new", content="c", source="b", relevance_score=0.5,
                metadata={"timestamp": now},
            ),
        ]
        ranked = RuleBasedReranker(now_ts=now).rerank("q", results)
        assert ranked[0].chunk_id == "new"

    def test_empty_results(self):
        assert RuleBasedReranker().rerank("q", []) == []


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class TestContextBuilder:
    def _mk(self, n=3, long=False):
        return [
            SearchResult(
                chunk_id=f"c{i}",
                content=("x" * 500 if long else f"content {i}"),
                source=f"src_{i}.md",
                relevance_score=1.0 - i * 0.1,
                rank=i + 1,
            )
            for i in range(n)
        ]

    def test_build_empty(self):
        built = ContextBuilder().build([])
        assert built.text == ""
        assert built.citations == []

    def test_token_budget_enforced(self):
        # Each chunk ~125 tokens. Budget for 2.
        results = self._mk(n=5, long=True)
        built = ContextBuilder(max_tokens=260, per_source_limit=0).build(results)
        assert len(built.used_chunks) <= 3
        assert built.total_tokens <= 260 * 1.5  # loose upper bound

    def test_per_source_dedup(self):
        results = [
            SearchResult(chunk_id="a", content="first", source="same.md", relevance_score=0.9),
            SearchResult(chunk_id="b", content="second", source="same.md", relevance_score=0.8),
            SearchResult(chunk_id="c", content="third", source="other.md", relevance_score=0.7),
        ]
        built = ContextBuilder(per_source_limit=1).build(results)
        # Only 2 chunks: one per source
        assert len(built.used_chunks) == 2
        sources = {c.source for c in built.used_chunks}
        assert sources == {"same.md", "other.md"}

    def test_per_source_unlimited(self):
        results = [
            SearchResult(chunk_id="a", content="first", source="same.md", relevance_score=0.9),
            SearchResult(chunk_id="b", content="second", source="same.md", relevance_score=0.8),
        ]
        built = ContextBuilder(per_source_limit=0).build(results)
        assert len(built.used_chunks) == 2

    def test_inline_citations(self):
        built = ContextBuilder(include_citations=True, per_source_limit=0).build(self._mk(2))
        assert "[1]" in built.text
        assert "[2]" in built.text
        assert len(built.citations) == 2
        assert all(isinstance(c, Citation) for c in built.citations)

    def test_no_citations(self):
        built = ContextBuilder(include_citations=False, per_source_limit=0).build(self._mk(2))
        assert built.citations == []
        assert "[1]" not in built.text

    def test_truncation_flag(self):
        # One huge chunk bigger than budget
        big = SearchResult(chunk_id="big", content="x" * 100_000, source="a.md", relevance_score=1.0)
        built = ContextBuilder(max_tokens=100).build([big])
        assert built.truncated

    def test_chronological_order(self):
        results = [
            SearchResult(
                chunk_id="old", content="old content", source="a.md",
                relevance_score=0.5, metadata={"timestamp": 1000},
            ),
            SearchResult(
                chunk_id="new", content="new content", source="b.md",
                relevance_score=0.9, metadata={"timestamp": 2000},
            ),
        ]
        built = ContextBuilder(order="chronological").build(results)
        assert built.used_chunks[0].chunk_id == "old"
        assert built.used_chunks[1].chunk_id == "new"


# ---------------------------------------------------------------------------
# Citation / CitationTracker
# ---------------------------------------------------------------------------

class TestCitation:
    def test_format_numeric(self):
        c = Citation(index=3, source="foo.md")
        assert c.format("numeric") == "[3]"

    def test_format_inline(self):
        c = Citation(index=1, source="foo.pdf", page=5, section="Intro")
        s = c.format("inline")
        assert "foo.pdf" in s and "p.5" in s and "Intro" in s

    def test_format_full(self):
        c = Citation(index=2, source="foo.md", page=10, section="Ch1")
        s = c.format("full")
        assert "[2]" in s and "foo.md" in s and "p.10" in s

    def test_to_dict(self):
        c = Citation(index=1, source="foo.md", relevance_score=0.5)
        d = c.to_dict()
        assert d["index"] == 1
        assert d["source"] == "foo.md"

    def test_relevance_score_default_documented_as_unscored(self):
        # A web-search-derived citation has no reranker score, so it keeps
        # this dataclass default; the docstring must say what 0.0 means here
        # (unscored, not "scored zero relevance") so a caller doesn't misread it.
        assert Citation(index=1, source="https://example.com").relevance_score == 0.0
        doc = Citation.__doc__
        assert "unscored" in doc


class TestCitationTracker:
    def test_add_and_sources(self):
        ct = CitationTracker()
        ct.add(Citation(index=1, source="a.md"))
        ct.add(Citation(index=2, source="b.md"))
        ct.add(Citation(index=3, source="a.md"))  # duplicate source
        assert ct.sources() == ["a.md", "b.md"]

    def test_extract_used_indices(self):
        ct = CitationTracker()
        answer = "This is supported by [1] and also [3], see [1] again."
        assert ct.extract_used_indices(answer) == [1, 3]

    def test_filter_used(self):
        ct = CitationTracker()
        ct.add(Citation(index=1, source="a.md"))
        ct.add(Citation(index=2, source="b.md"))
        ct.add(Citation(index=3, source="c.md"))
        answer = "Per [1] and [3]."
        used = ct.filter_used(answer)
        assert [c.index for c in used] == [1, 3]

    def test_filter_used_no_markers_returns_all(self):
        ct = CitationTracker()
        ct.add(Citation(index=1, source="a.md"))
        answer = "No markers at all."
        assert len(ct.filter_used(answer)) == 1

    def test_verify(self):
        ct = CitationTracker()
        ct.add(Citation(index=1, source="a.md", quote="scaling architecture uses sharding"))
        ct.add(Citation(index=2, source="b.md", quote="cafeteria lunch menu"))
        supporting = ct.verify("scaling architecture", min_overlap=0.5)
        assert len(supporting) == 1
        assert supporting[0].index == 1


# ---------------------------------------------------------------------------
# AgentResponse.citations/sources integration
# ---------------------------------------------------------------------------

class TestAgentResponseCitations:
    def test_has_citations_and_sources_fields(self):
        from effgen.core.agent import AgentResponse

        resp = AgentResponse(
            output="hello",
            citations=[Citation(index=1, source="a.md")],
            sources=["a.md"],
        )
        assert len(resp.citations) == 1
        assert resp.sources == ["a.md"]

    def test_to_dict_includes_citations(self):
        from effgen.core.agent import AgentResponse

        resp = AgentResponse(
            output="hello",
            citations=[Citation(index=1, source="a.md")],
            sources=["a.md"],
        )
        d = resp.to_dict()
        assert "citations" in d
        assert "sources" in d
        assert d["citations"][0]["index"] == 1


# ---------------------------------------------------------------------------
# RAG preset end-to-end
# ---------------------------------------------------------------------------

class TestRagPreset:
    def test_rag_preset_registered(self):
        from effgen.presets import list_presets

        assert "rag" in list_presets()

    def test_create_rag_agent_ingests_knowledge_base(self, tmp_path: Path):
        from effgen.presets import create_agent
        from tests.fixtures.mock_models import MockModel

        (tmp_path / "arch.md").write_text(
            "# Architecture\n\nWe use horizontal sharding for scaling."
        )
        (tmp_path / "notes.txt").write_text("General notes about the project.")

        agent = create_agent("rag", MockModel(responses=["ok"]), knowledge_base=str(tmp_path))
        # Retrieval tool should have been populated
        retrieval = next(
            (t for t in agent.tools.values() if t.metadata.name == "retrieval"),
            None,
        )
        assert retrieval is not None
        assert retrieval.num_documents > 0

    def test_rag_preset_broadens_retrieval_defaults(self, tmp_path: Path):
        # The knowledge-base agent widens retrieval breadth and enables
        # diversity so a multi-topic question keeps distinct passages instead
        # of losing a whole topic to near-duplicate chunks.
        from effgen.presets import create_agent
        from tests.fixtures.mock_models import MockModel

        (tmp_path / "a.txt").write_text("Some indexable content about systems.")
        agent = create_agent("rag", MockModel(responses=["ok"]), knowledge_base=str(tmp_path))
        retrieval = next(
            t for t in agent.tools.values() if t.metadata.name == "retrieval"
        )
        assert retrieval.default_top_k >= 8
        assert retrieval.diversity > 0.0
        specs = {p.name: p for p in retrieval.metadata.parameters}
        assert specs["top_k"].default == retrieval.default_top_k

    def test_create_rag_agent_without_knowledge_base_fails_loud(self):
        # Omitting knowledge_base= entirely must fail the same way an
        # explicit empty one does — never build a retrieval agent with zero
        # documents indexed (see TestRagKnowledgeBase for the mirrored case).
        import pytest

        from effgen.presets import create_agent
        from tests.fixtures.mock_models import MockModel

        with pytest.raises(ValueError, match="knowledge_base"):
            create_agent("rag", MockModel(responses=["ok"]))

    def test_create_rag_agent_missing_kb_dir_fails_loud(self, tmp_path: Path):
        # A knowledge_base path that doesn't exist is a typo, not an empty
        # index: it must raise a clear error rather than silently indexing the
        # literal path string (the silent-no-op trap).
        import pytest

        from effgen.presets import create_agent
        from tests.fixtures.mock_models import MockModel

        with pytest.raises(ValueError, match="looks like a file"):
            create_agent(
                "rag", MockModel(responses=["ok"]),
                knowledge_base=str(tmp_path / "nope"),
            )


# ---------------------------------------------------------------------------
# End-to-end pipeline test (the spec's acceptance test)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_spec_acceptance_test(self, tmp_path: Path):
        """The exact test from the v0.2.0 build spec."""
        # Create a docs/ directory with enough material to get 5 chunks
        for i in range(6):
            (tmp_path / f"doc{i}.md").write_text(
                f"# Document {i}\n\n"
                f"Scaling architecture requires careful thought. "
                f"Document {i} discusses different aspects of horizontal and "
                f"vertical scaling in distributed systems. "
                f"This is chunk number {i} with unique content {i*7}."
            )

        ingester = DocumentIngester(show_progress=False)
        chunks = ingester.ingest(tmp_path)
        assert len(chunks) > 0

        engine = HybridSearchEngine(chunks)
        results = engine.search("scaling architecture", top_k=5)
        assert len(results) == 5
        assert all(r.relevance_score > 0 for r in results)

    def test_full_pipeline_ingest_search_rerank_build(self, tmp_path: Path):
        (tmp_path / "a.md").write_text(
            "# Scaling\n\nHorizontal scaling with sharding is the standard approach."
        )
        (tmp_path / "b.md").write_text(
            "# Cafeteria\n\nLunch is served daily."
        )
        (tmp_path / "c.md").write_text(
            "# Architecture\n\nStateless services enable scaling across many nodes."
        )

        # 1. Ingest
        chunks = DocumentIngester(show_progress=False).ingest(tmp_path)
        assert len(chunks) >= 3

        # 2. Search
        engine = HybridSearchEngine(chunks)
        results = engine.search("scaling architecture", top_k=3)
        assert len(results) >= 1

        # 3. Rerank
        reranker = RuleBasedReranker()
        results = reranker.rerank("scaling architecture", results)

        # 4. Build context
        built = ContextBuilder(max_tokens=500, per_source_limit=0).build(results)
        assert built.text
        assert len(built.citations) >= 1
        assert built.total_tokens > 0

        # 5. Verify attribution works
        tracker = CitationTracker(citations=built.citations)
        assert len(tracker.sources()) >= 1
