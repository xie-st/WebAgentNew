"""Tests for PDFTool - round-trip and error handling."""

from __future__ import annotations

import io

import pytest

from effgen.errors import CorruptDocumentError
from effgen.tools.builtin import PDFTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_pdf(text: str = "Hello effGen PDF") -> bytes:
    """Generate a minimal valid single-page PDF containing *text*."""
    try:
        from reportlab.pdfgen import canvas as rl_canvas

        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf)
        c.drawString(72, 720, text)
        c.save()
        return buf.getvalue()
    except ImportError:
        pass

    # Fallback: hand-craft a minimal PDF with pypdf writer
    try:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except ImportError:
        pass

    # Last resort: return a known-good minimal PDF bytes (no text, just structure)
    # This is a valid 1-page PDF with no content - enough for metadata/structure tests.
    minimal = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
    )
    return minimal


def _make_table_pdf() -> bytes:
    """Generate a PDF containing a simple table that pdfplumber can extract."""
    reportlab = pytest.importorskip("reportlab")
    assert reportlab is not None
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    table = Table(
        [
            ["Item", "Qty", "Total"],
            ["Widget", "2", "20"],
            ["Gadget", "3", "45"],
        ]
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    doc.build(
        [
            Paragraph("EFFGEN_PDF_TABLE_TOKEN_2026", styles["Normal"]),
            Spacer(1, 12),
            table,
        ]
    )
    return buf.getvalue()


@pytest.fixture
def pdf_bytes():
    return _make_simple_pdf("effGen round-trip test PDF 12345")


@pytest.fixture
def pdf_file(tmp_path, pdf_bytes):
    p = tmp_path / "test.pdf"
    p.write_bytes(pdf_bytes)
    return str(p)


@pytest.fixture
def tool():
    return PDFTool()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pdf_tool_metadata_instance():
    t = PDFTool()
    assert t.metadata.name == "pdf"
    assert "pdf" in t.metadata.tags


@pytest.mark.unit
def test_corrupt_pdf_raises(tool):
    """Feeding garbage bytes should raise CorruptDocumentError."""
    with pytest.raises(CorruptDocumentError):
        import asyncio
        asyncio.run(tool._execute("text", path=None, pdf_bytes=b"NOT_A_PDF"))


@pytest.mark.unit
def test_missing_path_raises(tool):
    with pytest.raises(ValueError, match="path"):
        import asyncio
        asyncio.run(tool._execute("text"))


@pytest.mark.unit
def test_unknown_operation_raises(tool, pdf_file):
    with pytest.raises(ValueError, match="Unknown operation"):
        import asyncio
        asyncio.run(tool._execute("bogus", path=pdf_file))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_metadata_from_bytes(tool, pdf_bytes):
    result = await tool._execute("metadata", pdf_bytes=pdf_bytes)
    assert result["success"] is True
    assert "total_pages" in result["metadata"]
    assert result["metadata"]["total_pages"] >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_text_from_file(tool, pdf_file):
    result = await tool._execute("text", path=pdf_file)
    assert result["success"] is True
    assert "text" in result
    assert isinstance(result["text"], str)
    assert result["total_pages"] >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_text_page_filter(tool, pdf_bytes):
    result = await tool._execute("text", pdf_bytes=pdf_bytes, pages=[1])
    assert result["success"] is True
    assert len(result["pages"]) == 1
    assert result["pages"][0]["page"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_tables_from_bytes(tool, pdf_bytes):
    result = await tool._execute("tables", pdf_bytes=pdf_bytes)
    assert result["success"] is True
    assert "tables" in result
    assert isinstance(result["tables"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_tables_extract_rows(tool):
    result = await tool._execute("tables", pdf_bytes=_make_table_pdf())
    assert result["success"] is True
    assert result["table_count"] >= 1

    rows = result["tables"][0]["rows"]
    assert rows[0] == ["Item", "Qty", "Total"]
    assert ["Widget", "2", "20"] in rows
    assert ["Gadget", "3", "45"] in rows


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_extract_images_no_images(tool, pdf_bytes, tmp_path):
    dest = str(tmp_path / "images")
    result = await tool._execute("extract_images", pdf_bytes=pdf_bytes, dest_dir=dest)
    assert result["success"] is True
    assert result["image_count"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_roundtrip_with_text():
    """Generate a PDF with known text, extract and verify."""
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf)
        c.drawString(72, 720, "EFFGEN_PDF_ROUNDTRIP_TOKEN_42")
        c.save()
        pdf_data = buf.getvalue()
    except ImportError:
        pytest.skip("reportlab not installed; skipping text round-trip test")

    tool = PDFTool()
    result = await tool._execute("text", pdf_bytes=pdf_data)
    assert result["success"] is True
    assert "EFFGEN_PDF_ROUNDTRIP_TOKEN_42" in result["text"]
