"""Tests for ExcelTool - round-trip and error handling."""

from __future__ import annotations

import io

import pytest

from effgen.errors import CorruptDocumentError
from effgen.tools.builtin import ExcelTool

# openpyxl is an optional dependency (the ``tools-docs`` extra); these
# round-trip tests build real .xlsx fixtures, so skip cleanly when it is absent
# (e.g. the lean ``[dev]`` install used by the offline CI lane).
pytest.importorskip("openpyxl", reason="openpyxl not installed (pip install effgen[tools-docs])")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_xlsx(
    sheet_data: dict[str, list[list]] | None = None,
    author: str = "effGen Test",
    title: str = "effGen XLSX Round-trip",
) -> bytes:
    """Create an in-memory XLSX workbook with given sheet data."""
    import openpyxl

    wb = openpyxl.Workbook()
    wb.properties.creator = author
    wb.properties.title = title

    if sheet_data is None:
        sheet_data = {
            "Sheet1": [
                ["Name", "Score", "Grade"],
                ["Alice", 95, "A"],
                ["Bob", 82, "B"],
                ["XLSX_TOKEN_77", 100, "A+"],
            ]
        }

    first = True
    for sheet_name, rows in sheet_data.items():
        if first:
            ws = wb.active
            ws.title = sheet_name
            first = False
        else:
            ws = wb.create_sheet(sheet_name)
        for row in rows:
            ws.append(row)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def xlsx_bytes():
    return _make_xlsx()


@pytest.fixture
def multi_sheet_xlsx():
    return _make_xlsx(
        sheet_data={
            "Sales": [["Item", "Qty"], ["Widget", 10], ["Gadget", 5]],
            "Inventory": [["Part", "Stock"], ["X1", 100], ["X2", 200]],
        }
    )


@pytest.fixture
def xlsx_file(tmp_path, xlsx_bytes):
    p = tmp_path / "test.xlsx"
    p.write_bytes(xlsx_bytes)
    return str(p)


@pytest.fixture
def tool():
    return ExcelTool()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_excel_tool_metadata():
    t = ExcelTool()
    assert t.metadata.name == "excel"
    assert "excel" in t.metadata.tags


@pytest.mark.unit
def test_corrupt_xlsx_raises(tool):
    with pytest.raises(CorruptDocumentError):
        import asyncio
        asyncio.run(tool._execute("sheets", xlsx_bytes=b"NOT_AN_XLSX"))


@pytest.mark.unit
def test_missing_path_raises(tool):
    with pytest.raises(ValueError, match="path"):
        import asyncio
        asyncio.run(tool._execute("sheets"))


@pytest.mark.unit
def test_unknown_operation_raises(tool, xlsx_file):
    with pytest.raises(ValueError, match="Unknown operation"):
        import asyncio
        asyncio.run(tool._execute("bogus", path=xlsx_file))


@pytest.mark.unit
def test_xls_path_rejected(tool, tmp_path):
    p = tmp_path / "old.xls"
    p.write_bytes(b"dummy")
    with pytest.raises(ValueError, match=".xls"):
        import asyncio
        asyncio.run(tool._execute("sheets", path=str(p)))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_excel_sheets(tool, xlsx_bytes):
    result = await tool._execute("sheets", xlsx_bytes=xlsx_bytes)
    assert result["success"] is True
    assert "Sheet1" in result["sheets"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_excel_headers_roundtrip(tool, xlsx_bytes):
    result = await tool._execute("headers", xlsx_bytes=xlsx_bytes, sheet_name="Sheet1")
    assert result["success"] is True
    assert result["headers"] == ["Name", "Score", "Grade"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_excel_read_sheet_raw(tool, xlsx_bytes):
    result = await tool._execute("read_sheet", xlsx_bytes=xlsx_bytes, sheet_name="Sheet1")
    assert result["success"] is True
    all_values = [str(cell) for row in result["rows"] for cell in row if cell is not None]
    assert "XLSX_TOKEN_77" in all_values
    assert result["row_count"] == 4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_excel_read_sheet_as_dataframe(tool, xlsx_bytes):
    result = await tool._execute(
        "read_sheet", xlsx_bytes=xlsx_bytes, sheet_name="Sheet1", as_dataframe=True
    )
    assert result["success"] is True
    assert "records" in result
    assert len(result["records"]) == 3
    names = [r["Name"] for r in result["records"]]
    assert "XLSX_TOKEN_77" in names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_excel_metadata(tool, xlsx_bytes):
    result = await tool._execute("metadata", xlsx_bytes=xlsx_bytes)
    assert result["success"] is True
    assert result["metadata"]["author"] == "effGen Test"
    assert result["metadata"]["title"] == "effGen XLSX Round-trip"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_excel_multi_sheet(tool, multi_sheet_xlsx):
    sheets_result = await tool._execute("sheets", xlsx_bytes=multi_sheet_xlsx)
    assert set(sheets_result["sheets"]) == {"Sales", "Inventory"}

    inv_result = await tool._execute(
        "read_sheet", xlsx_bytes=multi_sheet_xlsx, sheet_name="Inventory"
    )
    assert inv_result["success"] is True
    all_vals = [str(c) for row in inv_result["rows"] for c in row if c is not None]
    assert "X1" in all_vals


@pytest.mark.integration
@pytest.mark.asyncio
async def test_excel_from_file(tool, xlsx_file):
    result = await tool._execute("sheets", path=xlsx_file)
    assert result["success"] is True
    assert len(result["sheets"]) >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_excel_missing_sheet_raises(tool, xlsx_bytes):
    with pytest.raises(ValueError, match="not found"):
        await tool._execute("read_sheet", xlsx_bytes=xlsx_bytes, sheet_name="NoSuchSheet")
