"""
PDF parsing tool for the effGen framework.

Primary backend : pypdf (fast text extraction).
Table backend   : pdfplumber (accurate table detection).

Operations
----------
- text           : Extract raw text from all or selected pages.
- metadata       : Extract document metadata (author, title, creation date, etc.).
- tables         : Extract tables from all or selected pages via pdfplumber.
- extract_images : Save embedded images to a destination directory.

Input: file path (str/Path) OR raw PDF bytes for all operations.
Output: structured dicts with success/data/error keys.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from pathlib import Path
from typing import Any

from effgen.errors import CorruptDocumentError

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._fs import confine_path, normalize_allowed_dirs

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_pypdf():
    """Temporarily silence pypdf's warning logger so that corrupt-input probes
    surface a clean CorruptDocumentError rather than noisy stderr from the
    parser's internal recovery attempts."""
    pypdf_logger = logging.getLogger("pypdf")
    prev = pypdf_logger.level
    pypdf_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        pypdf_logger.setLevel(prev)


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _resolve_pdf(source: str | bytes | Path) -> bytes:
    """Return raw PDF bytes from a file path or bytes input."""
    if isinstance(source, str | Path):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {source}")
        return p.read_bytes()
    if isinstance(source, bytes | bytearray):
        return bytes(source)
    raise TypeError(f"Expected str, Path, or bytes; got {type(source)}")


# ---------------------------------------------------------------------------
# pypdf helpers
# ---------------------------------------------------------------------------

def _pdf_text(source: str | bytes | Path, pages: list[int] | None = None) -> dict[str, Any]:
    """Extract text via pypdf."""
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDFTool. Install with: pip install 'effgen[documents]'"
        ) from exc

    raw = _resolve_pdf(source)
    try:
        with _quiet_pypdf():
            reader = pypdf.PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise CorruptDocumentError("pdf", str(exc)) from exc

    total = len(reader.pages)
    page_indices = [p - 1 for p in pages] if pages else list(range(total))
    page_texts: list[dict] = []

    for idx in page_indices:
        if idx < 0 or idx >= total:
            logger.warning("Page %d out of range (total %d); skipping.", idx + 1, total)
            continue
        try:
            text = reader.pages[idx].extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract text from page %d: %s", idx + 1, exc)
            text = ""
        page_texts.append({"page": idx + 1, "text": text})

    full_text = "\n".join(p["text"] for p in page_texts)
    return {
        "success": True,
        "data": {
            "text": full_text,
            "pages": page_texts,
            "total_pages": total,
        },
        "text": full_text,
        "pages": page_texts,
        "total_pages": total,
        "error": None,
    }


def _pdf_metadata(source: str | bytes | Path) -> dict[str, Any]:
    """Extract metadata via pypdf."""
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDFTool. Install with: pip install 'effgen[documents]'"
        ) from exc

    raw = _resolve_pdf(source)
    try:
        with _quiet_pypdf():
            reader = pypdf.PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise CorruptDocumentError("pdf", str(exc)) from exc

    meta = reader.metadata or {}
    info: dict[str, Any] = {
        "total_pages": len(reader.pages),
        "encrypted": reader.is_encrypted,
    }
    for key, val in meta.items():
        clean_key = str(key).lstrip("/")
        info[clean_key] = str(val) if val is not None else None

    return {
        "success": True,
        "data": info,
        "metadata": info,
        "error": None,
    }


def _pdf_tables(source: str | bytes | Path, pages: list[int] | None = None) -> dict[str, Any]:
    """Extract tables via pdfplumber."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for table extraction. "
            "Install with: pip install 'effgen[documents]'"
        ) from exc

    raw = _resolve_pdf(source)
    try:
        with _quiet_pypdf():
            pdf = pdfplumber.open(io.BytesIO(raw))
    except Exception as exc:
        raise CorruptDocumentError("pdf", str(exc)) from exc

    all_tables: list[dict] = []
    with pdf:
        total = len(pdf.pages)
        page_indices = [p - 1 for p in pages] if pages else list(range(total))
        for idx in page_indices:
            if idx < 0 or idx >= total:
                continue
            page_obj = pdf.pages[idx]
            try:
                tables = page_obj.extract_tables() or []
            except Exception as exc:
                logger.warning("Table extraction failed on page %d: %s", idx + 1, exc)
                tables = []
            for t_idx, table in enumerate(tables):
                all_tables.append({
                    "page": idx + 1,
                    "table_index": t_idx,
                    "rows": table,
                    "row_count": len(table),
                    "col_count": max((len(r) for r in table), default=0),
                })

    return {
        "success": True,
        "data": {"tables": all_tables, "table_count": len(all_tables)},
        "tables": all_tables,
        "table_count": len(all_tables),
        "error": None,
    }


def _pdf_extract_images(
    source: str | bytes | Path, dest_dir: str | Path
) -> dict[str, Any]:
    """Extract embedded images from PDF pages."""
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDFTool. Install with: pip install 'effgen[documents]'"
        ) from exc

    raw = _resolve_pdf(source)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise CorruptDocumentError("pdf", str(exc)) from exc

    saved: list[dict] = []
    for page_num, page in enumerate(reader.pages, start=1):
        if "/Resources" not in page:
            continue
        resources = page["/Resources"]
        xobject = resources.get("/XObject")
        if not xobject:
            continue
        for obj_name, obj_ref in xobject.items():
            obj = obj_ref.get_object() if hasattr(obj_ref, "get_object") else obj_ref
            if obj.get("/Subtype") != "/Image":
                continue
            try:
                data = obj.get_data()
                ext = "png"
                color_space = obj.get("/ColorSpace")
                if isinstance(color_space, str) and "CMYK" in color_space:
                    ext = "jpg"
                fname = f"page{page_num}_{str(obj_name).lstrip('/')}.{ext}"
                out_path = dest / fname
                out_path.write_bytes(data)
                saved.append({
                    "page": page_num,
                    "name": obj_name,
                    "path": str(out_path),
                    "size_bytes": len(data),
                })
            except Exception as exc:
                logger.warning("Could not save image %s on page %d: %s", obj_name, page_num, exc)

    return {
        "success": True,
        "data": {"images": saved, "image_count": len(saved)},
        "images": saved,
        "image_count": len(saved),
        "error": None,
    }


# ---------------------------------------------------------------------------
# BaseTool subclass
# ---------------------------------------------------------------------------

class PDFTool(BaseTool):
    """
    Parse PDF documents - extract text, metadata, tables, and images.

    Primary backend : pypdf  (text + metadata + images).
    Table backend   : pdfplumber (table extraction).

    Accepts file paths (str/Path) OR raw PDF bytes.
    Raises CorruptDocumentError on malformed input.
    """

    def __init__(self, allowed_directories: list[str] | None = None) -> None:
        """Args:
            allowed_directories: Roots a file path may be read from. By default
                any path is allowed except protected system and credential
                locations (/etc, /proc, ~/.ssh, cloud creds, …), which are
                always refused. Pass a list to confine reads to those roots only.
        """
        self._allowed_dirs = normalize_allowed_dirs(allowed_directories)
        super().__init__(
            metadata=ToolMetadata(
                name="pdf",
                description=(
                    "Parse PDF documents. Operations: text (extract text), "
                    "metadata (document info), tables (extract tabular data), "
                    "extract_images (save embedded images to a directory). "
                    "Accepts file paths or raw bytes."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'text', 'metadata', 'tables', or 'extract_images'.",
                        required=True,
                        enum=["text", "metadata", "tables", "extract_images"],
                    ),
                    ParameterSpec(
                        name="path",
                        type=ParameterType.STRING,
                        description="Path to the PDF file.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="pages",
                        type=ParameterType.ARRAY,
                        description=(
                            "1-based list of page numbers to process. "
                            "Omit to process all pages."
                        ),
                        required=False,
                        items_type=ParameterType.INTEGER,
                    ),
                    ParameterSpec(
                        name="dest_dir",
                        type=ParameterType.STRING,
                        description="Destination directory for extract_images operation.",
                        required=False,
                    ),
                ],
                timeout_seconds=60,
                tags=["pdf", "document", "parsing", "tables", "text-extraction"],
                examples=[
                    {"operation": "text", "path": "/path/to/doc.pdf"},
                    {"operation": "metadata", "path": "/path/to/doc.pdf"},
                    {"operation": "tables", "path": "/path/to/doc.pdf", "pages": [1, 2]},
                    {
                        "operation": "extract_images",
                        "path": "/path/to/doc.pdf",
                        "dest_dir": "/tmp/pdf_images",
                    },
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        path: str | None = None,
        pdf_bytes: bytes | None = None,
        pages: list[int] | None = None,
        dest_dir: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        source: str | bytes | Path
        if path:
            source = str(confine_path(path, self._allowed_dirs))
        elif pdf_bytes is not None:
            source = pdf_bytes
        else:
            raise ValueError("Provide 'path' or 'pdf_bytes'.")

        op = operation.lower()

        if op == "text":
            return await asyncio.to_thread(_pdf_text, source, pages)
        elif op == "metadata":
            return await asyncio.to_thread(_pdf_metadata, source)
        elif op == "tables":
            return await asyncio.to_thread(_pdf_tables, source, pages)
        elif op == "extract_images":
            if not dest_dir:
                raise ValueError("'dest_dir' is required for extract_images.")
            return await asyncio.to_thread(_pdf_extract_images, source, dest_dir)
        else:
            raise ValueError(
                f"Unknown operation: {operation!r}. "
                "Use 'text', 'metadata', 'tables', or 'extract_images'."
            )
