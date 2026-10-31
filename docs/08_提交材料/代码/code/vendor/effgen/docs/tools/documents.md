# Document Parsing Tools

effGen ships three document-parsing tools that work entirely locally - no network calls, no API keys required.

| Tool | Backends | Formats |
|------|----------|---------|
| `PDFTool` | pypdf (text/metadata/images) + pdfplumber (tables) | `.pdf` |
| `DOCXTool` | python-docx | `.docx` |
| `ExcelTool` | openpyxl + pandas | `.xlsx` |

All three tools accept **file paths** (str/Path) **or raw bytes**, and raise `CorruptDocumentError` for malformed input.

---

## Installation

```bash
pip install "effgen[documents]"
```

This installs: `pypdf`, `pdfplumber`, `python-docx`, `openpyxl`, `pandas`.

---

## PDFTool

### Operations

| Operation | Description |
|-----------|-------------|
| `text` | Extract raw text, optionally filtered to specific pages |
| `metadata` | Document properties (title, author, page count, encryption status) |
| `tables` | Extract tabular data using pdfplumber |
| `extract_images` | Save embedded raster images to a destination directory |

### Quick start

```python
import asyncio
from effgen.tools.builtin import PDFTool

tool = PDFTool()

# Extract text from all pages
result = asyncio.run(tool._execute("text", path="report.pdf"))
print(result["text"])

# Extract text from pages 1 and 3 only
result = asyncio.run(tool._execute("text", path="report.pdf", pages=[1, 3]))

# Document metadata
meta = asyncio.run(tool._execute("metadata", path="report.pdf"))
print(meta["metadata"])
# {"total_pages": 10, "encrypted": False, "Title": "Annual Report", ...}

# Table extraction
tables = asyncio.run(tool._execute("tables", path="report.pdf"))
for tbl in tables["tables"]:
    print(f"Page {tbl['page']}: {tbl['row_count']} rows x {tbl['col_count']} cols")

# Extract images
imgs = asyncio.run(tool._execute(
    "extract_images", path="report.pdf", dest_dir="/tmp/pdf_images"
))
print(f"Extracted {imgs['image_count']} images")
```

### From bytes

```python
pdf_bytes = open("report.pdf", "rb").read()
result = asyncio.run(tool._execute("text", pdf_bytes=pdf_bytes))
```

### Error handling

```python
from effgen.errors import CorruptDocumentError

try:
    result = asyncio.run(tool._execute("text", path="broken.pdf"))
except CorruptDocumentError as e:
    print(f"Bad PDF ({e.doc_type}): {e.detail}")
```

---

## DOCXTool

### Operations

| Operation | Description |
|-----------|-------------|
| `text` | Full document text as a single string |
| `paragraphs` | Structured list with text, style, bold/italic flags per paragraph |
| `tables` | All tables as list-of-rows (list of list of strings) |
| `metadata` | Core document properties (author, title, subject, keywords, dates) |

### Quick start

```python
import asyncio
from effgen.tools.builtin import DOCXTool

tool = DOCXTool()

# Full text
result = asyncio.run(tool._execute("text", path="document.docx"))
print(result["text"])

# Structured paragraphs
paras = asyncio.run(tool._execute("paragraphs", path="document.docx"))
for p in paras["paragraphs"]:
    print(f"[{p['style']}] {p['text']}")

# Tables
tables = asyncio.run(tool._execute("tables", path="document.docx"))
for tbl in tables["tables"]:
    for row in tbl["rows"]:
        print(" | ".join(row))

# Metadata
meta = asyncio.run(tool._execute("metadata", path="document.docx"))
print(meta["metadata"]["author"], meta["metadata"]["created"])
```

---

## ExcelTool

### Operations

| Operation | Description |
|-----------|-------------|
| `sheets` | List all sheet names |
| `read_sheet` | Read a sheet as raw rows or pandas-style records |
| `headers` | Return the first row (column headers) of a sheet |
| `metadata` | Workbook properties (author, title, sheet count, dates) |

### Quick start

```python
import asyncio
from effgen.tools.builtin import ExcelTool

tool = ExcelTool()

# List sheets
result = asyncio.run(tool._execute("sheets", path="data.xlsx"))
print(result["sheets"])  # ["Sheet1", "Revenue", "Expenses"]

# Read sheet as raw rows
data = asyncio.run(tool._execute("read_sheet", path="data.xlsx", sheet_name="Sheet1"))
for row in data["rows"]:
    print(row)

# Read sheet as pandas-style records (requires pandas)
data = asyncio.run(tool._execute(
    "read_sheet", path="data.xlsx", sheet_name="Sheet1", as_dataframe=True
))
for record in data["records"]:
    print(record)  # {"Name": "Alice", "Score": 95, ...}

# Column headers
headers = asyncio.run(tool._execute("headers", path="data.xlsx"))
print(headers["headers"])  # ["Name", "Score", "Grade"]

# Metadata
meta = asyncio.run(tool._execute("metadata", path="data.xlsx"))
print(meta["metadata"])
```

### .xls files

Legacy `.xls` files are **not supported** by openpyxl. Convert to `.xlsx` first:

```bash
# LibreOffice CLI
libreoffice --headless --convert-to xlsx old_file.xls

# Python (requires xlrd + openpyxl)
# pip install xlrd openpyxl
import xlrd, openpyxl
wb_old = xlrd.open_workbook("old.xls")
wb_new = openpyxl.Workbook()
# ... copy sheets manually or use a migration tool
```

---

## Using with presets

`PDFTool`, `DOCXTool`, and `ExcelTool` are included in the **`research`** and **`general`** presets:

```python
from effgen.presets import create_agent

agent = create_agent("research", model="gpt-4o-mini")
result = agent.run("Extract the key findings from /path/to/paper.pdf")
```

---

## Output schema

All operations return a consistent dict:

```python
{
    "success": True,
    "data": { ... },   # operation-specific structured data
    # top-level convenience aliases (same as data fields)
    "error": None,
}
```

On failure, `CorruptDocumentError` (or `FileNotFoundError` / `ValueError`) is raised - the tool never silently returns `success: False` for hard errors.
