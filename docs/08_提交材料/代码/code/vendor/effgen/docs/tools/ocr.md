# OCRTool

Extract text from images using OCR (Optical Character Recognition).

**Primary backend:** Tesseract (local, no network, no auth required).  
**Fallback backend:** [OCR.space](https://ocr.space/) cloud API (500 calls/day on the free tier; set `OCR_SPACE_API_KEY`).

Accepts image **file paths**, **raw bytes**, or **base64-encoded image strings** for all operations.

---

## Installation

### System dependency - Tesseract

Tesseract must be installed on your system before `pytesseract` can wrap it.

| OS | Command |
|----|---------|
| **Ubuntu / Debian** | `sudo apt install tesseract-ocr` |
| **macOS (Homebrew)** | `brew install tesseract` |
| **Windows (Chocolatey)** | `choco install tesseract` |
| **conda (any OS)** | `conda install -c conda-forge tesseract` |

Verify: `tesseract --version`

For additional languages:
```bash
# Ubuntu/Debian - e.g. French
sudo apt install tesseract-ocr-fra

# macOS
brew install tesseract-lang
```

### Python dependencies

```bash
pip install "effgen[tools]"
# installs: pytesseract>=0.3.10, Pillow>=12.3.0
```

Or as part of the full install:

```bash
pip install "effgen[all]"
```

### OCR.space fallback (optional)

If Tesseract is not available you can use the free [OCR.space](https://ocr.space/) cloud API:

```bash
export OCR_SPACE_API_KEY="your_key_here"
```

Register for a free key at <https://ocr.space/ocrapi> for 500 calls/day.

---

## Operations

### `extract` - full-image OCR

```python
import asyncio
from effgen.tools.builtin.ocr import OCRTool

tool = OCRTool()

# From a file path
result = asyncio.run(tool._execute(
    operation="extract",
    image_path="/path/to/scan.png",
    lang="eng",       # ISO 639-2/T or Tesseract lang code
    # tesseract_config="--oem 3 --psm 7",  # optional local Tesseract override
))

print(result["text"])        # extracted text
print(result["confidence"])  # average confidence 0.0-1.0
print(result["words"])       # list of word-level dicts
print(result["backend"])     # "tesseract" or "ocr.space"
```

From raw bytes:

```python
raw = Path("/path/to/image.png").read_bytes()

result = asyncio.run(tool._execute(
    operation="extract",
    image_bytes=raw,
))
```

From base64:

```python
import base64

with open("/path/to/image.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

result = asyncio.run(tool._execute(
    operation="extract",
    image_bytes=b64,
))
```

### `extract_regions` - OCR sub-regions of an image

```python
result = asyncio.run(tool._execute(
    operation="extract_regions",
    image_path="/path/to/doc.png",
    regions=[
        {"left": 0,   "top": 0,   "width": 500, "height": 80},   # header
        {"left": 0,   "top": 200, "width": 500, "height": 400},  # body
    ],
    lang="eng",
))

for region in result["regions"]:
    print(f"Region {region['region_index']}: {region['text']!r}")

print(result["combined_text"])  # all regions joined
```

---

## Output schema

```python
{
    "success": True,
    "text": str,          # extracted text
    "confidence": float,  # 0.0-1.0 (None for OCR.space)
    "words": [
        {
            "word": str,
            "confidence": float,   # None for OCR.space
            "left": int,
            "top": int,
            "width": int,
            "height": int,
            "block_num": int,      # Tesseract only
            "line_num": int,       # Tesseract only
        },
        ...
    ],
    "backend": str,       # "tesseract" or "ocr.space"
    "error": None,
}
```

For `extract_regions`:

```python
{
    "success": True,
    "regions": [
        {
            "region_index": int,
            "region": {"left": int, "top": int, "width": int, "height": int},
            "text": str,
            "confidence": float,
            "words": [...],
            "backend": str,
        },
        ...
    ],
    "combined_text": str,
    "error": None,
}
```

---

## Error handling

| Error | When raised |
|-------|------------|
| `OCRBackendUnavailable` | Neither Tesseract nor OCR.space is available |
| `MissingSystemDependency` | Reserved for system-dependency checks in other tools |
| `FileNotFoundError` | `image_path` does not exist |
| `ValueError` | Both/neither `image_path`/`image_bytes` provided; unknown operation |

```python
from effgen.errors import OCRBackendUnavailable, MissingSystemDependency

try:
    result = asyncio.run(tool._execute(operation="extract", image_path="scan.png"))
except OCRBackendUnavailable as e:
    print("No OCR backend available:", e)
except FileNotFoundError:
    print("Image file not found")
```

---

## Language codes

Use Tesseract's 3-letter codes (ISO 639-2):

| Language | Code |
|----------|------|
| English | `eng` |
| French | `fra` |
| German | `deu` |
| Spanish | `spa` |
| Chinese (Simplified) | `chi_sim` |
| Japanese | `jpn` |
| Arabic | `ara` |

Full list: `tesseract --list-langs`

---

## Preset integration

`OCRTool` is included in the **`general`** preset automatically:

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("cerebras/llama-3.3-70b")
agent = create_agent("general", model)
result = agent.run("Extract all text from /tmp/receipt.png")
```

---

## Tips

- For best results, use high-resolution images (300 DPI or higher).
- Increase contrast and denoise images before OCR for low-quality scans.
- Use `extract_regions` to isolate specific areas (e.g., invoice fields, table cells).
- The `backend` parameter can force `"tesseract"` or `"ocr.space"` if you need deterministic routing.
