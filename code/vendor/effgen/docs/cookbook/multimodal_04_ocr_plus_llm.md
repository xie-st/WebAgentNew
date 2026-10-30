# Cookbook 04 — OCR + LLM Structured Extraction

Extract machine-readable data from a scanned document image:
first run OCR to get raw text, then pass the text through the
`legal.contract_summarize.v1` prompt template to produce a structured JSON
summary.

## What you'll learn

- Using the `ocr` tool to pull text out of an image.
- Chaining OCR output into a prompt template from the effGen prompt library.
- Getting a structured JSON response without a schema framework.
- The two-step agent pattern: tool call → reasoning call.

## Prerequisites

```bash
pip install "effgen[all]"
# OCR engine (choose one):
#   Cloud OCR (default) — needs GOOGLE_API_KEY or OPENAI_API_KEY
#   Local Tesseract — sudo apt-get install tesseract-ocr  (Linux)
#                   — brew install tesseract              (macOS)
# LLM for extraction:
#   GOOGLE_API_KEY or OPENAI_API_KEY
```

## Quickstart

```python
"""ocr_plus_llm.py — OCR a document image, then extract structured data.

Run:
    python ocr_plus_llm.py path/to/document.png
    # without an argument, a synthetic PNG is generated on-the-fly.
"""

import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

from effgen.core.multimodal import image_from
from effgen.presets import create_agent

load_dotenv()


# ── 1. Resolve the document image ──────────────────────────────────────────
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Generate a minimal synthetic "contract" image for demonstration
    from PIL import Image, ImageDraw, ImageFont
    import tempfile

    img = Image.new("RGB", (640, 320), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    contract_text = [
        "SERVICE AGREEMENT",
        "",
        "Parties: Acme Corp. (Provider) and Beta LLC (Client).",
        "Effective Date: 2026-06-01",
        "Termination: Either party may terminate with 30 days written notice.",
        "Payment Terms: $5,000/month, due on the 1st.",
        "Governing Law: State of California.",
        "Confidentiality: Both parties agree to keep terms confidential.",
    ]
    y = 20
    for line in contract_text:
        draw.text((20, y), line, fill=(0, 0, 0), font=font)
        y += 32

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    image_path = tmp.name
    print(f"Generated synthetic contract image: {image_path}")

print(f"Document image: {image_path}")


# ── 2. Load a vision-capable model ─────────────────────────────────────────
if os.getenv("OPENAI_API_KEY"):
    from effgen.models.openai_adapter import OpenAIAdapter
    model = OpenAIAdapter(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
elif os.getenv("GOOGLE_API_KEY"):
    from effgen.models.gemini_adapter import GeminiAdapter
    model = GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
else:
    raise EnvironmentError("Set OPENAI_API_KEY or GOOGLE_API_KEY.")

model.load()


# ── 3. Build an image part ──────────────────────────────────────────────────
img_part = image_from(image_path)


# ── 4. Step 1 — OCR: extract raw text ──────────────────────────────────────
agent = create_agent("multimodal", model)

ocr_result = agent.run(
    "Use the ocr tool to extract all text from this document image. "
    "Return only the raw extracted text, no commentary.",
    inputs=[img_part],
)

assert ocr_result.success, f"OCR step failed: {ocr_result.output}"
raw_text = ocr_result.output
print("\n=== Raw OCR output ===")
print(raw_text)


# ── 5. Step 2 — Structured extraction via prompt template ──────────────────
from effgen.prompts.library import registry

template = registry.get("legal.contract_summarize.v1")

# Render the prompt with the OCR'd text
prompt_text = template.render(contract_text=raw_text)

# Direct model call — no tools needed for pure text extraction
from effgen.core.messages import Message, Role, TextPart
extraction_result = model.generate(
    [Message(role=Role.USER, content=[TextPart(text=prompt_text)])],
)

print("\n=== Structured extraction ===")
print(extraction_result.text)

# Parse and validate the JSON block if the model wrapped it
json_match = re.search(r"\{[\s\S]+\}", extraction_result.text)
assert json_match, f"Expected JSON object, got: {extraction_result.text}"
parsed = json.loads(json_match.group())
expected_keys = {"parties", "term", "obligations", "termination", "risks"}
missing_keys = expected_keys - set(parsed)
assert not missing_keys, f"Missing expected keys: {sorted(missing_keys)}"

print("\n=== Parsed JSON ===")
print(json.dumps(parsed, indent=2))
```

## How it works

### Step 1 — OCR via the `ocr` tool

The `multimodal` preset includes the `ocr` tool which:

1. Converts `ImagePart` bytes to a format accepted by the OCR engine.
2. For cloud providers (Gemini / OpenAI), sends the image and asks the model
   to return only the visible text.
3. For local Tesseract (if `pytesseract` is installed), calls the binary
   directly and returns the raw string.

```text
from effgen.tools.builtin.ocr import OCRTool

ocr = OCRTool()
text = ocr.run(image=img_part)
```

### Step 2 — Prompt template rendering

The `legal.contract_summarize.v1` template is a structured prompt that asks
the model to return JSON with keys like `parties`, `term`, `obligations`,
`termination`, and `risks`.

```text
template = TemplateManager().get("legal.contract_summarize.v1")
prompt_text = template.render(contract_text=raw_ocr_text)
```

### Why two steps?

- **Accuracy**: OCR tools are optimised for text extraction; LLMs are optimised
  for reasoning.  Separating the steps avoids asking the model to do both at
  once, which often leads to hallucinated text.
- **Reuse**: The extracted text can be saved and re-processed with different
  prompt templates without re-running OCR.

## Expected output

```
=== Raw OCR output ===
SERVICE AGREEMENT
Parties: Acme Corp. (Provider) and Beta LLC (Client).
Effective Date: 2026-06-01
...

=== Structured extraction ===
{
  "title": "Service Agreement",
  "parties": ["Acme Corp. (Provider)", "Beta LLC (Client)"],
  "effective_date": "2026-06-01",
  "payment_terms": "$5,000/month, due on the 1st",
  "termination_clause": "Either party may terminate with 30 days written notice",
  "governing_law": "State of California",
  "confidentiality": true
}
```

## Next steps

- **Cookbook 05** — Read a bar chart and answer comparison questions.
- **Cookbook 01** — General image Q&A without the two-step approach.
