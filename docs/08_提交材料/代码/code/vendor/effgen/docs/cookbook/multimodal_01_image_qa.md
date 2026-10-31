# Cookbook 01 — Image Q&A with effGen

Ask natural-language questions about any image - a URL, a local file, or an
in-memory PIL object — using the effGen multimodal agent.

## What you'll learn

- Constructing an `ImagePart` from a URL with `image_from()`.
- Sending a multimodal `Message` directly to a vision-capable model.
- Using the `multimodal` preset for automatic tool dispatch.
- Handling the result from `agent.run()`.

## Prerequisites

```bash
pip install "effgen[all]"
# Vision providers: set at least one of
#   GOOGLE_API_KEY (Gemini 3.x Flash-Lite — free tier)
#   OPENAI_API_KEY (gpt-4o-mini)
```

## Quickstart

```python
"""image_qa.py — Ask a question about an in-memory image.

Run:
    python image_qa.py
"""

import os
from dotenv import load_dotenv

from effgen.core.multimodal import image_from
from effgen.core.messages import Message, Role, TextPart
from effgen.presets import create_agent

load_dotenv()


# ── 1. Load a vision-capable model ─────────────────────────────────────────
if os.getenv("OPENAI_API_KEY"):
    from effgen.models.openai_adapter import OpenAIAdapter
    model = OpenAIAdapter(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
elif os.getenv("GOOGLE_API_KEY"):
    from effgen.models.gemini_adapter import GeminiAdapter
    model = GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
else:
    raise EnvironmentError("Set OPENAI_API_KEY or GOOGLE_API_KEY to run this example.")

model.load()


# ── 2. Build an image part from an in-memory PIL image ──────────────────────
from PIL import Image

image = Image.new("RGB", (128, 128), color=(255, 0, 0))
img_part = image_from(image)          # accepts URL, path, bytes, PIL.Image


# ── 3. Construct a multimodal message ──────────────────────────────────────
question = "What is the dominant color in this image? Answer in one sentence."
msg = Message(
    role=Role.USER,
    content=[img_part, TextPart(text=question)],
)


# ── 4. Option A — Direct model call (no agent, no tools) ───────────────────
result_direct = model.generate([msg])
print("=== Direct model call ===")
print(result_direct.text)


# ── 5. Option B — Agent with multimodal preset (includes tools) ────────────
agent = create_agent("multimodal", model)
result_agent = agent.run(question, inputs=[img_part])

print("\n=== Agent with multimodal preset ===")
print(result_agent.output)
assert result_agent.success, f"Agent failed: {result_agent.output}"
```

## How it works

### `image_from(source)`

The helper accepts any of:

| Source type | Example |
|---|---|
| URL string | `"https://example.com/photo.jpg"` |
| Local file path | `"/tmp/photo.png"` or `Path("/tmp/photo.png")` |
| Raw bytes | `open("photo.jpg", "rb").read()` |
| `PIL.Image` | `Image.open("photo.png")` |
| `numpy.ndarray` | `cv2.imread("photo.png")` |

It returns an `ImagePart` with the bytes, auto-detected MIME type, and size
metadata. MIME validation happens at construction — unsupported formats raise
`InvalidMultimodalContent` immediately.

### Provider routing

`create_agent("multimodal", model)` wires the Gemini-first fallback chain:

```
Gemini 3.x Flash-Lite (vision + audio + video)
  └→ OpenAI gpt-4o-mini (vision + text)
       └→ HF BLIP (vision-only)
```

If the model you pass doesn't support vision, the router raises
`CapabilityNotSupportedError` instead of silently stripping the image.

### Tool dispatch

With the `multimodal` preset the agent has access to:

- `multimodal_describe` — auto-pick between caption / OCR / audio based on
  part type; useful when the user doesn't specify which tool to use.
- `image_caption` — always generate a prose description.
- `ocr` — extract text from image.
- `image_info` — EXIF metadata, format, dimensions.

## Expected output

```
=== Direct model call ===
The image is a solid bright red square.

=== Agent with multimodal preset ===
The image is a plain red square with no other visible objects.
```

## Next steps

- **Cookbook 02** — Audio transcription and sentiment analysis.
- **Cookbook 04** — OCR a document image and extract structured data with a
  prompt template.
- **Cookbook 05** — Read a bar chart and answer comparison questions.
