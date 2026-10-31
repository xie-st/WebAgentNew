# Image Analysis Tools

Two image tools ship with effGen:

| Tool | Network | Auth required |
|------|---------|---------------|
| `ImageInfoTool` | None (local PIL) | No |
| `ImageCaptionTool` | Yes (vision LLM) | Vision provider API key |

Both accept image **file paths**, **raw bytes**, or **base64-encoded strings**.

---

## ImageInfoTool

Extract metadata and perform basic transformations on images.

**Backend:** Pillow only. Zero network, no API keys needed.

### Installation

```bash
pip install "effgen[tools]"
# Pillow is included in the [tools] extras
```

### Operations

| Operation | Description |
|-----------|-------------|
| `info` | Extract width/height/format/mode/EXIF/color stats |
| `resize` | Resize to exact pixel dimensions |
| `thumbnail` | Fit within `max_dim × max_dim` (aspect-ratio preserved) |

### Minimal example

```python
import asyncio
from effgen.tools.builtin.image_info import ImageInfoTool

tool = ImageInfoTool()

async def main():
    # Metadata
    result = await tool._execute(operation="info", image_path="photo.jpg")
    data = result["data"]
    print(f"{data['width']} × {data['height']}  {data['format']}  {data['mode']}")
    print("EXIF:", data["exif"])
    print("Colors:", data["color_stats"])

    # Create a thumbnail (result contains base64-encoded bytes)
    thumb = await tool._execute(operation="thumbnail", image_path="photo.jpg", max_dim=256)
    print(f"Thumbnail: {thumb['data']['width']} × {thumb['data']['height']}")

    # Save a resize to disk
    await tool._execute(
        operation="resize",
        image_path="photo.jpg",
        width=800, height=600,
        dest="photo_800x600.png",
    )

asyncio.run(main())
```

### With raw bytes

```python
with open("photo.jpg", "rb") as f:
    img_bytes = f.read()

result = await tool._execute(operation="info", image_bytes=img_bytes)
```

### Output schema — `info`

```json
{
  "success": true,
  "data": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG",
    "mode": "RGB",
    "n_frames": 1,
    "exif": {
      "Make": "Canon",
      "DateTime": "2024:01:15 12:30:00"
    },
    "color_stats": {
      "mean_r": 128.5,
      "mean_g": 110.2,
      "mean_b": 95.8,
      "stddev_r": 45.1,
      "stddev_g": 42.3,
      "stddev_b": 38.7
    }
  },
  "error": null
}
```

### Output schema — `resize` / `thumbnail`

```json
{
  "success": true,
  "data": {
    "width": 800,
    "height": 600,
    "format": "PNG",
    "saved_to": "/path/to/output.png",
    "bytes_b64": null
  },
  "error": null
}
```

When `dest` is omitted, `saved_to` is `null` and `bytes_b64` contains the base64-encoded image.

---

## ImageCaptionTool

Generate natural-language captions and descriptions for images using a vision LLM.

**Backend:** Auto-selects the cheapest available vision-capable provider:
1. `gemini/gemini-3.1-flash-lite` (set `GOOGLE_API_KEY`)
2. `openai/gpt-4o-mini` (set `OPENAI_API_KEY`)
3. `gemini/gemini-2.5-flash-lite`
4. `openai/gpt-4.1-mini`
5. `gemini/gemini-2.5-flash`

### Installation

```bash
pip install "effgen[tools]"
```

Set at least one provider key:

```bash
export OPENAI_API_KEY="sk-..."
# or
export GOOGLE_API_KEY="AIza..."
```

### Operations

| Operation | Description |
|-----------|-------------|
| `caption` | Concise one-sentence caption |
| `describe` | Detailed multi-sentence description |

### Minimal example

```python
import asyncio
from effgen.tools.builtin.image_caption import ImageCaptionTool

tool = ImageCaptionTool()

async def main():
    # Concise caption
    result = await tool._execute(operation="caption", image_path="photo.jpg")
    print(result["caption"])
    print(f"Provider: {result['provider']}/{result['model']}")

    # Detailed description
    result = await tool._execute(
        operation="describe",
        image_path="photo.jpg",
        detail="high",
    )
    print(result["caption"])

asyncio.run(main())
```

### Custom prompt

```python
result = await tool._execute(
    operation="caption",
    image_path="chart.png",
    prompt="Describe the trend shown in this chart in one sentence.",
)
```

### Force a specific provider

```python
tool = ImageCaptionTool(provider="openai", model_id="gpt-4o")
```

If `provider` is set without `model_id`, effGen uses that provider's default
cheap vision model (`gemini-2.5-flash-lite` for Gemini,
`gpt-4o-mini` for OpenAI).

### With raw bytes

```python
with open("photo.jpg", "rb") as f:
    img_bytes = f.read()

result = await tool._execute(operation="caption", image_bytes=img_bytes)
```

### Output schema

```json
{
  "success": true,
  "data": {
    "caption": "A serene mountain landscape at golden hour.",
    "prompt": "Provide a concise one-sentence caption for this image.",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "detail": "auto"
  },
  "caption": "A serene mountain landscape at golden hour.",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "error": null
}
```

### `NoVisionProviderAvailable` error

When no vision provider key is configured, the tool raises `NoVisionProviderAvailable`:

```python
from effgen.errors import NoVisionProviderAvailable

try:
    result = await tool._execute(operation="caption", image_path="photo.jpg")
except NoVisionProviderAvailable as exc:
    print("No vision provider:", exc)
    # Error message includes configuration instructions
```

---

## Preset integration

### `general` preset

`ImageInfoTool` is included in the `general` preset.

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("gpt-oss-120b", provider="cerebras")
agent = create_agent("general", model)
result = agent.run("What are the dimensions of /path/to/image.jpg?")
```

### `media` preset

Both tools are in the `media` preset (ImageCaptionTool requires a vision key).

```python
agent = create_agent("media", model)
result = agent.run("Caption this image: /path/to/photo.jpg")
```
