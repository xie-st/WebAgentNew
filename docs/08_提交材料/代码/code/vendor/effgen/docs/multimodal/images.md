# Image Input

effGen exposes a unified image input interface across all vision-capable providers.

## Quick Start

```python
from effgen.core.messages import Message, Role, TextPart
from effgen.core.multimodal import image_from
from effgen.models.openai_adapter import OpenAIAdapter

# Load image from any source: bytes, path, URL, PIL.Image, np.ndarray
img = image_from("path/to/photo.jpg")  # or image_from("https://...")

msg = Message(
    role=Role.USER,
    content=[img, TextPart(text="What's in this image?")],
)

adapter = OpenAIAdapter("gpt-4o-mini")
adapter.load()
result = adapter.generate(msg)
print(result.text)
adapter.unload()
```

## Supported Providers

| Provider | Model | Notes |
|---|---|---|
| **Gemini** | gemini-3.1-flash-lite, gemini-2.5-flash, ... | Native vision, inline JPEG/PNG/WEBP/GIF |
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-5 | data-URI base64 inline |
| **Groq** | qwen/qwen3.6-27b | Qwen3.6 27B, text and image input |
| **Together** | nim/meta/llama-3.2-90b-vision-instruct | Requires dedicated endpoint |
| **HF** | Qwen/Qwen3-VL-8B-Instruct, ... | Via HF Router, paid tier |
| **Anthropic** | claude-sonnet-4-6, claude-opus-4-7 | Base64 media blocks |

## image_from() Sources

```python
from effgen.core.multimodal import image_from

# From bytes
img = image_from(raw_bytes)

# From local path
img = image_from("/path/to/image.png")

# From URL
img = image_from("https://example.com/photo.jpg")

# From PIL.Image
from PIL import Image
pil_img = Image.open("photo.png")
img = image_from(pil_img)

# From numpy array (requires Pillow)
import numpy as np
arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
img = image_from(arr)
```

## Preprocessor

`effgen.multimodal.image_pre.prepare(part, provider, model)` automatically:

1. Converts unsupported MIME types (e.g. GIF → JPEG for Groq)
2. Resizes images that exceed provider pixel limits (Lanczos downscale)
3. Re-compresses oversized images to fit byte limits
4. Records all actions in `part.meta["preprocessing"]` for observability

This is called automatically by all adapters before sending to the API.

### Provider Constraints

| Provider | Max bytes | Max pixels | Allowed MIMEs |
|---|---|---|---|
| Gemini | 20 MB | 3072 px | JPEG, PNG, WEBP, GIF |
| OpenAI | 20 MB | 2000 px | JPEG, PNG, WEBP, GIF |
| Groq | 4 MB | 1568 px | JPEG, PNG, WEBP |
| Together | 8 MB | 2048 px | JPEG, PNG, WEBP |
| HF | 10 MB | 2048 px | JPEG, PNG, WEBP |
| Anthropic | 5 MB | 1568 px | JPEG, PNG, WEBP, GIF |

## Capability Gating

If you pass an image to a model that doesn't support vision, effGen raises
`CapabilityNotSupportedError` rather than silently downgrading:

```python
from effgen.errors import CapabilityNotSupportedError

try:
    adapter.generate(image_message)
except CapabilityNotSupportedError as e:
    print(e)  # "Capability 'vision' is not supported by provider 'groq'"
```

## Multi-turn Image Conversations

Pass a `list[Message]` for multi-turn conversations with images:

```python
history = [
    Message(role=Role.USER, content=[img, TextPart(text="Describe this image.")]),
    Message(role=Role.ASSISTANT, content=[TextPart(text="It shows an ant on stone.")]),
    Message(role=Role.USER, content=[TextPart(text="What colour is the ant?")]),
]
result = adapter.generate(history)
```
