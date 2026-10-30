# Multimodal Input — Overview

effGen v0.2.8 accepts image, audio, and video as input types. Every part of the stack — message schema, adapter translation, preprocessing, capability gating, and the agent preset — is designed around a single principle: **the adapter translates to the provider's format; your code only speaks effGen**.

---

## Architecture

```
User code
   │
   ▼
image_from() / audio_from() / video_from()
   │   (bytes, path, URL, PIL.Image, np.ndarray)
   ▼
Message(role=Role.USER, content=[ImagePart, TextPart, ...])
   │
   ▼
image_pre / audio_pre / video_pre   ← per-provider preprocessing
   │   (resize, downsample, frame-sample)
   ▼
Adapter.generate(messages)           ← provider-specific translation
   │   (Gemini inline_data, OpenAI image_url, Whisper, ...)
   ▼
Provider API
```

---

## Message Schema

`Message.content` is a typed `List[ContentPart]`:

```python
from effgen.core.messages import (
    Message, Role,
    TextPart, ImagePart, AudioPart, VideoPart,
    ToolCallPart, ToolResultPart,
    ContentPart,
)

# Structured construction
msg = Message(
    role=Role.USER,
    content=[
        ImagePart(image=b"...", mime="image/jpeg"),
        TextPart(text="What is in this image?"),
    ],
)

# Backwards-compatible — still works
msg = Message(role=Role.USER, content="What is in this image?")

# .text property joins all TextParts
print(msg.text)  # "What is in this image?"
```

### ContentPart types

| Type | Fields | Validated |
|------|--------|-----------|
| `TextPart` | `type="text"`, `text: str` | — |
| `ImagePart` | `type="image"`, `image: bytes`, `mime: str`, `meta: dict` | MIME ∈ {png, jpeg, gif, webp} |
| `AudioPart` | `type="audio"`, `audio: bytes`, `mime: str`, `duration_s: float \| None` | MIME ∈ {mp3, wav, flac, ogg, m4a} |
| `VideoPart` | `type="video_frames"`, `frames: List[bytes]`, `fps: float`, `mime: str` | frames non-empty |
| `ToolCallPart` | `type="tool_call"`, `id`, `name`, `arguments` | — |
| `ToolResultPart` | `type="tool_result"`, `id`, `content` | — |

Invalid MIME or empty frames raises `InvalidMultimodalContent`.

---

## Multimodal Helpers

```python
from effgen import image_from, audio_from, video_from

# image_from — bytes, path, URL, PIL.Image, np.ndarray
img = image_from("/tmp/photo.jpg")
img = image_from("https://example.com/image.png")
img = image_from(pil_image)          # PIL.Image.Image
img = image_from(numpy_array)        # np.ndarray HxWxC

# audio_from — bytes, path, URL
aud = audio_from("/tmp/recording.mp3")
aud = audio_from(b"\xff\xfb...")

# video_from — bytes, path, URL; samples keyframes
vid = video_from("/tmp/clip.mp4", fps=1)   # 1 frame/second, max 16 frames
```

---

## Agent API (recommended)

The simplest way to send multimodal input is through an `Agent`. Pass text plus
an `inputs=` list of parts, **or** pass a `Message` / `list[ContentPart]`
directly as the task — the agent extracts the text and routes any image/audio/
video parts through the multimodal path.

```python
from effgen import image_from
from effgen.core.agent import Agent, AgentConfig

agent = Agent(config=AgentConfig(
    name="vision", model="gemini-3.1-flash-lite", provider="gemini",
))

# 1) text + inputs= (the canonical form)
result = agent.run(
    "What single color dominates this image?",
    inputs=[image_from("photo.png")],
)
print(result.output)

# 2) a Message as the task — a bare string in the content list is fine
from effgen.core.messages import Message, Role
msg = Message(role=Role.USER, content=[image_from("photo.png"), "Describe this."])
result = agent.run(msg)

# 3) a list[ContentPart] as the task
result = agent.run([image_from("photo.png"), "What is this?"])
```

`inputs` is an explicit keyword parameter of `run()` (and `run_async()`), so it
shows up in IDE autocomplete and `inspect.signature`. Streaming (`agent.stream`)
is text-only; pass media through `run()` instead.

If a part targets a model without the matching capability, the agent surfaces a
clear `CapabilityNotSupportedError` message (with a suggested model) rather than
silently dropping the input.

## Preprocessing Pipeline

### Image preprocessing (`effgen/multimodal/image_pre.py`)

`prepare(part, provider, model) → ImagePart` enforces:

- **Pixel dimensions** — downscales to provider max (e.g. 2048×2048 for Gemini) using PIL Lanczos.
- **File size** — warns or converts format when over provider byte limit.
- **MIME** — converts PNG ↔ JPEG as needed.
- All steps recorded in `part.meta["preprocessing"]` for observability.

### Audio preprocessing (`effgen/multimodal/audio_pre.py`)

- Downsamples to 16 kHz mono when the provider requires it (OpenAI Whisper).
- Chunks clips longer than provider max duration (25 MB / 30 min) into sequential API calls; concatenates results.

### Video preprocessing (`effgen/multimodal/video_pre.py`)

```python
from effgen.multimodal.video_pre import VideoSource

vs = VideoSource("/tmp/clip.mp4")
frames = vs.sample_frames(fps=1, max_frames=16)   # → List[ImagePart]
audio  = vs.extract_audio()                        # → AudioPart | None
```

Requires `ffmpeg` on `PATH`. Raises `MissingSystemDependency("ffmpeg", install_hint=...)` with per-OS instructions when absent.

---

## Capability Gating

Every adapter checks `Capability.vision` / `Capability.audio_input` / `Capability.video_input` before sending:

```python
from effgen.errors import CapabilityNotSupportedError
from effgen.models.capabilities import Capability

try:
    result = model.generate(messages_with_image)
except CapabilityNotSupportedError as e:
    print(e.capability)   # Capability.vision
    print(e.provider)     # "cerebras"
```

No adapter silently downcasts an image to `"[image not supported]"`. If a model doesn't have the capability, the error is immediate and explicit.

---

## Provider Support Matrix

| Provider | Image | Audio | Video (native) | Video (frames) |
|----------|-------|-------|----------------|----------------|
| Gemini 2.x/3.x | ✅ | ✅ | ✅ | ✅ |
| OpenAI gpt-4o family | ✅ | ✅ (Whisper) | ❌ | ✅ |
| Groq (Llama 4 / 3.2-vision) | ✅ | ❌ | ❌ | ✅ |
| Anthropic (code-only, no live key) | ✅ | ❌ | ❌ | ❌ |
| Together (vision models) | ✅ | ❌ | ❌ | ✅ |
| HuggingFace Inference (BLIP/LLaVA) | ✅ | ✅ (ASR) | ❌ | ✅ |
| Cerebras | ❌ | ❌ | ❌ | ❌ |
| MLX-VLM (Apple Silicon only) | ✅ | ❌ | ❌ | ✅ |

---

## `multimodal` Preset

```python
from effgen import load_model
from effgen.presets import create_agent

model = load_model("gemini-3.1-flash-lite", provider="gemini")
agent = create_agent("multimodal", model)
```

The preset wires:
- **Primary** — Gemini Flash-Lite (vision + audio + video).
- **Fallback** — OpenAI gpt-4o-mini (vision), HF BLIP (vision-only).
- **Tools** — `ImageInfoTool`, `ImageCaptionTool`, `OCRTool`, `AudioTranscribeTool`, `PDFTool`, `WeatherTool`, `MultimodalDescribeTool`.

`MultimodalDescribeTool` inspects the input part type and automatically invokes the right tool — no manual routing needed.

---

## Per-modality Guides

- [Image input](images.md) — per-provider setup, MIME requirements, preprocessing config.
- [Audio input](audio.md) — Whisper, Gemini audio, HF ASR, chunking for long clips.
- [Video input](video.md) — native Gemini path, ffmpeg frame-sampling, audio extraction.

## Cookbook

Five end-to-end walkthroughs in [`docs/cookbook/`](../cookbook/README.md):

1. **Image Q&A** — ask questions about an image with Gemini and OpenAI.
2. **Audio transcribe + reason** — transcribe an audio clip then analyze its content.
3. **Video summarize** — sample keyframes from a video and produce a narrative summary.
4. **OCR + LLM** — extract text from an image then apply a structured prompt template.
5. **Chart reading** — read a bar chart from an image and answer comparison questions.
