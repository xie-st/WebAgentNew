# Video Input — effGen Multimodal

effGen supports video input via two strategies depending on the provider:

- **Native video** (Gemini 2.x/3.x): upload the raw MP4/WebM via the Gemini Files API and send a URI reference.
- **Frame-sampling fallback** (all other providers): sample JPEG keyframes at a chosen FPS and send them as a sequence of images in a single message.

Both strategies use the same `VideoPart` message schema — the adapter chooses the path automatically.

---

## Quick Start

### Frame-sampling (any vision provider)

```python
from effgen.core.messages import Message, Role, TextPart, VideoPart
from effgen.multimodal.video_pre import VideoSource
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.base import GenerationConfig

# Sample 5 frames at 1 fps from a local file
vs = VideoSource("clip.mp4")
frames = vs.sample_frames(fps=1.0, max_frames=5)

# Build a multimodal message
vp = VideoPart(frames=[f.image for f in frames], fps=1.0, mime="image/jpeg")
msg = Message(
    role=Role.USER,
    content=[vp, TextPart(text="Describe the main action in this video.")],
)

adapter = OpenAIAdapter(model_name="gpt-4o-mini")
adapter.load()
result = adapter.generate(msg, config=GenerationConfig(max_tokens=512))
print(result.text)
adapter.unload()
```

### Native video via `VideoPart.meta`

For Gemini vision+audio models with `supports_video=True`, you can attach a
local path or URL to a `VideoPart` via the `meta` dict — the adapter will
upload the source to the Gemini Files API automatically (and fall back to
frame-sampling if the upload fails):

```python
from effgen.core.messages import Message, Role, TextPart, VideoPart
from effgen.multimodal.video_pre import VideoSource
from effgen.models.gemini_adapter import GeminiAdapter

# We still need *some* frames to satisfy the VideoPart schema, but they're
# only used as the fallback. The adapter will prefer the native upload.
vs = VideoSource("clip.mp4")
frames = vs.sample_frames(fps=1.0, max_frames=3)
vp = VideoPart(
    frames=[f.image for f in frames],
    fps=1.0,
    mime="image/jpeg",
    meta={"video_path": "clip.mp4", "video_mime": "video/mp4"},
)
msg = Message(role=Role.USER, content=[vp, TextPart(text="Describe.")])

adapter = GeminiAdapter(model_name="gemini-3.1-flash-lite")
adapter.load()
print(adapter.generate(msg).text)
adapter.unload()
```

### Native video via Gemini Files API

```python
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.models.gemini_files import upload_file
from effgen.models.base import GenerationConfig
from effgen.core.messages import TextPart

api_key = "YOUR_GOOGLE_API_KEY"

# Upload the video — video/audio uploads auto-wait for the ACTIVE state.
fref = upload_file("clip.mp4", api_key=api_key, mime_type="video/mp4")

# Query with the uploaded video
adapter = GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key=api_key)
adapter.load()
result = adapter.generate(
    [TextPart(text="Describe the main action.")],
    files=[fref],
)
print(result.text)
adapter.unload()
```

---

## VideoSource

`effgen.multimodal.video_pre.VideoSource` wraps any video source and handles temp-file lifecycle.

```python
from effgen.multimodal.video_pre import VideoSource

vs = VideoSource(
    source,        # str path | Path | bytes | http(s) URL
    mime_type=None # override inferred MIME (default: video/mp4)
)
frames    = vs.sample_frames(fps=1.0, max_frames=16)  # → List[ImagePart]
audio     = vs.extract_audio(target_mime="audio/mp3") # → AudioPart | None
duration  = vs.duration_seconds()                     # → float | None
file_ref  = vs.to_gemini_file_ref(api_key=None)       # → FileRef
vs.cleanup()  # remove temp files (auto-called on GC)
```

### Convenience functions

```python
from effgen.multimodal.video_pre import sample_frames, extract_audio

frames = sample_frames("clip.mp4", fps=2.0, max_frames=8)
audio  = extract_audio("clip.mp4", target_mime="audio/mp3")
```

---

## Accepted Source Types

| Source | Example |
|--------|---------|
| Local file path (str) | `"clip.mp4"` |
| Local file path (Path) | `Path("clip.mp4")` |
| Raw bytes | `open("clip.mp4","rb").read()` |
| HTTP(S) URL | `"https://example.com/clip.mp4"` |

---

## Provider Support

| Provider | Strategy | Capability |
|----------|----------|------------|
| Gemini 2.x/3.x | Native video (Files API) + frame-sampling fallback | `Capability.video_input` |
| OpenAI (gpt-4o family) | Frame-sampling (sends as image_url sequence) | `Capability.video_input` |
| Groq | Frame-sampling (vision models only) | via vision |
| Together | Frame-sampling (vision models only) | via vision |
| HF Inference | Frame-sampling (vision models only) | via vision |
| Anthropic | Not supported — raises `CapabilityNotSupportedError` | — |

---

## Error Handling

```python
from effgen.errors import MissingSystemDependency, CapabilityNotSupportedError

try:
    frames = vs.sample_frames()
except MissingSystemDependency as e:
    print(e)  # includes install instructions: apt/brew/conda install ffmpeg

try:
    result = adapter.generate(video_message)
except CapabilityNotSupportedError as e:
    print(e)  # e.g. "Capability 'video_input' is not supported by provider 'anthropic'"
```

---

## Frame Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fps` | `1.0` | Frames per second to extract |
| `max_frames` | `16` | Hard cap on total frames |

**Recommended values:**
- Short clips (< 30 s): `fps=1.0`, `max_frames=8`
- Longer clips: `fps=0.5`, `max_frames=16`
- High detail: `fps=2.0`, `max_frames=16`

---

## Audio Extraction

Combine video understanding with audio transcription:

```python
vs = VideoSource("interview.mp4")
frames = vs.sample_frames(fps=0.5, max_frames=8)
audio  = vs.extract_audio()  # AudioPart or None

# Transcribe audio via OpenAI Whisper
from effgen.models.openai_adapter import OpenAIAdapter
oai = OpenAIAdapter()
oai.load()
transcript = oai.transcribe_audio(audio.audio, mime=audio.mime)
print("Transcript:", transcript)
```

---

## ffmpeg Requirement

`video_pre` requires **ffmpeg** on PATH. It is used for:
- Frame extraction (`sample_frames`)
- Audio extraction (`extract_audio`)
- Duration probing (`duration_seconds`, uses ffprobe)

Install:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# conda
conda install -c conda-forge ffmpeg

# Windows
choco install ffmpeg
```

A missing ffmpeg raises `MissingSystemDependency("ffmpeg", <install_hint>)` — never a silent failure.
