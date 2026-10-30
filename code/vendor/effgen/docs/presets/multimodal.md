# Multimodal Preset

The `multimodal` preset provides a ready-to-use agent that understands images, audio recordings, and video clips.

## Quick Start

```python
from effgen.presets import create_agent
from effgen.core.multimodal import audio_from, image_from, video_from

agent = create_agent("multimodal", model="gemini-3-flash-lite")

# Image
result = agent.run("Describe what's in this image.", inputs=[image_from("/path/to/photo.jpg")])

# Audio
result = agent.run("Transcribe and summarize this audio.", inputs=[audio_from("/path/to/speech.mp3")])

# Video
result = agent.run("What happens in this clip?", inputs=[video_from("/path/to/clip.mp4")])
```

## Models

| Priority | Provider | Model              | Modalities          |
|----------|----------|--------------------|---------------------|
| Primary  | Gemini   | gemini-3-flash-lite| image, audio, video |
| Fallback | OpenAI   | gpt-4o-mini        | image               |
| Fallback | HF/BLIP  | (vision only)      | image               |

The preset uses whatever model you pass to `create_agent`. Tool-level routing
(inside `ImageCaptionTool`, `AudioTranscribeTool`, etc.) picks the cheapest
available provider automatically.

## Tools Included

| Tool name             | Purpose                                       |
|-----------------------|-----------------------------------------------|
| `multimodal_describe` | Auto-dispatch hub for image / audio / video   |
| `image_caption`       | Vision captioning via best available provider |
| `image_info`          | EXIF, size, format metadata extraction        |
| `ocr`                 | Text extraction from images                   |
| `audio_transcribe`    | Speech-to-text (Whisper / Gemini)             |
| `pdf`                 | PDF parsing (for scanned documents)           |
| `weather`             | Geo / weather queries                         |

## `MultimodalDescribeTool`

The `multimodal_describe` tool is the main entry point for any media file.
It detects the media type from the file extension and routes to the right backend.

```python
from effgen.tools.builtin import MultimodalDescribeTool
import asyncio

tool = MultimodalDescribeTool()

# Image
result = asyncio.run(tool._execute(file_path="/path/to/photo.jpg"))
print(result["description"])

# Audio
result = asyncio.run(tool._execute(
    file_path="/path/to/speech.mp3",
    operation="transcribe",
))
print(result["transcript"])

# Video (requires ffmpeg)
result = asyncio.run(tool._execute(
    file_path="/path/to/clip.mp4",
    prompt="What is happening in this video?",
    max_frames=6,
))
print(result["description"])
```

### Parameters

| Parameter    | Type    | Default     | Description                                                 |
|--------------|---------|-------------|-------------------------------------------------------------|
| `file_path`  | string  | required    | Path to media file                                          |
| `media_type` | string  | `"auto"`    | `"image"`, `"audio"`, `"video"`, or `"auto"` (from ext)    |
| `prompt`     | string  | (per-type)  | Custom instruction for the model                            |
| `operation`  | string  | `"describe"`| `"describe"`, `"transcribe"`, or `"summarize"`              |
| `max_frames` | int     | `6`         | Max video frames to sample (video only)                     |

## Fallback Behaviour

- If the primary Gemini model hits quota or is unavailable, `ImageCaptionTool`
  falls back to OpenAI gpt-4o-mini automatically.
- Video processing requires **ffmpeg** on PATH. If absent, a clear
  `MissingSystemDependency` error is raised with install instructions.
- If no vision provider API key is found, `NoVisionProviderAvailable` is raised.

## System Requirements

| Feature       | Requirement                              |
|---------------|------------------------------------------|
| Image input   | `GOOGLE_API_KEY` or `OPENAI_API_KEY`     |
| Audio input   | `GOOGLE_API_KEY` or `OPENAI_API_KEY`     |
| Video input   | ffmpeg + `GOOGLE_API_KEY`/`OPENAI_API_KEY`|
| OCR           | Tesseract (optional, falls back to cloud) |

Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# conda
conda install -c conda-forge ffmpeg

# macOS
brew install ffmpeg
```

## Custom Configuration

```python
agent = create_agent(
    "multimodal",
    model,
    system_prompt="You are a forensic image analyst. Be precise.",
    max_iterations=12,
    temperature=0.1,
    extra_tools=[my_custom_tool],
)
```
