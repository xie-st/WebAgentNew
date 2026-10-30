# Cookbook 03 — Video Summarisation via Frame Sampling

Summarise a short video clip by sampling keyframes and sending them as a
sequence of images to a vision-capable model.  Works on any platform, even
when the provider does not support native video uploads.

## What you'll learn

- Creating a `VideoPart` from a local file with `video_from()`.
- Using the `multimodal_describe` tool which handles `VideoPart` automatically.
- Gemini native video path vs. frame-sampling fallback (OpenAI / Groq).
- Configuring frame rate and maximum frames via `video_from()` parameters.

## Prerequisites

```bash
pip install "effgen[all]"
# ffmpeg must be on PATH for frame sampling:
#   Ubuntu/Debian: sudo apt-get install ffmpeg
#   macOS:         brew install ffmpeg
# Providers:
#   GOOGLE_API_KEY (Gemini — supports native video inline)
#   OPENAI_API_KEY (gpt-4o-mini — uses frame-sampling fallback)
```

## Quickstart

```python
"""video_summarize.py — Summarise a video clip.

Run:
    python video_summarize.py path/to/clip.mp4
    # or use the built-in fixture:
    python video_summarize.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from effgen.core.multimodal import video_from
from effgen.presets import create_agent

load_dotenv()

# ── 1. Resolve the video file ───────────────────────────────────────────────
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    repo_root = Path(__file__).parent.parent.parent
    video_path = str(repo_root / "tests/fixtures/multimodal/sample_video.mp4")

print(f"Video file: {video_path}")


# ── 2. Load a vision-capable model ─────────────────────────────────────────
if os.getenv("OPENAI_API_KEY"):
    from effgen.models.openai_adapter import OpenAIAdapter
    model = OpenAIAdapter(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif os.getenv("GOOGLE_API_KEY"):
    from effgen.models.gemini_adapter import GeminiAdapter
    model = GeminiAdapter(
        model_name="gemini-3.1-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    raise EnvironmentError("Set OPENAI_API_KEY or GOOGLE_API_KEY.")

model.load()


# ── 3. Build a video part ───────────────────────────────────────────────────
#   fps=1  → one frame per second
#   max_frames=8  → cap at 8 frames regardless of clip length
video_part = video_from(video_path, fps=1, max_frames=8)
print(
    f"Video: {len(video_part.frames)} frame(s) sampled "
    f"at {video_part.fps} fps, MIME={video_part.mime}"
)


# ── 4. Run the agent ────────────────────────────────────────────────────────
agent = create_agent("multimodal", model)

result = agent.run(
    "Describe what happens in this video clip. "
    "What is the main subject and what action or motion can you see? "
    "Use the multimodal_describe tool to process the frames.",
    inputs=[video_part],
)

print("\n=== Agent output ===")
print(result.output)
assert result.success, f"Agent failed: {result.output}"
```

## How it works

### `video_from(source, fps=1, max_frames=16)`

Calls `ffmpeg` (subprocess) to extract frames at the requested rate, then
returns a `VideoPart(frames=[...ImagePart bytes...], fps=fps, mime="image/jpeg")`.

If `ffmpeg` is not on `PATH` a `MissingSystemDependency("ffmpeg", ...)` error
is raised with platform-specific install instructions.

```text
from effgen.core.multimodal import video_from

vp = video_from("clip.mp4", fps=0.5, max_frames=4)
print(len(vp.frames))   # <= 4
print(vp.fps)           # 0.5
```

### Gemini native video path

When the Gemini adapter receives a `VideoPart`, it inlines the frames as
`image/jpeg` parts in the request.  Gemini can reason over them in sequence,
preserving temporal order.

### Frame-sampling fallback (OpenAI / others)

Non-Gemini adapters that support vision receive each frame as a separate
`ImagePart` in the message.  The model sees them left-to-right and infers
motion from the sequence.

### Tip — longer videos

For clips longer than ~30 seconds, consider:

```text
vp = video_from("long_clip.mp4", fps=0.25, max_frames=16)
```

This gives one frame every 4 seconds, keeping the request size manageable.

## Expected output

```
Video file: .../sample_video.mp4
Video: 5 frame(s) sampled at 1.0 fps, MIME=image/jpeg

=== Agent output ===
The video clip shows a close-up of a person typing on a laptop keyboard.
The main action is hands moving across the keys; the background is a
blurred desk surface with soft ambient lighting.
```

## Next steps

- **Cookbook 01** — Image Q&A.
- **Cookbook 04** — OCR a document image and extract structured data.
