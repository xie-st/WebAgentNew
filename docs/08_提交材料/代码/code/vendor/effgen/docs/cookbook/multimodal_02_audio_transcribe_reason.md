# Cookbook 02 — Audio Transcription + Reasoning

Transcribe a speech recording and then reason over the transcript — find
sentiment, extract key points, or answer follow-up questions — all in a
single agent run.

## What you'll learn

- Creating an `AudioPart` from a local file or URL with `audio_from()`.
- Using the `audio_transcribe` tool via the `multimodal` preset.
- Chaining transcription → analysis in one `agent.run()` call.
- Handling chunked audio (files longer than 25 MB are split automatically).

## Prerequisites

```bash
pip install "effgen[all]"
# Audio providers: set at least one of
#   GOOGLE_API_KEY (Gemini native audio — free tier)
#   OPENAI_API_KEY (Whisper transcription)
```

## Quickstart

```python
"""audio_transcribe_reason.py — Transcribe audio, then analyse it.

Run:
    python audio_transcribe_reason.py path/to/recording.mp3
    # or use the built-in fixture:
    python audio_transcribe_reason.py  # uses sample audio from the repo
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from effgen.core.multimodal import audio_from
from effgen.presets import create_agent

load_dotenv()

# ── 1. Resolve the audio file ───────────────────────────────────────────────
if len(sys.argv) > 1:
    audio_path = sys.argv[1]
else:
    # Fall back to the repo fixture used in validation
    repo_root = Path(__file__).parent.parent.parent
    audio_path = str(repo_root / "tests/fixtures/multimodal/sample_audio.mp3")

print(f"Audio file: {audio_path}")


# ── 2. Load a model that supports audio ────────────────────────────────────
if os.getenv("OPENAI_API_KEY"):
    # OpenAI uses Whisper for the transcription step
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


# ── 3. Build an audio part ──────────────────────────────────────────────────
audio_part = audio_from(audio_path)        # bytes, path, or URL
print(f"Audio MIME: {audio_part.mime}, size: {len(audio_part.audio):,} bytes")


# ── 4. Run the agent: transcribe, then summarise ────────────────────────────
agent = create_agent("multimodal", model)

result = agent.run(
    "First transcribe this audio recording using the audio_transcribe tool. "
    "Then, in 2–3 sentences, summarise the main topic and the overall sentiment "
    "(positive, negative, or neutral).",
    inputs=[audio_part],
)

print("\n=== Agent output ===")
print(result.output)
assert result.success, f"Agent failed: {result.output}"
```

## How it works

### `audio_from(source)`

Accepts a local path, URL, or raw bytes and returns an `AudioPart` with:

- `audio` — the raw bytes.
- `mime` — detected from magic bytes or file extension (e.g. `audio/mpeg`).
- `duration_s` — optional, set when metadata is readable.

Supported formats: `mp3`, `wav`, `flac`, `ogg`, `m4a`, `webm`.

### Automatic chunking

`AudioPart` objects longer than 25 MB (OpenAI limit) or 30 minutes are
automatically chunked by `audio_pre.py` before sending, and the transcripts
are concatenated in order.  No code change is needed for long recordings.

### Downsample to 16 kHz

Some providers (OpenAI Whisper) require 16 kHz mono WAV.  The `audio_pre`
preprocessor converts automatically using `pydub` if the source file is a
different sample rate.

### Provider chain

| Provider | Transcription method |
|---|---|
| Gemini | Native `AudioPart` in the request; model returns transcript in the response. |
| OpenAI | Two-step: `Whisper /audio/transcriptions` → transcript text → follow-up reasoning call. |
| HF Inference | `automatic_speech_recognition` endpoint (Whisper-large-v3). |

## Expected output

```
Audio file: .../sample_audio.mp3
Audio MIME: audio/mpeg, size: 42,612 bytes

=== Agent output ===
Transcript: "Hello, this is a test recording demonstrating the effGen audio
transcription capability."

Summary: The recording is a brief technical demonstration of a text-to-speech
or audio transcription feature. The tone is neutral and informational.
```

## Next steps

- **Cookbook 01** — Image Q&A.
- **Cookbook 03** — Video summarisation via frame sampling.
