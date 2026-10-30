# AudioTranscribeTool

Transcribe audio files to text using local or cloud speech-to-text backends.

**Primary backend:** `faster-whisper` — runs fully locally on CPU or GPU, no auth required.  
**Fallback backend:** HuggingFace Inference API (`openai/whisper-large-v3`) — set `HF_TOKEN`.

---

## Installation

```bash
# Core dependencies
pip install "effgen[audio]"

# System dependency for non-WAV formats (MP3, OGG, FLAC, etc.)
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# conda
conda install -c conda-forge ffmpeg
```

If ffmpeg is absent, container formats that require it, such as M4A, MP4, WebM,
AAC, and WMA, raise a clean `MissingSystemDependency("ffmpeg")` error with the
install hint above. WAV files do not require ffmpeg.

---

## Quick Start

```python
import asyncio
from effgen.tools.builtin import AudioTranscribeTool

tool = AudioTranscribeTool()

result = asyncio.run(tool._execute(
    operation="transcribe",
    audio_path="/path/to/speech.mp3",
))

print(result["text"])
# → "Hello world, this is a test of the audio transcription tool."
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | str | required | Must be `"transcribe"`. |
| `audio_path` | str | — | Path to audio file. Mutually exclusive with `audio_bytes`. |
| `audio_bytes` | bytes | — | Raw audio bytes (programmatic use). Mutually exclusive with `audio_path`. |
| `model_size` | str | `"base"` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, and distil variants. |
| `language` | str | `None` | ISO 639-1 code (e.g. `"en"`, `"fr"`). Auto-detected when `None`. |
| `word_timestamps` | bool | `False` | Include per-word start/end timestamps (faster-whisper only). |
| `backend` | str | `"auto"` | `"faster-whisper"`, `"hf-inference"`, or `"auto"`. |

---

## Output Schema

```python
{
    "success": True,
    "text": "Hello world...",          # full transcript
    "segments": [                      # list of time-stamped segments
        {
            "id": 0,
            "start": 0.0,
            "end": 2.5,
            "text": "Hello world...",
            "avg_logprob": -0.23,
            "no_speech_prob": 0.01,
            # "words": [...] if word_timestamps=True
        }
    ],
    "language": "en",
    "backend": "faster-whisper",
    "error": None,
}
```

---

## Model Size Guide

| Model | Size | Speed (CPU) | Accuracy |
|-------|------|-------------|----------|
| `tiny` | ~75 MB | fastest | lowest |
| `base` | ~145 MB | fast | good |
| `small` | ~465 MB | moderate | better |
| `medium` | ~1.5 GB | slow | high |
| `large-v3` | ~3 GB | very slow on CPU | best |

For CPU inference, use `tiny` or `base`. For GPU inference, `large-v3` or
`distil-large-v3` give the best accuracy.

---

## GPU Detection

The tool checks for a CUDA GPU via `nvidia-smi` on startup. If found,
it uses `device="cuda"` with `compute_type="float16"`. Otherwise it falls
back to `device="cpu"` with `compute_type="int8"`.

A `RuntimeWarning` is emitted when running models larger than `base` on CPU.

---

## HuggingFace Inference Fallback

```bash
export HF_TOKEN=hf_your_token_here
```

```python
result = asyncio.run(tool._execute(
    operation="transcribe",
    audio_path="/path/to/speech.mp3",
    backend="hf-inference",   # explicit or let "auto" fall back
))
```

Uses `openai/whisper-large-v3` via effGen's `HFInferenceAdapter`, backed by
Hugging Face `InferenceClient.automatic_speech_recognition`.
Requires a Hugging Face account and available Inference credits.

---

## Media Preset

```python
from effgen.presets import create_agent
from effgen.models import load_model

model = load_model("your-model-id")
agent = create_agent("media", model)
result = agent.run("Transcribe /path/to/interview.mp3")
```

---

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `AudioBackendUnavailable` | No local or HF backend is available | `pip install "effgen[audio]"` or set `HF_TOKEN` |
| `MissingSystemDependency("ffmpeg")` | Compressed audio format and ffmpeg missing | Install ffmpeg |
| `RuntimeError("HF_TOKEN is not set")` | No cloud fallback token | Set `HF_TOKEN` env var |
| `FileNotFoundError` | Audio path doesn't exist | Check file path |
| ffmpeg error (via faster-whisper) | ffmpeg not on PATH | Install ffmpeg (see above) |
