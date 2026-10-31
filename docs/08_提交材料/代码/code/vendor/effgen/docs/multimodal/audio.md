# Audio Input

effGen supports audio input on Gemini, OpenAI (Whisper), and HF Inference.

## Quick Start

```python
from effgen.core.messages import AudioPart, Message, Role, TextPart

# Load audio from file
with open("speech.wav", "rb") as f:
    audio_bytes = f.read()

audio_part = AudioPart(audio=audio_bytes, mime="audio/wav")
msg = Message(
    role=Role.USER,
    content=[
        audio_part,
        TextPart(text="Transcribe this audio and give a brief summary."),
    ],
)
```

## Supported Providers

### Gemini — native audio understanding

Gemini Flash/Pro models natively understand audio. No pre-transcription step.

```python
from effgen.models.gemini_adapter import GeminiAdapter

adapter = GeminiAdapter("gemini-3.1-flash-lite")
adapter.load()
result = adapter.generate(msg)
print(result.text)
adapter.unload()
```

Supported MIMEs: `audio/mp3`, `audio/mpeg`, `audio/wav`, `audio/flac`, `audio/ogg`, `audio/m4a`, `audio/mp4`, `audio/mpga`, `audio/webm`.
Max inline: 20 MB (use the Files API for longer audio).

### OpenAI — Whisper transcription

Audio in chat messages is auto-transcribed via Whisper then injected as text.
Use `transcribe_audio()` for standalone transcription.

```python
from effgen.models.openai_adapter import OpenAIAdapter

adapter = OpenAIAdapter("gpt-4o-mini")
adapter.load()

# Standalone transcription
transcript = adapter.transcribe_audio(audio_bytes, mime="audio/wav")
print(transcript)

# Embedded in a chat message (auto-transcribes)
result = adapter.generate(msg)
print(result.text)

adapter.unload()
```

Supported MIMEs: `audio/mp3`, `audio/mpeg`, `audio/wav`, `audio/m4a`, `audio/ogg`, `audio/flac`, `audio/mp4`, `audio/mpga`, `audio/webm`.
Max file: 25 MB (chunked automatically by `audio_pre.chunk()`).

### HF Inference — Whisper ASR endpoint

Uses `InferenceClient.automatic_speech_recognition()` to transcribe, then sends the transcript to the chat model.

```python
from effgen.models.hf_inference_adapter import HFInferenceAdapter

adapter = HFInferenceAdapter("Qwen/Qwen2.5-7B-Instruct")
adapter.load()

# Standalone transcription
transcript = adapter.transcribe_audio(
    audio_bytes,
    mime="audio/wav",
    asr_model="openai/whisper-large-v3",
)
print(transcript)

# Embedded in a chat message
result = adapter.generate(msg)
print(result.text)

adapter.unload()
```

### Anthropic — not supported

Anthropic Claude does not currently support audio input. Sending an `AudioPart` to any Anthropic model raises `CapabilityNotSupportedError(Capability.audio_input)`.

## Supported MIME Types

`AudioPart` validates the MIME on construction:

| MIME | Notes |
|------|-------|
| `audio/mp3` | most providers |
| `audio/mpeg` | alias for mp3 |
| `audio/wav` | widely supported |
| `audio/x-wav` | alias for wav |
| `audio/flac` | lossless |
| `audio/ogg` | vorbis/opus |
| `audio/m4a` | AAC container |
| `audio/x-m4a` | alias for m4a |
| `audio/mp4` | MPEG-4 audio/video container |
| `audio/mpga` | MPEG audio alias |
| `audio/webm` | WebM audio container |

## Preprocessing

`effgen.multimodal.audio_pre` handles per-provider preprocessing:

```python
from effgen.multimodal.audio_pre import prepare, chunk, get_audio_duration
from effgen.core.messages import AudioPart

part = AudioPart(audio=audio_bytes, mime="audio/wav")

# Preprocess (resample to 16 kHz mono for HF providers)
prepared = prepare(part, provider="hf_inference")

# Split audio > provider limit into chunks
chunks = chunk(part, provider="openai")  # splits at 25 MB / 30 min boundary
for c in chunks:
    print(f"Chunk: {c.duration_s:.1f}s, {len(c.audio)} bytes")

# Get duration (requires pydub)
duration = get_audio_duration(audio_bytes, "audio/wav")
```

Provider constraints:

| Provider | Max bytes | Max duration | Resample |
|----------|-----------|--------------|---------|
| `gemini` | 20 MB | 60 min | native |
| `openai` | 25 MB | 30 min | native |
| `hf_inference` | 100 MB | 30 min | 16 kHz mono |

## Chunking Long Audio

Audio longer than a provider's limit is split automatically:

```python
from effgen.models.openai_adapter import OpenAIAdapter

adapter = OpenAIAdapter("gpt-4o-mini")
adapter.load()
full_transcript = adapter.transcribe_audio(long_audio, mime="audio/mp3")
adapter.unload()
```

## Capability Gating

Providers that don't support audio raise `CapabilityNotSupportedError`:

```python
from effgen.errors import CapabilityNotSupportedError
from effgen.models.capabilities import Capability

try:
    anthropic_adapter.generate(msg_with_audio)
except CapabilityNotSupportedError as e:
    assert e.capability == "audio_input"
    print(e)  # "Capability 'audio_input' is not supported by provider 'anthropic'."
```

## helpers.audio_from()

Use the `audio_from()` helper in `effgen.core.multimodal` to build an `AudioPart` from bytes, a file path, or a URL:

```python
from effgen.core.multimodal import audio_from

part = audio_from("path/to/speech.mp3")       # from file path
part = audio_from(b"...", mime="audio/wav")   # from bytes
```
