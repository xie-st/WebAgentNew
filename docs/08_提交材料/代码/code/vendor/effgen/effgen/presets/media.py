"""
Media preset — bundles audio, image, and media processing tools.

Exported symbol: MEDIA_PRESET (a PresetConfig instance).
The preset is automatically registered in the global PRESETS dict when this
module is imported.
"""

from __future__ import annotations

from .registry import PRESETS, PresetConfig

MEDIA_PRESET = PresetConfig(
    name="media",
    description=(
        "Media processing agent with AudioTranscribeTool (speech-to-text) and "
        "ImageCaptionTool (vision captioning via OpenAI/Gemini). "
        "Handles audio files (MP3, WAV, OGG, FLAC) and images (PNG, JPEG, WEBP)."
    ),
    tool_names=[
        "audio_transcribe",
        "image_caption",
    ],
    system_prompt=(
        "You are a media processing assistant. "
        "Use the audio_transcribe tool to convert speech audio files to text. "
        "Use the image_caption tool to describe or caption images. "
        "When given an audio file path, transcribe it and return the full transcript. "
        "When given an image path, caption or describe it as requested. "
        "For long recordings, summarize key points after transcribing."
    ),
    max_iterations=8,
    temperature=0.3,
    tags=["media", "audio", "transcription", "speech-to-text", "vision", "image", "caption"],
)

# Register the preset so create_agent("media") works
PRESETS["media"] = MEDIA_PRESET
