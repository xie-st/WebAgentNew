"""
Multimodal preset — image, audio, and video understanding in one agent.

Primary model : Gemini Flash (vision + audio + video).
Fallbacks     : OpenAI gpt-4o-mini (vision), HF BLIP (vision-only).
Tools         : ImageInfo, ImageCaption, OCR, AudioTranscribe, PDF, Weather,
                MultimodalDescribe (auto-dispatch hub for all modalities).

System prompt : modality-aware — describes images objectively, transcribes
                audio before reasoning, and samples video frames for analysis.

Exported symbol: MULTIMODAL_PRESET (PresetConfig).
The preset is automatically registered in the global PRESETS dict on import.
"""

from __future__ import annotations

from .registry import PRESETS, PresetConfig

MULTIMODAL_PRESET = PresetConfig(
    name="multimodal",
    description=(
        "Multimodal agent for image, audio, and video understanding. "
        "Uses Gemini Flash (primary) with OpenAI gpt-4o-mini and HF fallbacks. "
        "Equipped with ImageInfo, ImageCaption, OCR, AudioTranscribe, PDF, Weather, "
        "and MultimodalDescribe (auto-dispatch for any media type)."
    ),
    tool_names=[
        "multimodal_describe",  # auto-dispatch hub — handles image / audio / video
        "image_info",           # EXIF, size, format metadata
        "image_caption",        # vision captioning via best available provider
        "ocr",                  # text extraction from images/PDFs
        "audio_transcribe",     # speech-to-text (Whisper / Gemini)
        "pdf",                  # PDF parsing (useful for scanned docs)
        "weather",              # "where was this photo taken?" geo queries
    ],
    system_prompt=(
        "You are a multimodal reasoning assistant. "
        "You can process images, audio recordings, and video clips. "
        "\n\n"
        "Guidelines:\n"
        "- When given an IMAGE: use multimodal_describe or image_caption to obtain "
        "  a description, then reason over it. For text-heavy images use ocr first.\n"
        "- When given AUDIO: use audio_transcribe to get the transcript, then reason "
        "  over the transcript to answer the question.\n"
        "- When given VIDEO: use multimodal_describe to sample frames and get a "
        "  sequential description of what happens, then summarize.\n"
        "- Describe objectively — avoid interpreting beyond what is visible/audible.\n"
        "- If the user asks a follow-up question about the media, refer back to the "
        "  description or transcript from the previous step rather than re-processing.\n"
        "- When unsure of the media type, use multimodal_describe with media_type='auto'."
    ),
    max_iterations=10,
    temperature=0.3,
    tags=["multimodal", "image", "audio", "video", "vision", "transcription"],
)

# Register so create_agent("multimodal") works without explicit import
PRESETS["multimodal"] = MULTIMODAL_PRESET
