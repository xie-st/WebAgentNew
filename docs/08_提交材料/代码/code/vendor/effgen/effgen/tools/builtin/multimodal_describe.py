"""
MultimodalDescribeTool — auto-dispatch describing/transcription of multimodal inputs.

Inspects the input type (image bytes/path, audio bytes/path, video path) and
routes to the appropriate underlying tool or API:
  - image  → ImageCaptionTool (vision provider) or OCR fallback
  - audio  → AudioTranscribeTool (Whisper / Gemini audio)
  - video  → frame-sample + ImageCaptionTool per frame then summarise

Accepts:
  - file_path  : local path to image/audio/video file
  - media_type : "image", "audio", "video", or "auto" (default; inferred from path)
  - prompt     : instruction for the model (default varies by type)
  - operation  : "describe" | "transcribe" | "summarize" (default "describe")
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIME / type helpers
# ---------------------------------------------------------------------------

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _infer_media_type(path: str | Path) -> str:
    """Infer 'image', 'audio', or 'video' from file extension."""
    suffix = Path(str(path)).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    # Try mimetypes as a fallback
    mt, _ = mimetypes.guess_type(str(path))
    if mt:
        major = mt.split("/")[0]
        if major in ("image", "audio", "video"):
            return major
    return "image"  # safe fallback


# ---------------------------------------------------------------------------
# Backend dispatchers
# ---------------------------------------------------------------------------

def _describe_image_sync(file_path: str, prompt: str, operation: str) -> dict[str, Any]:
    """Describe an image using ImageCaptionTool."""
    from effgen.tools.builtin.image_caption import ImageCaptionTool

    tool = ImageCaptionTool()
    op = "describe" if operation in ("describe", "summarize") else "caption"
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            tool._execute(operation=op, image_path=file_path, prompt=prompt)
        )
    finally:
        loop.close()
    return result


def _transcribe_audio_sync(file_path: str, prompt: str) -> dict[str, Any]:
    """Transcribe audio using AudioTranscribeTool with optional Gemini reasoning."""
    from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

    tool = AudioTranscribeTool()
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            tool._execute(operation="transcribe", audio_path=file_path)
        )
    finally:
        loop.close()

    # If the prompt is non-trivial (not just "transcribe"), call a vision provider
    # to reason over the transcript.
    text = (result.get("data") or {}).get("text") or result.get("text") or ""
    if text and prompt.strip().lower() not in (
        "transcribe",
        "transcribe this audio",
        "",
    ):
        # Pass transcript + prompt to the LLM for reasoning
        try:
            reasoning = _reason_over_text(text, prompt)
            result["reasoning"] = reasoning
        except Exception as exc:
            logger.debug("Reasoning pass failed: %s", exc)

    return result


def _describe_video_sync(file_path: str, prompt: str, max_frames: int = 6) -> dict[str, Any]:
    """Sample frames from video and describe using ImageCaptionTool."""
    import tempfile

    from effgen.multimodal.video_pre import VideoSource
    from effgen.tools.builtin.image_caption import ImageCaptionTool

    vs = VideoSource(file_path)
    image_parts = vs.sample_frames(fps=1, max_frames=max_frames)

    if not image_parts:
        return {
            "success": False,
            "error": "No frames could be extracted from the video.",
            "description": "",
        }

    tool = ImageCaptionTool()
    descriptions: list[str] = []
    loop = asyncio.new_event_loop()
    try:
        for i, frame in enumerate(image_parts[:max_frames]):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                tmp.write(frame.image)
                tmp.close()
                result = loop.run_until_complete(
                    tool._execute(
                        operation="describe",
                        image_path=tmp.name,
                        prompt=f"Frame {i + 1}: {prompt}",
                    )
                )
                desc = (result.get("data") or {}).get("caption") or result.get("caption") or ""
                if desc:
                    descriptions.append(f"Frame {i + 1}: {desc}")
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except Exception:  # best-effort temp-file cleanup
                    pass
    finally:
        loop.close()

    combined = "\n".join(descriptions) if descriptions else "No description available."
    return {
        "success": True,
        "data": {
            "description": combined,
            "frames_sampled": len(descriptions),
            "prompt": prompt,
        },
        "description": combined,
        "error": None,
    }


def _reason_over_text(transcript: str, prompt: str) -> str:
    """Call a cheap LLM to reason over a transcript with the given prompt."""
    # Try Gemini Flash-Lite first, then OpenAI mini
    _providers = [
        ("gemini", "gemini-3.1-flash-lite"),
        ("gemini", "gemini-2.5-flash-lite"),
        ("openai", "gpt-4o-mini"),
    ]
    for provider, model_id in _providers:
        key_env = {"gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider, "")
        api_key = os.environ.get(key_env, "") if key_env else ""
        if not api_key:
            continue
        try:
            if provider == "gemini":
                from effgen.models.base import GenerationConfig
                from effgen.models.gemini_adapter import GeminiAdapter

                adapter = GeminiAdapter(model_name=model_id, api_key=api_key)
                adapter.load()
                full_prompt = f"Transcript:\n{transcript}\n\nTask: {prompt}"
                result = adapter.generate(
                    [{"type": "text", "text": full_prompt}],
                    GenerationConfig(max_tokens=512, temperature=0.3),
                )
                return result.text.strip()
            elif provider == "openai":
                from effgen.models.base import GenerationConfig
                from effgen.models.openai_adapter import OpenAIAdapter

                adapter = OpenAIAdapter(model_name=model_id, api_key=api_key)
                adapter.load()
                from effgen.core.messages import Message, Role, TextPart
                messages = [
                    Message(
                        role=Role.USER,
                        content=[TextPart(text=f"Transcript:\n{transcript}\n\nTask: {prompt}")]
                    )
                ]
                result = adapter.generate(messages, GenerationConfig(max_tokens=512, temperature=0.3))
                return result.text.strip()
        except Exception as exc:
            logger.debug("Reasoning with %s/%s failed: %s", provider, model_id, exc)
            continue
    return ""


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


class MultimodalDescribeTool(BaseTool):
    """
    Auto-dispatch describe / transcribe / summarize for image, audio, and video.

    Inspects the media type (from file extension or explicit media_type) and
    routes to the appropriate backend:
      - image  → vision model (ImageCaptionTool)
      - audio  → speech-to-text (AudioTranscribeTool) + optional LLM reasoning
      - video  → frame-sampling → vision model per frame, then summary

    Accepted input types: file paths (str/Path), bytes (via image_bytes field).
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="multimodal_describe",
                description=(
                    "Automatically describe, transcribe, or summarize image, audio, or "
                    "video inputs. Routes to the best available backend based on media type. "
                    "For images: uses a vision model (OpenAI/Gemini). "
                    "For audio: transcribes via Whisper/Gemini then optionally reasons. "
                    "For video: samples frames and describes each."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="file_path",
                        type=ParameterType.STRING,
                        description=(
                            "Path to the media file (image, audio, or video). "
                            "The media type is inferred from the file extension unless "
                            "media_type is explicitly specified."
                        ),
                        required=True,
                    ),
                    ParameterSpec(
                        name="media_type",
                        type=ParameterType.STRING,
                        description="Media type: 'image', 'audio', 'video', or 'auto' (default).",
                        required=False,
                        enum=["image", "audio", "video", "auto"],
                        default="auto",
                    ),
                    ParameterSpec(
                        name="prompt",
                        type=ParameterType.STRING,
                        description=(
                            "Instruction for the model. Defaults: "
                            "image→'Describe this image in detail.', "
                            "audio→'Transcribe this audio.', "
                            "video→'Describe what happens in this video.'."
                        ),
                        required=False,
                    ),
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'describe', 'transcribe', or 'summarize'.",
                        required=False,
                        enum=["describe", "transcribe", "summarize"],
                        default="describe",
                    ),
                    ParameterSpec(
                        name="max_frames",
                        type=ParameterType.INTEGER,
                        description="Maximum number of frames to sample for video (default 6).",
                        required=False,
                        default=6,
                    ),
                ],
                timeout_seconds=120,
                tags=["image", "audio", "video", "multimodal", "describe", "transcribe", "vision"],
                examples=[
                    {"file_path": "/path/to/photo.jpg", "operation": "describe"},
                    {"file_path": "/path/to/speech.mp3", "operation": "transcribe"},
                    {"file_path": "/path/to/clip.mp4", "operation": "summarize"},
                    {
                        "file_path": "/path/to/chart.png",
                        "prompt": "What data is shown in this chart?",
                    },
                ],
            )
        )

    async def _execute(
        self,
        file_path: str,
        media_type: str = "auto",
        prompt: str | None = None,
        operation: str = "describe",
        max_frames: int = 6,
        **kwargs: Any,
    ) -> dict[str, Any]:
        p = Path(file_path)
        if not p.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "description": "",
            }

        resolved_type = media_type if media_type != "auto" else _infer_media_type(p)

        default_prompts = {
            "image": "Describe this image in detail.",
            "audio": "Transcribe this audio.",
            "video": "Describe what happens in this video.",
        }
        effective_prompt = prompt or default_prompts.get(resolved_type, "Describe this media.")

        try:
            if resolved_type == "image":
                result = await asyncio.to_thread(
                    _describe_image_sync, str(p), effective_prompt, operation
                )
                # Normalise to common shape
                description = (
                    (result.get("data") or {}).get("caption")
                    or result.get("caption")
                    or result.get("description")
                    or ""
                )
                return {
                    "success": result.get("success", True),
                    "media_type": "image",
                    "description": description,
                    "raw": result,
                    "error": result.get("error"),
                }

            elif resolved_type == "audio":
                result = await asyncio.to_thread(
                    _transcribe_audio_sync, str(p), effective_prompt
                )
                transcript = (result.get("data") or {}).get("text") or result.get("text") or ""
                reasoning = result.get("reasoning", "")
                return {
                    "success": result.get("success", True),
                    "media_type": "audio",
                    "transcript": transcript,
                    "reasoning": reasoning,
                    "description": reasoning or transcript,
                    "raw": result,
                    "error": result.get("error"),
                }

            elif resolved_type == "video":
                result = await asyncio.to_thread(
                    _describe_video_sync, str(p), effective_prompt, max_frames
                )
                description = (result.get("data") or {}).get("description") or result.get("description") or ""
                return {
                    "success": result.get("success", True),
                    "media_type": "video",
                    "description": description,
                    "raw": result,
                    "error": result.get("error"),
                }

            else:
                return {
                    "success": False,
                    "error": f"Unknown media type: {resolved_type!r}",
                    "description": "",
                }

        except Exception as exc:
            logger.exception("MultimodalDescribeTool error for %s: %s", file_path, exc)
            return {
                "success": False,
                "error": str(exc),
                "description": "",
            }
