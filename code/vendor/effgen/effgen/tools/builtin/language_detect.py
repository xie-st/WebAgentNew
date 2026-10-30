"""
Language detection tool for the effGen framework.

Backend: langdetect (pure Python, no external API required).

Operations:
- detect: Detect the language of a single text string.
- detect_batch: Detect languages for a list of texts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


class InvalidInputError(ValueError):
    """Raised when language detection input is missing or invalid."""

# Map langdetect codes to human-readable names (common ones)
_LANG_NAMES: dict[str, str] = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ms": "Malay",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sr": "Serbian",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}


def _detect_one(text: str) -> dict[str, Any]:
    """Detect language of a single text."""
    try:
        from langdetect import DetectorFactory, detect_langs
    except ImportError as exc:
        raise RuntimeError(
            "langdetect is not installed. Install it with: pip install langdetect"
        ) from exc

    # Make langdetect deterministic across runs (its probabilistic core seeds
    # from random.random() by default — surprising for users comparing results).
    DetectorFactory.seed = 0

    text = text.strip()
    if not text:
        raise InvalidInputError(
            "Empty text provided: language detection requires non-empty "
            "input. Pass the text to identify in the 'text' argument."
        )

    try:
        results = detect_langs(text)
        top = results[0]
        lang_code = top.lang
        confidence = round(top.prob, 4)
        return {
            "success": True,
            "data": {
                "language": lang_code,
                "language_name": _LANG_NAMES.get(lang_code, lang_code),
                "confidence": confidence,
                "all_candidates": [
                    {
                        "language": r.lang,
                        "language_name": _LANG_NAMES.get(r.lang, r.lang),
                        "confidence": round(r.prob, 4),
                    }
                    for r in results
                ],
                "text_preview": text[:100],
            },
            "language": lang_code,
            "language_name": _LANG_NAMES.get(lang_code, lang_code),
            "confidence": confidence,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "data": {},
            "error": f"Detection failed: {exc}",
        }


def _detect_batch(texts: list[str]) -> dict[str, Any]:
    """Detect languages for multiple texts."""
    results = []
    for i, text in enumerate(texts):
        r = _detect_one(text)
        results.append({
            "index": i,
            "text_preview": text[:80],
            "language": r.get("language", ""),
            "language_name": r.get("language_name", ""),
            "confidence": r.get("confidence", 0.0),
            "success": r.get("success", False),
            "error": r.get("error"),
        })
    return {
        "success": True,
        "data": {"results": results, "count": len(results)},
        "results": results,
        "count": len(results),
        "error": None,
    }


class LanguageDetectTool(BaseTool):
    """Detect the language of text using langdetect (no external API required)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="language_detect",
                description=(
                    "Detect the language of text (fully offline, no external API). "
                    "Uses langdetect (pure Python). "
                    "Operations: detect (single text → language code + confidence), "
                    "detect_batch (list of texts)."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'detect' or 'detect_batch'.",
                        required=True,
                        enum=["detect", "detect_batch"],
                    ),
                    ParameterSpec(
                        name="text",
                        type=ParameterType.STRING,
                        description="Text to detect (required for 'detect').",
                        required=False,
                        max_length=5000,
                    ),
                    ParameterSpec(
                        name="texts",
                        type=ParameterType.ARRAY,
                        description="List of texts (required for 'detect_batch').",
                        required=False,
                        items_type=ParameterType.STRING,
                    ),
                ],
                timeout_seconds=30,
                tags=["language", "detect", "nlp", "offline", "free"],
                examples=[
                    {"operation": "detect", "text": "Hello, how are you?"},
                    {"operation": "detect", "text": "Bonjour le monde"},
                    {"operation": "detect_batch", "texts": ["Hello", "Bonjour", "Hola", "こんにちは"]},
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        text: str | None = None,
        texts: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        op = operation.lower()

        if op == "detect":
            if not text:
                raise InvalidInputError(
                    "'text' is required and must be non-empty for the detect "
                    "operation. Pass the text to identify, or use detect_batch "
                    "for several."
                )
            return await asyncio.to_thread(_detect_one, text)

        elif op == "detect_batch":
            if not texts:
                raise InvalidInputError(
                    "'texts' (list of strings) is required for detect_batch. "
                    "Pass a list of the strings to identify, or use detect for "
                    "one."
                )
            if not isinstance(texts, list):
                raise InvalidInputError(
                    "'texts' must be a list of strings. Pass each item as "
                    "text — wrap a single string in a list."
                )
            return await asyncio.to_thread(_detect_batch, texts)

        else:
            raise ValueError(f"Unknown operation: {operation!r}. Use 'detect' or 'detect_batch'.")
