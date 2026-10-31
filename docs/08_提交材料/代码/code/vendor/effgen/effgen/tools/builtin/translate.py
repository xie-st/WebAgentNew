"""
Translation tool for the effGen framework.

Primary backend: LibreTranslate (public instance or user-configured URL via
LIBRE_TRANSLATE_URL env var).  Fallback: argostranslate (fully local, requires
downloading language packs on first use).

Language packs for the Argos fallback are cached under ~/.effgen/argos/.

Operations:
- translate: Translate text from one language to another.
- available_pairs: List supported language pairs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

logger = logging.getLogger(__name__)

# Public LibreTranslate instances are flaky: hosts come and go without notice,
# some require API keys, others are intermittently DNS-unreachable. We try a
# small ranked list of known-good free public mirrors and stop on first hit.
# A user-supplied LIBRE_TRANSLATE_URL always takes precedence (single entry).
_DEFAULT_PUBLIC_LIBRE_URLS: tuple[str, ...] = (
    "https://translate.fedilab.app",
    "https://translate.argosopentech.com",
    "https://libretranslate.de",
)
_DEFAULT_TIMEOUT = 15

_ARGOS_CACHE_DIR = Path.home() / ".effgen" / "argos"


class InvalidInputError(ValueError):
    """Raised when translation input is missing or invalid."""


class TranslateServiceUnavailable(RuntimeError):
    """Raised when both LibreTranslate and Argos are unavailable."""


def _libre_urls() -> list[str]:
    """Return ordered list of LibreTranslate base URLs to try.

    If LIBRE_TRANSLATE_URL is set, only that URL is used. Otherwise we walk
    through the built-in public-mirror list until one responds.
    """
    user = os.environ.get("LIBRE_TRANSLATE_URL")
    if user:
        return [user.rstrip("/")]
    return [u.rstrip("/") for u in _DEFAULT_PUBLIC_LIBRE_URLS]


def _libre_url() -> str:
    """First-choice LibreTranslate base URL (for back-compat)."""
    return _libre_urls()[0]


def _operator_configured() -> bool:
    """True when the operator explicitly set LIBRE_TRANSLATE_URL.

    A self-hosted LibreTranslate on localhost is a legitimate, deliberate
    operator choice (not a model-suppliable value), so we relax the SSRF block
    for that case only; the default public mirrors are always SSRF-checked.
    """
    return bool(os.environ.get("LIBRE_TRANSLATE_URL"))


# ---------------------------------------------------------------------------
# LibreTranslate helpers
# ---------------------------------------------------------------------------

def _try_libre_post(base: str, path: str, body: dict, timeout: int) -> Any:
    url = f"{base}{path}"
    payload = json.dumps(body).encode()
    with safe_urlopen(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
        timeout=timeout,
        allow_private=_operator_configured(),
    ) as resp:
        return json.loads(resp.read().decode())


def _libre_translate(text: str, source: str, target: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Send a translation request to the first responsive LibreTranslate instance."""
    last_err: str = ""
    for base in _libre_urls():
        try:
            data = _try_libre_post(base, "/translate", {"q": text, "source": source, "target": target, "format": "text"}, timeout)
        except HTTPError as exc:
            # 429 / 403 / 5xx all warrant trying the next mirror.
            last_err = f"{base}: HTTP {exc.code} {exc.reason}"
            continue
        except URLError as exc:
            last_err = f"{base}: {exc.reason}"
            continue
        except Exception as exc:
            last_err = f"{base}: {exc}"
            continue

        # Some public instances respond with an error payload (e.g. API-key
        # requirement) — treat that as a soft failure and try the next mirror.
        if isinstance(data, dict) and data.get("error"):
            last_err = f"{base}: {data['error']}"
            continue
        translated = (data.get("translatedText") or data.get("translated_text") or "") if isinstance(data, dict) else ""
        if translated:
            return translated
        last_err = f"{base}: unexpected response: {data!r}"

    raise ConnectionError(f"LibreTranslate unavailable on all mirrors: {last_err}")


def _local_detect_language(text: str) -> str | None:
    """Detect source language locally via langdetect (offline)."""
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return None


def _libre_detect_language(text: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Detect source language (LibreTranslate first, langdetect fallback, 'en' as final default)."""
    for base in _libre_urls():
        try:
            data = _try_libre_post(base, "/detect", {"q": text}, timeout)
            if isinstance(data, list) and data:
                return data[0].get("language", "en")
            break
        except Exception:
            continue
    local = _local_detect_language(text)
    return local or "en"


def _libre_available_pairs(timeout: int = _DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
    """Fetch supported language list from LibreTranslate (first responsive mirror)."""
    last_err: str = ""
    for base in _libre_urls():
        try:
            with safe_urlopen(
                f"{base}/languages",
                timeout=timeout,
                allow_private=_operator_configured(),
            ) as resp:
                langs = json.loads(resp.read().decode())
            pairs = []
            for lang in langs:
                src = lang.get("code", "")
                for tgt in lang.get("targets", []):
                    pairs.append({"source": src, "target": tgt})
            if pairs:
                return pairs
            last_err = f"{base}: empty languages list"
        except Exception as exc:
            last_err = f"{base}: {exc}"
            continue
    raise ConnectionError(f"LibreTranslate /languages error: {last_err}")


# ---------------------------------------------------------------------------
# Argos Translate fallback helpers
# ---------------------------------------------------------------------------

def _ensure_argos_cache_dir() -> None:
    _ARGOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_ARGOS_CACHE_DIR / "cache" / "downloads").mkdir(parents=True, exist_ok=True)
    (_ARGOS_CACHE_DIR / "packages").mkdir(parents=True, exist_ok=True)


def _set_argos_data_dir() -> None:
    """Override argostranslate's data directory to use our cache."""
    try:
        import argostranslate.settings as at_settings
        _ensure_argos_cache_dir()
        data_dir = _ARGOS_CACHE_DIR
        cache_dir = data_dir / "cache"
        package_dir = data_dir / "packages"
        at_settings.data_dir = data_dir
        at_settings.cache_dir = cache_dir
        at_settings.downloads_dir = cache_dir / "downloads"
        at_settings.package_data_dir = package_dir
        at_settings.legacy_package_data_dir = package_dir
        at_settings.local_package_index = data_dir / "index.json"
        at_settings.package_dirs = [package_dir]
    except ImportError:  # argostranslate absent; caller checks availability
        pass


def _argos_translate(text: str, source: str, target: str) -> str:
    """Translate using argostranslate, downloading packs if needed."""
    _set_argos_data_dir()

    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError as exc:
        raise RuntimeError("argostranslate is not installed. Install it with: pip install argostranslate") from exc

    # Check if the pack is already installed
    installed = argostranslate.package.get_installed_packages()
    pack = next(
        (p for p in installed if p.from_code == source and p.to_code == target),
        None,
    )

    if pack is None:
        logger.info("Argos: downloading language pack %s→%s (one-time, cached in %s)", source, target, _ARGOS_CACHE_DIR)
        try:
            argostranslate.package.update_package_index()
            ok = argostranslate.package.install_package_for_language_pair(source, target)
            if not ok:
                raise RuntimeError(f"Argos: no language pack available for {source}→{target}")
        except Exception as exc:
            raise RuntimeError(f"Argos pack install failed for {source}→{target}: {exc}") from exc

    # Re-fetch installed list
    installed = argostranslate.package.get_installed_packages()
    pack = next(
        (p for p in installed if p.from_code == source and p.to_code == target),
        None,
    )
    if pack is None:
        raise RuntimeError(f"Argos: language pack {source}→{target} not found after install")

    return argostranslate.translate.translate(text, source, target)


def _argos_available_pairs() -> list[dict[str, Any]]:
    """List installed Argos language pairs."""
    try:
        import argostranslate.package
        _set_argos_data_dir()
        installed = argostranslate.package.get_installed_packages()
        return [{"source": p.from_code, "target": p.to_code} for p in installed]
    except ImportError:
        return []


# ---------------------------------------------------------------------------
# Combined translation logic
# ---------------------------------------------------------------------------

def _translate(text: str, source: str, target: str) -> dict[str, Any]:
    """Try LibreTranslate; fall back to Argos on failure."""
    text = text.strip()
    source = (source or "auto").strip()
    target = (target or "en").strip()

    if not text:
        raise InvalidInputError(
            "Translation requires non-empty text. Pass the text to translate "
            "in the 'text' argument."
        )
    if not source:
        raise InvalidInputError(
            "Translation source language must be non-empty. Pass a language "
            "code such as 'en', or 'auto' to detect it."
        )
    if not target:
        raise InvalidInputError(
            "Translation target language must be non-empty. Pass the language "
            "code to translate into, such as 'fr'."
        )

    if source == "auto":
        source = _libre_detect_language(text)

    if source == target:
        return {
            "success": True,
            "data": {
                "translated_text": text,
                "source_language": source,
                "target_language": target,
                "backend": "none",
                "original_text": text,
            },
            "translated_text": text,
            "source_language": source,
            "target_language": target,
            "backend": "none",
            "error": None,
        }

    libre_error_msg: str | None = None
    backend_used = "libretranslate"
    try:
        translated = _libre_translate(text, source, target)
        return {
            "success": True,
            "data": {
                "translated_text": translated,
                "source_language": source,
                "target_language": target,
                "backend": backend_used,
                "original_text": text,
            },
            "translated_text": translated,
            "source_language": source,
            "target_language": target,
            "backend": backend_used,
            "error": None,
        }
    except Exception as exc:
        libre_error_msg = str(exc)
        logger.warning("LibreTranslate failed (%s); trying Argos fallback", exc)

    backend_used = "argostranslate"
    try:
        translated = _argos_translate(text, source, target)
        return {
            "success": True,
            "data": {
                "translated_text": translated,
                "source_language": source,
                "target_language": target,
                "backend": backend_used,
                "original_text": text,
            },
            "translated_text": translated,
            "source_language": source,
            "target_language": target,
            "backend": backend_used,
            "error": None,
        }
    except Exception as argos_exc:
        raise TranslateServiceUnavailable(
            "Translation service unavailable. "
            f"LibreTranslate failed: {libre_error_msg}; "
            f"Argos fallback failed: {argos_exc}. "
            "Set LIBRE_TRANSLATE_URL to a reachable LibreTranslate instance, "
            "or install/download the required Argos language pack."
        ) from argos_exc


def _available_pairs() -> dict[str, Any]:
    """Return available language pairs from LibreTranslate + installed Argos packs."""
    pairs: list[dict] = []
    libre_error = None
    try:
        pairs = _libre_available_pairs()
        source = "libretranslate"
    except Exception as exc:
        libre_error = str(exc)
        source = "argostranslate"
        pairs = _argos_available_pairs()

    if not pairs:
        raise TranslateServiceUnavailable(
            "Translation service unavailable: no language pairs. "
            f"LibreTranslate failed: {libre_error}; "
            "no Argos language packs are installed. "
            "Set LIBRE_TRANSLATE_URL to a reachable LibreTranslate instance, "
            "or install an Argos language pack."
        )

    return {
        "success": True,
        "data": {
            "pairs": pairs,
            "count": len(pairs),
            "source": source,
        },
        "pairs": pairs,
        "count": len(pairs),
        "source": source,
        "error": libre_error,
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class TranslateTool(BaseTool):
    """Translate text between languages using LibreTranslate (with Argos local fallback)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="translate",
                description=(
                    "Translate text between languages. "
                    "Primary backend: LibreTranslate (configurable via LIBRE_TRANSLATE_URL env var). "
                    "Automatic offline fallback: argostranslate (downloads language packs on first use, "
                    "cached in ~/.effgen/argos/). "
                    "Operations: translate (translate text), available_pairs (list supported pairs)."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'translate' or 'available_pairs'.",
                        required=True,
                        enum=["translate", "available_pairs"],
                    ),
                    ParameterSpec(
                        name="text",
                        type=ParameterType.STRING,
                        description="Text to translate (required for 'translate').",
                        required=False,
                        max_length=5000,
                    ),
                    ParameterSpec(
                        name="source",
                        type=ParameterType.STRING,
                        description="Source language code (e.g. 'en', 'fr'). Use 'auto' to detect.",
                        required=False,
                        default="auto",
                        max_length=10,
                    ),
                    ParameterSpec(
                        name="target",
                        type=ParameterType.STRING,
                        description="Target language code (e.g. 'es', 'ja', 'de').",
                        required=False,
                        default="en",
                        max_length=10,
                    ),
                ],
                timeout_seconds=60,
                tags=["translate", "translation", "language", "nlp", "free", "offline"],
                examples=[
                    {"operation": "translate", "text": "Hello world", "source": "en", "target": "fr"},
                    {"operation": "translate", "text": "Bonjour le monde", "source": "auto", "target": "en"},
                    {"operation": "available_pairs"},
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        text: str | None = None,
        source: str = "auto",
        target: str = "en",
        **kwargs,
    ) -> dict[str, Any]:
        op = operation.lower()

        if op == "translate":
            if not text or not text.strip():
                raise InvalidInputError(
                    "'text' is required and must be non-empty for the "
                    "translate operation. Pass the text to translate."
                )
            return await asyncio.to_thread(_translate, text, source, target)

        elif op == "available_pairs":
            return await asyncio.to_thread(_available_pairs)

        else:
            raise ValueError(f"Unknown operation: {operation!r}. Use 'translate' or 'available_pairs'.")
