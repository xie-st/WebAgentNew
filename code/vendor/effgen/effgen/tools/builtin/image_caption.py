"""
ImageCaptionTool — vision-model image captioning via the effGen model router.

Uses the ProviderRegistry to discover vision-capable providers
(Capability.vision), then sends the image to the cheapest/most available one.

Supported providers (auto-detected when API keys are present):
  - OpenAI       : gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini
  - Google Gemini: gemini-3.1-flash-lite, gemini-2.5-flash (all support vision)
  - Replicate    : llava / moondream variants

Operations
----------
- caption   : short natural-language description
- describe  : detailed description (detail="high"|"low")
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any

from effgen.errors import NoVisionProviderAvailable

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._fs import confine_path, normalize_allowed_dirs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider preference order for vision — cheapest first
# ---------------------------------------------------------------------------

_VISION_PROVIDER_PREFERENCE: list[tuple[str, str]] = [
    # (provider_name, model_id) — cheap, stable vision models first
    ("gemini", "gemini-3.1-flash-lite"),
    ("openai", "gpt-4o-mini"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-4o"),
    ("openai", "gpt-4.1"),
    ("gemini", "gemini-2.5-flash"),
    ("gemini", "gemini-2.5-pro"),
]

_PROVIDER_DEFAULT_VISION_MODEL = dict(reversed(_VISION_PROVIDER_PREFERENCE))


def _ensure_provider_registry() -> None:
    """Ensure vision provider adapters are registered with ProviderRegistry.

    Importing an adapter module triggers its module-level ``_register()`` the
    first time only, so a registry that was emptied at runtime
    (``ProviderRegistry.clear()``) cannot be repopulated by importing again —
    the modules are already in ``sys.modules``. Re-running the built-in
    registrations restores whichever vision providers are missing.
    """
    from effgen.models.registry import ProviderRegistry

    if all(
        name in ProviderRegistry.list_providers()
        for name in ("gemini", "openai", "replicate")
    ):
        return
    try:
        ProviderRegistry.register_builtins()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not restore provider registrations for vision routing: %s", exc)


def _provider_api_key_env(provider_name: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "replicate": "REPLICATE_API_TOKEN",
    }.get(provider_name, "")


def _check_vision_candidate(
    provider_name: str,
    model_id: str,
) -> tuple[type | None, str | None]:
    """Return (adapter_cls, rejection_reason) for a provider/model candidate."""
    from effgen.models.capabilities import Capability
    from effgen.models.registry import ProviderRegistry

    if provider_name not in ProviderRegistry.list_providers():
        return None, f"{provider_name}(not registered)"
    if not ProviderRegistry.is_available(provider_name):
        key_env = _provider_api_key_env(provider_name)
        suffix = f"missing {key_env}" if key_env else "no key"
        return None, f"{provider_name}({suffix})"

    provider_caps = ProviderRegistry.get_capabilities(provider_name)
    if Capability.vision not in provider_caps:
        return None, f"{provider_name}(no vision capability)"

    try:
        _pname, adapter_cls, model_info = ProviderRegistry.lookup(
            model_id, provider=provider_name
        )
    except KeyError:
        return None, f"{provider_name}/{model_id}(not found)"

    if model_info.get("active") is False:
        return None, f"{provider_name}/{model_id}(inactive)"

    modality = str(model_info.get("modality", "")).lower()
    supports_v = model_info.get("supports_vision")
    if modality in {"embedding", "audio", "stt", "tts", "rerank"} or supports_v is False:
        return None, f"{provider_name}/{model_id}(no vision)"

    return adapter_cls, None


def _enumerate_vision_candidates(
    provider_name: str | None = None,
    model_id: str | None = None,
) -> list[tuple[str, str, type]]:
    """Return all (provider, model, adapter_cls) candidates ranked by preference.

    Empty list means no suitable provider/key is available — callers should
    raise NoVisionProviderAvailable.
    """
    from effgen.models.registry import ProviderRegistry

    _ensure_provider_registry()

    if provider_name and model_id:
        adapter_cls, _ = _check_vision_candidate(provider_name, model_id)
        if adapter_cls is None:
            return []
        return [(provider_name, model_id, adapter_cls)]

    if provider_name:
        default_model = _PROVIDER_DEFAULT_VISION_MODEL.get(provider_name)
        raw_candidates = [
            (p, m) for p, m in _VISION_PROVIDER_PREFERENCE if p == provider_name
        ]
        if default_model and (provider_name, default_model) not in raw_candidates:
            raw_candidates.insert(0, (provider_name, default_model))
    elif model_id:
        raw_candidates = []
        for provider in ProviderRegistry.list_providers():
            try:
                ProviderRegistry.lookup(model_id, provider=provider)
            except KeyError:
                continue
            raw_candidates.append((provider, model_id))
    else:
        raw_candidates = list(_VISION_PROVIDER_PREFERENCE)

    available: list[tuple[str, str, type]] = []
    for cand_p, cand_m in raw_candidates:
        adapter_cls, _ = _check_vision_candidate(cand_p, cand_m)
        if adapter_cls is None:
            continue
        available.append((cand_p, cand_m, adapter_cls))
    return available


def _pick_vision_provider(
    provider_name: str | None = None,
    model_id: str | None = None,
) -> tuple[str, str, type]:
    """Return (provider_name, model_id, adapter_cls) for the best available vision provider.

    Raises:
        NoVisionProviderAvailable: if no suitable provider/key is found.
    """
    from effgen.models.registry import ProviderRegistry

    _ensure_provider_registry()
    tried: list[str] = []

    if provider_name and model_id:
        adapter_cls, reason = _check_vision_candidate(provider_name, model_id)
        if adapter_cls is None:
            raise NoVisionProviderAvailable(tried_providers=[reason or provider_name])
        logger.debug("ImageCaptionTool: selected %s/%s", provider_name, model_id)
        return provider_name, model_id, adapter_cls

    if provider_name:
        default_model = _PROVIDER_DEFAULT_VISION_MODEL.get(provider_name)
        candidates = [
            (p, m)
            for p, m in _VISION_PROVIDER_PREFERENCE
            if p == provider_name
        ]
        if default_model and (provider_name, default_model) not in candidates:
            candidates.insert(0, (provider_name, default_model))
    elif model_id:
        candidates = []
        for provider in ProviderRegistry.list_providers():
            try:
                ProviderRegistry.lookup(model_id, provider=provider)
            except KeyError:
                continue
            candidates.append((provider, model_id))
    else:
        candidates = list(_VISION_PROVIDER_PREFERENCE)

    for candidate_provider, candidate_model in candidates:
        adapter_cls, reason = _check_vision_candidate(candidate_provider, candidate_model)
        if adapter_cls is None:
            tried.append(reason or f"{candidate_provider}/{candidate_model}")
            continue

        logger.debug(
            "ImageCaptionTool: selected %s/%s",
            candidate_provider,
            candidate_model,
        )
        return candidate_provider, candidate_model, adapter_cls

    raise NoVisionProviderAvailable(tried_providers=tried)


def _is_transient_provider_error(exc: BaseException) -> bool:
    """True if exc looks like a rate-limit / quota / temporary outage."""
    msg = str(exc).lower()
    transient_markers = (
        "429",
        "rate limit",
        "rate-limit",
        "ratelimit",
        "resource_exhausted",
        "quota",
        "503",
        "502",
        "504",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "try again",
    )
    return any(m in msg for m in transient_markers)


def _image_to_base64_url(source: str | bytes | Path) -> str:
    """Convert image source to a data-URI base64 string."""
    if isinstance(source, str | Path):
        raw = Path(str(source)).read_bytes()
    else:
        raw = bytes(source)
    b64 = base64.b64encode(raw).decode()
    # detect format from first bytes
    mime = "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        mime = "image/webp"
    elif raw[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    return f"data:{mime};base64,{b64}"


def _build_openai_vision_messages(
    prompt_text: str,
    image_source: str | bytes | Path,
) -> list[dict]:
    """Build an OpenAI-compatible multimodal messages list."""
    data_url = _image_to_base64_url(image_source)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
            ],
        }
    ]


def _call_openai_vision(
    client: Any,
    model_id: str,
    prompt: str,
    source: str | bytes | Path,
    detail: str,
) -> str:
    """Call OpenAI vision endpoint directly (bypasses validate_prompt on list)."""
    data_url = _image_to_base64_url(source)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url, "detail": detail}},
            ],
        }
    ]
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def _call_gemini_vision(
    adapter: Any,
    prompt: str,
    source: str | bytes | Path,
) -> str:
    """Call Gemini vision via the adapter's generate method."""
    import tempfile

    from effgen.models.base import GenerationConfig

    if isinstance(source, str | Path):
        img_arg = str(source)
        tmp_path = None
    else:
        # Write bytes to a temp file so Gemini adapter can read it
        suffix = ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(bytes(source))
            tmp_path = tmp.name
        img_arg = tmp_path

    try:
        multimodal = [prompt, {"image": img_arg}]
        gen_cfg = GenerationConfig(max_tokens=512, temperature=0.3)
        result = adapter.generate(multimodal, gen_cfg)
        return (result.text or "").strip()
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:  # best-effort temp-file cleanup
                pass


def _caption_sync(
    source: str | bytes | Path,
    prompt: str,
    detail: str = "auto",
    provider_name: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Call the best available vision provider synchronously.

    Falls back to the next candidate on transient errors (rate-limit,
    quota, 5xx). Raises NoVisionProviderAvailable if no candidates are
    available; re-raises the last error if all candidates fail.
    """
    candidates = _enumerate_vision_candidates(
        provider_name=provider_name,
        model_id=model_id,
    )
    if not candidates:
        # Reuse the precise error from _pick_vision_provider
        _pick_vision_provider(provider_name=provider_name, model_id=model_id)

    last_exc: Exception | None = None
    transient_failures: list[str] = []
    for chosen_provider, chosen_model, adapter_cls in candidates:
        logger.info("ImageCaptionTool: calling %s/%s", chosen_provider, chosen_model)

        api_key_env = _provider_api_key_env(chosen_provider)
        init_kwargs: dict[str, Any] = {"model_name": chosen_model}
        key = os.environ.get(api_key_env, "") if api_key_env else ""
        if key:
            init_kwargs["api_key"] = key

        try:
            adapter = adapter_cls(**init_kwargs)
            adapter.load()

            if chosen_provider == "openai":
                caption_text = _call_openai_vision(
                    adapter.client, chosen_model, prompt, source, detail
                )
            elif chosen_provider == "gemini":
                caption_text = _call_gemini_vision(adapter, prompt, source)
            else:
                from effgen.models.base import GenerationConfig
                data_url = _image_to_base64_url(source)
                multimodal: list[Any] = [
                    prompt,
                    {"type": "image_url", "image_url": {"url": data_url, "detail": detail}},
                ]
                gen_cfg = GenerationConfig(max_tokens=512, temperature=0.3)
                result = adapter.generate(multimodal, gen_cfg)
                caption_text = (result.text or "").strip()
        except Exception as exc:
            last_exc = exc
            if _is_transient_provider_error(exc) and len(candidates) > 1:
                transient_failures.append(f"{chosen_provider}/{chosen_model}")
                logger.warning(
                    "ImageCaptionTool: %s/%s transient failure (%s); "
                    "trying next provider",
                    chosen_provider,
                    chosen_model,
                    type(exc).__name__,
                )
                continue
            raise

        return {
            "success": True,
            "data": {
                "caption": caption_text,
                "prompt": prompt,
                "provider": chosen_provider,
                "model": chosen_model,
                "detail": detail,
                "fallbacks_used": transient_failures or None,
            },
            "caption": caption_text,
            "provider": chosen_provider,
            "model": chosen_model,
            "error": None,
        }

    # All candidates failed with transient errors
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# BaseTool subclass
# ---------------------------------------------------------------------------

class ImageCaptionTool(BaseTool):
    """
    Generate natural-language captions for images using a vision model.

    Automatically selects the best available vision-capable provider via the
    effGen model router. Requires at least one of:
      - OPENAI_API_KEY   (gpt-4o-mini)
      - GOOGLE_API_KEY   (gemini-3.1-flash-lite)

    Accepts image file paths OR raw bytes.

    Operations:
      - caption  : concise one-sentence description
      - describe : detailed multi-sentence description
    """

    def __init__(
        self,
        provider: str | None = None,
        model_id: str | None = None,
        allowed_directories: list[str] | None = None,
    ) -> None:
        """
        Args:
            provider: Force a specific provider (e.g. "openai", "gemini").
            model_id: Force a specific model (e.g. "gpt-4o-mini").
                      If provider is given but model_id is not, the default
                      vision model for that provider is auto-selected.
            allowed_directories: Roots an image path may be read from. By
                default any path is allowed except protected system and
                credential locations (/etc, /proc, ~/.ssh, cloud creds, …),
                which are always refused. Pass a list to confine reads to
                those roots only. This bounds what bytes may be uploaded to
                the vision provider.
        """
        self._forced_provider = provider
        self._forced_model = model_id
        self._allowed_dirs = normalize_allowed_dirs(allowed_directories)

        super().__init__(
            metadata=ToolMetadata(
                name="image_caption",
                description=(
                    "Generate image captions using a vision model (OpenAI gpt-4o-mini, "
                    "Google Gemini 2.0-flash, etc.). "
                    "Auto-selects the best available vision provider. "
                    "Operations: caption (brief), describe (detailed). "
                    "Accepts file paths or raw image bytes."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'caption' or 'describe'.",
                        required=True,
                        enum=["caption", "describe"],
                    ),
                    ParameterSpec(
                        name="image_path",
                        type=ParameterType.STRING,
                        description="Path to the image. Mutually exclusive with image_bytes.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="image_bytes",
                        type=ParameterType.STRING,
                        description="Base64-encoded image. Mutually exclusive with image_path.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="prompt",
                        type=ParameterType.STRING,
                        description="Custom instruction for the vision model. Defaults to a description prompt.",
                        required=False,
                        default="Describe this image.",
                    ),
                    ParameterSpec(
                        name="detail",
                        type=ParameterType.STRING,
                        description="Detail level: 'high', 'low', or 'auto' (default).",
                        required=False,
                        enum=["high", "low", "auto"],
                        default="auto",
                    ),
                ],
                timeout_seconds=60,
                tags=["image", "vision", "caption", "multimodal", "llm"],
                examples=[
                    {"operation": "caption", "image_path": "/path/to/photo.jpg"},
                    {"operation": "describe", "image_path": "/path/to/photo.jpg", "detail": "high"},
                ],
            )
        )

    def validate_parameters(self, **kwargs: Any) -> tuple[bool, str | None]:
        """Validate parameters, letting raw ``image_bytes`` pass the string check."""
        if isinstance(kwargs.get("image_bytes"), bytes | bytearray):
            kwargs = dict(kwargs)
            kwargs["image_bytes"] = "__raw_bytes__"
        return super().validate_parameters(**kwargs)

    def _resolve_source(
        self,
        image_path: str | None,
        image_bytes: str | bytes | bytearray | None,
    ) -> str | bytes:
        if image_path and image_bytes:
            raise ValueError("Provide image_path OR image_bytes, not both.")
        if image_path:
            return str(confine_path(image_path, self._allowed_dirs))
        if image_bytes is not None:
            if isinstance(image_bytes, bytes | bytearray):
                return bytes(image_bytes)
            encoded = image_bytes.strip()
            if encoded.startswith("data:image/") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            import binascii
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image_bytes must be raw bytes or a base64-encoded string.") from exc
        raise ValueError("Either image_path or image_bytes is required.")

    async def _execute(
        self,
        operation: str,
        image_path: str | None = None,
        image_bytes: str | bytes | bytearray | None = None,
        prompt: str = "Describe this image.",
        detail: str = "auto",
        **kwargs: Any,
    ) -> dict[str, Any]:
        source = self._resolve_source(image_path, image_bytes)
        op = operation.lower()

        if op == "caption":
            actual_prompt = prompt if prompt != "Describe this image." else "Provide a concise one-sentence caption for this image."
        elif op == "describe":
            actual_prompt = prompt if prompt != "Describe this image." else (
                "Provide a detailed description of this image. "
                "Cover the main subjects, setting, colors, actions, and any notable details."
            )
        else:
            raise ValueError(f"Unknown operation: {operation!r}. Use 'caption' or 'describe'.")

        return await asyncio.to_thread(
            _caption_sync,
            source,
            actual_prompt,
            detail,
            self._forced_provider,
            self._forced_model,
        )
