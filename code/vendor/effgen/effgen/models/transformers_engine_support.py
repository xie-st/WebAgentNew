"""Cache, offline-mode and special-token helpers for the Transformers engine.

Module-level helpers shared by the engine's loading, generation and streaming
concerns. Nothing here imports torch or transformers, so it stays importable on
an install without the local-inference extras.
"""

from __future__ import annotations

import os
from contextlib import contextmanager


class GPUPlacementError(RuntimeError):
    """Raised when ``require_gpu`` is set but the model can't fit on the GPU."""


class ModelNotCachedError(RuntimeError):
    """Raised when a model isn't in the local cache and offline mode is set."""


def _offline_mode_active() -> bool:
    """True if HuggingFace offline mode is set via the environment."""
    return any(
        os.environ.get(var, "").strip().lower() in ("1", "true", "yes", "on")
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _list_cached_model_repos(limit: int = 20) -> list[str]:
    """Return locally-cached HuggingFace model repo ids (best effort, sorted)."""
    try:
        from huggingface_hub import scan_cache_dir

        repos = sorted(
            r.repo_id for r in scan_cache_dir().repos if r.repo_type == "model"
        )
        return repos[:limit]
    except Exception:
        return []


def _is_cache_miss_error(exc: Exception) -> bool:
    """True if *exc* is a HuggingFace "not found locally" / offline error."""
    text = str(exc).lower()
    return (
        "couldn't find them in the cached files" in text
        or ("can't load" in text and "offline" in text)
        or "offlinemodeisenabled" in text
        or "localentrynotfound" in text
        or type(exc).__name__ in ("LocalEntryNotFoundError", "OfflineModeIsEnabled")
    )


def _reraise_if_classified(exc: Exception) -> None:
    """Re-raise *exc* unwrapped when it already carries retry classification.

    A timeout raised by ``effgen.reliability.timeouts.with_timeout()`` around
    a local generate call must propagate as-is instead of being flattened
    into a generic ``RuntimeError`` by the callers' blanket exception
    handlers below — flattening discards the type information
    ``is_transient_error()`` relies on to retry it correctly.
    """
    from effgen.reliability.timeouts import TimeoutError as EffGenTimeoutError

    if isinstance(exc, EffGenTimeoutError):
        raise exc


# Native tool-call delimiters that the downstream parser (core.tool_calling)
# needs to see; these must survive the special-token strip on the tool path.
_TOOL_CALL_DELIMITERS = frozenset({
    "<tool_call>", "</tool_call>", "<|python_tag|>", "[TOOL_CALLS]", "<function=",
})
# Markers that render as literal text on some tokenizers even though they are
# not always registered as special tokens (belt-and-suspenders).
_COMMON_END_MARKERS = frozenset({
    "<|im_end|>", "</s>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>",
})


def _strip_special_tokens_keep_tool_calls(text: str, tokenizer) -> str:
    """Remove chat-template special tokens from decoded tool-path text.

    On the native-tool-calling path the model output is decoded with
    ``skip_special_tokens=False`` so that tool-call delimiters survive for the
    parser. That also lets chat-template turn/end markers through, which must
    not reach the user-visible answer. The strip set is derived from the
    tokenizer's own ``all_special_tokens`` (minus the tool-call delimiters) so
    it stays correct across model families — e.g. Gemma's ``<end_of_turn>`` /
    ``<eos>`` / ``<start_of_turn>`` and Qwen's ``<|im_start|>``, which a fixed
    list silently let leak through.
    """
    strip = set(getattr(tokenizer, "all_special_tokens", ()) or ())
    strip -= _TOOL_CALL_DELIMITERS
    strip |= _COMMON_END_MARKERS
    # Strip longest-first so a marker that contains another is removed whole.
    for marker in sorted(strip, key=len, reverse=True):
        if marker:
            text = text.replace(marker, "")
    return text.strip()


@contextmanager
def _quiet_model_load():
    """Silence Transformers' load-time chatter (weight-loading progress bars and
    INFO reports) for the duration of a model load, restoring the previous state
    afterwards. Loading diagnostics belong in logs at debug level, not on stdout
    for every inference call.
    """
    try:
        from transformers.utils import logging as hf_logging
    except Exception:  # pragma: no cover - transformers always present here
        yield
        return

    prev_verbosity = hf_logging.get_verbosity()
    try:
        prev_progress = hf_logging.is_progress_bar_enabled()
    except Exception:
        prev_progress = True
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(prev_verbosity)
        if prev_progress:
            hf_logging.enable_progress_bar()
