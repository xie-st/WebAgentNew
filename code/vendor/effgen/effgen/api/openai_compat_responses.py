"""Response and usage builders for the OpenAI-compatible API.

Builds the ``chat.completion``, ``chat.completion.chunk``, usage-only chunk and
``text_completion`` payloads, and counts tokens with a real tokenizer when one
is available. Every name is re-exported from ``effgen.api.openai_compat``;
import from there.
"""
from __future__ import annotations

import time
import uuid
from typing import Any


def _now() -> int:
    return int(time.time())


def _chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _cmpl_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def _get_tiktoken_encoder() -> Any:
    """Return the cached ``cl100k_base`` encoding, or ``None`` if it is unavailable.

    Unavailable covers both tiktoken not being installed and its BPE data being
    neither cached nor reachable; the adapters resolve it the same way.
    """
    from effgen.models._adapter_utils import get_bpe_encoding

    return get_bpe_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in *text* using a real tokenizer when available.

    Prefers a BPE tokenizer (tiktoken ``cl100k_base``) over the legacy
    ``len(text) // 4`` heuristic so usage/cost numbers track real counts, and
    falls back to that heuristic when the encoding cannot be loaded. The server
    runner should supply provider- or model-reported usage when it has it; this
    is the estimate of last resort.
    """
    from effgen.models._adapter_utils import estimate_tokens

    return estimate_tokens(text)


def _approx_tokens(text: str) -> int:  # retained for back-compat
    return count_tokens(text)


def build_chat_completion(
    model: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int | None = None,
    finish_reason: str = "stop",
    effgen_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-format chat.completion response dict.

    ``completion_tokens`` may be supplied directly (e.g. provider-reported or
    tokenizer counts); when ``None`` it is estimated from ``content``.
    ``effgen_meta`` adds a non-standard ``effgen`` object documenting the
    requested vs. resolved model (OpenAI clients ignore unknown top-level keys).

    Args:
        model: The model id reported in the response.
        content: The assistant's answer text.
        tool_calls: Tool calls the turn made, in OpenAI's shape.
        prompt_tokens: Input tokens the call consumed.
        completion_tokens: Output tokens, estimated from *content* when ``None``.
        finish_reason: Why the turn ended.
        effgen_meta: Extra effGen fields, such as the resolved model and cost.

    Returns:
        The response body, ready to serialize.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    if completion_tokens is None:
        completion_tokens = _approx_tokens(content or "")
    resp: dict[str, Any] = {
        "id": _chat_id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if effgen_meta:
        resp["effgen"] = effgen_meta
    return resp


def build_chat_chunk(
    model: str,
    delta_content: str,
    *,
    chat_id: str | None = None,
    finish_reason: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Build one chat.completion.chunk for SSE streaming.

    Args:
        model: The model id reported in the chunk.
        delta_content: The text this chunk adds.
        chat_id: The id shared by every chunk of one response, generated when
            absent.
        finish_reason: Set on the last chunk to say why the turn ended.
        role: Set on the first chunk to open the assistant message.

    Returns:
        One chunk body, ready to serialize into an SSE event.
    """
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if delta_content:
        delta["content"] = delta_content
    return {
        "id": chat_id or _chat_id(),
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def build_usage_chunk(
    model: str,
    *,
    chat_id: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    effgen_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a final usage-only streaming chunk (OpenAI ``include_usage`` form).

    Per the OpenAI streaming spec, when ``stream_options.include_usage`` is set
    the server emits one extra chunk after the content chunks whose ``choices``
    is empty and which carries the ``usage`` totals so clients can reconcile
    billing for streamed requests. ``effgen_meta`` adds the same non-standard
    ``effgen`` object the non-streaming response carries — including
    ``cost_usd`` — so a client tallying spend reads one number from the server
    on both paths instead of re-deriving a price from the catalog itself.

    Args:
        model: The model id reported in the chunk.
        chat_id: The id shared by every chunk of one response.
        prompt_tokens: Input tokens the call consumed.
        completion_tokens: Output tokens the call produced.
        effgen_meta: Extra effGen fields, such as the resolved model and cost.

    Returns:
        The trailing usage-only chunk, ready to serialize.
    """
    chunk: dict[str, Any] = {
        "id": chat_id or _chat_id(),
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if effgen_meta:
        chunk["effgen"] = effgen_meta
    return chunk


def build_text_completion(
    model: str,
    text: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a legacy ``text_completion`` response envelope for *text*.

    Args:
        model: The model id reported in the response.
        text: The completion text.
        prompt_tokens: Input tokens the call consumed.
        completion_tokens: Output tokens, estimated from *text* when ``None``.
        finish_reason: Why the completion ended.

    Returns:
        The response body, ready to serialize.
    """
    if completion_tokens is None:
        completion_tokens = _approx_tokens(text)
    return {
        "id": _cmpl_id(),
        "object": "text_completion",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
