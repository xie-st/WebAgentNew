"""SSE streaming generators for the OpenAI-compatible API.

Produces the ``text/event-stream`` bodies for ``/v1/chat/completions`` and
``/v1/completions``: incremental content chunks, an optional final usage-only
chunk (``stream_options.include_usage``), and a terminal error event on a
mid-stream failure instead of a truncated stream. Every name is re-exported
from ``effgen.api.openai_compat``; import from there.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


def chat_sse_iter(
    result: Any,
    *,
    model: str,
    chat_id: str,
    prompt_tokens: int,
    effgen_meta: dict[str, Any],
    include_usage: bool,
) -> Iterator[str]:
    """Yield the SSE events for one streamed ``/v1/chat/completions`` response.

    ``result`` is what the runner returned for a streaming request — normally
    an iterator of string chunks, possibly exposing the completed call's real
    usage as ``.usage`` once exhausted; a non-iterable result is stringified
    into a single chunk.

    Args:
        result: What the runner returned for the streaming request.
        model: The model id reported in each chunk.
        chat_id: The id shared by every chunk of this response.
        prompt_tokens: Input tokens the call consumed.
        effgen_meta: Extra effGen fields carried on the usage chunk.
        include_usage: Emit the trailing usage-only chunk.
    """
    # Helper builders are looked up through the facade module at generation
    # time so a monkeypatch on ``effgen.api.openai_compat`` stays effective.
    from effgen.api import openai_compat as _compat

    first = _compat.build_chat_chunk(
        model, "", chat_id=chat_id, role="assistant"
    )
    first["effgen"] = effgen_meta
    yield f"data: {json.dumps(first)}\n\n"
    completion_text_parts: list[str] = []
    provider_completion_tokens: int | None = None
    started = False

    def _error_event(exc: Exception) -> str:
        """Terminal SSE event describing a mid-stream failure."""
        _, _etype, _ecode = _compat._classify_http(exc)
        return json.dumps({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": _compat._now(),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": _compat._error_payload(str(exc), _etype, _ecode)["error"],
        })

    try:
        for chunk in result:
            started = True
            text = str(chunk)
            completion_text_parts.append(text)
            payload = _compat.build_chat_chunk(
                model, text, chat_id=chat_id
            )
            yield f"data: {json.dumps(payload)}\n\n"
    except Exception as e:  # noqa: BLE001
        if isinstance(e, TypeError) and not started:
            # ``result`` was not iterable (a bare object) — stringify once.
            text = str(result)
            completion_text_parts.append(text)
            payload = _compat.build_chat_chunk(model, text, chat_id=chat_id)
            yield f"data: {json.dumps(payload)}\n\n"
        else:
            # Mid-stream failure: emit a terminal error event
            # (redacted) instead of truncating the stream or
            # buffering the error into a content chunk.
            yield f"data: {_error_event(e)}\n\n"
            yield "data: [DONE]\n\n"
            return
    # A streaming runner may expose the completed call's real usage
    # once exhausted (as ``.usage`` on the iterator), including the
    # cost the provider's token counts price out to.
    _stream_usage = getattr(result, "usage", None)
    _stream_cost: float | None = None
    _prompt_tokens = prompt_tokens
    if isinstance(_stream_usage, dict):
        provider_completion_tokens = _stream_usage.get("completion_tokens")
        _reported_prompt = _stream_usage.get("prompt_tokens")
        if _reported_prompt is not None:
            _prompt_tokens = int(_reported_prompt)
        _stream_cost = _stream_usage.get("cost_usd")
    # A runner that ended the run without an answer reports it here, so the
    # last chunk says the generation was cut short rather than completed.
    _finish_reason = str(getattr(result, "finish_reason", "stop") or "stop")
    final = _compat.build_chat_chunk(
        model, "", chat_id=chat_id, finish_reason=_finish_reason
    )
    yield f"data: {json.dumps(final)}\n\n"
    # Emit a final usage-only chunk when the client opts in via
    # stream_options.include_usage so streamed requests can be
    # reconciled for billing.
    if include_usage:
        completion_tokens = (
            provider_completion_tokens
            if provider_completion_tokens is not None
            else _compat.count_tokens("".join(completion_text_parts))
        )
        usage_meta = dict(effgen_meta)
        if _stream_cost is not None:
            usage_meta["cost_usd"] = _stream_cost
        usage_chunk = _compat.build_usage_chunk(
            model,
            chat_id=chat_id,
            prompt_tokens=_prompt_tokens,
            completion_tokens=completion_tokens,
            effgen_meta=usage_meta,
        )
        yield f"data: {json.dumps(usage_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def completion_sse_iter(
    result: Any,
    *,
    model: str,
    cmpl_id: str,
) -> Iterator[str]:
    """Yield the SSE events for one streamed ``/v1/completions`` response.

    ``result`` is what the runner returned for a streaming request — normally
    an iterator of string chunks; a non-iterable result is stringified into a
    single chunk.

    Args:
        result: What the runner returned for the streaming request.
        model: The model id reported in each chunk.
        cmpl_id: The id shared by every chunk of this response.
    """
    from effgen.api import openai_compat as _compat

    started = False

    def _chunk(text: str, finish_reason: str | None) -> dict:
        return {
            "id": cmpl_id,
            "object": "text_completion",
            "created": _compat._now(),
            "model": model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

    try:
        for chunk in result:
            started = True
            yield f"data: {json.dumps(_chunk(str(chunk), None))}\n\n"
    except Exception as e:  # noqa: BLE001
        if isinstance(e, TypeError) and not started:
            # ``result`` was not iterable (a bare object) — stringify once.
            yield f"data: {json.dumps(_chunk(str(result), 'stop'))}\n\n"
        else:
            # Mid-stream failure: emit a terminal error event
            # (redacted) rather than truncating the stream, matching
            # /v1/chat/completions.
            _, _etype, _ecode = _compat._classify_http(e)
            err_evt = _chunk("", "error")
            err_evt["error"] = _compat._error_payload(str(e), _etype, _ecode)["error"]
            yield f"data: {json.dumps(err_evt)}\n\n"
            yield "data: [DONE]\n\n"
            return
    yield "data: [DONE]\n\n"
