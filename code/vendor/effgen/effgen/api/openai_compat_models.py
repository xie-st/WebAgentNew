"""Request schemas, model aliases and the runner result for the OpenAI-compatible API.

Defines the pydantic request models for ``/v1/chat/completions`` and
``/v1/completions``, the OpenAI-name → local-model alias tables, and
:class:`RunnerResult` — the structured value a runner returns so responses can
carry real usage and cost. Every name is re-exported from
``effgen.api.openai_compat``; import from there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore

    def Field(default: Any = None, **kwargs: Any) -> Any:  # type: ignore
        """Stand-in for ``pydantic.Field`` when pydantic is not installed."""
        return default


# ---------------------------------------------------------------------------
# Model aliasing
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "gpt-4": "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4-turbo": "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4o": "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4o-mini": "Qwen/Qwen2.5-3B-Instruct",
    "gpt-3.5-turbo": "Qwen/Qwen2.5-3B-Instruct",
    "gpt-3.5-turbo-instruct": "Qwen/Qwen2.5-3B-Instruct",
}

# Names that route to the server's configured default model. A caller that has
# no particular model in mind (including the native client's zero-argument
# ``chat("hi")``) can send ``"effgen-default"`` or ``"default"`` and get an
# answer without knowing a concrete id. The target is read from
# ``EFFGEN_DEFAULT_MODEL`` at request time, falling back to a small local model.
DEFAULT_MODEL_ALIASES: frozenset[str] = frozenset({"effgen-default", "default"})
_FALLBACK_DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def default_model_id() -> str:
    """Return the model id the ``effgen-default``/``default`` names resolve to.

    Controlled by ``EFFGEN_DEFAULT_MODEL``; defaults to a small local model when
    the variable is unset or blank.
    """
    return os.environ.get("EFFGEN_DEFAULT_MODEL", "").strip() or _FALLBACK_DEFAULT_MODEL


def resolve_model_alias(model: str) -> str:
    """Resolve an OpenAI model name to a local effGen model id."""
    if model in DEFAULT_MODEL_ALIASES:
        return default_model_id()
    return MODEL_ALIASES.get(model, model)


# ---------------------------------------------------------------------------
# Runner result
# ---------------------------------------------------------------------------


@dataclass
class RunnerResult:
    """Structured result a runner may return instead of a bare string.

    Returning this (rather than a plain ``str``) lets the OpenAI-compatible
    layer surface **real** usage — provider-reported counts when the upstream
    API returns them, tokenizer counts for local models — instead of
    re-estimating from the response text. It also carries the *resolved* effGen
    model so the response can document which model actually ran when an OpenAI
    alias (e.g. ``gpt-4``) was mapped to a local model.

    A runner may still return a plain ``str`` (or, for streaming, an iterator of
    string chunks); the router falls back to a tokenizer estimate in that case.
    """

    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    resolved_model: str | None = None
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):  # type: ignore[misc]
    """One chat message in an OpenAI-compatible request."""

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):  # type: ignore[misc]
    """Request body for ``POST /v1/chat/completions``."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    max_tokens: int | None = Field(default=None, gt=0)
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    # OpenAI streaming option: {"include_usage": true} requests a final
    # usage-only chunk (choices: []) after the content chunks.
    stream_options: dict[str, Any] | None = None


class CompletionRequest(BaseModel):  # type: ignore[misc]
    """Request body for the legacy ``POST /v1/completions``."""

    model: str
    prompt: str | list[str]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    max_tokens: int | None = Field(default=16, gt=0)
    stop: str | list[str] | None = None
    user: str | None = None
    seed: int | None = None
