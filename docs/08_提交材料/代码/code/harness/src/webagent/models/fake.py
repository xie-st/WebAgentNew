"""Offline scripted model for A1/A2 tests.

This is an EffGen ``BaseModel`` so A2's interactive entry can run the real
framework path without any network or model weights. Scripts are plain text
responses; tool-call parsing stays in the framework/A2, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from effgen.models.base import BaseModel, GenerationConfig, GenerationResult, ModelType, TokenCount


@dataclass(frozen=True)
class FakeResponse:
    text: str = ""
    finish_reason: str = "stop"
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("fake response text must be a string")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("fake response error must be a string")
        if self.error is not None and self.text:
            raise ValueError("a fake response cannot be both text and error")

    @classmethod
    def from_value(cls, value: Any) -> "FakeResponse":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, Mapping):
            return cls(
                text=value.get("text", ""),
                finish_reason=value.get("finish_reason", "stop"),
                error=value.get("error"),
                metadata=value.get("metadata") or {},
            )
        raise ValueError(f"unsupported fake response value {type(value).__name__}")


@dataclass(frozen=True)
class FakeCall:
    index: int
    prompt: str
    response_index: int | None
    response_text: str | None
    error: str | None
    config: GenerationConfig | None = None


class ScriptedModel(BaseModel):
    """A deterministic text model with a scripted sequence of responses."""

    def __init__(
        self,
        responses: Iterable[str | Mapping[str, Any] | FakeResponse] = (),
        *,
        model_name: str = "fake-scripted",
        context_length: int = 32_768,
        model_type: ModelType = ModelType.TRANSFORMERS,
    ) -> None:
        super().__init__(model_name, model_type, context_length=context_length)
        self._responses = tuple(FakeResponse.from_value(value) for value in responses)
        self._index = 0
        self.calls: list[FakeCall] = []
        self._stream_chunks: list[str] = []

    @classmethod
    def from_script(cls, path: str | Path) -> "ScriptedModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            responses = payload.get("responses", [])
            model_name = payload.get("model_name", "fake-scripted")
            context_length = int(payload.get("context_length", 32_768))
        elif isinstance(payload, list):
            responses = payload
            model_name = "fake-scripted"
            context_length = 32_768
        else:
            raise ValueError("fake script must be a list or an object with responses")
        if not isinstance(responses, list):
            raise ValueError("fake script responses must be a list")
        return cls(responses, model_name=model_name, context_length=context_length)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(call.prompt for call in self.calls)

    @property
    def remaining_responses(self) -> int:
        return len(self._responses) - self._index

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def _next_response(self) -> tuple[int, FakeResponse]:
        if self._index >= len(self._responses):
            raise RuntimeError(
                f"fake model script exhausted after {self._index} response(s); no network fallback"
            )
        response_index = self._index
        response = self._responses[self._index]
        self._index += 1
        return response_index, response

    def _record(
        self,
        prompt: str,
        config: GenerationConfig | None,
        response_index: int | None,
        response: FakeResponse | None,
        error: str | None,
    ) -> None:
        self.calls.append(
            FakeCall(
                index=len(self.calls),
                prompt=prompt,
                response_index=response_index,
                response_text=None if response is None else response.text,
                error=error,
                config=config,
            )
        )

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        response_index, response = self._next_response()
        self._record(prompt, config, response_index, response, response.error)
        if response.error is not None:
            raise RuntimeError(response.error)
        tokens = self.count_tokens(prompt).count + self.count_tokens(response.text).count
        metadata = dict(response.metadata)
        metadata.setdefault("fake_response_index", response_index)
        return GenerationResult(
            text=response.text,
            tokens_used=tokens,
            finish_reason=response.finish_reason,
            model_name=self.model_name,
            metadata=metadata,
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterable[str]:
        response_index, response = self._next_response()
        self._record(prompt, config, response_index, response, response.error)
        if response.error is not None:
            raise RuntimeError(response.error)
        if not response.text:
            return
        midpoint = max(1, len(response.text) // 2)
        yield response.text[:midpoint]
        yield response.text[midpoint:]

    def count_tokens(self, text: str) -> TokenCount:
        # Deterministic local count for the fake; it is not a target-model count.
        return TokenCount(count=0 if not text else max(1, (len(text) + 3) // 4), model_name=self.model_name)

    def get_context_length(self) -> int:
        return int(self._context_length or 32_768)

    def supports_tool_calling(self) -> bool:
        return False

    def supports_conversation(self) -> bool:
        return False
