"""Official-platform text model adapter for EffGen.

Every request ends at the frozen SDK's ``third_party_agent.model.generate``.
The platform owns the model, sampling settings, token accounting and pricing, so
this adapter deliberately passes only ``messages``: no model name, temperature,
output limit, tools, images or tool-role messages. The complete EffGen prompt is
one string user message; an explicitly supplied system prompt remains a system
message rather than being flattened into the user turn.

The SDK returns only ``.text`` and ``.request_id`` and reports no token usage.
Exact input/output counts are therefore ``None`` (not zero), while local
estimates are recorded in separate fields. Every actual call, including an
explicit retry after an SDK failure, gets one :class:`OfficialCallRecord` and one
``model_call`` event through the optional call sink.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping

from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)

from ..budget import BudgetLimits, enforce_request_limits
from ..config import ModelConfig
from ..errors import RequestLimitExceeded
from ..redaction import DEFAULT_REDACTOR, Redactor

_PLATFORM_USAGE_SOURCE = "missing"
_PLATFORM_TOKEN_SOURCE = "platform_usage_unavailable"


class OfficialModelError(RuntimeError):
    """The platform model call failed or returned an unusable result.

    Messages name only the exception type and never echo a prompt, token or
    provider response body.
    """


@dataclass(frozen=True)
class OfficialCallRecord:
    """One actual call to ``third_party_agent.model.generate``.

    ``request_id`` is the platform-issued id. ``local_request_id`` identifies the
    call even when the SDK raised before returning an id. Usage is always
    ``None``/``missing``: the official SDK exposes no token accounting. Estimates
    are separate fields and are always labelled as estimates.
    """

    index: int
    request_id: str | None
    local_request_id: str
    status: str
    latency_ms: float
    messages_count: int
    has_system_message: bool
    input_chars: int
    output_chars: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    usage_source: str = _PLATFORM_USAGE_SOURCE
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None
    finish_reason: str | None = None
    failure_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OfficialModel(BaseModel):
    """EffGen ``BaseModel`` carrying the official platform's text gateway."""

    # The bridge must not write a token-less placeholder event before the call:
    # this adapter emits the real per-call event through its sink afterwards.
    records_model_calls = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        model_api: Any | None = None,
        system_prompt: str | None = None,
        call_sink: Callable[[OfficialCallRecord], None] | None = None,
        redactor: Redactor | None = None,
        token_count_source: str | None = None,
        limits: BudgetLimits | None = None,
        mandatory_provider: Callable[[], Any] | None = None,
        limit_sink: Callable[[RequestLimitExceeded], None] | None = None,
    ) -> None:
        if config.provider != "official":
            raise ValueError("OfficialModel needs model.provider=official")
        super().__init__(config.name, ModelType.OPENAI, context_length=config.context_length)
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string or None")
        self.config = config
        self._model_api = model_api
        self._system_prompt = system_prompt if system_prompt else None
        self._call_sink = call_sink
        self._limits = limits or BudgetLimits()
        self._mandatory_provider = mandatory_provider
        self._limit_sink = limit_sink
        self.redactor = redactor or DEFAULT_REDACTOR
        self.token_count_source = token_count_source or config.token_count_source
        self.calls: list[OfficialCallRecord] = []
        self._api: Any | None = None
        self._encoder: Any = None
        self._closed = False

    @classmethod
    def from_app_config(cls, app_config: Any, **kwargs: Any) -> "OfficialModel":
        return cls(app_config.model, **kwargs)

    # --------------------------------------------------------------- lifecycle
    def set_call_sink(self, sink: Callable[[OfficialCallRecord], None] | None) -> None:
        self._call_sink = sink

    def set_request_limits(
        self,
        limits: BudgetLimits,
        mandatory_provider: Callable[[], Any] | None = None,
        limit_sink: Callable[[RequestLimitExceeded], None] | None = None,
    ) -> None:
        """Install the project request limits checked at the final SDK exit."""

        if not isinstance(limits, BudgetLimits):
            raise ValueError("limits must be BudgetLimits")
        self._limits = limits
        self._mandatory_provider = mandatory_provider
        self._limit_sink = limit_sink

    def load(self) -> None:
        if self._closed:
            raise OfficialModelError("official model has been unloaded")
        if self._is_loaded:
            return
        self._api = self._resolve_api()
        self._is_loaded = True

    def unload(self) -> None:
        self._api = None
        self._is_loaded = False
        self._closed = True

    # ---------------------------------------------------------------- messages
    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    def build_messages(self, prompt: Any) -> list[dict[str, str]]:
        """Build the SDK message list, preserving supplied roles.

        EffGen sends one flat string, which becomes a user message; an explicit
        system prompt stays a system message. A caller may also supply a list of
        role/content text messages directly (for diagnostics); tool/image/other
        roles and non-text content are still refused.
        """

        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            messages: list[dict[str, str]] = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, (list, tuple)):
            messages = []
            for item in prompt:
                if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
                    raise OfficialModelError(
                        "official messages may only contain role and content"
                    )
                messages.append({"role": item["role"], "content": item["content"]})
            if not messages:
                raise ValueError("messages must not be empty")
        else:
            raise ValueError("prompt must be a string or a list of role/content messages")
        # This fixed shape is the contract: role/content text only.
        for message in messages:
            if set(message) != {"role", "content"}:
                raise OfficialModelError("official messages may only contain role and content")
            if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                raise OfficialModelError("official message role/content must be strings")
            if message["role"] not in {"system", "user", "assistant"}:
                raise OfficialModelError("official messages cannot use a tool role")
            if not message["content"].strip():
                raise ValueError("official message content must not be empty")
        return messages

    # ---------------------------------------------------------------- generation
    def generate(
        self,
        prompt: Any,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        # GenerationConfig is intentionally ignored: the platform fixes the model
        # and every inference parameter. Accepting it keeps the BaseModel contract.
        del config, kwargs
        messages = self.build_messages(prompt)
        local_request_id = "oreq-" + uuid.uuid4().hex
        # Final request exit: count messages and serialize the body once, before
        # the SDK is called. No message is trimmed here; an overflow is refused.
        measurement = self._check_request_limits(messages, local_request_id)
        if not self._is_loaded:
            self.load()
        api = self._api
        if api is None or not callable(getattr(api, "generate", None)):
            raise OfficialModelError("third_party_agent.model.generate is unavailable")
        started = time.monotonic()
        try:
            response = api.generate(messages)
        except Exception as exc:  # noqa: BLE001 - recorded and reported by type
            latency_ms = (time.monotonic() - started) * 1000.0
            record = self._record(
                request_id=None,
                local_request_id=local_request_id,
                status="error",
                latency_ms=latency_ms,
                messages_count=len(messages),
                has_system_message=any(message["role"] == "system" for message in messages),
                input_chars=sum(len(message["content"]) for message in messages),
                output_chars=0,
                failure_type=type(exc).__name__,
                error_message=f"platform model call failed: {type(exc).__name__}",
                estimated_input=self._estimate_messages(messages),
            )
            raise OfficialModelError(
                f"platform model call failed ({record.failure_type}); "
                f"platform request id unavailable"
            ) from exc

        latency_ms = (time.monotonic() - started) * 1000.0
        text = getattr(response, "text", None)
        platform_request_id = getattr(response, "request_id", None)
        if not isinstance(text, str):
            record = self._record(
                request_id=platform_request_id if isinstance(platform_request_id, str) else None,
                local_request_id=local_request_id,
                status="invalid_response",
                latency_ms=latency_ms,
                messages_count=len(messages),
                has_system_message=any(message["role"] == "system" for message in messages),
                input_chars=sum(len(message["content"]) for message in messages),
                output_chars=0,
                failure_type="invalid_response",
                error_message="platform model result has no text string",
                estimated_input=self._estimate_messages(messages),
            )
            raise OfficialModelError(
                f"platform model result has no text string; local request {record.local_request_id}"
            )
        request_id = platform_request_id if isinstance(platform_request_id, str) and platform_request_id else None
        record = self._record(
            request_id=request_id,
            local_request_id=local_request_id,
            status="ok",
            latency_ms=latency_ms,
            messages_count=len(messages),
            has_system_message=any(message["role"] == "system" for message in messages),
            input_chars=sum(len(message["content"]) for message in messages),
            output_chars=len(text),
            finish_reason="stop",
            estimated_input=self._estimate_messages(messages),
            estimated_output=self._estimate(text),
        )
        metadata = {
            "request_id": request_id,
            "local_request_id": local_request_id,
            "request_id_source": "platform" if request_id else "unavailable",
            "request_message_count": measurement.message_count,
            "request_bytes": measurement.request_bytes,
            "request_mandatory_bytes": measurement.mandatory_request_bytes,
            "usage_source": _PLATFORM_USAGE_SOURCE,
            "token_count_source": _PLATFORM_TOKEN_SOURCE,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "estimated_input_tokens": record.estimated_input_tokens,
            "estimated_output_tokens": record.estimated_output_tokens,
            "official_model_call": record.to_dict(),
            "retries": 0,
        }
        return GenerationResult(
            text=text,
            tokens_used=0,
            finish_reason="stop",
            model_name=self.model_name,
            metadata=metadata,
        )

    def generate_stream(
        self,
        prompt: Any,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterable[str]:
        """One request, exposed as one chunk; phase 1 is non-streaming."""

        result = self.generate(prompt, config, **kwargs)
        return iter((result.text,) if result.text else ())

    # --------------------------------------------------------------- accounting
    def count_tokens(self, text: str) -> TokenCount:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return TokenCount(count=self._encode_count(text), model_name=self.model_name)

    def get_context_length(self) -> int:
        return int(self.config.context_length)

    def supports_tool_calling(self) -> bool:
        return False

    def supports_conversation(self) -> bool:
        return False

    def tool_call_support(self) -> str:
        return "none"

    def prompt_detail(self) -> str:
        return "full"

    # ---------------------------------------------------------------- internals
    def _check_request_limits(
        self,
        messages: Iterable[Mapping[str, str]],
        local_request_id: str,
    ) -> Any:
        mandatory = None
        if self._mandatory_provider is not None:
            try:
                supplied = self._mandatory_provider()
            except Exception:  # noqa: BLE001 - the limit still applies without it
                supplied = None
            if supplied is not None:
                mandatory = list(supplied)
        try:
            return enforce_request_limits(messages, self._limits, mandatory=mandatory)
        except RequestLimitExceeded as exc:
            body = list(messages)
            self._record(
                request_id=None,
                local_request_id=local_request_id,
                status="rejected",
                latency_ms=0.0,
                messages_count=len(body),
                has_system_message=any(message["role"] == "system" for message in body),
                input_chars=sum(len(message["content"]) for message in body),
                output_chars=0,
                failure_type="request_limit",
                error_message=f"{exc.limit_name} limit exceeded: {exc}",
                estimated_input=self._estimate_messages(body),
            )
            if self._limit_sink is not None:
                try:
                    self._limit_sink(exc)
                except Exception:  # noqa: BLE001 - reporting never replaces the error
                    pass
            # EffGen retries "unknown" errors three times. A project limit is
            # deterministic, so mark it invalid_request through EffGen's own
            # documented error context; the refusal then fails fast.
            try:
                from effgen.models.errors import error_context_dict

                exc.error_context = error_context_dict(
                    "official", self.model_name, "generate", "invalid_request"
                )
            except Exception:  # noqa: BLE001 - classification is best-effort
                pass
            raise

    def _resolve_api(self) -> Any:
        if self._model_api is not None:
            api = self._model_api
        else:
            try:
                from third_party_agent import model as api  # type: ignore[import-not-found]
            except Exception as exc:  # noqa: BLE001 - import failures are actionable
                raise OfficialModelError(
                    "cannot import third_party_agent.model; run the client from "
                    "code/official_client or pass model_api explicitly"
                ) from exc
            self._model_api = api
        if not callable(getattr(api, "generate", None)):
            raise OfficialModelError("model_api must expose generate(messages)")
        return api

    def _record(
        self,
        *,
        request_id: str | None,
        local_request_id: str,
        status: str,
        latency_ms: float,
        messages_count: int,
        has_system_message: bool,
        input_chars: int,
        output_chars: int,
        finish_reason: str | None = None,
        failure_type: str | None = None,
        error_message: str | None = None,
        estimated_input: int | None = None,
        estimated_output: int | None = None,
    ) -> OfficialCallRecord:
        record = OfficialCallRecord(
            index=len(self.calls),
            request_id=request_id,
            local_request_id=local_request_id,
            status=status,
            latency_ms=round(float(latency_ms), 3),
            messages_count=messages_count,
            has_system_message=has_system_message,
            input_chars=input_chars,
            output_chars=output_chars,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            usage_source=_PLATFORM_USAGE_SOURCE,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            finish_reason=finish_reason,
            failure_type=failure_type,
            error_message=error_message,
        )
        payload = self.redactor.redact(record.to_dict())
        record = OfficialCallRecord(**payload)
        self.calls.append(record)
        if self._call_sink is not None:
            self._call_sink(record)
        return record

    def _estimate_messages(self, messages: Iterable[Mapping[str, str]]) -> int:
        combined = "\n".join(str(message.get("content", "")) for message in messages)
        return self._estimate(combined)

    def _estimate(self, text: str) -> int:
        try:
            return self._encode_count(text)
        except Exception:  # noqa: BLE001 - an estimate never blocks a call
            return 0 if not text else max(1, (len(text) + 3) // 4)

    def _encode_count(self, text: str) -> int:
        value = self._encoder
        if value is None:
            tokenizer = self.config.tokenizer
            if tokenizer.startswith("tiktoken:"):
                import tiktoken

                value = tiktoken.get_encoding(tokenizer.split(":", 1)[1])
            else:
                value = False
            self._encoder = value
        if value is False:
            return 0 if not text else max(1, (len(text) + 3) // 4)
        return len(value.encode(text, disallowed_special=()))  # type: ignore[union-attr]


__all__ = ["OfficialCallRecord", "OfficialModel", "OfficialModelError"]
