"""Real-model adapter for the phase-1 unified text endpoint.

The adapter implements EffGen's :class:`BaseModel` contract without exposing any
native tool/vision capability. It speaks the OpenAI-compatible
``/chat/completions`` text interface and deliberately sends one user message
containing the rendered ReAct prompt: the framework owns tool parsing, not the
provider.

Credentials are read from environment variables named by the configuration.
Values are never stored on the instance in a readable form, never included in a
call record, and never logged. Every HTTP attempt (including retries) produces
one :class:`ModelCallRecord`, with request id, latency and usage source; missing
usage is ``None``, never zero.

The module is offline-testable with a local fake HTTP service. No function here
falls back to a real service or model weights.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

import httpx
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)

from ..config import ConfigError, ModelConfig
from ..redaction import DEFAULT_REDACTOR, Redactor

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 2


class RealModelError(RuntimeError):
    """A model request failed after the configured retries.

    Messages are built from status codes and exception types, not response
    bodies or secret-bearing headers.
    """


def _endpoint_from_base_url(base_url: str) -> str:
    """Append ``/chat/completions`` unless the URL already names the endpoint."""

    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError("model base URL must be an absolute http(s) URL")
    path = parts.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = path + "/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _safe_endpoint(endpoint: str) -> str:
    """Endpoint for records: scheme/host/path only, no userinfo/query/fragment."""

    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


@dataclass(frozen=True)
class ModelCallRecord:
    """One HTTP attempt, including failed retries.

    ``input_tokens``/``output_tokens`` are ``None`` when the provider reported no
    usage. ``usage_source`` is ``provider`` or ``missing``; locally estimated
    token counts are kept separate in :attr:`estimated_input_tokens`.
    """

    index: int
    request_id: str | None
    provider_request_id: str | None
    model: str
    endpoint: str
    attempt: int
    retries_before: int
    status: str
    http_status: int | None
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    usage_source: str = "missing"
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None
    finish_reason: str | None = None
    failure_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _AttemptOutcome:
    text: str
    finish_reason: str
    provider_request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RealModel(BaseModel):
    """OpenAI-compatible text model with explicit call accounting."""

    # The bridge uses this marker to avoid writing a second, token-less
    # ``model_call`` event before the real request runs.
    records_model_calls = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = 0.5,
        transport: httpx.BaseTransport | None = None,
        call_sink: Callable[[ModelCallRecord], None] | None = None,
        deadline_check: Callable[[str], None] | None = None,
        redactor: Redactor | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(config.name, ModelType.OPENAI, context_length=config.context_length)
        if config.provider != "openai_compatible":
            raise ConfigError("RealModel needs model.provider=openai_compatible")
        if not config.base_url_env or not config.api_key_env:
            raise ConfigError("RealModel needs base_url_env and api_key_env names")
        if max_retries < 0:
            raise ConfigError("max_retries must be >= 0")
        if timeout_seconds <= 0:
            raise ConfigError("timeout_seconds must be > 0")
        self.config = config
        self.environ = environ if environ is not None else os.environ
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._transport = transport
        self._call_sink = call_sink
        self._deadline_check = deadline_check
        self.redactor = redactor or DEFAULT_REDACTOR
        self._sleep = sleep
        self._client: httpx.Client | None = None
        self._endpoint: str | None = None
        self._api_key: str | None = None
        self.calls: list[ModelCallRecord] = []
        self._closed = False

    # --------------------------------------------------------------- lifecycle
    @classmethod
    def from_app_config(
        cls,
        app_config: Any,
        **kwargs: Any,
    ) -> "RealModel":
        return cls(app_config.model, **kwargs)

    def set_call_sink(self, sink: Callable[[ModelCallRecord], None] | None) -> None:
        self._call_sink = sink

    def set_deadline_check(self, check: Callable[[str], None] | None) -> None:
        self._deadline_check = check

    def load(self) -> None:
        if self._closed:
            raise RealModelError("model has been unloaded")
        if self._is_loaded:
            return
        base_url = self.environ.get(self.config.base_url_env or "", "")
        api_key = self.environ.get(self.config.api_key_env or "", "")
        if not base_url:
            raise ConfigError(
                f"environment variable {self.config.base_url_env} is not set"
            )
        if not api_key:
            raise ConfigError(
                f"environment variable {self.config.api_key_env} is not set"
            )
        self._endpoint = _endpoint_from_base_url(base_url)
        # Keep the value out of repr/records; only the Authorization header sees it.
        self._api_key = api_key
        self._client = httpx.Client(
            timeout=self.timeout_seconds,
            trust_env=False,
            transport=self._transport,
        )
        self._is_loaded = True

    def unload(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._api_key = None
        self._is_loaded = False
        self._closed = True

    # ---------------------------------------------------------------- generation
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not self._is_loaded:
            self.load()
        assert self._client is not None and self._endpoint is not None
        settings = config or GenerationConfig()
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 2):
            if self._deadline_check is not None:
                self._deadline_check("model call")
            request_id = f"ireq-{uuid.uuid4().hex}"
            started = time.monotonic()
            try:
                response = self._client.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=self._request_body(prompt, settings),
                )
            except httpx.HTTPError as exc:
                latency_ms = (time.monotonic() - started) * 1000.0
                record = self._record(
                    request_id=request_id,
                    provider_request_id=None,
                    attempt=attempt,
                    status="network_error",
                    http_status=None,
                    latency_ms=latency_ms,
                    failure_type=type(exc).__name__,
                    error_message=self.redactor.redact_exception(exc),
                    estimated_input=self._estimate(prompt),
                )
                last_error = record.error_message
                if attempt <= self.max_retries:
                    self._backoff(attempt)
                    continue
                raise RealModelError(
                    f"model request failed after {attempt} attempt(s): "
                    f"{type(exc).__name__}"
                ) from exc

            latency_ms = (time.monotonic() - started) * 1000.0
            provider_request_id = self._provider_request_id(response)
            if response.status_code >= 400:
                failure_type = f"http_{response.status_code}"
                self._record(
                    request_id=request_id,
                    provider_request_id=provider_request_id,
                    attempt=attempt,
                    status="http_error",
                    http_status=response.status_code,
                    latency_ms=latency_ms,
                    failure_type=failure_type,
                    error_message=f"model endpoint returned HTTP {response.status_code}",
                    estimated_input=self._estimate(prompt),
                )
                last_error = failure_type
                if response.status_code in _RETRYABLE_STATUS and attempt <= self.max_retries:
                    self._backoff(attempt)
                    continue
                raise RealModelError(
                    f"model endpoint returned HTTP {response.status_code} "
                    f"after {attempt} attempt(s)"
                )

            try:
                outcome = self._parse_response(response)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._record(
                    request_id=request_id,
                    provider_request_id=provider_request_id,
                    attempt=attempt,
                    status="invalid_response",
                    http_status=response.status_code,
                    latency_ms=latency_ms,
                    failure_type="invalid_response",
                    error_message=f"invalid model response: {type(exc).__name__}",
                    estimated_input=self._estimate(prompt),
                )
                last_error = "invalid_response"
                if attempt <= self.max_retries:
                    self._backoff(attempt)
                    continue
                raise RealModelError(
                    f"invalid model response after {attempt} attempt(s)"
                ) from exc

            record = self._record(
                request_id=request_id,
                provider_request_id=provider_request_id,
                attempt=attempt,
                status="ok",
                http_status=response.status_code,
                latency_ms=latency_ms,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                total_tokens=outcome.total_tokens,
                usage_source=outcome.usage_source,
                finish_reason=outcome.finish_reason,
                estimated_input=self._estimate(prompt),
                estimated_output=self._estimate(outcome.text),
            )
            metadata = dict(outcome.metadata)
            metadata.update(
                {
                    "request_id": request_id,
                    "provider_request_id": provider_request_id,
                    "latency_ms": latency_ms,
                    "usage_source": outcome.usage_source,
                    "retries": attempt - 1,
                    "real_model_call": record.to_dict(),
                }
            )
            tokens_used = (
                outcome.total_tokens
                if outcome.total_tokens is not None
                else (record.estimated_input_tokens or 0)
                + (record.estimated_output_tokens or 0)
            )
            return GenerationResult(
                text=outcome.text,
                tokens_used=int(tokens_used),
                finish_reason=outcome.finish_reason,
                model_name=self.model_name,
                metadata=metadata,
            )
        raise RealModelError(f"model request failed: {last_error or 'unknown error'}")

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterable[str]:
        """The phase-1 adapter is non-streaming; expose exactly one chunk.

        EffGen's interactive entry is configured non-streaming. Keeping the
        method functional satisfies the BaseModel contract without a second
        provider request or a hidden streaming mode.
        """

        result = self.generate(prompt, config, **kwargs)
        return iter((result.text,) if result.text else ())

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
    def _request_body(self, prompt: str, settings: GenerationConfig) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if settings.max_tokens is not None:
            body["max_tokens"] = int(settings.max_tokens)
        if settings.temperature is not None:
            body["temperature"] = float(settings.temperature)
        return body

    @staticmethod
    def _provider_request_id(response: httpx.Response) -> str | None:
        for name in ("x-request-id", "request-id", "x-amzn-requestid"):
            value = response.headers.get(name)
            if value:
                return str(value)
        return None

    @staticmethod
    def _parse_response(response: httpx.Response) -> _AttemptOutcome:
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("response body is not an object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("response has no choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("choice is not an object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("choice has no message")
        content = message.get("content")
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
            content = "".join(pieces)
        if not isinstance(content, str):
            raise ValueError("message.content is not text")
        finish_reason = first.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            finish_reason = "stop"
        usage = payload.get("usage")
        input_tokens = output_tokens = total_tokens = None
        source = "missing"
        if isinstance(usage, Mapping):
            prompt_value = usage.get("prompt_tokens", usage.get("input_tokens"))
            completion_value = usage.get("completion_tokens", usage.get("output_tokens"))
            total_value = usage.get("total_tokens")
            if isinstance(prompt_value, int) and not isinstance(prompt_value, bool):
                input_tokens = prompt_value
            if isinstance(completion_value, int) and not isinstance(completion_value, bool):
                output_tokens = completion_value
            if isinstance(total_value, int) and not isinstance(total_value, bool):
                total_tokens = total_value
            elif input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            if input_tokens is not None or output_tokens is not None or total_tokens is not None:
                source = "provider"
        provider_id = payload.get("id")
        metadata: dict[str, Any] = {}
        if isinstance(provider_id, str) and provider_id:
            metadata["provider_response_id"] = provider_id
        return _AttemptOutcome(
            text=content,
            finish_reason=finish_reason,
            provider_request_id=provider_id if isinstance(provider_id, str) else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_source=source,
            metadata=metadata,
        )

    def _record(
        self,
        *,
        request_id: str,
        provider_request_id: str | None,
        attempt: int,
        status: str,
        http_status: int | None,
        latency_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        usage_source: str = "missing",
        finish_reason: str | None = None,
        failure_type: str | None = None,
        error_message: str | None = None,
        estimated_input: int | None = None,
        estimated_output: int | None = None,
    ) -> ModelCallRecord:
        record = ModelCallRecord(
            index=len(self.calls),
            request_id=request_id,
            provider_request_id=provider_request_id,
            model=self.model_name,
            endpoint=_safe_endpoint(self._endpoint or ""),
            attempt=attempt,
            retries_before=attempt - 1,
            status=status,
            http_status=http_status,
            latency_ms=round(float(latency_ms), 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_source=usage_source,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            finish_reason=finish_reason,
            failure_type=failure_type,
            error_message=error_message,
        )
        record = self._redact_record(record)
        self.calls.append(record)
        if self._call_sink is not None:
            self._call_sink(record)
        return record

    def _redact_record(self, record: ModelCallRecord) -> ModelCallRecord:
        payload = self.redactor.redact(record.to_dict())
        return ModelCallRecord(**payload)

    def _backoff(self, attempt: int) -> None:
        if self.retry_backoff_seconds > 0:
            self._sleep(self.retry_backoff_seconds * attempt)

    def _estimate(self, text: str) -> int:
        try:
            return self._encode_count(text)
        except Exception:  # noqa: BLE001 - a counter never blocks a request
            return max(0, (len(text) + 3) // 4)

    def _encode_count(self, text: str) -> int:
        value = getattr(self, "_encoder", None)
        if value is None:
            tokenizer = self.config.tokenizer
            if tokenizer.startswith("tiktoken:"):
                import tiktoken

                value = tiktoken.get_encoding(tokenizer.split(":", 1)[1])
            else:
                value = False  # sentinel: use heuristic
            self._encoder = value
        if value is False:
            return 0 if not text else max(1, (len(text) + 3) // 4)
        return len(value.encode(text, disallowed_special=()))  # type: ignore[union-attr]


__all__ = ["ModelCallRecord", "RealModel", "RealModelError"]
