"""A model wrapper that records every call through the pause.

The wrapper exists so a model call that returns while the task is paused cannot
lose its cost or its latency. Phase-1 real/official adapters already record their
own calls (``records_model_calls=True``); the interactive driver skips its
pre-call placeholder for them. This wrapper does the same for scripted/offline
models:

* the driver does not add a second placeholder;
* each attempt is recorded once through ``Session.record_model_call`` with a
  synthetic platform request id on success and no request id on failure;
* ``ControlCoordinator.record_model_result`` keeps the three counters and the
  latency sum, and marks a result that arrived after the control epoch changed.

It is a normal EffGen ``BaseModel`` subclass so the agent accepts it as a model.
Nothing here changes prompt/tool serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from effgen.models.base import BaseModel, GenerationConfig, GenerationResult, TokenCount

from ..clock import Clock, SystemClock
from ..session import Session
from .coordinator import ControlCoordinator


@dataclass
class ControlModelCall:
    call_id: str
    control_epoch: int | None
    request_id: str | None
    latency_ms: float
    failure_type: str | None
    paused_at_return: bool
    discarded: bool = False


class ControlAwareModel(BaseModel):
    """Delegating model with per-call timing and control accounting."""

    records_model_calls = True

    def __init__(
        self,
        inner: BaseModel,
        *,
        session: Session,
        coordinator: ControlCoordinator | None = None,
        clock: Clock | None = None,
        request_prefix: str = "ireq-model",
    ) -> None:
        if not isinstance(inner, BaseModel):
            raise TypeError("ControlAwareModel wraps an EffGen BaseModel")
        if not isinstance(session, Session):
            raise TypeError("ControlAwareModel needs the run Session")
        super().__init__(
            inner.model_name,
            inner.model_type,
            context_length=inner.get_context_length(),
        )
        self.inner = inner
        self.session = session
        self.coordinator = coordinator
        self.clock = clock or session.clock
        self.request_prefix = request_prefix
        self.calls: list[ControlModelCall] = []
        self._index = 0

    # ------------------------------------------------------------- accounting
    def _next_call_id(self) -> str:
        self._index += 1
        return f"{self.request_prefix}-{self._index}"

    def _count(self, text: str) -> int:
        try:
            return int(self.inner.count_tokens(text).count)
        except Exception:  # noqa: BLE001 - accounting must not change generation
            return 0

    def _record(
        self,
        *,
        call_id: str,
        control_epoch: int | None,
        prompt: str,
        result: GenerationResult | None,
        error: str | None,
        latency_ms: float,
    ) -> None:
        paused_at_return = self.coordinator is not None and (
            self.coordinator.state.value not in ("running", "refreshing")
            or (
                control_epoch is not None
                and control_epoch != self.coordinator.control_epoch
            )
        )
        # A failed attempt has no confirmed platform request id. A successful
        # offline call gets a synthetic id so the three counters remain visible
        # separately; real adapters pass their own ids.
        request_id = None if error else f"req-{call_id}"
        try:
            self.session.record_model_call(
                request_id=request_id,
                input_tokens=self._count(prompt) or None,
                output_tokens=(
                    None if result is None else self._count(result.text) or None
                ),
                token_count_source="estimate:fake_model_chars_div_4",
                latency_ms=latency_ms,
                failure_type=error,
                model_name=self.model_name,
            )
        except Exception as exc:  # noqa: BLE001 - never hide the model result
            try:
                self.session.log_exception("model_accounting", exc)
            except Exception:  # noqa: BLE001
                pass
        if self.coordinator is not None:
            try:
                self.coordinator.record_model_result(
                    call_id=call_id,
                    control_epoch=control_epoch,
                    request_id=request_id,
                    latency_ms=latency_ms,
                    failure_type=error,
                    paused_at_return=paused_at_return,
                )
            except Exception:  # noqa: BLE001 - accounting is diagnostics
                pass
        self.calls.append(
            ControlModelCall(
                call_id=call_id,
                control_epoch=control_epoch,
                request_id=request_id,
                latency_ms=latency_ms,
                failure_type=error,
                paused_at_return=paused_at_return,
            )
        )

    def _epoch(self) -> int | None:
        return None if self.coordinator is None else self.coordinator.control_epoch

    # ---------------------------------------------------------------- abstract
    def load(self) -> None:
        self.inner.load()
        self._is_loaded = True

    def unload(self) -> None:
        self.inner.unload()
        self._is_loaded = False

    def count_tokens(self, text: str) -> TokenCount:
        return self.inner.count_tokens(text)

    def get_context_length(self) -> int:
        return self.inner.get_context_length()

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        call_id = self._next_call_id()
        epoch = self._epoch()
        started = self.clock.monotonic()
        result: GenerationResult | None = None
        error: str | None = None
        try:
            result = self.inner.generate(prompt, config, **kwargs)
            return result
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            latency_ms = max(0.0, (self.clock.monotonic() - started) * 1000.0)
            self._record(
                call_id=call_id,
                control_epoch=epoch,
                prompt=prompt,
                result=result,
                error=error,
                latency_ms=latency_ms,
            )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterable[str]:
        call_id = self._next_call_id()
        epoch = self._epoch()
        started = self.clock.monotonic()
        error: str | None = None

        def _iterator() -> Iterable[str]:
            nonlocal error
            try:
                yield from self.inner.generate_stream(prompt, config, **kwargs)
            except Exception as exc:  # noqa: BLE001 - recorded then re-raised
                error = type(exc).__name__
                raise
            finally:
                latency_ms = max(0.0, (self.clock.monotonic() - started) * 1000.0)
                self._record(
                    call_id=call_id,
                    control_epoch=epoch,
                    prompt=prompt,
                    result=None,
                    error=error,
                    latency_ms=latency_ms,
                )

        return _iterator()

    # --------------------------------------------------------------- delegates
    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[GenerationResult]:
        method = getattr(self.inner, "generate_batch", None)
        if method is None:
            return [self.generate(prompt, config, **kwargs) for prompt in prompts]
        return method(prompts, config, **kwargs)

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        method = getattr(self.inner, "generate_with_tools", None)
        if method is None:
            raise NotImplementedError(
                f"{type(self.inner).__name__} has no generate_with_tools"
            )
        return method(prompt, tools, config, **kwargs)

    def supports_function_calling(self) -> bool:
        return bool(getattr(self.inner, "supports_function_calling", lambda: False)())

    def supports_tool_calling(self) -> bool:
        return bool(getattr(self.inner, "supports_tool_calling", lambda: False)())

    def supports_conversation(self) -> bool:
        return bool(getattr(self.inner, "supports_conversation", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        # Attributes this wrapper does not own (provider hints, cache policy,
        # etc.) remain the wrapped model's.
        return getattr(self.inner, name)


__all__ = ["ControlAwareModel", "ControlModelCall"]
