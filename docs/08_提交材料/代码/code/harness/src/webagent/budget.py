"""Deadlines, token counting and context packing.

The input budget (24K by default) and the planned 32K context are engineering
settings, not a claim about the unified model's official window. The token
counter records its source; a target-model tokenizer is not verified in A1, so
tiktoken counts are still labelled as estimates.

Packing never truncates the task, current AX text or pending confirmation. It
compresses old history first and raises :class:`ContextOverflow` when the
mandatory content itself does not fit.

The deadline margin defaults to zero: keep using the remaining platform time
instead of ending early. A positive value is an explicit opt-in to the legacy
early-close policy; the absolute platform deadline is always enforced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .clock import Clock, SystemClock
from .errors import BudgetExceeded, ContextOverflow, DeadlineExceeded, RequestLimitExceeded, ProtocolError
from .protocol import Observation

DEFAULT_INPUT_TOKENS = 24_000
DEFAULT_PLANNED_CONTEXT_TOKENS = 32_768
DEFAULT_MAX_MESSAGES = 256
DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
DEFAULT_DEADLINE_MARGIN_SECONDS = 0.0


@dataclass(frozen=True)
class BudgetLimits:
    input_tokens: int = DEFAULT_INPUT_TOKENS
    planned_context_tokens: int = DEFAULT_PLANNED_CONTEXT_TOKENS
    max_messages: int = DEFAULT_MAX_MESSAGES
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_model_calls: int | None = None
    deadline_margin_seconds: float = DEFAULT_DEADLINE_MARGIN_SECONDS

    def __post_init__(self) -> None:
        for name in ("input_tokens", "planned_context_tokens", "max_messages", "max_request_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ProtocolError(f"budget.{name} must be a positive integer")
        if self.input_tokens > self.planned_context_tokens:
            raise ProtocolError("budget.input_tokens must not exceed planned_context_tokens")
        if self.max_model_calls is not None:
            if (
                not isinstance(self.max_model_calls, int)
                or isinstance(self.max_model_calls, bool)
                or self.max_model_calls <= 0
            ):
                raise ProtocolError("budget.max_model_calls must be a positive integer or null")
        margin = self.deadline_margin_seconds
        if not isinstance(margin, (int, float)) or isinstance(margin, bool) or margin < 0:
            raise ProtocolError("budget.deadline_margin_seconds must be a non-negative number")


class Deadline:
    def __init__(self, epoch_seconds: float, *, clock: Clock | None = None) -> None:
        if not isinstance(epoch_seconds, (int, float)) or isinstance(epoch_seconds, bool):
            raise ProtocolError("deadline must be an epoch timestamp")
        self.epoch_seconds = float(epoch_seconds)
        self.clock = clock or SystemClock()

    def remaining_seconds(self) -> float:
        return self.epoch_seconds - self.clock.now()

    def expired(self) -> bool:
        return self.remaining_seconds() <= 0

    def require(self, stage: str = "work") -> None:
        if self.expired():
            raise DeadlineExceeded(
                f"task deadline reached before {stage}; no further model, download or action work"
            )


class DeadlineMarginReached(DeadlineExceeded):
    """A new model call was refused because the safety margin was reached.

    This is a deliberate close-out signal, not a failure: the remaining time is
    shorter than the expected call latency, so the run stops with a terminal
    ``deadline`` state instead of starting a call the platform would kill.
    """

    def __init__(self, stage: str, *, remaining_seconds: float, margin_seconds: float) -> None:
        self.stage = str(stage)
        self.remaining_seconds = float(remaining_seconds)
        self.margin_seconds = float(margin_seconds)
        super().__init__(
            f"task has {self.remaining_seconds:.1f}s left before {stage}, below the "
            f"{self.margin_seconds:.1f}s deadline margin; no new model call is started"
        )


class TokenCounter:
    source: str = "unknown"

    def count_text(self, text: str) -> int:
        raise NotImplementedError


class HeuristicTokenCounter(TokenCounter):
    """Fallback estimate, explicitly not a tokenizer."""

    source = "estimate:chars_div_4"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)


class TiktokenTokenCounter(TokenCounter):
    """tiktoken count of a local encoding.

    The target unified model's tokenizer is not verified in A1; the source name
    therefore still says estimated.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.source = f"tiktoken:{encoding_name}:estimated_for_target_model"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))


class ProvidedTokenCounter(TokenCounter):
    """A counter whose source the caller has verified (exact for that model)."""

    def __init__(self, counter: Callable[[str], int], *, source: str) -> None:
        if not callable(counter):
            raise ProtocolError("counter must be callable")
        if not isinstance(source, str) or not source:
            raise ProtocolError("token counter source must be non-empty")
        self._counter = counter
        self.source = source

    def count_text(self, text: str) -> int:
        return int(self._counter(text))


def default_token_counter() -> TokenCounter:
    try:
        return TiktokenTokenCounter()
    except Exception:
        return HeuristicTokenCounter()


@dataclass(frozen=True)
class HistoryEntry:
    text: str
    kind: str = "step"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ProtocolError("history text must be a string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ProtocolError("history kind must be a non-empty string")


def _normalise_history(history: Iterable[str | HistoryEntry]) -> tuple[HistoryEntry, ...]:
    entries: list[HistoryEntry] = []
    for item in history:
        if isinstance(item, HistoryEntry):
            entries.append(item)
        elif isinstance(item, str):
            entries.append(HistoryEntry(item))
        else:
            raise ProtocolError("history entries must be strings or HistoryEntry")
    return tuple(entries)


def render_observation(observation: Observation) -> str:
    """Readable prompt view; no localization and no truncation of AX text."""

    lines = [
        "CURRENT OBSERVATION",
        f"page_id={observation.page_id}",
        f"observation_id={observation.observation_id}",
        f"url={observation.url}",
        "AX TEXT:",
        observation.ax_text,
        "ELEMENTS:",
    ]
    for number in sorted(observation.elements, key=int):
        element = observation.elements[number]
        lines.append(f"[{number}] {element.role} {element.name!r}")
    if observation.feedback is not None:
        lines.append(
            "PREVIOUS FEEDBACK: "
            f"{observation.feedback.status.value}: {observation.feedback.message}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class RequestMeasurement:
    """What one serialized prompt request costs against the project limits.

    ``mandatory_*`` is filled only when the caller supplied the must-keep task /
    current-observation / pending-confirmation view, so an overflow can say
    whether the mandatory content alone already exceeds the limit.
    """

    message_count: int
    request_bytes: int
    mandatory_message_count: int | None = None
    mandatory_request_bytes: int | None = None


class MandatoryContentOverflow(RequestLimitExceeded):
    """The task, current observation and pending confirmation alone exceed a limit.

    The request is refused instead of dropping or truncating those items. This is
    deliberately a distinct type from :class:`RequestLimitExceeded` so callers and
    evidence can tell "the request grew" from "the required content cannot be
    sent at all".
    """

    scope = "mandatory"


def serialized_request_bytes(messages: Sequence[Mapping[str, str]]) -> int:
    """Bytes of the request body as this harness serializes it (UTF-8 JSON)."""

    encoded = json.dumps(
        list(messages), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return len(encoded)


def enforce_request_limits(
    messages: Sequence[Mapping[str, str]],
    limits: BudgetLimits,
    *,
    mandatory: Sequence[Mapping[str, str]] | None = None,
) -> RequestMeasurement:
    """Check the message count and serialized byte size, naming the exceeded item.

    Mandatory content is measured first: if it cannot fit, no trimming of it can
    make the request valid, so that is reported separately. Nothing is trimmed
    here; the caller must not shrink the mandatory items to pass the check.
    """

    message_count = len(messages)
    request_bytes = serialized_request_bytes(messages)
    mandatory_count = None if mandatory is None else len(mandatory)
    mandatory_bytes = None if mandatory is None else serialized_request_bytes(mandatory)
    if mandatory is not None:
        if mandatory_count > limits.max_messages:
            raise MandatoryContentOverflow(
                f"the task, current observation and pending confirmation alone exceed "
                f"max_messages ({mandatory_count} > {limits.max_messages}); they are "
                "never dropped to fit a request",
                limit_name="max_messages",
                limit=limits.max_messages,
                actual=mandatory_count,
            )
        if mandatory_bytes > limits.max_request_bytes:
            raise MandatoryContentOverflow(
                f"the task, current observation and pending confirmation alone exceed "
                f"max_request_bytes ({mandatory_bytes} > {limits.max_request_bytes}); "
                "they are never dropped to fit a request",
                limit_name="max_request_bytes",
                limit=limits.max_request_bytes,
                actual=mandatory_bytes,
            )
    if message_count > limits.max_messages:
        raise RequestLimitExceeded(
            f"prompt messages exceed max_messages ({message_count} > "
            f"{limits.max_messages}); messages are not trimmed at the request exit",
            limit_name="max_messages",
            limit=limits.max_messages,
            actual=message_count,
        )
    if request_bytes > limits.max_request_bytes:
        raise RequestLimitExceeded(
            f"serialized prompt bytes exceed max_request_bytes ({request_bytes} > "
            f"{limits.max_request_bytes}); messages are not trimmed at the request exit",
            limit_name="max_request_bytes",
            limit=limits.max_request_bytes,
            actual=request_bytes,
        )
    return RequestMeasurement(
        message_count=message_count,
        request_bytes=request_bytes,
        mandatory_message_count=mandatory_count,
        mandatory_request_bytes=mandatory_bytes,
    )


@dataclass(frozen=True)
class PackedContext:
    messages: tuple[dict[str, str], ...]
    token_count: int
    token_count_source: str
    request_bytes: int
    dropped_history_entries: int
    compressed: bool

    @property
    def message_count(self) -> int:
        return len(self.messages)


def build_prompt_messages(
    *,
    task: str,
    observation: Observation,
    pending: str | None = None,
    history: Sequence[HistoryEntry] = (),
    system: str | None = None,
    dropped: int = 0,
) -> list[dict[str, str]]:
    """The canonical prompt view: task, current observation and pending stay whole.

    This is also the must-keep view a backend can measure against its request
    limits, so "the required content alone does not fit" is judged from the same
    message shape the packer uses.
    """

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": "TASK:\n" + task})
    if history or dropped:
        lines: list[str] = []
        if dropped:
            lines.append(f"[omitted {dropped} earlier history entries]")
        lines.extend(f"- {entry.kind}: {entry.text}" for entry in history)
        messages.append({"role": "user", "content": "HISTORY:\n" + "\n".join(lines)})
    messages.append({"role": "user", "content": render_observation(observation)})
    if pending:
        messages.append({"role": "user", "content": "PENDING CONFIRMATION:\n" + pending})
    return messages


class ContextPacker:
    def __init__(
        self,
        limits: BudgetLimits | None = None,
        *,
        counter: TokenCounter | None = None,
    ) -> None:
        self.limits = limits or BudgetLimits()
        self.counter = counter or default_token_counter()

    def _messages(
        self,
        *,
        task: str,
        observation: Observation,
        pending: str | None,
        history: Sequence[HistoryEntry],
        system: str | None,
        dropped: int,
    ) -> list[dict[str, str]]:
        return build_prompt_messages(
            task=task,
            observation=observation,
            pending=pending,
            history=history,
            system=system,
            dropped=dropped,
        )

    def _measure(self, messages: Sequence[Mapping[str, str]]) -> tuple[int, int]:
        combined = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        tokens = self.counter.count_text(combined)
        return tokens, serialized_request_bytes(messages)

    def pack(
        self,
        *,
        task: str,
        observation: Observation,
        pending: str | None = None,
        history: Iterable[str | HistoryEntry] = (),
        system: str | None = None,
    ) -> PackedContext:
        if not isinstance(task, str) or not task.strip():
            raise ProtocolError("task must be a non-empty string")
        if not isinstance(observation, Observation):
            raise ProtocolError("packer needs an Observation")
        if pending is not None and not isinstance(pending, str):
            raise ProtocolError("pending must be a string or null")
        entries = _normalise_history(history)
        source = self.counter.source

        mandatory = self._messages(
            task=task,
            observation=observation,
            pending=pending,
            history=(),
            system=system,
            dropped=0,
        )
        mandatory_tokens, mandatory_bytes = self._measure(mandatory)
        if mandatory_tokens > self.limits.input_tokens:
            raise ContextOverflow(
                "task, current observation and pending confirmation cannot fit the input budget; "
                "the task and current AX text are not truncated",
                required_tokens=mandatory_tokens,
                available_tokens=self.limits.input_tokens,
                token_count_source=source,
            )
        if len(mandatory) > self.limits.max_messages:
            raise RequestLimitExceeded(
                "mandatory prompt messages exceed the message limit",
                limit_name="max_messages",
                limit=self.limits.max_messages,
                actual=len(mandatory),
            )
        if mandatory_bytes > self.limits.max_request_bytes:
            raise RequestLimitExceeded(
                "mandatory prompt bytes exceed the request size limit",
                limit_name="max_request_bytes",
                limit=self.limits.max_request_bytes,
                actual=mandatory_bytes,
            )

        for keep in range(len(entries), -1, -1):
            recent = entries[len(entries) - keep :] if keep else ()
            for include_marker in ((True, False) if len(entries) - keep else (True,)):
                dropped = len(entries) - keep
                messages = self._messages(
                    task=task,
                    observation=observation,
                    pending=pending,
                    history=recent,
                    system=system,
                    dropped=dropped if include_marker else 0,
                )
                tokens, request_bytes = self._measure(messages)
                if tokens > self.limits.input_tokens:
                    continue
                if request_bytes > self.limits.max_request_bytes:
                    continue
                if len(messages) > self.limits.max_messages:
                    continue
                return PackedContext(
                    messages=tuple(messages),
                    token_count=tokens,
                    token_count_source=source,
                    request_bytes=request_bytes,
                    dropped_history_entries=dropped,
                    compressed=dropped > 0,
                )
        raise ContextOverflow(
            "history could not be compressed into the input budget",
            required_tokens=mandatory_tokens,
            available_tokens=self.limits.input_tokens,
            token_count_source=source,
        )


class Budget:
    """Deadline and model-call checks used by the A2 driver.

    There is deliberately no default action/iteration cap: EffGen's default
    10-iteration stop is replaced by the task deadline in the interactive entry.
    """

    def __init__(
        self,
        limits: BudgetLimits | None = None,
        *,
        deadline: Deadline | None = None,
        counter: TokenCounter | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.limits = limits or BudgetLimits()
        self.deadline = deadline
        self.counter = counter or default_token_counter()
        self._clock = clock or (deadline.clock if deadline is not None else SystemClock())
        self.model_calls = 0

    def check_deadline(self, stage: str = "work") -> None:
        if self.deadline is not None:
            self.deadline.require(stage)

    def require_model_call_margin(self, stage: str = "model call") -> None:
        """Refuse a new model call when less time is left than the safety margin.

        The margin is a startup cost, not new work: a call that cannot finish
        before the platform revokes execution would be killed mid-flight and
        leave no local terminal record. The caller turns this into the normal
        deadline close-out.
        """

        margin = float(self.limits.deadline_margin_seconds)
        if self.deadline is None or margin <= 0:
            return
        remaining = self.deadline.remaining_seconds()
        if remaining <= margin:
            raise DeadlineMarginReached(
                stage, remaining_seconds=remaining, margin_seconds=margin
            )

    def begin_model_call(self) -> int:
        self.check_deadline("model call")
        if self.limits.max_model_calls is not None and self.model_calls >= self.limits.max_model_calls:
            raise BudgetExceeded(
                f"model call budget exhausted ({self.limits.max_model_calls})"
            )
        self.model_calls += 1
        return self.model_calls

    def remaining_seconds(self) -> float | None:
        return None if self.deadline is None else self.deadline.remaining_seconds()
