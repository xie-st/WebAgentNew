"""Deadlines, token counting and context packing.

The input budget (24K by default) and the planned 32K context are engineering
settings, not a claim about the unified model's official window. The token
counter records its source; a target-model tokenizer is not verified in A1, so
tiktoken counts are still labelled as estimates.

Packing never truncates the task, current AX text or pending confirmation. It
compresses old history first and raises :class:`ContextOverflow` when the
mandatory content itself does not fit.
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


@dataclass(frozen=True)
class BudgetLimits:
    input_tokens: int = DEFAULT_INPUT_TOKENS
    planned_context_tokens: int = DEFAULT_PLANNED_CONTEXT_TOKENS
    max_messages: int = DEFAULT_MAX_MESSAGES
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_model_calls: int | None = None

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

    def _measure(self, messages: Sequence[Mapping[str, str]]) -> tuple[int, int]:
        combined = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        tokens = self.counter.count_text(combined)
        encoded = json.dumps(list(messages), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return tokens, len(encoded)

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
