"""Shared record types.

Benchmarks turn raw data into `Sample`s. Frameworks turn a `Sample` into an
`Attempt`. The runner scores the attempt and writes a `Record`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Sample:
    """One benchmark item, framework independent."""

    sample_id: str
    question: str
    answer: Any
    # Extra context the agent is allowed to see, e.g. a conversation history for
    # the memory benchmarks or the option list for a multiple-choice question.
    context: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    name: str
    arguments: Any
    result: str = ""


@dataclass
class Attempt:
    """What a framework produced for one sample."""

    output: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Some frameworks report only how many tool calls happened, not what they
    # were. When that is all we get, the count lands here and `tool_calls` stays
    # empty. None means "the list is the whole story".
    tool_call_count: int | None = None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None
    # Why the framework's loop ended, when it says. Not every framework
    # reports one. This exists because a run that stopped at the iteration cap
    # and a run that answered look identical in the accuracy column: on the
    # 15 Aug sweep a quarter of one cell was the string "Maximum iterations
    # reached without final answer." being scored as an answer, and nothing in
    # the summary said so.
    stop_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tools_used(self) -> int:
        if self.tool_call_count is not None:
            return self.tool_call_count
        return len(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["tools_used"] = self.tools_used
        return d


@dataclass
class Record:
    """One scored sample, written as a line of records.jsonl."""

    sample_id: str
    question: str
    ground_truth: Any
    prediction: Any
    output: str
    correct: bool
    attempt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
