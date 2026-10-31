"""The final-send measurement: one pure check of the exact request about to go out.

The check runs on the serialized prompt the EffGen loop is about to hand to the
model adapter, which for the official entry is the same one user message that
``OfficialModel.build_messages`` forwards to ``third_party_agent.model.generate``.
It answers three questions in a fixed order:

1. are all four necessary regions present byte-for-byte,
2. does the request stay inside the project input budget,
3. does it stay inside the official message/byte limits.

Nothing here trims a message: a violation produces a report and the caller
refuses the send. The only shortening allowed happens earlier, when the harness
builds its own ledger block and drops optional detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..budget import BudgetLimits, TokenCounter, serialized_request_bytes
from ..errors import ProtocolError
from .sections import (
    SECTION_LABELS,
    SECTION_ORDER,
    NecessarySection,
    SectionKind,
)

VERDICT_OK = "ok"
VERDICT_MISSING = "missing_necessary"
VERDICT_OVER_LIMIT = "over_limit"


@dataclass(frozen=True)
class SectionCheck:
    """What happened to one necessary region in one final send."""

    kind: SectionKind
    label: str
    present: bool
    expected_bytes: int
    matched_bytes: int
    offset: int | None = None

    @property
    def byte_complete(self) -> bool:
        return self.present and self.matched_bytes == self.expected_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "label": self.label,
            "present": self.present,
            "expected_bytes": self.expected_bytes,
            "matched_bytes": self.matched_bytes,
            "offset": self.offset,
            "byte_complete": self.byte_complete,
        }


@dataclass(frozen=True)
class ContextCheckReport:
    """One deterministic measurement of one final send.

    ``mandatory_intact`` is the protection verdict; ``over_limit`` is the budget
    verdict. Both can be true at once: an over-long request still reports the four
    regions byte-complete instead of claiming they were trimmed.
    """

    sections: tuple[SectionCheck, ...]
    tokens: int
    token_count_source: str
    request_bytes: int
    message_count: int
    max_input_tokens: int
    max_request_bytes: int
    max_messages: int
    verdict: str
    trimmed_optional_chars: int = 0
    optional_bytes: int = 0
    revision: int | None = None
    plan_version: int | None = None
    branch: str | None = None

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            section.kind.value for section in self.sections if not section.present
        )

    @property
    def mandatory_intact(self) -> bool:
        return all(section.byte_complete for section in self.sections)

    @property
    def over_limit(self) -> bool:
        return (
            self.tokens > self.max_input_tokens
            or self.request_bytes > self.max_request_bytes
            or self.message_count > self.max_messages
        )

    @property
    def protected_bytes(self) -> int:
        return sum(section.expected_bytes for section in self.sections)

    def section(self, kind: SectionKind) -> SectionCheck:
        for item in self.sections:
            if item.kind is kind:
                return item
        raise KeyError(kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "mandatory_intact": self.mandatory_intact,
            "over_limit": self.over_limit,
            "missing": list(self.missing),
            "sections": [section.to_dict() for section in self.sections],
            "tokens": self.tokens,
            "token_count_source": self.token_count_source,
            "request_bytes": self.request_bytes,
            "message_count": self.message_count,
            "max_input_tokens": self.max_input_tokens,
            "max_request_bytes": self.max_request_bytes,
            "max_messages": self.max_messages,
            "protected_bytes": self.protected_bytes,
            "optional_bytes": self.optional_bytes,
            "trimmed_optional_chars": self.trimmed_optional_chars,
            "revision": self.revision,
            "plan_version": self.plan_version,
            "branch": self.branch,
        }


def prompt_messages_and_text(prompt: Any) -> tuple[list[dict[str, str]], str]:
    """The message list and text of the exact prompt about to be sent.

    The official adapter turns a string prompt into one user message, so a string
    is measured the same way. A caller-supplied role/content list is accepted as
    EffGen passes it through unchanged.
    """

    if isinstance(prompt, str):
        if not prompt.strip():
            raise ProtocolError("the final prompt is empty")
        return [{"role": "user", "content": prompt}], prompt
    if isinstance(prompt, (list, tuple)):
        messages: list[dict[str, str]] = []
        for item in prompt:
            if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
                raise ProtocolError("final prompt messages must be role/content pairs")
            messages.append({"role": str(item["role"]), "content": str(item["content"])})
        if not messages:
            raise ProtocolError("the final prompt has no messages")
        text = "\n".join(message["content"] for message in messages)
        return messages, text
    raise ProtocolError("the final prompt must be a string or role/content messages")


def check_sections(
    text: str, sections: Sequence[NecessarySection]
) -> tuple[SectionCheck, ...]:
    """Byte-exact containment for each region, in canonical order."""

    checks: list[SectionCheck] = []
    for kind in SECTION_ORDER:
        section = next((item for item in sections if item.kind is kind), None)
        if section is None:
            checks.append(
                SectionCheck(
                    kind=kind,
                    label=SECTION_LABELS[kind],
                    present=False,
                    expected_bytes=0,
                    matched_bytes=0,
                )
            )
            continue
        offset = text.find(section.text)
        present = offset >= 0
        checks.append(
            SectionCheck(
                kind=kind,
                label=section.label,
                present=present,
                expected_bytes=section.byte_length,
                matched_bytes=section.byte_length if present else 0,
                offset=offset if present else None,
            )
        )
    return tuple(checks)


def evaluate(
    *,
    prompt: Any,
    sections: Sequence[NecessarySection],
    limits: BudgetLimits,
    counter: TokenCounter,
    trimmed_optional_chars: int = 0,
    optional_bytes: int = 0,
    state: Mapping[str, Any] | None = None,
) -> ContextCheckReport:
    """Measure one send and return the report; never raises for a violation."""

    if not isinstance(limits, BudgetLimits):
        raise TypeError("limits must be BudgetLimits")
    if not isinstance(counter, TokenCounter):
        raise TypeError("counter must be a TokenCounter")
    messages, text = prompt_messages_and_text(prompt)
    checks = check_sections(text, sections)
    tokens = int(counter.count_text(text))
    request_bytes = serialized_request_bytes(messages)
    verdict = VERDICT_OK
    if not all(check.byte_complete for check in checks):
        verdict = VERDICT_MISSING
    elif (
        tokens > limits.input_tokens
        or request_bytes > limits.max_request_bytes
        or len(messages) > limits.max_messages
    ):
        verdict = VERDICT_OVER_LIMIT
    info = dict(state or {})
    return ContextCheckReport(
        sections=checks,
        tokens=tokens,
        token_count_source=counter.source,
        request_bytes=request_bytes,
        message_count=len(messages),
        max_input_tokens=limits.input_tokens,
        max_request_bytes=limits.max_request_bytes,
        max_messages=limits.max_messages,
        verdict=verdict,
        trimmed_optional_chars=int(trimmed_optional_chars),
        optional_bytes=int(optional_bytes),
        revision=info.get("revision"),
        plan_version=info.get("plan_version"),
        branch=info.get("branch"),
    )


def describe(report: ContextCheckReport) -> str:
    """A short, evidence-ready explanation of one report."""

    parts = [
        f"{section.kind.value}={section.matched_bytes}/{section.expected_bytes}B"
        for section in report.sections
    ]
    head = (
        f"final send {report.tokens} tokens ({report.token_count_source}), "
        f"{report.request_bytes} bytes, {report.message_count} message(s); "
        f"limits tokens={report.max_input_tokens} bytes={report.max_request_bytes} "
        f"messages={report.max_messages}"
    )
    return head + "; necessary regions " + ", ".join(parts)


__all__ = [
    "VERDICT_MISSING",
    "VERDICT_OK",
    "VERDICT_OVER_LIMIT",
    "ContextCheckReport",
    "SectionCheck",
    "check_sections",
    "describe",
    "evaluate",
    "prompt_messages_and_text",
]
