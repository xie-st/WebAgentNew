"""Semantic verification through the task's one model entry.

A semantic condition is a judgement that cannot be reduced to a field equality
(relevance, tone, whether visible text meets a natural-language requirement).
The judge calls the caller's model, counts the call through the session budget
and event log, and requires a JSON answer. If the answer is absent, malformed or
does not identify a result, the condition is ``unknown``; it is never forced to
pass or fail.

The prompt contains the current normal observation and the evidence ids already
in scope. It never contains hidden judge truth or a site database.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..session import Session
from .types import (
    ConditionResult,
    VerificationCondition,
    VerificationRequest,
    VerificationStatus,
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_VERDICT_WORDS = frozenset({"pass", "fail", "unknown"})

SYSTEM_PROMPT = (
    "You are a local verification check, not the task agent. Judge only the "
    "single condition against the supplied current page observation and "
    "evidence ids. Return exactly one JSON object: "
    '{"status":"pass|fail|unknown","reason":"short reason",'
    '"evidence_ids":["existing id"]}. '
    "Use unknown when the page does not contain enough information. Never "
    "invent an element number, evidence id or completed result."
)


class SemanticVerificationError(RuntimeError):
    """The semantic judge could not produce a usable answer."""


@dataclass(frozen=True)
class SemanticJudgement:
    status: VerificationStatus
    reason: str
    evidence_ids: tuple[str, ...] = ()
    model_calls: int = 0


def _prompt(request: VerificationRequest, condition: VerificationCondition) -> str:
    evidence_lines = []
    for ref in request.evidence_scope:
        evidence_lines.append(
            f"- {ref.evidence_id}: kind={ref.kind or 'unknown'} "
            f"observation={ref.observation_id or 'unknown'} "
            f"reference={ref.reference}"
        )
    evidence = "\n".join(evidence_lines) or "- (none)"
    expected = condition.expected
    if not isinstance(expected, str):
        expected = json.dumps(expected, ensure_ascii=False, sort_keys=True, default=str)
    return (
        f"CONDITION ID: {condition.condition_id}\n"
        f"CONDITION: {condition.description}\n"
        f"EXPECTED JUDGEMENT: {expected}\n"
        f"SUBJECT OBJECT: {request.subject_id}\n"
        f"CURRENT URL: {request.observation.url}\n"
        f"CURRENT OBSERVATION ID: {request.observation.observation_id}\n"
        "CURRENT PAGE OBSERVATION:\n"
        f"{request.observation.ax_text}\n"
        "EVIDENCE IN SCOPE:\n"
        f"{evidence}\n"
    )


class ModelSemanticJudge:
    """Call the unified model once per semantic condition and bill the call."""

    def __init__(
        self,
        model: Any,
        *,
        session: Session | None = None,
        model_name: str | None = None,
    ) -> None:
        if model is None or not hasattr(model, "generate"):
            raise SemanticVerificationError("semantic judge needs a model with generate()")
        self.model = model
        self.session = session
        self.model_name = model_name or str(getattr(model, "model_name", "unified-model"))
        self.calls = 0
        self.last_prompt: str | None = None

    def evaluate(
        self,
        condition: VerificationCondition,
        request: VerificationRequest,
    ) -> ConditionResult:
        prompt = _prompt(request, condition)
        self.last_prompt = prompt
        if self.session is not None:
            self.session.check_deadline("semantic verification")
            self.session.check_model_call_margin("semantic verification")
            try:
                count = self.session.budget.counter.count_text(SYSTEM_PROMPT + "\n" + prompt)
                source = self.session.budget.counter.source
            except Exception:  # noqa: BLE001 - counting never blocks the check
                count, source = None, "unknown"
            self.session.record_model_call(
                request_id=f"verify-{condition.condition_id}-{self.calls + 1}",
                input_tokens=count,
                output_tokens=None,
                token_count_source=source,
                latency_ms=None,
                model_name=self.model_name,
            )
        self.calls += 1
        try:
            result = self.model.generate(SYSTEM_PROMPT + "\n\n" + prompt)
        except Exception as exc:  # noqa: BLE001 - verification must report unknown
            if self.session is not None:
                self.session.log_exception("semantic_verification", exc)
            return ConditionResult(
                condition_id=condition.condition_id,
                status=VerificationStatus.UNKNOWN,
                detail=f"semantic model call failed: {type(exc).__name__}: {exc}",
                method="semantic",
                model_calls=1,
            )
        text = getattr(result, "text", result if isinstance(result, str) else "")
        judgement = parse_semantic_answer(str(text or ""))
        if judgement.status is VerificationStatus.PASS:
            known = {ref.evidence_id for ref in request.evidence_scope}
            unknown_ids = [item for item in judgement.evidence_ids if item not in known]
            if unknown_ids:
                return ConditionResult(
                    condition_id=condition.condition_id,
                    status=VerificationStatus.UNKNOWN,
                    detail=(
                        "semantic answer cited evidence outside the request scope: "
                        + ", ".join(unknown_ids)
                    ),
                    evidence=tuple(
                        ref
                        for ref in request.evidence_scope
                        if ref.evidence_id in judgement.evidence_ids
                    ),
                    method="semantic",
                    model_calls=1,
                )
        cited = tuple(
            ref
            for ref in request.evidence_scope
            if ref.evidence_id in set(judgement.evidence_ids)
        )
        return ConditionResult(
            condition_id=condition.condition_id,
            status=judgement.status,
            detail=judgement.reason,
            evidence=cited,
            method="semantic",
            model_calls=1,
        )


def parse_semantic_answer(text: str) -> SemanticJudgement:
    """Parse one JSON answer; anything unusable is unknown, never guessed."""

    match = _JSON_OBJECT.search(text or "")
    if match is None:
        return SemanticJudgement(
            VerificationStatus.UNKNOWN,
            "semantic answer did not contain a JSON object",
        )
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return SemanticJudgement(
            VerificationStatus.UNKNOWN,
            f"semantic answer is not valid JSON: {exc}",
        )
    if not isinstance(payload, Mapping):
        return SemanticJudgement(VerificationStatus.UNKNOWN, "semantic answer is not an object")
    raw_status = str(payload.get("status", payload.get("verdict", ""))).strip().lower()
    if raw_status not in _VERDICT_WORDS:
        return SemanticJudgement(
            VerificationStatus.UNKNOWN,
            f"semantic answer has no valid status: {raw_status!r}",
        )
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        reason = "semantic model did not give a reason"
    raw_ids = payload.get("evidence_ids") or ()
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    evidence_ids: list[str] = []
    if isinstance(raw_ids, (list, tuple)):
        for item in raw_ids:
            text_id = str(item).strip()
            if text_id:
                evidence_ids.append(text_id)
    return SemanticJudgement(
        VerificationStatus(raw_status),
        reason,
        tuple(dict.fromkeys(evidence_ids)),
    )


__all__ = [
    "ModelSemanticJudge",
    "SemanticJudgement",
    "SemanticVerificationError",
    "SYSTEM_PROMPT",
    "parse_semantic_answer",
]
