"""Deterministic verification checks over one normal observation and its evidence.

These checks never navigate, capture, download, click or read a site database.
They inspect:

* the current model-visible AX text and URL;
* the accessible role/name/value table of the current observation;
* evidence files the run already saved (existence, size and SHA-256).

A successful screenshot save is only evidence that a screenshot was saved. It
never proves what text is inside the image, so text conditions use the current
observation and are not satisfied by a screenshot record.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..protocol import Observation
from ..state.records import EvidenceRef, Fact
from ..state.store import StateStore
from .ax_official import (
    child_static_text_value,
    find_element_line,
    name_matches,
)
from .types import (
    ConditionResult,
    VerificationCondition,
    VerificationRequest,
    VerificationStatus,
)

_VALUE_IN_AX = re.compile(r'\bvalue=("(?:[^"\\]|\\.)*")')


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass(frozen=True)
class CheckOutcome:
    status: VerificationStatus
    detail: str
    evidence: tuple[EvidenceRef, ...] = ()

    def as_condition(self, condition: VerificationCondition) -> ConditionResult:
        return ConditionResult(
            condition_id=condition.condition_id,
            status=self.status,
            detail=self.detail,
            evidence=self.evidence,
            method="deterministic",
            model_calls=0,
        )


def _unwrap_quote(raw: str) -> str:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip('"')
    return str(value)


def _element_line(observation: Observation, number: str) -> str:
    """Return the numbered AX line, tolerating the official tree's indentation."""

    return find_element_line(observation.ax_text, number)


def visible_value(observation: Observation, *, role: str, name: str) -> str | None:
    """Return the visible value of the first matching accessible element.

    The browser extractor already records an input's current value in the normal
    observation. Using that field does not fetch the page again; parsing the AX
    line is the fallback for another backend that omits localization metadata.

    Official observations need two adaptations: the element name carries an
    attribute suffix (``'Biography' required: False ...``), and the value is the
    quoted child ``StaticText`` on a more deeply indented line. The child scan is
    limited to descendants of the matched element, so a same-named label printed
    next to the input is never returned as the field's value.
    """

    wanted_role = role.casefold()
    for number in sorted(observation.elements, key=int):
        element = observation.elements[number]
        if element.role.casefold() != wanted_role:
            continue
        if not name_matches(element.name, name):
            continue
        localization = element.localization or {}
        value = localization.get("value")
        if value is not None:
            return str(value)
        line = _element_line(observation, number)
        match = _VALUE_IN_AX.search(line)
        if match is not None:
            return _unwrap_quote(match.group(1))
        child_value = child_static_text_value(observation.ax_text, number)
        if child_value is not None:
            return child_value
        return ""
    return None


def visible_element(observation: Observation, *, role: str, name: str) -> tuple[str, Any] | None:
    wanted_role = role.casefold()
    for number in sorted(observation.elements, key=int):
        element = observation.elements[number]
        if element.role.casefold() == wanted_role and name_matches(element.name, name):
            return number, element
    return None


class DeterministicChecker:
    """Evaluate the deterministic condition vocabulary for one request."""

    def __init__(self, store: StateStore) -> None:
        if not isinstance(store, StateStore):
            raise TypeError("DeterministicChecker needs a StateStore")
        self.store = store

    # ---------------------------------------------------------------- dispatch
    def evaluate(
        self,
        condition: VerificationCondition,
        request: VerificationRequest,
    ) -> CheckOutcome:
        check = condition.check
        if check == "visible_value_equals":
            return self._visible_value(condition, request, equals=True)
        if check == "visible_value_contains":
            return self._visible_value(condition, request, equals=False)
        if check == "page_text_contains":
            return self._page_text(condition, request)
        if check == "page_url_contains":
            return self._page_url(condition, request)
        if check == "screenshot_saved":
            return self._screenshot(condition, request)
        if check == "evidence_saved":
            return self._evidence(condition, request)
        if check == "fact_value_equals":
            return self._fact(condition)
        if check == "goal_objects_complete":
            return self._objects(condition, request)
        return CheckOutcome(
            VerificationStatus.UNKNOWN,
            f"no deterministic checker for {check!r}",
        )

    # ------------------------------------------------------------------ fields
    def _visible_value(
        self,
        condition: VerificationCondition,
        request: VerificationRequest,
        *,
        equals: bool,
    ) -> CheckOutcome:
        actual = visible_value(
            request.observation,
            role=condition.role or "",
            name=condition.name or "",
        )
        if actual is None:
            return CheckOutcome(
                VerificationStatus.UNKNOWN,
                f"visible field role={condition.role!r} name={condition.name!r} "
                "is not present in the current observation",
            )
        expected = _norm(condition.expected)
        if equals:
            passed = _norm(actual) == expected
            detail = f"visible value {_norm(actual)!r} == {expected!r}: {passed}"
        else:
            passed = expected in _norm(actual)
            detail = f"visible value {_norm(actual)!r} contains {expected!r}: {passed}"
        return CheckOutcome(
            VerificationStatus.PASS if passed else VerificationStatus.FAIL,
            detail,
            self._condition_refs(condition, request),
        )

    def _page_text(
        self, condition: VerificationCondition, request: VerificationRequest
    ) -> CheckOutcome:
        expected = _norm(condition.expected)
        if not expected:
            return CheckOutcome(VerificationStatus.FAIL, "page text expectation is empty")
        passed = expected in _norm(request.observation.ax_text)
        return CheckOutcome(
            VerificationStatus.PASS if passed else VerificationStatus.FAIL,
            f"current page text contains {expected!r}: {passed}",
            self._condition_refs(condition, request),
        )

    def _page_url(
        self, condition: VerificationCondition, request: VerificationRequest
    ) -> CheckOutcome:
        expected = _norm(condition.expected)
        if not expected:
            return CheckOutcome(VerificationStatus.FAIL, "URL expectation is empty")
        passed = expected in request.observation.url
        return CheckOutcome(
            VerificationStatus.PASS if passed else VerificationStatus.FAIL,
            f"current URL {request.observation.url!r} contains {expected!r}: {passed}",
            self._condition_refs(condition, request),
        )

    # ----------------------------------------------------------------- evidence
    def _condition_refs(
        self, condition: VerificationCondition, request: VerificationRequest
    ) -> tuple[EvidenceRef, ...]:
        index = request.evidence_index()
        return tuple(index[evidence_id] for evidence_id in condition.evidence_ids if evidence_id in index)

    def _scope_refs(
        self,
        condition: VerificationCondition,
        request: VerificationRequest,
        *,
        kinds: frozenset[str] | None = None,
    ) -> tuple[EvidenceRef, ...]:
        refs = list(self._condition_refs(condition, request))
        if not refs:
            refs = list(request.evidence_scope)
        if kinds is not None:
            refs = [ref for ref in refs if ref.kind in kinds]
        return tuple(refs)

    def _screenshot(
        self, condition: VerificationCondition, request: VerificationRequest
    ) -> CheckOutcome:
        refs = self._scope_refs(condition, request, kinds=frozenset({"screenshot"}))
        if not refs:
            artifact = request.observation.artifacts.get("screenshot")
            if artifact is None or not artifact.available:
                reason = "no screenshot artifact" if artifact is None else artifact.reason
                return CheckOutcome(
                    VerificationStatus.FAIL,
                    f"screenshot is not saved for observation "
                    f"{request.observation.observation_id}: {reason}",
                )
            from ..state.integration import evidence_ref_from_artifact

            ref = evidence_ref_from_artifact(
                artifact,
                run_id=request.observation.run_id,
                branch_id=request.branch,
                evidence_id=f"ev-{request.observation.observation_id}-screenshot",
                observation_id=request.observation.observation_id,
            )
            refs = (ref,)
        for ref in refs:
            try:
                self.store.verify_evidence(ref)
            except Exception as exc:  # noqa: BLE001 - an invalid citation is a failure
                return CheckOutcome(
                    VerificationStatus.FAIL,
                    f"screenshot evidence {ref.evidence_id!r} is not valid: {exc}",
                    refs,
                )
        # Only the save/hash is checked here, never pixels or their text.
        return CheckOutcome(
            VerificationStatus.PASS,
            f"{len(refs)} saved screenshot artifact(s) verified",
            refs,
        )

    def _evidence(
        self, condition: VerificationCondition, request: VerificationRequest
    ) -> CheckOutcome:
        refs = self._scope_refs(condition, request)
        if not refs:
            return CheckOutcome(VerificationStatus.FAIL, "no evidence reference in scope")
        for ref in refs:
            try:
                self.store.verify_evidence(ref)
            except Exception as exc:  # noqa: BLE001 - invalid evidence cannot pass
                return CheckOutcome(
                    VerificationStatus.FAIL,
                    f"evidence {ref.evidence_id!r} is not valid: {exc}",
                    refs,
                )
        return CheckOutcome(
            VerificationStatus.PASS,
            f"{len(refs)} evidence artifact(s) verified",
            refs,
        )

    # -------------------------------------------------------------------- facts
    def _fact(self, condition: VerificationCondition) -> CheckOutcome:
        fact_id = str(condition.params.get("fact_id") or condition.expected or "").strip()
        if not fact_id:
            return CheckOutcome(VerificationStatus.FAIL, "fact_value_equals needs fact_id")
        fact: Fact | None = self.store.state.facts.get(fact_id)
        if fact is None:
            return CheckOutcome(VerificationStatus.UNKNOWN, f"fact {fact_id!r} is not recorded")
        expected = condition.params.get("expected", True)
        passed = _norm(fact.value) == _norm(expected)
        return CheckOutcome(
            VerificationStatus.PASS if passed else VerificationStatus.FAIL,
            f"fact {fact_id!r} value {fact.value!r} equals {expected!r}: {passed}",
            fact.evidence,
        )

    def _objects(
        self, condition: VerificationCondition, request: VerificationRequest
    ) -> CheckOutcome:
        goal = self.store.state.goals.get(request.goal_id)
        if goal is None:
            return CheckOutcome(VerificationStatus.UNKNOWN, f"goal {request.goal_id!r} is absent")
        if not goal.object_scope:
            return CheckOutcome(VerificationStatus.PASS, "goal declares no object scope")
        completed = set(goal.completed_object_ids)
        if request.completed_object_ids is not None:
            completed = set(request.completed_object_ids)
        missing = sorted(set(goal.object_scope) - completed)
        return CheckOutcome(
            VerificationStatus.PASS if not missing else VerificationStatus.FAIL,
            f"unaccounted objects: {missing}",
        )


__all__ = [
    "CheckOutcome",
    "DeterministicChecker",
    "visible_element",
    "visible_value",
]
