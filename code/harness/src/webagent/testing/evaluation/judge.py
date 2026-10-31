"""External judge for the local L01-L03 cases.

The judge is called after a run. It reads hidden site state through a
read-only SQLite connection and combines that with an explicit evidence
bundle (screenshots) supplied by the caller. It never calls the site's HTTP
surface and never exposes hidden answers to the running Agent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..cases import CaseDefinition
from .evidence import EvidenceBundle, EvidenceRecord
from .hidden import HiddenStateReader

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF\u2600-\u27BF\uFE0F]"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    gap: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {"name": self.name, "passed": self.passed, "detail": self.detail}
        if self.gap:
            payload["gap"] = self.gap
        return payload


@dataclass(frozen=True)
class JudgeResult:
    case_id: str
    case_version: str
    seed: int
    checks: tuple[CheckResult, ...]
    evidence_count: int
    hidden_state_hash: str

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "fail"

    def failed_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_version": self.case_version,
            "seed": self.seed,
            "verdict": self.verdict,
            "passed": self.passed,
            "evidence_count": self.evidence_count,
            "hidden_state_hash": self.hidden_state_hash,
            "checks": [check.to_dict() for check in self.checks],
        }


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))


def _preview(values: Iterable[str], limit: int = 8) -> str:
    items = sorted(str(value) for value in values)
    head = ", ".join(items[:limit])
    if len(items) > limit:
        head += f", … (+{len(items) - limit})"
    return head


class JudgeDispatchError(RuntimeError):
    """The case declares no registered external judge checker."""


Checker = Callable[["LocalJudge", EvidenceBundle, Any], list[CheckResult]]
_CHECKER_REGISTRY: dict[str, Checker] = {}


def register_judge_checker(name: str) -> Callable[[Checker], Checker]:
    """Register one named external checker used by the unified dispatch."""

    def decorator(func: Checker) -> Checker:
        key = str(name).strip()
        if not key:
            raise ValueError("judge checker name must not be empty")
        _CHECKER_REGISTRY[key] = func
        return func

    return decorator


def judge_checker_name(case: CaseDefinition) -> str:
    """The checker a case declares, defaulting to its case id."""

    declared = case.judge_config.get("checker")
    if declared is None:
        return case.case_id.lower()
    name = str(declared).strip()
    if not name:
        raise JudgeDispatchError(f"{case.case_id}: hidden.judge.checker is empty")
    return name


def resolve_judge_checker(case: CaseDefinition) -> Checker:
    """Resolve a case through the single judge dispatch table."""

    name = judge_checker_name(case)
    checker = _CHECKER_REGISTRY.get(name)
    if checker is None:
        available = ", ".join(sorted(_CHECKER_REGISTRY)) or "none"
        raise JudgeDispatchError(
            f"{case.case_id}: no judge checker registered for {name!r}; "
            f"available: {available}"
        )
    return checker


class LocalJudge:
    def __init__(
        self,
        case: CaseDefinition,
        hidden: HiddenStateReader,
        *,
        evidence_root: str | Path | None = None,
    ) -> None:
        self.case = case
        self.hidden = hidden
        self.config = case.judge_config
        self.evidence_root = Path(evidence_root) if evidence_root is not None else Path.cwd()

    # --------------------------------------------------------------- evidence
    def _verify_screenshot(self, record: EvidenceRecord) -> str | None:
        reference = record.reference
        path = Path(reference)
        if not path.is_absolute():
            path = (self.evidence_root / path).resolve()
            if not path.is_file() and Path(reference).is_file():
                path = Path(reference).resolve()
        if not path.is_file():
            return f"missing file {reference}"
        try:
            data = path.read_bytes()
        except OSError as exc:
            return f"unreadable {reference}: {exc}"
        if not data.startswith(_PNG_SIGNATURE):
            return f"not a PNG {reference}"
        if record.sha256 is not None:
            digest = hashlib.sha256(data).hexdigest()
            if digest != record.sha256:
                return f"sha256 mismatch {reference}"
        return None

    def _screenshot_audit(
        self,
        expected: Sequence[str],
        object_type: str,
        evidence: EvidenceBundle,
        *,
        required: bool,
    ) -> tuple[list[str], list[str]]:
        missing: list[str] = []
        invalid: list[str] = []
        for object_id in expected:
            records = evidence.for_object(object_type, object_id, kind="screenshot")
            if not records:
                if required:
                    missing.append(object_id)
                continue
            problems = [
                problem
                for record in records
                if (problem := self._verify_screenshot(record)) is not None
            ]
            if len(problems) == len(records):
                invalid.append(f"{object_id}: {problems[0]}")
        return missing, invalid

    # ----------------------------------------------------------------- answer
    @staticmethod
    def _answer_tokens(answer: Any, field: str) -> tuple[list[Any] | None, str]:
        if answer is None:
            return None, "no answer supplied"
        if isinstance(answer, Mapping):
            value = answer.get(field)
            if value is None:
                value = answer.get("items")
            if value is None:
                value = answer.get("objects")
            if value is None:
                if "id" in answer:
                    return [answer], ""
                return None, f"answer mapping has no {field!r} items"
        elif isinstance(answer, (list, tuple)):
            value = answer
        elif isinstance(answer, str):
            cleaned = answer.strip().strip("[]")
            value = [part for part in re.split(r"[,;\n]", cleaned) if part.strip()]
        else:
            return None, f"unsupported answer type {type(answer).__name__}"
        if not isinstance(value, (list, tuple)):
            return None, f"answer field {field!r} must be a list"
        return list(value), ""

    @staticmethod
    def _product_lookup(products: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for product in products:
            for key in ("id", "sku", "name"):
                value = product.get(key)
                if value is not None:
                    lookup.setdefault(str(value), str(product["id"]))
                    lookup.setdefault(str(value).casefold(), str(product["id"]))
        return lookup

    def _resolve_products(
        self, tokens: Sequence[Any], products: Sequence[Mapping[str, Any]]
    ) -> tuple[list[str], list[str]]:
        lookup = self._product_lookup(products)
        resolved: list[str] = []
        unresolved: list[str] = []
        for token in tokens:
            candidate: Any = token
            if isinstance(token, Mapping):
                candidate = token.get("id") or token.get("sku") or token.get("name")
            text = str(candidate).strip().strip("\"'[]")
            found = lookup.get(text) or lookup.get(text.casefold())
            if found is None:
                unresolved.append(text)
            else:
                resolved.append(found)
        return resolved, unresolved

    # ----------------------------------------------------------------- checks
    def evaluate(self, *, evidence: Any = None, answer: Any = None) -> JudgeResult:
        bundle = EvidenceBundle.coerce(evidence)
        checks = resolve_judge_checker(self.case)(self, bundle, answer)
        return JudgeResult(
            case_id=self.case.case_id,
            case_version=self.case.case_version,
            seed=int(self.hidden.meta().get("seed", 0)),
            checks=tuple(checks),
            evidence_count=len(bundle),
            hidden_state_hash=self.hidden.snapshot_hash(),
        )

    def _expected(self, object_type: str) -> list[str]:
        records = self.hidden.hidden_truth(
            role=str(self.config["role"]), item_type=object_type
        )
        return [str(record["item_id"]) for record in records]

    def _judge_l01(self, evidence: EvidenceBundle, answer: Any) -> list[CheckResult]:
        object_type = str(self.config["object_type"])
        expected = self._expected(object_type)
        checks = [
            CheckResult(
                "target_count",
                len(expected) == int(self.config.get("expected_count", 1)),
                f"hidden targets: {_preview(expected)}",
            )
        ]
        if not expected:
            checks.extend(
                CheckResult(name, False, "no hidden target")
                for name in ("target_viewed", "draft_text", "draft_cancelled", "screenshot")
            )
            checks.append(self._submission_check("submissions_zero"))
            return checks
        target = expected[0]
        viewed = self.hidden.viewed_ids(object_type)
        checks.append(
            CheckResult(
                "target_viewed",
                target in viewed,
                f"target {target} viewed={target in viewed}",
            )
        )
        expected_text = _norm(self.config.get("expected_draft_text", ""))
        texts = [_norm(text) for text in self.hidden.draft_texts(object_type, target)]
        last_text = texts[-1] if texts else ""
        checks.append(
            CheckResult(
                "draft_text",
                bool(expected_text) and last_text == expected_text,
                f"expected {expected_text!r}; latest recorded {last_text!r}",
            )
        )
        checks.append(
            CheckResult(
                "draft_cancelled",
                self.hidden.cancelled(object_type, target),
                f"target {target} cancelled="
                f"{self.hidden.cancelled(object_type, target)}",
            )
        )
        missing, invalid = self._screenshot_audit(
            expected,
            object_type,
            evidence,
            required=bool(self.config.get("require_screenshot", True)),
        )
        checks.append(
            CheckResult(
                "screenshot",
                not missing and not invalid,
                f"missing={missing} invalid={invalid}",
            )
        )
        checks.append(self._submission_check("submissions_zero"))
        if answer is not None:
            products = self.hidden.products()
            tokens, error = self._answer_tokens(answer, "product_ids")
            if tokens is None:
                checks.append(CheckResult("answer_target", False, error))
            else:
                resolved, unresolved = self._resolve_products(tokens, products)
                checks.append(
                    CheckResult(
                        "answer_target",
                        target in resolved and not unresolved,
                        f"resolved={resolved} unresolved={unresolved}",
                    )
                )
        return checks

    def _judge_l02(self, evidence: EvidenceBundle, answer: Any) -> list[CheckResult]:
        object_type = str(self.config["object_type"])
        expected = self._expected(object_type)
        expected_count = int(self.config.get("expected_count", 5))
        checks = [
            CheckResult(
                "expected_count",
                len(expected) == expected_count,
                f"hidden targets {len(expected)} expected {expected_count}",
            )
        ]
        viewed = self.hidden.viewed_ids(object_type)
        missing_views = [item for item in expected if item not in viewed]
        checks.append(
            CheckResult(
                "viewed_all",
                not missing_views,
                f"not viewed: {_preview(missing_views)}",
            )
        )
        products = self.hidden.products()
        tokens, error = self._answer_tokens(answer, "product_ids")
        if tokens is None:
            checks.append(CheckResult("answer_exact", False, error))
            checks.append(CheckResult("answer_distinct", False, error))
        else:
            resolved, unresolved = self._resolve_products(tokens, products)
            resolved_set = set(resolved)
            exact = not unresolved and resolved_set == set(expected) and len(resolved) == len(tokens)
            checks.append(
                CheckResult(
                    "answer_exact",
                    exact,
                    f"reported={_preview(resolved_set)} expected={_preview(expected)} "
                    f"unresolved={_preview(unresolved)}",
                )
            )
            distinct = len(resolved) == len(set(resolved)) and len(resolved) == len(tokens)
            checks.append(
                CheckResult(
                    "answer_distinct",
                    distinct,
                    f"tokens={len(tokens)} resolved={len(resolved)} unique={len(set(resolved))}",
                )
            )
        missing, invalid = self._screenshot_audit(
            expected,
            object_type,
            evidence,
            required=bool(self.config.get("require_evidence_each", True)),
        )
        checks.append(
            CheckResult(
                "screenshot",
                not missing and not invalid,
                f"missing={missing} invalid={invalid}",
            )
        )
        checks.append(self._submission_check("submissions_zero"))
        return checks

    def _judge_l03(self, evidence: EvidenceBundle, answer: Any) -> list[CheckResult]:
        object_type = str(self.config["object_type"])
        expected = self._expected(object_type)
        expected_count = int(self.config.get("expected_count", 20))
        min_chars = int(self.config.get("min_draft_chars", 1))
        require_emoji = bool(self.config.get("require_emoji", True))
        checks = [
            CheckResult(
                "expected_count",
                len(expected) == expected_count,
                f"hidden targets {len(expected)} expected {expected_count}",
            )
        ]
        viewed = self.hidden.viewed_ids(object_type)
        not_viewed = [item for item in expected if item not in viewed]
        checks.append(
            CheckResult(
                "viewed_all",
                not not_viewed,
                f"not viewed: {_preview(not_viewed)}",
            )
        )
        edited: list[str] = []
        short: list[str] = []
        no_emoji: list[str] = []
        cancelled: list[str] = []
        for object_id in expected:
            texts = self.hidden.draft_texts(object_type, object_id)
            valid_texts = [text for text in texts if len(_norm(text)) >= min_chars]
            emoji_texts = [text for text in valid_texts if _has_emoji(text)]
            if valid_texts:
                edited.append(object_id)
            else:
                short.append(object_id)
            if require_emoji and not emoji_texts:
                no_emoji.append(object_id)
            if self.hidden.cancelled(object_type, object_id):
                cancelled.append(object_id)
        checks.append(
            CheckResult(
                "edited_all",
                not short,
                f"missing or too short: {_preview(short)}",
            )
        )
        checks.append(
            CheckResult(
                "emoji_all",
                not no_emoji,
                f"missing text+emoji: {_preview(no_emoji)}",
            )
        )
        not_cancelled = [item for item in expected if item not in cancelled]
        checks.append(
            CheckResult(
                "cancelled_all",
                not not_cancelled,
                f"not cancelled: {_preview(not_cancelled)}",
            )
        )
        missing, invalid = self._screenshot_audit(
            expected,
            object_type,
            evidence,
            required=bool(self.config.get("require_screenshot", True)),
        )
        checks.append(
            CheckResult(
                "screenshot",
                not missing and not invalid,
                f"missing={_preview(missing)} invalid={_preview(invalid)}",
            )
        )
        checks.append(self._submission_check("submissions_zero"))
        return checks

    def _submission_check(self, name: str) -> CheckResult:
        maximum = int(self.config.get("max_submissions", 0))
        total = self.hidden.submission_count()
        return CheckResult(
            name,
            total <= maximum,
            f"submission_count={total} max={maximum}",
        )


def _field_value_gap(actual: str, expected: str, replaced: str) -> str:
    """Concrete reason why a replaced field value is not the requested value."""

    if not actual:
        return f"field has no saved value; expected {expected!r}"
    if actual == expected:
        return ""
    if replaced:
        if actual == replaced:
            return f"field was not updated; it still equals baseline {replaced!r}"
        if actual.startswith(replaced) and actual.endswith(expected):
            return (
                f"leftover baseline {replaced!r} kept before the requested value; "
                "the value was appended, not replaced"
            )
        if actual.startswith(expected) and actual.endswith(replaced):
            return (
                f"leftover baseline {replaced!r} kept after the requested value; "
                "the value was appended, not replaced"
            )
        if replaced in actual:
            return f"leftover baseline {replaced!r} remains in the saved value"
    if expected and expected in actual:
        return "requested text was saved with extra text instead of replacing the field"
    return f"saved value {actual!r} does not equal requested text {expected!r}"


@register_judge_checker("field_replacement")
def _check_field_replacement(
    judge: "LocalJudge", evidence: EvidenceBundle, answer: Any
) -> list[CheckResult]:
    """Generic external checker: replace a saved field and correct leftovers.

    It consumes only the hidden site database (view/save activity and the
    persisted field value) and the caller-supplied screenshot bundle. ``answer``
    and any Agent state are deliberately not read.
    """

    config = judge.config
    object_type = str(config["object_type"])
    expected_targets = judge._expected(object_type)
    expected_count = int(config.get("expected_count", 1))
    expected_text = _norm(config.get("expected_text", ""))
    replaced_text = _norm(config.get("replaced_text", ""))
    label = str(config.get("field_label", "field"))
    min_saves = int(config.get("min_saves", 1))
    checks = [
        CheckResult(
            "expected_count",
            len(expected_targets) == expected_count,
            f"hidden targets: {_preview(expected_targets)}",
        )
    ]
    if not expected_targets:
        checks.extend(
            CheckResult(name, False, "no hidden target", gap="no hidden target")
            for name in (
                "target_viewed",
                "baseline_saved",
                "field_saved",
                "field_value_exact",
                "field_replaced",
                "screenshot",
            )
        )
        checks.append(judge._submission_check("submissions_zero"))
        return checks

    target = expected_targets[0]
    viewed = target in judge.hidden.viewed_ids(object_type)
    checks.append(
        CheckResult(
            "target_viewed",
            viewed,
            f"target {target} viewed={viewed}",
        )
    )

    save_texts = [_norm(text) for text in judge.hidden.saved_draft_texts(object_type, target)]
    current_raw = judge.hidden.current_draft(object_type, target)
    current_text = _norm(current_raw or "")
    latest_saved = save_texts[-1] if save_texts else ""
    baseline_saved = not replaced_text or any(
        text == replaced_text for text in save_texts
    )
    checks.append(
        CheckResult(
            "baseline_saved",
            baseline_saved,
            f"baseline {replaced_text!r} in saved values {_preview(save_texts)}",
            gap="" if baseline_saved else f"baseline value {replaced_text!r} was never saved",
        )
    )

    exact = bool(expected_text) and current_text == expected_text
    gap = _field_value_gap(current_text, expected_text, replaced_text)
    checks.append(
        CheckResult(
            "field_saved",
            len(save_texts) >= min_saves and latest_saved == expected_text and exact,
            f"{label}: saves={len(save_texts)} latest={latest_saved!r} "
            f"current={current_text!r}",
            gap="" if exact and latest_saved == expected_text else gap,
        )
    )
    checks.append(
        CheckResult(
            "field_value_exact",
            exact,
            f"{label}: expected {expected_text!r}; actual {current_text!r}",
            gap=gap,
        )
    )
    checks.append(
        CheckResult(
            "field_replaced",
            exact and not gap,
            f"{label}: expected exactly {expected_text!r}; actual {current_text!r}; "
            f"gap={gap or 'none'}",
            gap=gap,
        )
    )

    missing, invalid = judge._screenshot_audit(
        expected_targets,
        object_type,
        evidence,
        required=bool(config.get("require_screenshot", True)),
    )
    checks.append(
        CheckResult(
            "screenshot",
            not missing and not invalid,
            f"missing={missing} invalid={invalid}",
            gap="" if not missing and not invalid else "final screenshot missing or invalid",
        )
    )
    checks.append(judge._submission_check("submissions_zero"))
    return checks


def _register_builtin_case_checkers() -> None:
    for name in ("l01", "l02", "l03"):
        _CHECKER_REGISTRY.setdefault(name, getattr(LocalJudge, f"_judge_{name}"))


_register_builtin_case_checkers()


def judge(
    prepared: Any,
    *,
    evidence: Any = None,
    answer: Any = None,
    evidence_root: str | Path | None = None,
) -> JudgeResult:
    """Judge a prepared case using hidden state and external evidence."""

    case = prepared.case
    hidden = HiddenStateReader(prepared.db_path)
    return LocalJudge(
        case, hidden, evidence_root=evidence_root or prepared.run_dir
    ).evaluate(evidence=evidence, answer=answer)


__all__ = [
    "CheckResult",
    "JudgeDispatchError",
    "JudgeResult",
    "LocalJudge",
    "judge",
    "judge_checker_name",
    "register_judge_checker",
    "resolve_judge_checker",
]
