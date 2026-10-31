"""Programmatic withdrawal of a completion claim (phase-2 E1/E03 entry).

Recovery needs one ledger transition that the model-facing tools must never
reach: a goal that already reached ``verified`` can be withdrawn to
``invalidated`` or ``awaiting_verification`` without deleting the record, its
requirement, its acceptance criteria, its object scope, its completed objects or
the evidence that supported the earlier claim. Withdrawing appends the new
(conflicting) evidence to the existing references, so old and new sources stay
visible, and writes one audit record that remains readable after a reload.

The transition is exposed as :meth:`webagent.state.store.StateStore.invalidate_goal`
and deliberately has no tool wrapper: ``state_update`` / ``plan_update`` still
refuse to change a verified goal, so an action proposal cannot withdraw a
verified claim.

Withdrawal is idempotent. A repeated call for a goal already in the requested
status returns the recorded result without increasing the state revision and
without appending a second audit record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .records import GoalStatus, StateValidationError

#: Statuses a withdrawal may start from: a completion claim, not ordinary work.
WITHDRAWABLE_FROM: frozenset[GoalStatus] = frozenset(
    {GoalStatus.VERIFIED, GoalStatus.AWAITING_VERIFICATION}
)

#: Statuses a withdrawal may target.
WITHDRAWAL_TARGETS: frozenset[GoalStatus] = frozenset(
    {GoalStatus.INVALIDATED, GoalStatus.AWAITING_VERIFICATION}
)

_INVALIDATION_FILE = "invalidations.jsonl"
_UNSAFE_ID = frozenset("\r\n\t")


def require_goal_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StateValidationError("goal_id must be a non-empty string")
    if any(char in value for char in _UNSAFE_ID):
        raise StateValidationError("goal_id must not contain control line breaks")
    return value


def require_reason(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StateValidationError("a withdrawal needs a non-empty reason")
    return value


def coerce_target_status(
    value: GoalStatus | str | None,
    *,
    has_evidence: bool,
) -> GoalStatus:
    """Resolve the withdrawal target.

    With new conflicting evidence the claim is ``invalidated``. Without new
    evidence the claim is only withdrawn to ``awaiting_verification``: the old
    verified result may no longer be trusted, but no cause is invented.
    """

    if value is None:
        target = (
            GoalStatus.INVALIDATED if has_evidence else GoalStatus.AWAITING_VERIFICATION
        )
    else:
        try:
            target = value if isinstance(value, GoalStatus) else GoalStatus(value)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(status.value for status in sorted(WITHDRAWAL_TARGETS))
            raise StateValidationError(
                f"withdrawal target must be one of: {allowed}"
            ) from exc
    if target not in WITHDRAWAL_TARGETS:
        allowed = ", ".join(status.value for status in sorted(WITHDRAWAL_TARGETS))
        raise StateValidationError(f"withdrawal target must be one of: {allowed}")
    return target


@dataclass(frozen=True)
class GoalInvalidation:
    """One programmatic withdrawal, as returned and as written to the audit log."""

    goal_id: str
    reason: str
    from_status: GoalStatus
    to_status: GoalStatus
    changed: bool
    state_revision_before: int
    state_revision_after: int
    plan_version: int
    source_branch: str
    previous_evidence_ids: tuple[str, ...]
    appended_evidence_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    recorded_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", require_goal_id(self.goal_id))
        object.__setattr__(self, "reason", require_reason(self.reason))
        object.__setattr__(
            self, "from_status", GoalStatus(self.from_status)
        )
        object.__setattr__(self, "to_status", GoalStatus(self.to_status))
        if not isinstance(self.changed, bool):
            raise StateValidationError("invalidation.changed must be a bool")
        for name in (
            "state_revision_before",
            "state_revision_after",
            "plan_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise StateValidationError(f"invalidation.{name} must be an integer >= 0")
        if not isinstance(self.source_branch, str) or not self.source_branch.strip():
            raise StateValidationError("invalidation.source_branch must be a non-empty string")
        for name in ("previous_evidence_ids", "appended_evidence_ids", "evidence_ids"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise StateValidationError(f"invalidation.{name} must be a tuple")
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise StateValidationError(
                        f"invalidation.{name} must contain non-empty ids"
                    )
        if not isinstance(self.recorded_at, str) or not self.recorded_at.strip():
            raise StateValidationError("invalidation.recorded_at must be a timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "goal_invalidated",
            "goal_id": self.goal_id,
            "reason": self.reason,
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "changed": self.changed,
            "state_revision_before": self.state_revision_before,
            "state_revision_after": self.state_revision_after,
            "plan_version": self.plan_version,
            "source_branch": self.source_branch,
            "previous_evidence_ids": list(self.previous_evidence_ids),
            "appended_evidence_ids": list(self.appended_evidence_ids),
            "evidence_ids": list(self.evidence_ids),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalInvalidation":
        if not isinstance(data, Mapping):
            raise StateValidationError("invalidation record must be an object")
        try:
            return cls(
                goal_id=data["goal_id"],
                reason=data["reason"],
                from_status=data["from_status"],
                to_status=data["to_status"],
                changed=data.get("changed", True),
                state_revision_before=data["state_revision_before"],
                state_revision_after=data["state_revision_after"],
                plan_version=data["plan_version"],
                source_branch=data["source_branch"],
                previous_evidence_ids=tuple(data.get("previous_evidence_ids") or ()),
                appended_evidence_ids=tuple(data.get("appended_evidence_ids") or ()),
                evidence_ids=tuple(data.get("evidence_ids") or ()),
                recorded_at=data["recorded_at"],
            )
        except KeyError as exc:
            raise StateValidationError(
                f"invalidation record is missing {exc.args[0]}"
            ) from exc


class InvalidationLog:
    """Append-only audit of withdrawals inside ``.../state/``.

    The log is diagnostics, not a second source of ledger truth: the committed
    snapshot keeps the goal's status and evidence. It exists so a caller can ask
    what was withdrawn, why and with which sources after a reload.
    """

    FILENAME = _INVALIDATION_FILE

    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / self.FILENAME

    def append(self, record: GoalInvalidation) -> Path:
        if not isinstance(record, GoalInvalidation):
            raise StateValidationError("the invalidation log stores GoalInvalidation records")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self.path

    def read(self) -> tuple[GoalInvalidation, ...]:
        if not self.path.is_file():
            return ()
        records: list[GoalInvalidation] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StateValidationError(
                    f"invalid JSON on invalidation line {line_number}: {exc}"
                ) from exc
            records.append(GoalInvalidation.from_dict(value))
        return tuple(records)


def read_invalidations(state_dir: str | Path) -> tuple[GoalInvalidation, ...]:
    """Read the audit records written under one run's ``state/`` directory."""

    path = Path(state_dir)
    if path.is_file():
        return InvalidationLog(path.parent).read()
    return InvalidationLog(path).read()


__all__ = [
    "GoalInvalidation",
    "InvalidationLog",
    "WITHDRAWABLE_FROM",
    "WITHDRAWAL_TARGETS",
    "coerce_target_status",
    "read_invalidations",
    "require_goal_id",
    "require_reason",
]
