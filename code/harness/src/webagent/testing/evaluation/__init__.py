"""External judging for the local test environment."""

from .evidence import EvidenceBundle, EvidenceError, EvidenceRecord
from .hidden import HiddenStateError, HiddenStateReader
from .judge import (
    CheckResult,
    JudgeDispatchError,
    JudgeResult,
    LocalJudge,
    judge,
    judge_checker_name,
    register_judge_checker,
    resolve_judge_checker,
)

__all__ = [
    "CheckResult",
    "EvidenceBundle",
    "EvidenceError",
    "EvidenceRecord",
    "HiddenStateError",
    "HiddenStateReader",
    "JudgeDispatchError",
    "JudgeResult",
    "LocalJudge",
    "judge",
    "judge_checker_name",
    "register_judge_checker",
    "resolve_judge_checker",
]
