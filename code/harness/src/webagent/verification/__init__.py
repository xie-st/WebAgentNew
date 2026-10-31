"""Phase-2 E1 verification: evidence-backed goal results and finish checks."""

from .checks import CheckOutcome, DeterministicChecker, visible_element, visible_value
from .completion import CompletionCheck, CompletionChecker, make_finish_gate
from .semantic import (
    ModelSemanticJudge,
    SemanticJudgement,
    SemanticVerificationError,
    parse_semantic_answer,
)
from .tools import (
    COMPLETION_TOOL_NAMES,
    CompletionToolRuntime,
    RequestFinishTool,
    build_completion_tools,
)
from .types import (
    CheckKind,
    CompletionDecision,
    ConditionResult,
    DETERMINISTIC_CHECKS,
    SEMANTIC_CHECKS,
    SUPPORTED_CHECKS,
    VerificationCondition,
    VerificationError,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
    VerificationTrigger,
)
from .verifier import GoalVerifier

__all__ = [
    "COMPLETION_TOOL_NAMES",
    "CheckKind",
    "CheckOutcome",
    "CompletionCheck",
    "CompletionChecker",
    "CompletionDecision",
    "CompletionToolRuntime",
    "ConditionResult",
    "DETERMINISTIC_CHECKS",
    "DeterministicChecker",
    "GoalVerifier",
    "ModelSemanticJudge",
    "RequestFinishTool",
    "SEMANTIC_CHECKS",
    "SUPPORTED_CHECKS",
    "SemanticJudgement",
    "SemanticVerificationError",
    "VerificationCondition",
    "VerificationError",
    "VerificationRequest",
    "VerificationResult",
    "VerificationStatus",
    "VerificationTrigger",
    "build_completion_tools",
    "make_finish_gate",
    "parse_semantic_answer",
    "visible_element",
    "visible_value",
]
