"""Phase-2 E1 recovery: classification, closure, unknown effects and restart."""

from .classify import classify_condition, classify_failure, classify_feedback
from .controller import (
    RecoveryController,
    RecoveryLimitExceeded,
    RecoveryLimits,
    RecoveryNeedsLedgerSupport,
)
from .coordinator import RecoveryCoordinator, RecoveryOutcome, StagnationRecoveryResult
from .intake import FeedbackIntake, IntakeResult, RecoveryIntake
from .planner import RecoveryPlanner, reverse_dependency_closure
from .restart import RestartRecovery
from .stagnation import ProgressRecord, StagnationDetector, StagnationSignal
from .wiring import OfficialRecoveryRuntime, RecoveryNote
from .types import (
    EffectExpectation,
    EffectOutcome,
    EffectResolution,
    FailureEvidence,
    RecoveryAnchor,
    RecoveryAttempt,
    RecoveryClass,
    RecoveryError,
    RecoveryPlan,
    RecoveryResult,
    RestartResult,
)
from .unknown_effect import UnknownEffectResolver

__all__ = [
    "EffectExpectation",
    "EffectOutcome",
    "EffectResolution",
    "FailureEvidence",
    "FeedbackIntake",
    "IntakeResult",
    "ProgressRecord",
    "RecoveryAnchor",
    "RecoveryAttempt",
    "RecoveryClass",
    "RecoveryController",
    "RecoveryCoordinator",
    "RecoveryError",
    "RecoveryIntake",
    "RecoveryLimitExceeded",
    "RecoveryLimits",
    "RecoveryNeedsLedgerSupport",
    "RecoveryPlan",
    "RecoveryPlanner",
    "RecoveryResult",
    "RestartRecovery",
    "RestartResult",
    "StagnationDetector",
    "StagnationRecoveryResult",
    "StagnationSignal",
    "UnknownEffectResolver",
    "classify_condition",
    "classify_failure",
    "classify_feedback",
    "reverse_dependency_closure",
]
