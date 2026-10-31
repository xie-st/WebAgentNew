"""Phase-2 task state ledger (batch D2).

Public surface used by later batches:

* :class:`TaskState` and its record types (:mod:`webagent.state.records`).
* :class:`StateStore` for atomic snapshots, version checks and pending effects.
* :class:`StateUpdate` / :class:`PlanUpdate` mutation requests.
* ``state_read`` / ``state_update`` / ``plan_update`` EffGen tools.
"""

from .ledger import PlanUpdate, StateUpdate
from .invalidation import (
    GoalInvalidation,
    InvalidationLog,
    WITHDRAWABLE_FROM,
    WITHDRAWAL_TARGETS,
    read_invalidations,
)
from .records import (
    EffectStatus,
    EvidenceRef,
    EvidenceReferenceError,
    EvidenceResolver,
    Fact,
    FactKind,
    FactValidity,
    Goal,
    GoalStatus,
    PendingEffect,
    PlanNode,
    PlanNodeStatus,
    PlanRejected,
    StateConflict,
    StateError,
    StateValidationError,
    TASK_STATE_SCHEMA_VERSION,
    TaskState,
)
from .store import StateNotFound, StatePersistenceError, StateStore
from .tools import STATE_TOOL_NAMES, StateToolRuntime, build_state_tools
from .integration import (
    DEFAULT_EFFECT_RULES,
    EffectClass,
    EffectIntent,
    EffectPolicy,
    EffectRule,
    ObservedObject,
    StatefulLoopRunner,
    create_stateful_driver,
    evidence_ref_from_artifact,
    observed_page_fact_id,
    record_observed_object,
)

__all__ = [
    "DEFAULT_EFFECT_RULES",
    "EffectClass",
    "EffectIntent",
    "EffectPolicy",
    "EffectRule",
    "EffectStatus",
    "EvidenceRef",
    "EvidenceReferenceError",
    "EvidenceResolver",
    "Fact",
    "FactKind",
    "FactValidity",
    "Goal",
    "GoalInvalidation",
    "GoalStatus",
    "InvalidationLog",
    "PendingEffect",
    "PlanNode",
    "PlanNodeStatus",
    "PlanRejected",
    "PlanUpdate",
    "STATE_TOOL_NAMES",
    "StateConflict",
    "ObservedObject",
    "StateError",
    "StateNotFound",
    "StatePersistenceError",
    "StateStore",
    "StateToolRuntime",
    "StateUpdate",
    "StateValidationError",
    "StatefulLoopRunner",
    "TASK_STATE_SCHEMA_VERSION",
    "TaskState",
    "WITHDRAWABLE_FROM",
    "WITHDRAWAL_TARGETS",
    "build_state_tools",
    "create_stateful_driver",
    "evidence_ref_from_artifact",
    "observed_page_fact_id",
    "read_invalidations",
    "record_observed_object",
]
