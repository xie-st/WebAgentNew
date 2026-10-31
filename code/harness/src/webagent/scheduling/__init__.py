"""F-wave scheduling skeleton.

Public surface:

* :class:`TaskCoordinator` - default serial; explicit ``parallel`` config turns
  on independent read-only branches; shared writes degrade the run to serial.
* :class:`BranchSpec`, :class:`BranchContext`, :class:`BranchReport`,
  :class:`CoordinatorReport` - the branch contract and its evidence.
* :class:`SharedRateLimiter` - one site schedule and shared cooldown.
* :class:`TabPool` / :class:`TabLease` - Tab ownership and one in-flight action.
* :class:`RootBudgetLedger` - atomic root accounting for every model call.
* :data:`CONTROL_HOOK_EVENTS` / :class:`ControlHookEvent` - the F07 hook shape;
  ``pause``/``resume``/``drain`` raise ``NotImplementedError`` until the R wave
  is merged.
"""

from .coordinator import (
    CONTROL_HOOK_EVENTS,
    F07_NOT_IMPLEMENTED,
    BranchContext,
    BranchReport,
    BranchSpec,
    ControlHookEvent,
    CoordinatorReport,
    LockWait,
    TaskCoordinator,
)
from ..config import ParallelConfig
from .errors import (
    CoordinatorError,
    RateLimited,
    SchedulingError,
    TabUnavailable,
    WaitCancelled,
)
from .ledger import ModelCallRecord, RootBudgetLedger
from .limiter import RateWait, SharedRateLimiter
from .tabs import TabLease, TabPool

__all__ = [
    "CONTROL_HOOK_EVENTS",
    "F07_NOT_IMPLEMENTED",
    "BranchContext",
    "BranchReport",
    "BranchSpec",
    "ControlHookEvent",
    "CoordinatorError",
    "CoordinatorReport",
    "LockWait",
    "ModelCallRecord",
    "ParallelConfig",
    "RateLimited",
    "RateWait",
    "RootBudgetLedger",
    "SchedulingError",
    "SharedRateLimiter",
    "TabLease",
    "TabPool",
    "TabUnavailable",
    "TaskCoordinator",
    "WaitCancelled",
]
