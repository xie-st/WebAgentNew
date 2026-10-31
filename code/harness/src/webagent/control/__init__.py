"""Phase-2 control layer: takeover detection, pause/drain/handoff/resume.

The phase-1 file-based ``RunControl`` (``pause`` / ``resume`` / ``cancel`` in
``control.jsonl``) is preserved as ``ControlSource``/``FileControlSource`` input;
this package adds the root task state, visible-state takeover detection, the
dispatch gate, separate timing, independent control events, model accounting
that survives a pause, the controlled loop and restart-from-snapshot.
"""

from .accounting import ModelCallAccounting, ModelCallRecord
from .coordinator import ControlCoordinator
from .detection import TakeoverSignal, detect_takeover, takeover_reason
from .errors import (
    ControlCancelled,
    ControlError,
    ControlPaused,
    ControlStateError,
    ControlWaitTimeout,
)
from .events import (
    CONTROL_EVENT_SCHEMA_VERSION,
    ControlEventLog,
    read_control_events,
)
from .file_control import RunControl
from .loop import ControlledLoopRunner
from .model import ControlAwareModel, ControlModelCall
from .restart import (
    RestartLoad,
    RestartResult,
    load_restart_snapshot,
    restart_from_snapshot,
)
from .signals import (
    ControlSource,
    FileControlSource,
    MemoryControlSource,
    SIGNAL_COMMANDS,
)
from .state import (
    ACTION_STATES,
    ControlSnapshot,
    ControlState,
    HandoffReason,
    TERMINAL_CONTROL_STATES,
)
from .timing import ControlTiming, TimingSnapshot

__all__ = [
    "ACTION_STATES",
    "CONTROL_EVENT_SCHEMA_VERSION",
    "ControlAwareModel",
    "ControlCancelled",
    "ControlCoordinator",
    "ControlError",
    "ControlEventLog",
    "ControlModelCall",
    "ControlPaused",
    "ControlSnapshot",
    "ControlSource",
    "ControlState",
    "ControlStateError",
    "ControlTiming",
    "ControlWaitTimeout",
    "ControlledLoopRunner",
    "FileControlSource",
    "HandoffReason",
    "MemoryControlSource",
    "ModelCallAccounting",
    "ModelCallRecord",
    "RestartLoad",
    "RestartResult",
    "RunControl",
    "SIGNAL_COMMANDS",
    "TERMINAL_CONTROL_STATES",
    "TakeoverSignal",
    "TimingSnapshot",
    "detect_takeover",
    "load_restart_snapshot",
    "read_control_events",
    "restart_from_snapshot",
    "takeover_reason",
]
