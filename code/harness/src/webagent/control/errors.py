"""Control-layer errors.

``ControlPaused`` and ``ControlCancelled`` derive from
:class:`~webagent.errors.ActionRejected` so they can travel through the same
dispatch gate as the phase-1 session rejections. They are not protocol errors:
the model must not format-correct a pause into a new action.
"""

from __future__ import annotations

from ..errors import ActionRejected, RejectReason


class ControlError(RuntimeError):
    """A control-layer operation was used outside its contract."""


class ControlStateError(ControlError):
    """The root control state does not permit this transition."""


class ControlWaitTimeout(ControlError):
    """No explicit user resume arrived before the wait timeout.

    This is deliberately not an automatic resume: the run stays in
    ``user_control`` and the caller decides whether to keep waiting, cancel or
    report the handoff as unfinished.
    """


class ControlPaused(ActionRejected):
    """A page action may not be dispatched while the task is handed off."""

    def __init__(self, message: str = "the task is paused for user control") -> None:
        super().__init__(
            RejectReason.PAUSED,
            details=[message],
            correction=(
                "The task is paused and the browser belongs to the user. No action "
                "may be dispatched until an explicit resume signal arrives; after "
                "resume the page is re-observed and old element numbers are void."
            ),
        )


class ControlCancelled(ActionRejected):
    """The task was cancelled from the control layer."""

    def __init__(self, message: str = "the task was cancelled") -> None:
        super().__init__(
            RejectReason.CLOSED,
            details=[message],
            correction="The task is cancelled; no further page action is possible.",
        )


__all__ = [
    "ControlCancelled",
    "ControlError",
    "ControlPaused",
    "ControlStateError",
    "ControlWaitTimeout",
]
