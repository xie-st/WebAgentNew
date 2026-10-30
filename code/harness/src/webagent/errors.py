"""Harness error types.

Errors carry only structured, non-secret metadata. Logging code is expected to
pass messages through :class:`webagent.redaction.Redactor` before writing.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class WebAgentError(Exception):
    """Base class for harness errors."""


class ProtocolError(WebAgentError):
    """A protocol value is invalid or internally inconsistent."""


class ConfigError(WebAgentError):
    """Configuration is missing, malformed, or contains credentials."""


class SessionStateError(WebAgentError):
    """The session is not in a state that permits the requested transition."""


class DeadlineExceeded(WebAgentError):
    """The task deadline has passed; no further model/action work is allowed."""


class BudgetExceeded(WebAgentError):
    """A configured token, message or request-size limit was exceeded."""


class ContextOverflow(BudgetExceeded):
    """The mandatory task/observation/pending content cannot fit the budget.

    The packer refuses rather than silently truncating the task or the current
    observation. ``required_tokens`` and ``available_tokens`` are estimates with
    the source recorded by the caller.
    """

    def __init__(
        self,
        message: str,
        *,
        required_tokens: int,
        available_tokens: int,
        token_count_source: str,
    ) -> None:
        super().__init__(message)
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens
        self.token_count_source = token_count_source


class RequestLimitExceeded(BudgetExceeded):
    """Serialized request exceeds the official message or byte limit."""

    def __init__(self, message: str, *, limit_name: str, limit: int, actual: int) -> None:
        super().__init__(message)
        self.limit_name = limit_name
        self.limit = limit
        self.actual = actual


class EvidenceError(WebAgentError):
    """An evidence artifact could not be saved or verified."""


class RejectReason(str, Enum):
    """Why an action request was not allowed to become a WebAction."""

    MULTIPLE_ACTIONS = "multiple_actions"
    UNKNOWN_TOOL = "unknown_tool"
    STALE_OBSERVATION = "stale_observation"
    WRONG_PAGE = "wrong_page"
    WRONG_RUN = "wrong_run"
    UNKNOWN_ELEMENT = "unknown_element"
    INVALID_ARGUMENTS = "invalid_arguments"
    EXTERNAL_ORIGIN = "external_origin"
    NO_OBSERVATION = "no_observation"
    PAUSED = "paused"
    CLOSED = "closed"
    DUPLICATE_CONSUMPTION = "duplicate_consumption"
    PENDING_ACTION = "pending_action"
    NO_PENDING_ACTION = "no_pending_action"
    IN_FLIGHT = "in_flight"
    SESSION_STATE = "session_state"


class ActionRejected(WebAgentError):
    """An action request is refused before dispatch.

    ``correction`` is the model-facing explanation. It contains the current
    observation binding and the available element numbers so the model can fix
    the request instead of retrying with a stale one.
    """

    def __init__(
        self,
        reason: RejectReason | str,
        *,
        details: Iterable[str] = (),
        correction: str = "",
    ) -> None:
        self.reason = RejectReason(reason) if not isinstance(reason, RejectReason) else reason
        self.details = tuple(str(item) for item in details)
        self.correction = str(correction)
        message = f"action rejected ({self.reason.value})"
        if self.details:
            message += ": " + "; ".join(self.details)
        super().__init__(message)

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "details": list(self.details),
            "correction": self.correction,
        }
