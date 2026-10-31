"""Evidence-aware failure classification.

The classifier does not turn every error into a network problem. It uses the
feedback status, the error type, the page text and whether the action was a
declared write. When none of those support a narrower class, the result stays
``unknown``.
"""

from __future__ import annotations

from typing import Any

from ..protocol import ActionOp, Feedback, FeedbackStatus
from .types import RecoveryClass

_AUTH_WORDS = (
    "sign in required",
    "verification required",
    "session has expired",
    "verify your account",
)
_TIMEOUT_WORDS = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection reset",
    "network",
    "503",
    "502",
    "429",
)
_PROTOCOL_WORDS = (
    "unknown tool",
    "invalid argument",
    "element is not in this observation",
    "stale observation",
    "protocol",
)


def classify_failure(
    *,
    status: FeedbackStatus | str,
    message: str = "",
    error_type: str | None = None,
    exception_type: str | None = None,
    action_op: ActionOp | str | None = None,
    page_text: str = "",
    url: str = "",
    explicit_class: RecoveryClass | str | None = None,
) -> RecoveryClass:
    """Classify one failure without inventing a cause.

    ``explicit_class`` is used by a caller that already has evidence (for
    example a verified login overlay detector). It still cannot hide an unknown
    result: a caller may explicitly say ``unknown``.
    """

    if explicit_class is not None:
        return RecoveryClass(explicit_class)

    status = FeedbackStatus(status)
    text = f"{message} {page_text} {url}".casefold()
    error = f"{error_type or ''} {exception_type or ''}".casefold()
    op = ActionOp(action_op) if isinstance(action_op, str) else action_op
    write = bool(op is not None and op in (ActionOp.CLICK, ActionOp.TYPE))

    if any(word in text for word in _AUTH_WORDS) or "authentication" in error or "login" in error:
        return RecoveryClass.AUTHENTICATION

    if status is FeedbackStatus.UNKNOWN:
        if write:
            return RecoveryClass.UNKNOWN_EFFECT
        return RecoveryClass.TRANSIENT

    if status is FeedbackStatus.ERROR:
        if "user_pause" in error or "userpause" in error:
            return RecoveryClass.USER_PAUSE
        if any(word in error for word in _PROTOCOL_WORDS):
            return RecoveryClass.PROTOCOL
        if any(word in error for word in _TIMEOUT_WORDS) or any(
            word in text for word in _TIMEOUT_WORDS
        ):
            return RecoveryClass.TRANSIENT
        if write:
            # A dispatched write that returned a non-auth, non-protocol error is
            # still an unknown business outcome until the page is inspected.
            return RecoveryClass.UNKNOWN_EFFECT
        return RecoveryClass.UNKNOWN

    return RecoveryClass.UNKNOWN


def classify_feedback(
    feedback: Feedback,
    *,
    action_op: ActionOp | str | None = None,
    page_text: str = "",
    url: str = "",
) -> RecoveryClass:
    return classify_failure(
        status=feedback.status,
        message=feedback.message,
        error_type=feedback.error_type,
        action_op=action_op,
        page_text=page_text,
        url=url,
    )


def classify_condition(*, passed: bool | None) -> RecoveryClass:
    """A deterministic condition mismatch is condition_failed; no evidence is unknown."""

    if passed is True:
        raise ValueError("a passing condition is not a failure")
    if passed is False:
        return RecoveryClass.CONDITION_FAILED
    return RecoveryClass.UNKNOWN


__all__ = ["classify_condition", "classify_failure", "classify_feedback"]
