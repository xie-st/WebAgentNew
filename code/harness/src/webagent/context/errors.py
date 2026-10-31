"""Errors raised by the final-send context check.

They live beside the ``context`` package because the rest of the harness only
needs to know that a send was refused, never how the view was assembled. Both
types carry the check report so evidence can name the exact regions and limits
without re-measuring the prompt.
"""

from __future__ import annotations

from typing import Any

from ..errors import BudgetExceeded, WebAgentError


class ContextError(WebAgentError):
    """A necessary-context region could not be protected on the final send."""

    def __init__(self, message: str, *, report: Any | None = None) -> None:
        super().__init__(message)
        self.report = report


class ContextProtectionError(ContextError):
    """One of the four necessary regions is missing or was changed in the send.

    This is deliberately not a budget error: the request may fit comfortably and
    still violate the contract because a region was dropped or rewritten. The
    send is refused instead of being accepted with the protected content gone.
    """


class ContextBudgetExceeded(ContextError, BudgetExceeded):
    """The final send exceeds the project input budget.

    ``report.over_limit`` is true and every necessary section in
    ``report.sections`` stays byte-complete: the refusal reports the overflow
    rather than silently trimming the task, AX text, ledger view or effect
    records.
    """
