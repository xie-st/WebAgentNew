"""Errors used by the F-wave scheduling skeleton.

They are separate from :mod:`webagent.errors` so the scheduling package can be
reviewed on its own; all of them are ordinary harness errors and carry no
credential-bearing text.
"""

from __future__ import annotations

from ..errors import WebAgentError


class SchedulingError(WebAgentError):
    """Base class for coordinator and branch failures."""


class CoordinatorError(SchedulingError):
    """The coordinator was used in a state that permits no scheduling."""


class WaitCancelled(SchedulingError):
    """A pacing or lock wait was cancelled before it could continue."""

    def __init__(self, message: str = "scheduling wait was cancelled") -> None:
        super().__init__(message)


class TabUnavailable(SchedulingError):
    """No free Tab/page is available, or the requested one belongs elsewhere."""


class RateLimited(SchedulingError):
    """A page action asked for the shared site backoff.

    The branch raises this from a page action when the site answers 429 or shows
    a cooling hint. ``retry_after`` is the site's own Retry-After value when it
    supplied one; otherwise the coordinator uses the configured shared backoff.
    The coordinator applies the cooldown globally, so every other branch waiting
    for the same site resumes at the same time.
    """

    def __init__(self, message: str = "site rate limit", *, retry_after: float | None = None) -> None:
        self.retry_after = None if retry_after is None else float(retry_after)
        super().__init__(message)
