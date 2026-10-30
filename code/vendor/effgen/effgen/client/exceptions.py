"""Typed exceptions for the effGen client SDK."""
from __future__ import annotations

from typing import Any

from effgen.errors import quote_for_message, with_next_step


class EffGenClientError(Exception):
    """Base class for all effGen client errors."""


class EffGenConnectionError(EffGenClientError):
    """Raised when the client cannot connect to the server."""


class EffGenTimeoutError(EffGenClientError):
    """Raised when a request times out."""


class EffGenAPIError(EffGenClientError):
    """Raised when the server returns a non-2xx response.

    The server's own message is redacted and bounded before it is quoted, then
    followed by what the caller can do about that status.

    Attributes:
        status_code: The HTTP status the server returned, when one was read.
        payload: The decoded response body, when the server sent one.
    """

    #: What to do about each status the client classifies.
    _FOLLOW_ON_BY_STATUS = {
        401: "Check the API key the client was built with.",
        403: "The key is valid but not permitted this action — check the "
             "principal's roles.",
        404: "Check the route and the model id in the request.",
        429: "Slow the request rate, or raise the server's limit for this "
             "principal.",
    }
    _DEFAULT_FOLLOW_ON = (
        "Check the server logs for the matching request, and retry if the "
        "status is a 5xx."
    )

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        payload: Any | None = None,
    ) -> None:
        follow_on = self._FOLLOW_ON_BY_STATUS.get(
            status_code or 0, self._DEFAULT_FOLLOW_ON
        )
        super().__init__(with_next_step(quote_for_message(message), follow_on))
        self.status_code = status_code
        self.payload = payload


class EffGenAuthError(EffGenAPIError):
    """Raised on 401/403 — authentication or authorization failed."""


class EffGenRateLimitError(EffGenAPIError):
    """Raised on 429 — rate limit exceeded."""


class EffGenServerError(EffGenAPIError):
    """Raised on 5xx — server-side error."""
