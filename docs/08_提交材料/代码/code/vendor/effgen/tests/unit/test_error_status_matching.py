"""A status code is read from the status, not from digits elsewhere in a message.

An adapter decides whether a failure is an auth failure, a rate limit or a
not-found by the HTTP status. Testing for the bare digits anywhere in the
message text reads the port in a URL, a request id or a byte count as a status:
a 413 answered on port 40312 classified as a 403, so a request that was too
large was reported as a credential problem and never retried differently.
"""

from __future__ import annotations

import pytest

from effgen.models.errors import error_has_status


class _Plain(Exception):
    pass


class _WithStatus(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status_code = status


# The message shape an httpx-backed SDK raises, with an ephemeral port that
# happens to contain the digits of an auth status.
_413_ON_PORT_40312 = (
    "Client error '413 Payload Too Large' for url "
    "'http://127.0.0.1:40312/inference/v1/chat/completions'\n"
    "For more information check: "
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/413"
)


@pytest.mark.parametrize("code", [401, 403, 404, 429])
def test_digits_inside_a_port_are_not_a_status(code) -> None:
    message = (
        f"Client error '413 Payload Too Large' for url "
        f"'http://127.0.0.1:{code}12/v1/chat/completions'"
    )
    assert not error_has_status(_Plain(message), code)


def test_the_real_status_is_still_found() -> None:
    assert error_has_status(_Plain(_413_ON_PORT_40312), 413)
    assert not error_has_status(_Plain(_413_ON_PORT_40312), 403)
    assert not error_has_status(_Plain(_413_ON_PORT_40312), 401)


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("Client error '401 Unauthorized' for url 'http://x/y'", 401),
        ("Error code: 429 - rate limit reached", 429),
        ("403 Forbidden", 403),
        ("HTTP 404: model not found", 404),
    ],
)
def test_a_status_in_the_message_is_found(message, code) -> None:
    assert error_has_status(_Plain(message), code)


def test_a_recorded_status_wins_over_the_message() -> None:
    exc = _WithStatus("mentions 403 and 401 in passing", 413)
    assert error_has_status(exc, 413)
    assert not error_has_status(exc, 403)
    assert not error_has_status(exc, 401)


def test_a_non_numeric_code_attribute_does_not_decide() -> None:
    """OpenAI puts a string tag on ``.code``; it must not be read as a status."""

    class _Tagged(Exception):
        code = "invalid_api_key"

    assert error_has_status(_Tagged("Client error '401 Unauthorized'"), 401)
    assert not error_has_status(_Tagged("Client error '401 Unauthorized'"), 429)


def test_a_response_status_is_used_when_present() -> None:
    class _Response:
        status_code = 429

    class _Exc(Exception):
        response = _Response()

    assert error_has_status(_Exc("no digits here"), 429)
    assert not error_has_status(_Exc("no digits here"), 401)
