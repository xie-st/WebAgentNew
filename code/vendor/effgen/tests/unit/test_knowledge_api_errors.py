"""A rate-limited knowledge API says so, rather than reporting HTTP 400.

The Stack Exchange API answers a caller over its anonymous quota with HTTP 400
and the reason in the body — ``throttle_violation`` and how long to wait. GitHub
does the same with a ``message``. Reading only the status leaves the caller with
"HTTP 400 Bad Request" for what is a wait-and-retry condition, indistinguishable
from a query the API rejected.

The bodies below are the shapes those APIs document and return.
"""

from __future__ import annotations

import pytest

from effgen.tools.builtin.knowledge import _api_error_detail


class TestTheReasonIsReadOffTheBody:
    def test_a_stack_exchange_throttle_names_itself(self):
        body = (
            b'{"error_id":502,"error_message":"Violation of backoff parameter",'
            b'"error_name":"throttle_violation"}'
        )
        detail = _api_error_detail(body)
        assert "throttle_violation" in detail
        assert "Violation of backoff parameter" in detail

    def test_a_backoff_says_how_long_to_wait(self):
        body = b'{"error_name":"throttle_violation","backoff":10}'
        assert "retry after 10s" in _api_error_detail(body)

    def test_a_github_rate_limit_carries_its_message(self):
        body = (
            b'{"message":"API rate limit exceeded for 203.0.113.7.",'
            b'"documentation_url":"https://docs.github.com/rest"}'
        )
        assert "API rate limit exceeded" in _api_error_detail(body)


class TestAnUnreadableBodyIsNotAnError:
    @pytest.mark.parametrize(
        "body",
        [b"", b"<html>502 Bad Gateway</html>", b"[1, 2, 3]", b"\xff\xfe not utf-8", b"null"],
        ids=["empty", "html", "json-array", "undecodable", "json-null"],
    )
    def test_it_reports_nothing_rather_than_raising(self, body):
        assert _api_error_detail(body) == ""

    def test_a_json_object_with_no_known_key_reports_nothing(self):
        assert _api_error_detail(b'{"quota_remaining": 0}') == ""
