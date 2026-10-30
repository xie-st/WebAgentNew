"""The shared self-containment check distinguishes text from fetched positions.

The surfaces that promise to open offline used to assert that "https://"
appeared nowhere in the output. A provider whose error text names its billing
console then turned a green anchor red on an ordinary rate limit — while the
page still fetched nothing.
"""

from __future__ import annotations

import pytest

from tests._harness.offline_assets import assert_self_contained, external_references


class TestTextIsNotAReference:
    def test_a_provider_error_quoting_a_url_is_not_a_fetch(self):
        page = (
            "<div>Rate limit reached. Retry today at "
            "https://console.groq.com/settings/billing</div>"
        )
        assert external_references(page) == []
        assert_self_contained(page)

    def test_an_escaped_url_in_text_is_not_a_fetch(self):
        page = "<div>see https://example.com/docs&#x27;, code: rate_limit</div>"
        assert external_references(page) == []

    def test_an_inlined_data_uri_is_not_a_fetch(self):
        assert external_references('<img src="data:image/png;base64,iVBOR">') == []


class TestFetchedPositionsAreCaught:
    @pytest.mark.parametrize(
        "page, expected",
        [
            ('<script src="https://cdn.example/x.js"></script>', "https://cdn.example/x.js"),
            ('<link href="https://fonts.example/f.css">', "https://fonts.example/f.css"),
            ('<style>body{background:url("https://i.example/b.png")}</style>',
             "https://i.example/b.png"),
            ('<style>@import "https://x.example/a.css";</style>', "https://x.example/a.css"),
            ('<script>fetch("https://api.example/d")</script>', "https://api.example/d"),
            ('<img srcset="https://i.example/2x.png 2x">', "https://i.example/2x.png"),
            ('<script>new WebSocket("wss://s.example/w")</script>', "wss://s.example/w"),
            ('<script>new EventSource("https://e.example/s")</script>', "https://e.example/s"),
        ],
    )
    def test_each_position_is_found(self, page, expected):
        assert external_references(page) == [expected]

    def test_the_failure_names_what_would_be_fetched(self):
        page = '<script src="https://cdn.example/x.js"></script>'
        with pytest.raises(AssertionError) as caught:
            assert_self_contained(page, "the report")
        assert "the report" in str(caught.value)
        assert "cdn.example" in str(caught.value)

    def test_duplicates_are_reported_once(self):
        page = '<img src="https://i.example/a.png"><img src="https://i.example/a.png">'
        assert external_references(page) == ["https://i.example/a.png"]
