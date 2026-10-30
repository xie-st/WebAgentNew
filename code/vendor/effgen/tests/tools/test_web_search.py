"""Tests for the WebSearch tool's backend options.

The DuckDuckGo backend reads ``region`` and ``timelimit`` as strings. Passing
an explicit ``None`` makes it fail while preparing the filter, and the search
then reports an attribute error on ``None`` instead of the network or service
problem that actually stopped it — so the tool sends only the filters that
have a value.
"""

from __future__ import annotations

from effgen.tools.builtin.web_search import _duckduckgo_options


def test_no_option_is_sent_as_none():
    options = _duckduckgo_options(5, None, "en", None)
    assert options == {"max_results": 5}
    assert None not in options.values()


def test_unset_filters_are_omitted_not_nulled():
    for time_range, region in ((None, None), ("", None), (None, ""), ("", "")):
        options = _duckduckgo_options(3, time_range, "en", region)
        assert "region" not in options
        assert "timelimit" not in options


def test_time_range_maps_to_the_backend_code():
    assert _duckduckgo_options(5, "day", "en", None)["timelimit"] == "d"
    assert _duckduckgo_options(5, "week", "en", None)["timelimit"] == "w"
    assert _duckduckgo_options(5, "month", "en", None)["timelimit"] == "m"
    assert _duckduckgo_options(5, "year", "en", None)["timelimit"] == "y"


def test_an_unknown_time_range_is_dropped_rather_than_sent_as_none():
    assert "timelimit" not in _duckduckgo_options(5, "fortnight", "en", None)


def test_region_is_combined_with_the_language():
    assert _duckduckgo_options(5, None, "en", "uk")["region"] == "uk-en"


def test_result_count_is_always_passed_through():
    assert _duckduckgo_options(42, "day", "de", "de")["max_results"] == 42
