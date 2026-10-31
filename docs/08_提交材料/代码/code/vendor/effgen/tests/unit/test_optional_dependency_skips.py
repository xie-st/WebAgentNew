"""An absent optional extra is a skip; anything else is still a failure.

The rule and the reasoning live in ``tests/_harness/optional_deps.py``; the hook
that applies it is in ``tests/conftest.py``. These tests hold both to their
narrow contract, because a hook that turns failures into skips would hide real
breakage if it were even slightly too broad.
"""
from __future__ import annotations

import pytest

from tests._harness.optional_deps import (
    absent_optional_dependency,
    optional_dependency_names,
)


def test_the_set_is_read_from_the_project_not_hand_written():
    names = optional_dependency_names()
    # Packages that really are optional extras.
    for package in ("together", "replicate", "pypdf", "feedparser", "staticmap"):
        assert package in names, package
    # Core requirements and the test tooling never qualify: their absence is a
    # real problem and must keep failing.
    for package in ("pytest", "httpx", "pydantic", "effgen"):
        assert package not in names, package


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ModuleNotFoundError: No module named 'together'", "together"),
        ("ModuleNotFoundError: No module named 'yt_dlp'", "yt_dlp"),
        ("AssertionError: tool failed: feedparser is required: pip install feedparser",
         "feedparser"),
        ("staticmap is not installed. Install it with: pip install staticmap",
         "staticmap"),
    ],
)
def test_the_two_shapes_that_mean_not_installed_are_recognised(text, expected):
    assert absent_optional_dependency(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "assert 1 == 2",
        "ModuleNotFoundError: No module named 'effgen.core.base'",
        "ModuleNotFoundError: No module named 'pydantic'",
        "AssertionError: the answer is required",
        "",
    ],
)
def test_an_ordinary_failure_is_left_alone(text):
    assert absent_optional_dependency(text) is None


class TestAConvertedSkipDoesNotCrashTheReporter:
    """The conversion must not leave a ``wasxfail`` attribute behind.

    pytest decides a report is an xfail with ``hasattr(report, "wasxfail")`` and
    then calls ``.startswith()`` on the value. A ``None`` left there raises
    inside the terminal reporter — an INTERNALERROR that ends the whole session
    with exit code 3 and discards every passing result with it. It fires only on
    the verbose reporting path, so a quiet local run stays green while CI dies,
    which is exactly how it reached CI.
    """

    def test_the_conversion_removes_the_attribute(self):
        """Drive the hook itself and inspect the report it produces."""
        import tests.conftest as root_conftest

        class _Report:
            when = "call"
            failed = True
            outcome = "failed"
            longrepr = "ModuleNotFoundError: No module named 'vllm'"

        class _Outcome:
            def __init__(self, report):
                self._report = report

            def get_result(self):
                return self._report

        report = _Report()
        # pytest sets this on the instance; a previous xfail marking is exactly
        # what the conversion has to clear.
        report.wasxfail = "reason: something earlier"
        wrapper = root_conftest.pytest_runtest_makereport(
            item=type("I", (), {"fspath": "f.py", "location": ("f.py", 3, "t")})(),
            call=None,
        )
        next(wrapper)                       # run up to the yield
        try:
            wrapper.send(_Outcome(report))  # hand it the outcome
        except StopIteration:
            pass

        assert report.outcome == "skipped"
        # The attribute must be GONE, not None: pytest tests for it with
        # hasattr and then calls .startswith on the value.
        assert not hasattr(report, "wasxfail")

    def test_a_report_with_wasxfail_set_to_none_is_what_pytest_chokes_on(self):
        """The mechanism itself, without a subprocess."""
        from _pytest.terminal import _get_raw_skip_reason

        class _Report:
            skipped = True
            longrepr = ("f.py", 1, "Skipped: because")

        report = _Report()
        # With the attribute absent the reason comes from longrepr.
        assert _get_raw_skip_reason(report) == "because"
        # Present-but-None is the shape that raises.
        report.wasxfail = None
        with pytest.raises(AttributeError):
            _get_raw_skip_reason(report)
