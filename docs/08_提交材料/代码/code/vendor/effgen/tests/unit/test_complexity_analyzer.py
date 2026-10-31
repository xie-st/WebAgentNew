"""What the complexity analyzer reads out of a task.

The tool indicators used to be matched as substrings, so ordinary prose was
scored as needing tools: "api" was found inside "capital" and "graph" inside
"paragraph". The identified tools feed sub-agent routing, so the effect was a
routing score too high for plain text.
"""

from __future__ import annotations

import pytest

from effgen.core.complexity_analyzer import ComplexityAnalyzer


@pytest.fixture
def analyzer() -> ComplexityAnalyzer:
    return ComplexityAnalyzer()


def tools_for(analyzer: ComplexityAnalyzer, task: str) -> list[str]:
    return analyzer.analyze(task).breakdown["tools_needed"]


class TestIndicatorsMatchWholeWords:
    """A word inside a longer word is not a mention of it."""

    @pytest.mark.parametrize(
        "task, must_not_contain",
        [
            ("What is the capital of France?", "api"),
            ("Summarize this paragraph", "image"),
            ("Rewrite the paragraph below", "image"),
        ],
    )
    def test_a_substring_is_not_a_mention(self, analyzer, task, must_not_contain):
        assert must_not_contain not in tools_for(analyzer, task)

    def test_prose_needing_nothing_scores_no_tool_requirement(self, analyzer):
        score = analyzer.analyze("Summarize this paragraph")
        assert score.tool_requirements == 0.0


class TestRealMentionsStillCount:
    """The fix must not cost the analyzer its actual job."""

    def test_a_named_tool_is_still_identified(self, analyzer):
        assert "api" in tools_for(analyzer, "Call the API for me")

    def test_several_tools_are_identified_together(self, analyzer):
        tools = tools_for(analyzer, "Call the API and search the web")
        assert {"api", "web_search"} <= set(tools)

    def test_a_multi_word_indicator_still_matches(self, analyzer):
        """Word boundaries sit at the ends of the phrase, not inside it."""
        assert tools_for(analyzer, "Please read the file report.txt")

    def test_more_tools_score_higher_than_fewer(self, analyzer):
        one = analyzer.analyze("Call the API for me").tool_requirements
        many = analyzer.analyze(
            "Call the API, search the web and read the file"
        ).tool_requirements
        assert many > one
