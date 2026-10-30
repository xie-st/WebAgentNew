"""Unit tests for the continuation line that closes a native/hybrid tool prompt.

The native/hybrid prompt ends with the last tool's observation, so the closing
line is the final thing a model reads before it answers. After a retrieval or
search tool that observation is a block of source passages; the closing line
restates the answer contract there so the passages are not returned as the
answer. Every other tool keeps the original wording.

The line says what the passages are and what not to do with them, and nothing
about the form of the answer: it is the last thing the model reads, so a form
demand there overrules the task, which is the one statement of shape that knows
the question. A schema the caller declared is stated in the same closing
position, ahead of it.

Inline ``[1]`` markers are part of that line only when the caller asked for
them: a marker the caller did not request ends up inside the answer text.
"""

from __future__ import annotations

import pytest

from effgen.core.agent_react import AgentReActMixin
from effgen.tools.base_tool import ToolCategory

GENERIC = "Continue solving the task. If you have the final answer, state it clearly."


class _StubTool:
    def __init__(self, category=None):
        self.metadata = type("Meta", (), {"category": category})()


class _StubAgent(AgentReActMixin):
    """Minimal carrier for the tool registry the instruction selector reads."""

    def __init__(self, tools=None):
        self.tools = tools or {}


def _instruction(agent, *actions, **kwargs):
    return agent._continuation_instruction([(a, "{}") for a in actions], **kwargs)


class TestContinuationInstruction:
    def test_no_tool_has_run_yet_keeps_generic_line(self):
        assert _StubAgent()._continuation_instruction([]) == GENERIC

    @pytest.mark.parametrize("action", ["calculator", "python_repl", "bash", "unknown_tool"])
    def test_non_retrieval_tool_keeps_generic_line(self, action):
        assert _instruction(_StubAgent(), action) == GENERIC

    @pytest.mark.parametrize(
        "action", ["retrieval", "web_search", "search", "knowledge_base"]
    )
    def test_retrieval_tool_restates_the_answer_contract(self, action):
        text = _instruction(_StubAgent(), action)
        assert text != GENERIC
        low = text.lower()
        # Names the passages as source material and forbids returning one as
        # the answer -- the two things only the framework knows.
        assert "source material" in low
        assert "do not return a passage as the answer" in low
        # Keeps the fail-closed instruction for an unanswerable question.
        assert "do not answer it" in low or "not answer it" in low or "missing" in low

    @pytest.mark.parametrize(
        "action", ["retrieval", "web_search", "search", "knowledge_base"]
    )
    def test_retrieval_close_states_no_form_of_its_own(self, action):
        """The close describes the passages, never the shape of the answer.

        It is the last thing the model reads, so a form demand here overrules
        a task that already asked for a letter, a number or a value -- and
        buys several times the output tokens for an answer already stated.
        """
        low = _instruction(_StubAgent(), action).lower()
        for demand in (
            "your own sentences",
            "in your own words",
            "do not copy",
            "covering only what the question asks",
            "be concise",
            "as few words",
        ):
            assert demand not in low, demand
        # It defers instead: the question decides the form.
        assert "in the form the question asks for" in low

    @pytest.mark.parametrize(
        "action", ["retrieval", "web_search", "search", "knowledge_base"]
    )
    def test_inline_markers_are_not_asked_for_by_default(self, action):
        """A caller who said nothing about citations is not asked for markers."""
        text = _instruction(_StubAgent(), action)
        assert "[1]" not in text
        assert "cite each passage" not in text.lower()

    @pytest.mark.parametrize(
        "action", ["retrieval", "web_search", "search", "knowledge_base"]
    )
    def test_inline_markers_are_asked_for_when_requested(self, action):
        text = _instruction(
            _StubAgent(), action, cite_sources=True, numbered_passages=3
        )
        assert "[1]" in text
        assert "cite each passage" in text.lower()
        # The request is added to the default line, not swapped for it.
        assert "source material" in text.lower()

    def test_markers_are_not_asked_for_with_nothing_numbered(self):
        """A retrieval tool that offered no numbered passage is not cited."""
        text = _instruction(
            _StubAgent(), "retrieval", cite_sources=True, numbered_passages=0
        )
        assert "[1]" not in text

    def test_category_marks_a_custom_tool_as_retrieval(self):
        agent = _StubAgent(
            {"house_docs": _StubTool(ToolCategory.INFORMATION_RETRIEVAL)}
        )
        assert _instruction(agent, "house_docs") != GENERIC

    def test_category_leaves_a_non_retrieval_custom_tool_alone(self):
        agent = _StubAgent({"invoice_math": _StubTool(ToolCategory.DATA_PROCESSING)})
        assert _instruction(agent, "invoice_math") == GENERIC

    def test_only_the_most_recent_tool_decides(self):
        agent = _StubAgent()
        # Retrieval ran earlier, but a calculator produced the latest observation.
        assert _instruction(agent, "retrieval", "calculator") == GENERIC
        # ... and the other way round.
        assert _instruction(agent, "calculator", "retrieval") != GENERIC

    def test_generic_line_is_unchanged_for_default_agents(self):
        """A tool-using agent that never retrieves sees the original prompt."""
        assert _instruction(_StubAgent(), "calculator") == GENERIC
