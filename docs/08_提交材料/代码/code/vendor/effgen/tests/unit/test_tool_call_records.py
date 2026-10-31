"""``AgentResponse.tool_calls`` reports which calls a run made, not just how many.

The field carries a record per call — tool, arguments, result, duration, error —
while still comparing and casting as the count, so code written against the
pre-1.0 integer keeps working.
"""

from __future__ import annotations

import pytest

from effgen.core.agent_response import AgentResponse
from effgen.core.agent_tool_loop import NativeToolLoop
from effgen.core.tool_call_record import (
    MAX_RESULT_CHARS,
    ToolCall,
    ToolCallList,
    coerce_tool_calls,
    truncate_result,
)


def _calls() -> ToolCallList:
    return ToolCallList([
        ToolCall("calculator", "6*7", "42", duration=0.01, iteration=1),
        ToolCall("web_search", {"q": "effgen"}, error="Error executing tool", iteration=2),
    ])


class TestTheCallsThemselves:
    """What the report says about each call."""

    def test_a_run_s_calls_can_be_iterated(self):
        assert [call.name for call in _calls()] == ["calculator", "web_search"]

    def test_iterating_a_count_only_report_does_not_raise(self):
        # The pre-1.0 int raised TypeError here, which is what sent callers
        # looking for the detail in the first place.
        assert list(coerce_tool_calls(3)) == []

    def test_each_record_carries_what_was_passed_and_returned(self):
        first = _calls()[0]
        assert first.name == "calculator"
        assert first.arguments == "6*7"
        assert first.result == "42"
        assert first.duration == pytest.approx(0.01)

    def test_a_failed_call_is_recorded_with_its_error(self):
        failed = _calls()[1]
        assert failed.ok is False
        assert failed.error == "Error executing tool"

    def test_a_successful_call_reports_no_error(self):
        assert _calls()[0].ok is True

    def test_the_failed_calls_can_be_asked_for_on_their_own(self):
        assert _calls().failed.names == ["web_search"]

    def test_the_calls_to_one_tool_can_be_asked_for(self):
        assert _calls().by_name("calculator").names == ["calculator"]

    def test_a_long_result_is_kept_short_enough_to_read_back(self):
        kept = truncate_result("x" * (MAX_RESULT_CHARS + 500))
        assert len(kept) < MAX_RESULT_CHARS + 100
        assert "chars)" in kept

    def test_a_record_survives_a_round_trip_through_plain_data(self):
        original = _calls()[0]
        assert ToolCall.from_dict(original.to_dict()) == original


class TestTheCountContract:
    """The number-like behaviour the field had before it carried records."""

    def test_it_compares_equal_to_the_number_of_calls(self):
        assert _calls() == 2

    def test_it_compares_greater_than_zero_when_a_tool_ran(self):
        assert _calls() > 0
        assert _calls() >= 2

    def test_no_calls_compares_equal_to_zero_and_is_falsey(self):
        assert ToolCallList() == 0
        assert not ToolCallList()

    def test_it_casts_to_the_count(self):
        assert int(_calls()) == 2

    def test_counts_from_several_runs_still_sum(self):
        assert sum([ToolCallList(total=2), ToolCallList(total=3)]) == 5

    def test_a_path_reporting_only_a_count_still_reports_it(self):
        counted = coerce_tool_calls(3)
        assert counted == 3
        assert counted.total == 3

    def test_the_count_matches_the_records_when_they_were_captured(self):
        assert _calls().total == len(_calls())


class TestOnTheResponse:
    """How a response presents the field."""

    def test_a_count_passed_by_an_older_path_is_accepted(self):
        assert AgentResponse(output="x", tool_calls=2).tool_calls == 2

    def test_records_passed_by_the_loop_are_kept(self):
        response = AgentResponse(output="x", tool_calls=_calls())
        assert response.tool_calls.names == ["calculator", "web_search"]

    def test_the_default_is_no_calls(self):
        assert AgentResponse(output="x").tool_calls == 0

    def test_the_count_is_also_available_under_a_plain_name(self):
        assert AgentResponse(output="x", tool_calls=_calls()).tool_call_count == 2

    def test_a_saved_run_keeps_the_count_under_its_original_key(self):
        document = AgentResponse(output="x", tool_calls=_calls()).to_dict()
        assert document["tool_calls"] == 2

    def test_a_saved_run_carries_the_calls_alongside_the_count(self):
        document = AgentResponse(output="x", tool_calls=_calls()).to_dict()
        assert [c["name"] for c in document["tool_call_details"]] == [
            "calculator", "web_search",
        ]

    def test_a_saved_run_reads_back_into_records(self):
        document = AgentResponse(output="x", tool_calls=_calls()).to_dict()
        restored = AgentResponse(output="x", tool_calls=document["tool_call_details"])
        assert restored.tool_calls.names == ["calculator", "web_search"]


class TestTheLoopRecordsWhatItDispatched:
    """The loop policy both the blocking and streaming loops share."""

    def test_a_dispatched_call_becomes_a_record(self):
        loop = NativeToolLoop(tools={})
        loop.record_execution(
            "calculator", arguments="6*7", result="42", duration=0.5, iteration=1,
        )
        assert loop.calls[0].name == "calculator"
        assert loop.calls[0].result == "42"
        assert loop.calls[0].duration == 0.5

    def test_a_failed_dispatch_is_recorded_as_a_call_that_errored(self):
        loop = NativeToolLoop(tools={})
        loop.record_execution("web_search", result="Error executing tool 'web_search'")
        assert loop.calls[0].ok is False

    def test_the_tool_is_still_marked_as_having_run(self):
        loop = NativeToolLoop(tools={})
        loop.record_execution("calculator")
        assert loop.tool_ran("calculator")


class TestThroughARealRun:
    """The records a tool-using run actually produces."""

    def test_a_run_that_used_a_tool_reports_which_tool_and_with_what(self):
        from effgen import Agent, AgentConfig
        from effgen.tools import get_registry
        from tests.fixtures.mock_models import MockModel

        model = MockModel([
            "Thought: I should compute this.\nAction: calculator\nAction Input: 6*7",
            "Thought: I have the result.\nFinal Answer: 42",
        ])
        calculator = get_registry().get_tool_sync("calculator")
        agent = Agent(AgentConfig(
            model=model, tools=[calculator], max_iterations=3, raise_on_error=False,
        ))
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()

        assert response.tool_calls > 0, "the run reported no tool call"
        assert "calculator" in response.tool_calls.names
        first = response.tool_calls[0]
        assert first.arguments is not None, "the arguments were not recorded"
        assert first.result is not None, "the result was not recorded"
        assert first.duration is not None, "the duration was not recorded"
        assert first.iteration == 1

    def test_the_recorded_result_is_what_the_tool_returned(self):
        from effgen import Agent, AgentConfig
        from effgen.tools import get_registry
        from tests.fixtures.mock_models import MockModel

        model = MockModel([
            "Thought: compute.\nAction: calculator\nAction Input: 6*7",
            "Thought: done.\nFinal Answer: 42",
        ])
        agent = Agent(AgentConfig(
            model=model, tools=[get_registry().get_tool_sync("calculator")],
            max_iterations=3, raise_on_error=False,
        ))
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert "42" in (response.tool_calls[0].result or "")
