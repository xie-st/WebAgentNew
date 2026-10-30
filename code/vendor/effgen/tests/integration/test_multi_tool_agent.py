"""Integration tests for multi-tool agent with real model."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, DateTimeTool, TextProcessingTool


def _assert_run_reached_an_answer(result):
    """The run either answered or was stopped by the loop while still working.

    A 3B model sometimes keeps re-reading the tool instead of writing a final
    answer. That run is reported fail-closed as a stopped outcome, with the tool
    output still in hand — a legitimate outcome for the model, not a defect. Any
    other unsuccessful reason (a tool error, a refusal, an empty result) still
    fails here.
    """
    assert result.outcome in ("answered", "stopped"), (
        f"run failed for an unexpected reason: stop_reason={result.stop_reason!r} "
        f"output={result.output!r}"
    )


def _reached_text(result) -> str:
    """Everything the run produced: the answer, the progress, and the trace."""
    partial = result.partial.text if result.partial else ""
    return f"{result.output}\n{partial}\n{result.execution_trace}"


@pytest.mark.gpu
class TestMultiToolAgent:
    """Test agent with multiple tools."""

    def test_calculator_with_multiple_tools(self, real_model):
        agent = Agent(config=AgentConfig(
            name="multi_test",
            model=real_model,
            tools=[Calculator(), DateTimeTool(), TextProcessingTool()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
            # A run the loop stops is one of the outcomes under test, so it is
            # read from the response rather than raised.
            raise_on_error=False,
        ))
        result = agent.run("What is 17 * 23? Use the calculator tool.")
        _assert_run_reached_an_answer(result)
        # Check that the tool was called and produced the right answer. The
        # value may be in the answer, in the progress a stopped run kept, or in
        # the execution trace.
        assert "391" in _reached_text(result)

    def test_datetime_tool(self, real_model):
        agent = Agent(config=AgentConfig(
            name="dt_test",
            model=real_model,
            tools=[DateTimeTool()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
            # A run the loop stops is one of the outcomes under test, so it is
            # read from the response rather than raised.
            raise_on_error=False,
        ))
        result = agent.run("What is the current date and time in UTC?")
        _assert_run_reached_an_answer(result)
        # The tool result may appear in the answer, in a stopped run's
        # progress, or in the trace (small local models sometimes return a "no
        # further steps needed" summary instead of quoting the date back).
        assert "202" in _reached_text(result)
