"""
OpenAI code_interpreter native tool example.

Demonstrates using OpenAI's hosted Python sandbox for computation,
data analysis, and problem solving — no local execution required.

Run:
    python examples/openai/native_tools_code_interpreter.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.openai_native import OpenAICodeInterpreterTool

MODEL = "gpt-5.4-nano"


def demo_fibonacci():
    """Compute large Fibonacci numbers using code_interpreter."""
    print("=" * 60)
    print("Demo 1: Fibonacci computation via code_interpreter")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    tool = OpenAICodeInterpreterTool()
    result = adapter.generate_with_native_tools(
        prompt=(
            "Compute the 100th Fibonacci number (F(100), where F(0)=0 and F(1)=1). "
            "Write Python code, run it, and tell me the exact result."
        ),
        native_tool_specs=[tool.to_openai_tool_spec()],
    )
    print(f"Response:\n{result.text}")
    adapter.unload()


def demo_statistics():
    """Statistical analysis via code_interpreter."""
    print("\n" + "=" * 60)
    print("Demo 2: Statistical analysis")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    tool = OpenAICodeInterpreterTool()
    data = [23, 45, 12, 67, 34, 89, 56, 78, 90, 11, 44, 66]

    result = adapter.generate_with_native_tools(
        prompt=(
            f"Given this dataset: {data}\n"
            "Using Python, compute: mean, median, standard deviation, min, max. "
            "Show the code and formatted results."
        ),
        native_tool_specs=[tool.to_openai_tool_spec()],
    )
    print(f"Response:\n{result.text}")
    adapter.unload()


def demo_algorithm():
    """Implement and benchmark a sorting algorithm."""
    print("\n" + "=" * 60)
    print("Demo 3: Algorithm implementation")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    tool = OpenAICodeInterpreterTool()
    result = adapter.generate_with_native_tools(
        prompt=(
            "Implement quicksort in Python, sort a list of 1000 random integers, "
            "and verify it produces a correctly sorted result. "
            "Show: the implementation, a timing measurement, and confirmation the sort is correct."
        ),
        native_tool_specs=[tool.to_openai_tool_spec()],
    )
    print(f"Response:\n{result.text}")
    adapter.unload()


def demo_agent_coding_tasks():
    """Agent that uses code_interpreter for hard math/coding tasks."""
    print("\n" + "=" * 60)
    print("Demo 4: Agent with code_interpreter for agentic coding tasks")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    agent = Agent(AgentConfig(
        name="coding-agent",
        model=adapter,
        tools=[OpenAICodeInterpreterTool()],
        tool_calling_mode="native",
        system_prompt=(
            "You are an expert Python programmer and mathematician. "
            "Write clean, correct Python code and verify results by running them."
        ),
    ))

    tasks = [
        "What is the sum of all prime numbers below 1000?",
        "Implement binary search and find the index of 42 in a sorted list from 1 to 100.",
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        result = agent.run(task)
        print(f"Answer: {result.output}")
        print(f"Iterations: {result.iterations} | Tool calls: {result.tool_calls}")

    agent.close()
    adapter.unload()


if __name__ == "__main__":
    demo_fibonacci()
    demo_statistics()
    demo_algorithm()
    demo_agent_coding_tasks()
    print("\nAll code_interpreter demos complete.")
