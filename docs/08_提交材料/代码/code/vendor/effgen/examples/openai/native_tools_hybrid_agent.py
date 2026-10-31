"""
Hybrid agent: OpenAI native tools + effGen local tools.

Demonstrates a hard agentic task where the agent:
1. Uses web_search to get live data
2. Uses code_interpreter to process/analyze it
3. Uses effGen's Calculator for arithmetic verification
4. Produces a grounded, computed final answer

Run:
    python examples/openai/native_tools_hybrid_agent.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.openai_native import OpenAICodeInterpreterTool, OpenAIWebSearchTool

MODEL = "gpt-5.4-nano"


def demo_research_and_compute():
    """Agent that researches a topic and computes answers from it."""
    print("=" * 60)
    print("Demo 1: Research + Compute hybrid task")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    agent = Agent(AgentConfig(
        name="research-compute-agent",
        model=adapter,
        tools=[
            OpenAIWebSearchTool(search_context_size="low"),
            OpenAICodeInterpreterTool(),
            Calculator(),
        ],
        tool_calling_mode="auto",
        max_iterations=8,
        system_prompt=(
            "You are a research analyst who can search the web for information "
            "and perform precise computations. Use web search for current facts, "
            "code interpreter for data processing, and calculator for arithmetic. "
            "Always show your reasoning and cite sources."
        ),
    ))

    task = (
        "What is the current approximate market cap of Nvidia? "
        "If it were to grow by 15%, what would the new market cap be? "
        "Use web search for the current value, then compute the 15% growth."
    )
    print(f"\nTask: {task}")
    result = agent.run(task)
    print(f"\nAnswer: {result.output}")
    print(f"Iterations: {result.iterations} | Tool calls: {result.tool_calls}")
    print(f"Cost: ${adapter.get_total_cost():.6f}")

    agent.close()
    adapter.unload()


def demo_math_challenge():
    """Agent solves a multi-step math challenge using code + calculator."""
    print("\n" + "=" * 60)
    print("Demo 2: Hard multi-step math challenge")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    agent = Agent(AgentConfig(
        name="math-agent",
        model=adapter,
        tools=[
            OpenAICodeInterpreterTool(),
            Calculator(),
        ],
        tool_calling_mode="auto",
        max_iterations=10,
        system_prompt=(
            "You are a mathematician. Solve problems step by step. "
            "Use code interpreter for complex calculations and the calculator for simple arithmetic. "
            "Verify your answers."
        ),
    ))

    tasks = [
        (
            "Find all prime numbers between 1000 and 1100, then compute their sum. "
            "Use Python to find the primes."
        ),
        (
            "Compute the determinant of the 3×3 matrix: "
            "[[1,2,3],[4,5,6],[7,8,9]]. Then verify: what is 100 * 357?"
        ),
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        result = agent.run(task)
        print(f"Answer: {result.output}")
        print(f"Tool calls: {result.tool_calls}")

    agent.close()
    adapter.unload()


def demo_incompatibility_guard():
    """Show that ToolIncompatibleError fires cleanly at init."""
    print("\n" + "=" * 60)
    print("Demo 3: Incompatibility guard — non-OpenAI model")
    print("=" * 60)

    from effgen.models.errors import ToolIncompatibleError

    # Try to mix Cerebras with OpenAI native tools
    try:
        from effgen.models.cerebras_adapter import CerebrasAdapter
        cerebras = CerebrasAdapter(model_name="gpt-oss-120b")

        Agent(AgentConfig(
            name="bad-hybrid",
            model=cerebras,
            tools=[OpenAIWebSearchTool(), Calculator()],
        ))
        print("ERROR: Should have raised ToolIncompatibleError")
    except ToolIncompatibleError as e:
        print("Caught ToolIncompatibleError (expected):")
        print(f"  {e}")
        print("Guard works correctly — error at init, not at run time.")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")


def demo_agentic_coding_and_search():
    """Hard agentic task: write code, test it, and explain using web search."""
    print("\n" + "=" * 60)
    print("Demo 4: Hard agentic task — coding + web research")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    agent = Agent(AgentConfig(
        name="senior-engineer-agent",
        model=adapter,
        tools=[
            OpenAICodeInterpreterTool(),
            OpenAIWebSearchTool(search_context_size="low"),
        ],
        tool_calling_mode="auto",
        max_iterations=12,
        system_prompt=(
            "You are a senior software engineer. "
            "Write clean, tested Python code. "
            "Use web search to verify technical facts. "
            "Always test your code by running it."
        ),
    ))

    task = (
        "Implement a simple LRU (Least Recently Used) cache in Python "
        "with get() and put() operations. Test it with at least 5 test cases. "
        "Run the tests and confirm they all pass."
    )
    print(f"\nTask: {task}")
    result = agent.run(task)
    print(f"\nAnswer:\n{result.output[:800]}")
    print(f"\nIterations: {result.iterations} | Tool calls: {result.tool_calls}")
    print(f"Cost: ${adapter.get_total_cost():.6f}")

    agent.close()
    adapter.unload()


if __name__ == "__main__":
    demo_incompatibility_guard()
    demo_math_challenge()
    demo_research_and_compute()
    demo_agentic_coding_and_search()
    print("\nAll hybrid agent demos complete.")
