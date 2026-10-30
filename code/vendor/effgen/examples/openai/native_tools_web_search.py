"""
OpenAI web_search native tool example.

Demonstrates using OpenAI's hosted web search tool for real-time,
cited information retrieval — no API key for a search engine needed.

Cost note: web_search_preview adds ~$30/1k calls on top of token costs.

Run:
    python examples/openai/native_tools_web_search.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

MODEL = "gpt-5.4-nano"


def demo_direct_api():
    """Call the Responses API directly with web_search."""
    print("=" * 60)
    print("Demo 1: Direct Responses API with web_search")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    tool = OpenAIWebSearchTool(search_context_size="medium")
    spec = tool.to_openai_tool_spec()
    print(f"Tool spec: {spec}")

    response = client.responses.create(
        model=MODEL,
        tools=[spec],
        input="What are the top AI announcements from the past 7 days? Summarize in 3 bullet points.",
    )

    for item in response.output:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", []):
                if getattr(block, "type", None) in ("output_text", "text"):
                    print("\nResponse:")
                    print(getattr(block, "text", ""))


def demo_effgen_adapter():
    """Use OpenAIAdapter.generate_with_native_tools."""
    print("\n" + "=" * 60)
    print("Demo 2: OpenAIAdapter with web_search native tool")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    tool = OpenAIWebSearchTool(search_context_size="low")
    spec = tool.to_openai_tool_spec()

    result = adapter.generate_with_native_tools(
        prompt="What is the latest version of Python released? Just the version number.",
        native_tool_specs=[spec],
    )
    print(f"Response: {result.text}")
    print(f"Cost: ${result.metadata['cost']:.6f}")
    adapter.unload()


def demo_agent_web_search():
    """Full Agent with web_search — asking research questions."""
    print("\n" + "=" * 60)
    print("Demo 3: Agent with web_search for research tasks")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    agent = Agent(AgentConfig(
        name="research-agent",
        model=adapter,
        tools=[OpenAIWebSearchTool()],
        tool_calling_mode="native",
        system_prompt=(
            "You are a research assistant. Use web search to find current, "
            "accurate information. Always cite your sources."
        ),
    ))

    questions = [
        "What did Anthropic announce recently about Claude?",
        "What is the current price of gold per ounce?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        result = agent.run(q)
        print(f"Answer: {result.output[:500]}")
        print(f"Iterations: {result.iterations} | Tool calls: {result.tool_calls}")

    agent.close()
    adapter.unload()


def demo_geo_targeted_search():
    """web_search with user_location for geo-relevant results."""
    print("\n" + "=" * 60)
    print("Demo 4: Geo-targeted web search")
    print("=" * 60)

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    tool = OpenAIWebSearchTool(
        search_context_size="medium",
        user_location={
            "type": "approximate",
            "country": "US",
            "city": "San Francisco",
        },
    )

    result = adapter.generate_with_native_tools(
        prompt="What is the weather like in San Francisco this week?",
        native_tool_specs=[tool.to_openai_tool_spec()],
    )
    print(f"Response: {result.text[:400]}")
    adapter.unload()


if __name__ == "__main__":
    demo_direct_api()
    demo_effgen_adapter()
    demo_agent_web_search()
    demo_geo_targeted_search()
    print("\nAll web_search demos complete.")
