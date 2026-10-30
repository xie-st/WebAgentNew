"""
OpenAI file_search native tool example.

Demonstrates using OpenAI's hosted vector-search (RAG) tool for
grounded question-answering over uploaded documents.

Run:
    python examples/openai/native_tools_file_search.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.openai_native import OpenAIFileSearchTool
from openai import OpenAI

MODEL = "gpt-5.4-nano"

# Sample documentation to upload
SAMPLE_DOC = """
# effGen Configuration Reference

## Environment Variables

OPENAI_API_KEY: Your OpenAI API key. Required for OpenAI models.
CEREBRAS_API_KEY: Your Cerebras Cloud API key. Required for Cerebras models.
ANTHROPIC_API_KEY: Your Anthropic API key. Required for Claude models.

## Supported Models

### OpenAI Models
- gpt-5.4-nano: Cheapest chat model, 1M token context, $0.20/1M input
- gpt-5.4-mini: Mid-tier chat model, 1M token context, $0.75/1M input
- gpt-5.4: High-performance chat, 1M token context, $2.50/1M input
- o4-mini: Reasoning model, 200K context, $1.10/1M input
- o3: High-capability reasoning, 200K context, $2.00/1M input

### Cerebras Models (Free Tier)
- gpt-oss-120b: Fast, lightweight model, 128K context
- zai-glm-4.7: Large MoE model, 128K context

## AgentConfig Options

name: Agent identifier (required)
model: Model instance or model name string
tools: List of BaseTool instances
system_prompt: System-level instructions
max_iterations: Maximum ReAct loop iterations (default: 10)
tool_calling_mode: "auto" | "native" | "react" | "hybrid"
temperature: Sampling temperature (default: 0.7)
enable_memory: Enable conversation memory (default: True)

## Native Tool Usage

OpenAI native tools require an OpenAI model and the Native strategy.
Use ToolIncompatibleError to detect misconfiguration at init time.

## Cost Tracking

Every API call logs token usage and cost at INFO level.
Access via: adapter.get_total_cost() and adapter.get_total_tokens()
"""


def setup_vector_store(client: OpenAI, content: str) -> tuple[str, str, str]:
    """Upload content and create a vector store. Returns (vs_id, file_id, tmp_path)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        tmp_path = f.name

    with open(tmp_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="assistants")
    file_id = file_obj.id

    vs = client.vector_stores.create(name="effgen-docs-example")
    vs_id = vs.id
    client.vector_stores.files.create(vector_store_id=vs_id, file_id=file_id)

    print(f"  Uploaded file: {file_id}")
    print(f"  Vector store: {vs_id}")
    print("  Waiting for indexing...", end="", flush=True)
    for _ in range(30):
        status = client.vector_stores.retrieve(vs_id)
        if status.file_counts.completed > 0:
            break
        time.sleep(1)
        print(".", end="", flush=True)
    print(" ready.")

    return vs_id, file_id, tmp_path


def cleanup(client: OpenAI, vs_id: str, file_id: str, tmp_path: str) -> None:
    """Clean up uploaded resources."""
    try:
        client.vector_stores.delete(vs_id)
        client.files.delete(file_id)
        os.unlink(tmp_path)
        print("  Cleaned up vector store and file.")
    except Exception as e:
        print(f"  Cleanup warning: {e}")


def demo_direct_file_search():
    """Query documentation using file_search via the Responses API."""
    print("=" * 60)
    print("Demo 1: Direct file_search over documentation")
    print("=" * 60)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vs_id, file_id, tmp_path = setup_vector_store(client, SAMPLE_DOC)

    try:
        tool = OpenAIFileSearchTool(
            vector_store_ids=[vs_id],
            max_num_results=5,
        )
        spec = tool.to_openai_tool_spec()

        questions = [
            "What is the cost per million input tokens for gpt-5.4-nano?",
            "Which Cerebras model is the largest?",
            "What environment variable is needed for Anthropic models?",
        ]

        for q in questions:
            print(f"\nQ: {q}")
            response = client.responses.create(
                model=MODEL,
                tools=[spec],
                input=q,
            )
            text = ""
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for block in getattr(item, "content", []):
                        if getattr(block, "type", None) in ("output_text", "text"):
                            text += getattr(block, "text", "")
            print(f"A: {text.strip()[:300]}")
    finally:
        cleanup(client, vs_id, file_id, tmp_path)


def demo_agent_with_file_search():
    """Agent that answers questions grounded in uploaded docs."""
    print("\n" + "=" * 60)
    print("Demo 2: Agent with file_search for grounded Q&A")
    print("=" * 60)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vs_id, file_id, tmp_path = setup_vector_store(client, SAMPLE_DOC)

    try:
        adapter = OpenAIAdapter(model_name=MODEL)
        adapter.load()

        tool = OpenAIFileSearchTool(vector_store_ids=[vs_id])

        agent = Agent(AgentConfig(
            name="doc-qa-agent",
            model=adapter,
            tools=[tool],
            tool_calling_mode="native",
            system_prompt=(
                "You are a documentation assistant. Answer questions using only "
                "the provided documentation. If the answer isn't in the docs, say so."
            ),
        ))

        queries = [
            "What tool_calling_mode options are available in AgentConfig?",
            "What is the default max_iterations for an agent?",
            "How do I access the total API cost after running the agent?",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            result = agent.run(query)
            print(f"Answer: {result.output}")

        agent.close()
        adapter.unload()
    finally:
        cleanup(client, vs_id, file_id, tmp_path)


def demo_dynamic_vector_store():
    """Dynamically add/remove vector stores from a file_search tool."""
    print("\n" + "=" * 60)
    print("Demo 3: Dynamic vector store management")
    print("=" * 60)

    tool = OpenAIFileSearchTool(max_num_results=10)
    print(f"Initial stores: {tool.vector_store_ids}")

    tool.add_vector_store("vs_store_a")
    tool.add_vector_store("vs_store_b")
    tool.add_vector_store("vs_store_a")  # duplicate — idempotent
    print(f"After adding: {tool.vector_store_ids}")
    assert tool.vector_store_ids.count("vs_store_a") == 1

    tool.remove_vector_store("vs_store_a")
    print(f"After removing vs_store_a: {tool.vector_store_ids}")
    assert "vs_store_a" not in tool.vector_store_ids
    assert "vs_store_b" in tool.vector_store_ids

    spec = tool.to_openai_tool_spec()
    print(f"Tool spec: {spec}")
    print("Dynamic vector store management PASS")


if __name__ == "__main__":
    demo_direct_file_search()
    demo_agent_with_file_search()
    demo_dynamic_vector_store()
    print("\nAll file_search demos complete.")
