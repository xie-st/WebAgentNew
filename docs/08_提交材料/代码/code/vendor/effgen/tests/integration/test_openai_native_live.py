"""
Integration tests for OpenAI native tools.

These tests make real API calls and are skipped if OPENAI_API_KEY is absent.
Run with: pytest tests/integration/test_openai_native_live.py -v
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set — skipping live OpenAI native tool tests",
)

MODEL = "gpt-5.4-nano"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def openai_adapter():
    from effgen.models.openai_adapter import OpenAIAdapter
    adapter = OpenAIAdapter(model_name=MODEL, api_key=OPENAI_API_KEY)
    adapter.load()
    yield adapter
    adapter.unload()


@pytest.fixture(scope="module")
def openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Tool spec round-trip tests (no API call)
# ---------------------------------------------------------------------------

def test_web_search_spec_round_trip():
    from effgen.tools.builtin.openai_native import OpenAIWebSearchTool
    tool = OpenAIWebSearchTool(search_context_size="low")
    spec = tool.to_openai_tool_spec()
    assert spec["type"] == "web_search_preview"
    assert spec["search_context_size"] == "low"


def test_code_interpreter_spec_round_trip():
    from effgen.tools.builtin.openai_native import OpenAICodeInterpreterTool
    tool = OpenAICodeInterpreterTool()
    spec = tool.to_openai_tool_spec()
    assert spec["type"] == "code_interpreter"
    assert "container" in spec


def test_file_search_spec_round_trip():
    from effgen.tools.builtin.openai_native import OpenAIFileSearchTool
    tool = OpenAIFileSearchTool(vector_store_ids=["vs_test"], max_num_results=5)
    spec = tool.to_openai_tool_spec()
    assert spec["type"] == "file_search"
    assert spec["vector_store_ids"] == ["vs_test"]
    assert spec["max_num_results"] == 5


# ---------------------------------------------------------------------------
# Live API tests
# ---------------------------------------------------------------------------

def test_live_web_search(openai_client):
    from effgen.tools.builtin.openai_native import OpenAIWebSearchTool
    tool = OpenAIWebSearchTool()
    spec = tool.to_openai_tool_spec()

    response = openai_client.responses.create(
        model=MODEL,
        tools=[spec],
        input="What is the capital of France? (one word answer)",
    )

    text = ""
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", []):
                if getattr(block, "type", None) in ("output_text", "text"):
                    text += getattr(block, "text", "")

    assert "paris" in text.lower() or "Paris" in text, f"Expected Paris in: {text}"


def test_live_code_interpreter(openai_client):
    from effgen.tools.builtin.openai_native import OpenAICodeInterpreterTool
    tool = OpenAICodeInterpreterTool()
    spec = tool.to_openai_tool_spec()

    response = openai_client.responses.create(
        model=MODEL,
        tools=[spec],
        input="Compute 2 ** 20 using Python and tell me the result.",
    )

    text = ""
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", []):
                if getattr(block, "type", None) in ("output_text", "text"):
                    text += getattr(block, "text", "")

    normalized = text.replace("{", "").replace("}", "").replace(",", "").replace(" ", "")
    assert "1048576" in normalized, f"Expected 2**20=1048576 in: {text}"


@pytest.mark.timeout(180)
def test_live_file_search(openai_client):
    from effgen.tools.builtin.openai_native import OpenAIFileSearchTool

    content = "The effGen mascot is a purple fox named AURORA.\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    file_id = None
    vs_id = None
    try:
        # A 5xx here is upstream vector-store unavailability, not an effGen
        # bug — skip rather than fail (mirrors the Gemini/Cerebras live tests).
        from openai import InternalServerError

        try:
            with open(tmp_path, "rb") as f:
                uploaded = openai_client.files.create(file=f, purpose="assistants")
            file_id = uploaded.id

            vs = openai_client.vector_stores.create(name="effgen-integration-test")
            vs_id = vs.id
            openai_client.vector_stores.files.create(vector_store_id=vs_id, file_id=file_id)
        except InternalServerError as exc:
            pytest.skip(f"OpenAI vector-store backend unavailable (5xx) — {exc}")

        for _ in range(60):
            status = openai_client.vector_stores.retrieve(vs_id)
            if status.file_counts.completed > 0:
                break
            time.sleep(1)
        else:
            pytest.skip("Vector store indexing did not complete within 60s")

        tool = OpenAIFileSearchTool(vector_store_ids=[vs_id])
        spec = tool.to_openai_tool_spec()

        response = openai_client.responses.create(
            model=MODEL,
            tools=[spec],
            input=(
                "Use the file_search tool to find the answer in the attached document. "
                "What is the name of the effGen mascot?"
            ),
        )

        text = ""
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for block in getattr(item, "content", []):
                    if getattr(block, "type", None) in ("output_text", "text"):
                        text += getattr(block, "text", "")

        assert "AURORA" in text or "aurora" in text.lower(), f"Expected AURORA in: {text}"
    finally:
        try:
            if vs_id:
                openai_client.vector_stores.delete(vs_id)
            if file_id:
                openai_client.files.delete(file_id)
        except Exception:
            pass
        os.unlink(tmp_path)


def test_tool_incompatible_error_with_cerebras_model():
    """ToolIncompatibleError fires at Agent init, not at run time."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.cerebras_adapter import CerebrasAdapter
    from effgen.models.errors import ToolIncompatibleError
    from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

    # Create a Cerebras adapter (no real connection needed just for init check)
    adapter = CerebrasAdapter(model_name="gpt-oss-120b")

    with pytest.raises(ToolIncompatibleError) as exc_info:
        Agent(AgentConfig(
            name="bad-agent",
            model=adapter,
            tools=[OpenAIWebSearchTool()],
        ))

    err = exc_info.value
    assert err.tool_name == "openai_web_search"
    assert "gpt-oss-120b" in err.model_name


def test_live_agent_with_web_search_tool():
    """End-to-end: Agent + OpenAI model + OpenAIWebSearchTool."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.openai_adapter import OpenAIAdapter
    from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

    adapter = OpenAIAdapter(model_name=MODEL, api_key=OPENAI_API_KEY)
    adapter.load()

    try:
        agent = Agent(AgentConfig(
            name="web-search-agent",
            model=adapter,
            tools=[OpenAIWebSearchTool()],
            tool_calling_mode="native",
        ))
        result = agent.run("What is 2+2? No web search needed — answer directly.")
        assert result.success
        assert "4" in result.output
        agent.close()
    finally:
        adapter.unload()


def test_live_agent_native_plus_effgen_tools():
    """Agent with both OpenAI native tools AND effGen local tools."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.openai_adapter import OpenAIAdapter
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

    adapter = OpenAIAdapter(model_name=MODEL, api_key=OPENAI_API_KEY)
    adapter.load()

    try:
        agent = Agent(AgentConfig(
            name="hybrid-agent",
            model=adapter,
            tools=[OpenAIWebSearchTool(), Calculator()],
            tool_calling_mode="auto",
        ))
        result = agent.run("What is 15 * 7? Use the calculator tool.")
        assert result.success
        assert "105" in result.output
        agent.close()
    finally:
        adapter.unload()
