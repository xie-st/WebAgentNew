"""
Live integration tests for Gemini native tools.

Requires GOOGLE_API_KEY.  All tests are skipped automatically if the key is absent.
Run with a real Gemini model (gemini-3.1-flash-lite per project rules).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load from project .env first, then ~/.effgen/.env
load_dotenv(Path(__file__).parents[2] / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SKIP = pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="GOOGLE_API_KEY not available — skipping live Gemini native tools tests",
)

MODEL = "gemini-3.1-flash-lite"


def _skip_if_gemini_transient(value: object) -> None:
    msg = str(value)
    msg_lower = msg.lower()
    if (
        "503 unavailable" in msg_lower
        or "high demand" in msg_lower
        or "try again later" in msg_lower
        or "resource_exhausted" in msg_lower
        or "rate limit" in msg_lower
        or "timeout" in msg_lower
        or "timed out" in msg_lower
    ):
        pytest.skip(f"Gemini transient service pressure: {msg[:200]}")


def _run_agent_or_skip_transient(agent, prompt: str):
    try:
        return agent.run(prompt)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        _skip_if_gemini_transient(exc)
        raise


@SKIP
def test_google_search_tool_live():
    """GoogleSearchTool returns a non-empty answer about a current event."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.tools.builtin.gemini_native import GoogleSearchTool

    model = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    model.load()

    agent = Agent(AgentConfig(
        name="gemini-search-agent",
        model=model,
        tools=[GoogleSearchTool()],
    ))

    try:
        response = _run_agent_or_skip_transient(
            agent,
            "What is the latest major AI model released by Google in 2025?",
        )
        _skip_if_gemini_transient(response.output)
        assert response.output.strip(), "Expected non-empty answer"
        print("GoogleSearchTool response:", response.output[:300])
    finally:
        model.unload()


@SKIP
def test_url_context_tool_live():
    """GeminiUrlContextTool can summarize a URL."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.tools.builtin.gemini_native import GeminiUrlContextTool

    model = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    model.load()

    agent = Agent(AgentConfig(
        name="gemini-url-agent",
        model=model,
        tools=[GeminiUrlContextTool()],
    ))

    try:
        response = _run_agent_or_skip_transient(
            agent,
            "Please summarize what this page is about: "
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        )
        _skip_if_gemini_transient(response.output)
        assert response.output.strip(), "Expected non-empty summary"
        assert len(response.output) > 50, "Summary is too short"
        print("GeminiUrlContextTool response:", response.output[:300])
    finally:
        model.unload()


@SKIP
def test_code_execution_tool_live():
    """GeminiCodeExecutionTool can compute 100! exactly."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.tools.builtin.gemini_native import GeminiCodeExecutionTool

    model = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    model.load()

    agent = Agent(AgentConfig(
        name="gemini-code-agent",
        model=model,
        tools=[GeminiCodeExecutionTool()],
    ))

    try:
        response = _run_agent_or_skip_transient(
            agent,
            "Compute 100! (100 factorial) exactly.",
        )
        _skip_if_gemini_transient(response.output)
        FACTORIAL_100 = "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000"
        assert FACTORIAL_100 in response.output, (
            f"Expected 100! value in response, got: {response.output[:200]}"
        )
        print("GeminiCodeExecutionTool 100! response:", response.output[:200])
    finally:
        model.unload()


@SKIP
def test_parallel_function_calls_live():
    """Parallel function calling: Agent + Calculator on two expressions in one turn."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.tools.builtin.calculator import Calculator

    model = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    model.load()

    agent = Agent(AgentConfig(
        name="gemini-parallel-agent",
        model=model,
        tools=[Calculator()],
    ))

    # The question asks for two independent calculations — model should use two calls
    try:
        response = _run_agent_or_skip_transient(
            agent,
            "What is 2+3? Also, what is 7*8? Give me both answers.",
        )
        _skip_if_gemini_transient(response.output)
        assert response.output.strip(), "Expected non-empty answer"
        # Both results should appear somewhere in the final answer
        assert "5" in response.output, "Expected 2+3=5 in answer"
        assert "56" in response.output, "Expected 7*8=56 in answer"
        print("Parallel function calls response:", response.output[:300])
    finally:
        model.unload()


@SKIP
def test_general_preset_array_param_tools_accepted_live():
    """The general preset attaches tools whose parameters include a plain
    ``type: array`` field (map markers, email attachments). Gemini rejects an
    array property without ``items``; the schema sanitizer defaults it so the
    request is accepted and the preset runs instead of failing with a 400."""
    from effgen.presets import create_agent

    agent = create_agent("general", f"gemini:{MODEL}")
    response = _run_agent_or_skip_transient(agent, "Say hello in one word.")
    _skip_if_gemini_transient(response.output)
    # The prior failure surfaced as a 400 "properties[markers].items: missing
    # field" before generation — assert the request was accepted and answered.
    assert response.success, f"general preset failed on Gemini: {response.metadata.get('error')}"
    assert response.output.strip(), "Expected a non-empty answer"
    print("general preset on Gemini:", response.output[:120])
