"""Structural + behavioural tests for the ``agent.py`` decomposition.

``Agent`` was split into focused modules (``agent_runtime``, ``agent_generation``,
``agent_react``) mixed back into the public ``Agent`` facade. These tests pin the
decomposition so the split stays a *no behaviour change* refactor:

* the extracted modules import cleanly and expose their mixins,
* ``Agent`` still resolves every extracted method (instance + static surface
  that callers and the existing test-suite rely on),
* the public symbols remain importable from ``effgen.core.agent``,
* the pure helpers behave identically in isolation.
"""

from __future__ import annotations

import inspect

import pytest

from effgen.core import agent as agent_mod
from effgen.core import agent_generation, agent_react, agent_runtime
from effgen.core.agent import (
    Agent,
    AgentConfig,
    AgentMode,
    AgentResponse,
    sanitize_final_answer,
)


def test_extracted_modules_expose_mixins():
    assert inspect.isclass(agent_runtime.AgentRuntimeMixin)
    assert inspect.isclass(agent_generation.AgentGenerationMixin)
    assert inspect.isclass(agent_react.AgentReActMixin)


def test_agent_inherits_all_three_mixins():
    mro = Agent.__mro__
    assert agent_generation.AgentGenerationMixin in mro
    assert agent_react.AgentReActMixin in mro
    assert agent_runtime.AgentRuntimeMixin in mro


def test_public_surface_still_importable():
    # The four public names + the module-level sanitizer must remain importable
    # from effgen.core.agent exactly as before the split.
    assert Agent.__name__ == "Agent"
    assert issubclass(AgentConfig, object)
    assert AgentMode.SINGLE.value == "single"
    assert AgentResponse(output="hi").success is True
    assert callable(sanitize_final_answer)


@pytest.mark.parametrize(
    "method_name",
    [
        # runtime helpers
        "_clean_json_input",
        "_extract_task_preview",
        "_coerce_task_input",
        "_prompt_to_task_hint",
        "_run_coroutine_sync",
        "_sanitize_tool_input",
        "_humanize_observation",
        "_resolve_guardrails",
        "_build_multimodal_prompt",
        # generation
        "_generate",
        "_run_direct_inference",
        "_apply_structured_output",
        "_generation_failure_response",
        "_generate_speculative",
        "_model_provider",
        "_parse_with_pydantic",
        "_reconstruct_error",
        # react
        "_run_single_agent",
        "_parse_react_response",
        "_execute_tool",
        "_execute_tool_once",
        "_map_input_to_parameters",
        "_collect_citations",
        "_attach_citations",
        "_native_tool_loop_hint",
        "_run_with_native_tools",
        "_run_with_gemini_native_tools",
        "_run_with_sub_agents",
        "_extract_partial_answer",
    ],
)
def test_extracted_method_reachable_on_agent(method_name):
    assert hasattr(Agent, method_name), f"{method_name} not reachable on Agent"


def test_module_helpers_live_in_runtime_module():
    # The free functions moved to the leaf runtime module; agent.py re-exports
    # what it still uses.
    for name in (
        "sanitize_final_answer",
        "_infer_provider_from_model",
        "_safe_int_or_none",
        "_safe_float_or_none",
    ):
        assert hasattr(agent_runtime, name)
    # Re-exported on agent.py for backward compatibility.
    assert agent_mod.sanitize_final_answer is agent_runtime.sanitize_final_answer


def test_runtime_module_is_a_leaf():
    # The runtime module must not import agent.py (keeps the dependency graph
    # acyclic; generation/react are allowed to import agent for the dataclasses).
    src = inspect.getsource(agent_runtime)
    assert "from .agent import" not in src
    assert "import agent\n" not in src


def test_static_helpers_behaviour_unchanged():
    # JSON fence stripping
    assert Agent._clean_json_input('```json\n{"a": 1}\n```') == '{"a": 1}'
    # Preview truncation honours the limit
    assert Agent._extract_task_preview("x" * 50, limit=10) == "x" * 10
    # Provider inference from a bare id
    assert agent_runtime._infer_provider_from_model(None) == "unknown"
    assert agent_runtime._infer_provider_from_model(object(), "gpt-4o-mini") == "openai"
    assert agent_runtime._infer_provider_from_model(object(), "gemini-3.1-flash") == "google"
    # Numeric coercion never raises
    assert agent_runtime._safe_int_or_none("not-a-number") is None
    assert agent_runtime._safe_int_or_none(3.9) == 3
    assert agent_runtime._safe_float_or_none(None) is None


def test_sanitize_final_answer_still_strips_scaffolding():
    assert sanitize_final_answer("Canberra\nFinal Answer: Canberra") == "Canberra"
    assert sanitize_final_answer(None) is None
