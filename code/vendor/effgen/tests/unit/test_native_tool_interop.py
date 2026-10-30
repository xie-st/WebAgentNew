"""Native + effGen tool interop, no-network unit coverage.

Covers the hardening done for native/custom tool interop:
  * provider-native abstract spec methods raise an *actionable* NotImplementedError
  * the ReAct loop returns a recovery hint (not a raw RuntimeError) for any
    provider-native tool that reaches it
  * the OpenAI / Gemini native paths surface a structured, redacted failure
    (metadata["error"]={type,category,provider,model,message,retryable})
  * Gemini's INVALID_ARGUMENT (a generic 400) is classified as an invalid
    request, not an auth error
  * mixing a Gemini built-in tool with custom function tools yields a clear
    actionable error instead of a raw provider 400
"""
from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin.anthropic_native import (
    AnthropicBashTool,
    AnthropicNativeTool,
)
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.gemini_native import GeminiNativeTool, GoogleSearchTool
from effgen.tools.builtin.openai_native import OpenAINativeTool, OpenAIWebSearchTool


# ---------------------------------------------------------------------------
# Abstract native-spec methods: actionable NotImplementedError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls, method",
    [
        (OpenAINativeTool, "to_openai_tool_spec"),
        (GeminiNativeTool, "to_gemini_tool"),
        (AnthropicNativeTool, "to_anthropic_tool_spec"),
    ],
)
def test_abstract_native_spec_is_actionable(cls, method):
    inst = object.__new__(cls)  # bypass __init__ (abstract base)
    with pytest.raises(NotImplementedError) as exc:
        getattr(inst, method)()
    msg = str(exc.value)
    assert "must implement" in msg
    assert method in msg
    assert msg.strip() != ""  # not a bare NotImplementedError


# ---------------------------------------------------------------------------
# ReAct loop recovery hint for provider-native tools
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tool, vendor",
    [
        (OpenAIWebSearchTool(), "OpenAI"),
        (GoogleSearchTool(), "Google"),
        (AnthropicBashTool(), "Anthropic"),
    ],
)
def test_native_loop_hint_for_each_vendor(tool, vendor):
    hint = Agent._native_tool_loop_hint(tool, tool.name)
    assert hint is not None
    assert vendor in hint
    assert tool.name in hint
    assert "cannot be executed locally" in hint


def test_native_loop_hint_none_for_ordinary_tool():
    assert Agent._native_tool_loop_hint(Calculator(), "calculator") is None


# ---------------------------------------------------------------------------
# Gemini INVALID_ARGUMENT classification (not auth)
# ---------------------------------------------------------------------------
def test_gemini_invalid_argument_is_invalid_request(monkeypatch):
    from effgen.models import gemini_adapter as ga
    from effgen.models.errors import InvalidRequestError, ModelAuthError

    adapter = ga.GeminiAdapter(model_name="gemini-2.5-flash", api_key="k")
    adapter._is_loaded = True

    class _FakeModels:
        def generate_content(self, **kw):
            raise RuntimeError(
                "400 INVALID_ARGUMENT. {'error': {'code': 400, "
                "'message': 'Built-in tools and Function Calling cannot be combined'}}"
            )

    class _FakeClient:
        models = _FakeModels()

    adapter.client = _FakeClient()
    with pytest.raises(InvalidRequestError) as exc:
        adapter.generate("hi")
    assert not isinstance(exc.value, ModelAuthError)
    assert exc.value.error_context["category"] == "invalid_request"


def test_gemini_api_key_invalid_is_auth(monkeypatch):
    from effgen.models import gemini_adapter as ga
    from effgen.models.errors import ModelAuthError

    adapter = ga.GeminiAdapter(model_name="gemini-2.5-flash", api_key="k")
    adapter._is_loaded = True

    class _FakeModels:
        def generate_content(self, **kw):
            raise RuntimeError("400 INVALID_ARGUMENT API key not valid. Please pass a valid API key.")

    class _FakeClient:
        models = _FakeModels()

    adapter.client = _FakeClient()
    with pytest.raises(ModelAuthError):
        adapter.generate("hi")


# ---------------------------------------------------------------------------
# OpenAI native path: structured, redacted failure
# ---------------------------------------------------------------------------
def test_openai_native_failure_is_structured_and_redacted():
    from effgen.models.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(model_name="gpt-5-nano", api_key="sk-secret-LEAKME-123456")
    adapter._is_loaded = True

    def _boom(**kwargs):
        raise RuntimeError("nope sk-secret-LEAKME-123456 failed")

    adapter.generate_with_native_tools = _boom  # type: ignore[assignment]

    agent = Agent(AgentConfig(name="t", model=adapter, tools=[OpenAIWebSearchTool()],
                              raise_on_error=False,
                              tool_calling_mode="native"))
    resp = agent.run("do something")
    agent.close()

    assert resp.success is False
    err = resp.metadata.get("error")
    assert isinstance(err, dict)
    assert set(err) >= {"type", "category", "provider", "model", "message", "retryable"}
    assert resp.metadata.get("reason") == "generation_failed"
    # secret must not leak anywhere in the response
    import json as _json
    assert "sk-secret-LEAKME-123456" not in _json.dumps(resp.metadata)
    assert "sk-secret-LEAKME-123456" not in resp.output


# ---------------------------------------------------------------------------
# Gemini mixed native + custom: clear actionable error
# ---------------------------------------------------------------------------
def test_gemini_mixed_native_and_custom_clear_error():
    from effgen.models.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter(model_name="gemini-2.5-flash", api_key="k")
    adapter._is_loaded = True

    def _boom(prompt, **kwargs):
        from effgen.models.errors import InvalidRequestError
        raise InvalidRequestError(
            "gemini", "gemini-2.5-flash",
            "400 INVALID_ARGUMENT Built-in tools ({google_search}) and Function "
            "Calling cannot be combined in the same request.",
        )

    adapter.generate = _boom  # type: ignore[assignment]

    agent = Agent(AgentConfig(name="gm", model=adapter, raise_on_error=False,
                              tools=[GoogleSearchTool(), Calculator()]))
    resp = agent.run("compute 1+1 with the calculator")
    agent.close()

    assert resp.success is False
    assert "cannot use its built-in tool" in resp.output
    assert "google_search" in resp.output
    err = resp.metadata.get("error")
    assert isinstance(err, dict)
    assert err["category"] == "invalid_request"
    assert err["retryable"] is False
