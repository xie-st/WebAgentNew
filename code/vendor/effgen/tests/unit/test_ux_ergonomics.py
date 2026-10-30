"""UX/usability contracts (no live calls).

Covers the user-facing ergonomics fixed in the 0.3.0 polish work:
result ``__str__``/aliases, the ``@tool`` / ``Tool.from_function`` helper, typed
preset errors with fuzzy suggestions, zero-config guidance, model-less-agent
construction failure, and the centralized "did you mean" model suggestion.
"""

import asyncio
from typing import Literal

import pytest


# --------------------------------------------------------------------------- #
# Result ergonomics
# --------------------------------------------------------------------------- #
def test_agent_response_str_is_the_answer():
    from effgen.core.agent import AgentResponse

    r = AgentResponse(output="42.5", success=True)
    assert str(r) == "42.5"
    # repr still carries the structured detail.
    assert "AgentResponse(" in repr(r)
    assert "success=True" in repr(r)


def test_agent_response_text_content_aliases():
    from effgen.core.agent import AgentResponse

    r = AgentResponse(output="hello")
    assert r.text == "hello"
    assert r.content == "hello"
    assert r.text == r.output == r.content


def test_agent_response_str_handles_none_output():
    from effgen.core.agent import AgentResponse

    r = AgentResponse(output=None)  # type: ignore[arg-type]
    assert str(r) == ""


# --------------------------------------------------------------------------- #
# @tool / Tool.from_function
# --------------------------------------------------------------------------- #
def test_tool_decorator_derives_schema_from_signature_and_docstring():
    from effgen import FunctionTool, tool

    @tool
    def word_count(text: str) -> int:
        """Count the words in a piece of text.

        Args:
            text: The text to count words in.
        """
        return len(text.split())

    assert isinstance(word_count, FunctionTool)
    assert word_count.name == "word_count"
    assert word_count.description == "Count the words in a piece of text."
    schema = word_count.metadata.to_json_schema()
    props = schema["parameters"]["properties"]
    assert props["text"]["type"] == "string"
    assert props["text"]["description"] == "The text to count words in."
    assert schema["parameters"]["required"] == ["text"]


def test_tool_decorator_executes_sync_function():
    from effgen import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    result = asyncio.run(add.execute(a=2, b=3))
    assert result.success is True
    assert result.output == 5


def test_tool_decorator_executes_async_function():
    from effgen import tool

    @tool
    async def double(x: int) -> int:
        """Double x."""
        await asyncio.sleep(0)
        return x * 2

    result = asyncio.run(double.execute(x=21))
    assert result.success is True
    assert result.output == 42


def test_tool_decorator_optional_and_literal_params():
    from effgen import tool

    @tool(name="fmt", category="computation")
    def fmt(value: int, style: Literal["hex", "dec"] = "dec", label: str | None = None) -> str:
        """Format a number.

        Args:
            value: the number to format.
            style: output style.
            label: an optional label.
        """
        return label or str(value)

    schema = fmt.metadata.to_json_schema()
    props = schema["parameters"]["properties"]
    # required = only the parameter without a default
    assert schema["parameters"]["required"] == ["value"]
    assert props["style"]["enum"] == ["hex", "dec"]
    assert props["label"]["type"] == "string"  # Optional[str] -> string


def test_tool_validation_reports_missing_required():
    from effgen import tool

    @tool
    def needs_text(text: str) -> str:
        """Echo text."""
        return text

    result = asyncio.run(needs_text.execute())
    assert result.success is False
    assert "required" in (result.error or "").lower()


def test_from_function_equivalent_to_decorator():
    from effgen import Tool

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    t = Tool.from_function(add)
    assert t.name == "add"
    assert asyncio.run(t.execute(a=4, b=5)).output == 9


def test_tool_rejects_lambda_without_name():
    from effgen import Tool

    with pytest.raises(ValueError, match="name="):
        Tool.from_function(lambda x: x)


def test_tool_decorator_requires_approval_reaches_metadata():
    """@tool(requires_approval=True) is reachable through ToolMetadata."""
    from effgen import tool

    @tool
    def order_lookup(order_id: str) -> str:
        """Look up an order."""
        return order_id

    @tool(requires_approval=True)
    def issue_refund(order_id: str) -> str:
        """Refund an order."""
        return order_id

    assert order_lookup.metadata.requires_approval is False
    assert issue_refund.metadata.requires_approval is True


def test_from_function_requires_approval_and_cost_timeout_overrides():
    from effgen import Tool

    def cancel_subscription(account_id: str) -> str:
        """Cancel a subscription."""
        return account_id

    t = Tool.from_function(
        cancel_subscription,
        requires_approval=True,
        cost_estimate="high",
        timeout_seconds=5,
    )
    assert t.metadata.requires_approval is True
    assert t.metadata.cost_estimate == "high"
    assert t.metadata.timeout_seconds == 5


def test_dangerous_tool_requires_approval_gates_dangerous_only_mode():
    """A tool marked requires_approval is reachable by ApprovalManager
    (approval_mode="dangerous_only"), the mechanism `issue_refund` needs."""
    from effgen import tool
    from effgen.core.human_loop import ApprovalManager, ApprovalMode

    @tool(requires_approval=True)
    def issue_refund(order_id: str) -> str:
        """Refund an order."""
        return order_id

    mgr = ApprovalManager(mode=ApprovalMode.DANGEROUS_ONLY)
    assert mgr.should_request_approval(
        issue_refund.name, issue_refund.metadata.requires_approval
    )


def test_dangerous_tool_keywords_cover_common_mutating_verbs():
    """Money-moving/mutating tool names are gated under dangerous_only even
    without an explicit requires_approval=True, as defense in depth."""
    from effgen.core.human_loop import is_tool_dangerous

    for name in (
        "issue_refund", "refund_order", "charge_card",
        "cancel_subscription", "delete_account", "transfer_funds",
    ):
        assert is_tool_dangerous(name), f"{name} should be flagged dangerous"
    for name in ("order_lookup", "calculator", "web_search"):
        assert not is_tool_dangerous(name), f"{name} should not be flagged dangerous"


def test_tool_usable_in_agent_config():
    """A decorated tool drops straight into AgentConfig(tools=[...])."""
    from effgen import tool
    from effgen.core.agent import AgentConfig

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    cfg = AgentConfig(name="t", model=None, tools=[ping], require_model=False)
    assert "ping" in {t.name for t in cfg.tools}


def test_tool_array_param_carries_element_type():
    """list[str]/list[int] populate schema `items`, not a bare `array`."""
    from effgen import tool

    @tool
    def join_tags(tags: list[str]) -> str:
        """Join tags.

        Args:
            tags: The tags to join.
        """
        return ",".join(tags)

    @tool
    def sum_ints(values: list[int]) -> int:
        """Sum integers.

        Args:
            values: The integers to sum.
        """
        return sum(values)

    tags_schema = join_tags.metadata.to_json_schema()["parameters"]["properties"]["tags"]
    assert tags_schema["items"] == {"type": "string"}

    values_schema = sum_ints.metadata.to_json_schema()["parameters"]["properties"]["values"]
    assert values_schema["items"] == {"type": "integer"}


def test_tool_untyped_array_param_has_no_items():
    """A bare `list` annotation (no element type) still emits no `items` — no regression."""
    from effgen import tool

    @tool
    def echo_list(items: list) -> list:
        """Echo a list.

        Args:
            items: items to echo.
        """
        return items

    schema = echo_list.metadata.to_json_schema()["parameters"]["properties"]["items"]
    assert schema["type"] == "array"
    assert "items" not in schema


def test_tool_unknown_category_warns_and_falls_back_to_system(caplog):
    """@tool(category="typo") warns naming the bad value + valid ones, then falls back."""
    from effgen import tool
    from effgen.tools.base_tool import ToolCategory

    with caplog.at_level("WARNING"):
        @tool(category="not_a_real_category")
        def f(x: int) -> int:
            """f.

            Args:
                x: x
            """
            return x

    assert f.metadata.category == ToolCategory.SYSTEM
    assert any("not_a_real_category" in rec.message for rec in caplog.records)
    assert any("computation" in rec.message for rec in caplog.records)  # a valid value is named


def test_tool_known_category_string_is_quiet(caplog):
    from effgen import tool
    from effgen.tools.base_tool import ToolCategory

    with caplog.at_level("WARNING"):
        @tool(category="computation")
        def g(x: int) -> int:
            """g.

            Args:
                x: x
            """
            return x

    assert g.metadata.category == ToolCategory.COMPUTATION
    assert not caplog.records


# --------------------------------------------------------------------------- #
# Preset errors + zero-config guidance
# --------------------------------------------------------------------------- #
def test_unknown_preset_is_typed_with_fuzzy_suggestion():
    from effgen.presets import UnknownPresetError, get_preset

    with pytest.raises(UnknownPresetError) as ei:
        get_preset("maths")
    msg = str(ei.value)
    assert "Unknown preset 'maths'" in msg
    assert "Did you mean 'math'?" in msg
    # Clean message (no KeyError repr-quoting), but still compatible.
    assert not msg.startswith('"')
    assert isinstance(ei.value, ValueError | KeyError)


def test_create_agent_without_model_gives_guidance():
    from effgen.presets import create_agent

    with pytest.raises(ValueError) as ei:
        create_agent("math")
    msg = str(ei.value)
    assert "effgen models list" in msg
    assert "gpt-5-nano" in msg


def test_create_agent_env_default_model(monkeypatch):
    from effgen.presets.registry import _resolve_default_model

    monkeypatch.setenv("EFFGEN_DEFAULT_MODEL", "my-default-model")
    assert _resolve_default_model("math") == "my-default-model"


def test_create_agent_docstring_lists_all_presets():
    from effgen.presets import create_agent, list_presets

    doc = create_agent.__doc__ or ""
    for name in list_presets():
        assert name in doc, f"preset {name!r} missing from create_agent docstring"
    # steers newcomers
    assert "kitchen sink" in doc


# --------------------------------------------------------------------------- #
# Model-less agent fails at construction
# --------------------------------------------------------------------------- #
def test_modelless_agent_raises_at_construction():
    from effgen.core.agent import Agent, AgentConfig

    with pytest.raises(ValueError, match="without a model"):
        Agent(AgentConfig(name="x", model=None))  # require_model defaults True


def test_modelless_agent_allowed_when_require_model_false():
    from effgen.core.agent import Agent, AgentConfig

    agent = Agent(AgentConfig(name="x", model=None, require_model=False))
    assert agent.model is None


# --------------------------------------------------------------------------- #
# Centralized "did you mean" model suggestion
# --------------------------------------------------------------------------- #
def test_not_found_error_appends_live_suggestion():
    from effgen.models._adapter_utils import provider_runtime_error

    class _Fake404(Exception):
        status_code = 404

        def __str__(self) -> str:
            return "model_not_found: gpt-5-nanoo does not exist"

    err = provider_runtime_error("openai", "gpt-5-nanoo", "generate", _Fake404())
    text = str(err)
    assert "Did you mean" in text
    assert "gpt-5-nano" in text
    assert err.error_context["category"] == "not_found"
