"""
Unit tests for Anthropic native computer-use tool stubs.

Tests cover:
- Tool spec shapes (to_anthropic_tool_spec)
- ToolIncompatibleError raised at Agent init for non-Anthropic models
- Local execution raises RuntimeError
- IS_ANTHROPIC_NATIVE sentinel present
- EXPERIMENTAL flag visible in docstrings
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.errors import ToolIncompatibleError
from effgen.tools.builtin.anthropic_native import (
    AnthropicBashTool,
    AnthropicComputerTool,
    AnthropicNativeTool,
    AnthropicTextEditorTool,
)

# ---------------------------------------------------------------------------
# Tool spec shapes
# ---------------------------------------------------------------------------

class TestAnthropicBashToolSpec:
    def test_type_field(self):
        spec = AnthropicBashTool().to_anthropic_tool_spec()
        assert spec["type"] == "bash_20250124"

    def test_name_field(self):
        spec = AnthropicBashTool().to_anthropic_tool_spec()
        assert spec["name"] == "bash"

    def test_is_dict(self):
        assert isinstance(AnthropicBashTool().to_anthropic_tool_spec(), dict)


class TestAnthropicTextEditorToolSpec:
    def test_type_field(self):
        spec = AnthropicTextEditorTool().to_anthropic_tool_spec()
        assert spec["type"] == "text_editor_20250728"

    def test_name_field(self):
        spec = AnthropicTextEditorTool().to_anthropic_tool_spec()
        assert spec["name"] == "str_replace_based_edit_tool"

    def test_is_dict(self):
        assert isinstance(AnthropicTextEditorTool().to_anthropic_tool_spec(), dict)


class TestAnthropicComputerToolSpec:
    def test_type_field(self):
        spec = AnthropicComputerTool().to_anthropic_tool_spec()
        assert spec["type"] == "computer_20251124"

    def test_name_field(self):
        spec = AnthropicComputerTool().to_anthropic_tool_spec()
        assert spec["name"] == "computer"

    def test_default_display_dimensions(self):
        spec = AnthropicComputerTool().to_anthropic_tool_spec()
        assert spec["display_width_px"] == 1024
        assert spec["display_height_px"] == 768

    def test_custom_display_dimensions(self):
        spec = AnthropicComputerTool(display_width_px=1920, display_height_px=1080).to_anthropic_tool_spec()
        assert spec["display_width_px"] == 1920
        assert spec["display_height_px"] == 1080

    def test_display_number_included_by_default(self):
        spec = AnthropicComputerTool().to_anthropic_tool_spec()
        assert "display_number" in spec
        assert spec["display_number"] == 1

    def test_display_number_none_excluded(self):
        spec = AnthropicComputerTool(display_number=None).to_anthropic_tool_spec()
        assert "display_number" not in spec


# ---------------------------------------------------------------------------
# Sentinel and metadata
# ---------------------------------------------------------------------------

class TestNativeToolSentinel:
    @pytest.mark.parametrize("cls", [AnthropicBashTool, AnthropicTextEditorTool, AnthropicComputerTool])
    def test_is_anthropic_native_sentinel(self, cls):
        assert getattr(cls(), "IS_ANTHROPIC_NATIVE", False) is True

    @pytest.mark.parametrize("cls", [AnthropicBashTool, AnthropicTextEditorTool, AnthropicComputerTool])
    def test_is_subclass_of_anthropic_native_tool(self, cls):
        assert isinstance(cls(), AnthropicNativeTool)

    @pytest.mark.parametrize("cls", [AnthropicBashTool, AnthropicTextEditorTool, AnthropicComputerTool])
    def test_experimental_in_docstring(self, cls):
        doc = cls.__doc__ or ""
        assert "EXPERIMENTAL" in doc.upper()

    @pytest.mark.parametrize("cls", [AnthropicBashTool, AnthropicTextEditorTool, AnthropicComputerTool])
    def test_has_name(self, cls):
        tool = cls()
        assert tool.name


# ---------------------------------------------------------------------------
# Local execution raises RuntimeError
# ---------------------------------------------------------------------------

class TestLocalExecutionForbidden:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", [AnthropicBashTool, AnthropicTextEditorTool, AnthropicComputerTool])
    async def test_execute_raises_runtime_error(self, cls):
        tool = cls()
        with pytest.raises(RuntimeError, match="server-side"):
            await tool._execute()


# ---------------------------------------------------------------------------
# ToolIncompatibleError at Agent init for non-Anthropic model
# ---------------------------------------------------------------------------

class TestToolIncompatibleError:
    def _make_mock_non_anthropic_model(self, model_name: str = "gpt-4o"):
        model = MagicMock()
        model.model_name = model_name
        model._provider = "openai"
        # Make isinstance check fail for AnthropicAdapter
        model.__class__ = type("FakeModel", (), {})
        return model

    def _make_mock_anthropic_model(self, model_name: str = "claude-sonnet-4-6"):
        """Return a mock that passes the AnthropicAdapter isinstance check."""
        from effgen.models.anthropic_adapter import AnthropicAdapter
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            a = AnthropicAdapter(model_name=model_name)
        a._is_loaded = False
        a.client = None
        return a

    def _make_agent_with_tools(self, model, tools):
        """Call _validate_native_tool_compatibility directly."""
        from effgen.core.agent import Agent
        agent = object.__new__(Agent)
        agent.model = model
        agent._validate_native_tool_compatibility(tools)

    def test_bash_tool_with_openai_model_raises(self):
        model = self._make_mock_non_anthropic_model("gpt-4o")
        with pytest.raises(ToolIncompatibleError) as exc_info:
            self._make_agent_with_tools(model, [AnthropicBashTool()])
        assert "anthropic" in str(exc_info.value).lower() or "AnthropicAdapter" in str(exc_info.value)

    def test_text_editor_tool_with_openai_model_raises(self):
        model = self._make_mock_non_anthropic_model("gpt-4o")
        with pytest.raises(ToolIncompatibleError):
            self._make_agent_with_tools(model, [AnthropicTextEditorTool()])

    def test_computer_tool_with_openai_model_raises(self):
        model = self._make_mock_non_anthropic_model("gpt-4o")
        with pytest.raises(ToolIncompatibleError):
            self._make_agent_with_tools(model, [AnthropicComputerTool()])

    def test_bash_tool_with_gemini_model_raises(self):
        model = self._make_mock_non_anthropic_model("gemini-2.5-flash")
        model._provider = "gemini"
        with pytest.raises(ToolIncompatibleError):
            self._make_agent_with_tools(model, [AnthropicBashTool()])

    def test_anthropic_tools_compatible_with_anthropic_model(self):
        """No error when tools are used with an AnthropicAdapter."""
        model = self._make_mock_anthropic_model("claude-sonnet-4-6")
        # Should not raise
        self._make_agent_with_tools(model, [AnthropicBashTool()])

    def test_non_native_tools_no_error_with_any_model(self):
        """Regular (non-native) tools don't trigger the Anthropic guard."""
        from effgen.tools.builtin.calculator import Calculator
        model = self._make_mock_non_anthropic_model("gpt-4o")
        # Should not raise for a regular tool
        self._make_agent_with_tools(model, [Calculator()])


# ---------------------------------------------------------------------------
# Not registered in any preset
# ---------------------------------------------------------------------------

class TestNotInPresets:
    def test_anthropic_native_tools_not_in_default_tools(self):
        """Native tools must NOT appear in default tool presets."""
        try:
            from effgen.tools.presets import DEFAULT_TOOLS
            tool_names = [t.name if hasattr(t, "name") else str(t) for t in DEFAULT_TOOLS]
            assert "anthropic_bash" not in tool_names
            assert "anthropic_text_editor" not in tool_names
            assert "anthropic_computer" not in tool_names
        except ImportError:
            pytest.skip("No tool presets module found")


# ---------------------------------------------------------------------------
# Integration skip guard
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="SKIPPED: ANTHROPIC_API_KEY not in ~/.effgen/.env",
)
class TestAnthropicNativeIntegration:
    def test_live_bash_tool_spec_accepted(self):
        """Live: verify the bash tool spec can be sent to the API without rejection."""
        from pathlib import Path

        from dotenv import load_dotenv
        load_dotenv(Path.home() / ".effgen" / ".env", override=False)

        from effgen.models.anthropic_adapter import AnthropicAdapter
        a = AnthropicAdapter(model_name="claude-sonnet-4-6")
        a.load()
        try:
            spec = AnthropicBashTool().to_anthropic_tool_spec()
            result = a.generate_with_tools(
                "What is 2+2?",
                tools=[spec],
                extra_headers={"anthropic-beta": "computer-use-2025-11-24"},
            )
            assert result.text or result.metadata.get("tool_uses")
        finally:
            a.unload()
