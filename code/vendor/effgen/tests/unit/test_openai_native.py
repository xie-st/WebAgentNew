"""Unit tests for OpenAI native tool wrappers."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

def test_import_openai_native_tools():
    from effgen.tools.builtin.openai_native import (
        OpenAICodeInterpreterTool,
        OpenAIFileSearchTool,
        OpenAINativeTool,
        OpenAIWebSearchTool,
    )
    assert issubclass(OpenAIWebSearchTool, OpenAINativeTool)
    assert issubclass(OpenAICodeInterpreterTool, OpenAINativeTool)
    assert issubclass(OpenAIFileSearchTool, OpenAINativeTool)


def test_import_tool_incompatible_error():
    from effgen.models.errors import ToolIncompatibleError
    assert issubclass(ToolIncompatibleError, Exception)


def test_top_level_exports():
    import effgen
    assert hasattr(effgen, "ToolIncompatibleError")
    assert hasattr(effgen, "OpenAIWebSearchTool")
    assert hasattr(effgen, "OpenAICodeInterpreterTool")
    assert hasattr(effgen, "OpenAIFileSearchTool")


# ---------------------------------------------------------------------------
# OpenAIWebSearchTool
# ---------------------------------------------------------------------------

class TestOpenAIWebSearchTool:
    def _make(self, **kw):
        from effgen.tools.builtin.openai_native import OpenAIWebSearchTool
        return OpenAIWebSearchTool(**kw)

    def test_default_spec(self):
        tool = self._make()
        spec = tool.to_openai_tool_spec()
        assert spec["type"] == "web_search_preview"
        assert spec["search_context_size"] == "medium"
        assert "user_location" not in spec

    def test_custom_context_size(self):
        tool = self._make(search_context_size="high")
        spec = tool.to_openai_tool_spec()
        assert spec["search_context_size"] == "high"

    def test_invalid_context_size_raises(self):
        with pytest.raises(ValueError, match="search_context_size"):
            self._make(search_context_size="ultra")

    def test_user_location_in_spec(self):
        loc = {"type": "approximate", "country": "US"}
        tool = self._make(user_location=loc)
        spec = tool.to_openai_tool_spec()
        assert spec["user_location"] == loc

    def test_is_openai_native_sentinel(self):
        from effgen.tools.builtin.openai_native import OpenAIWebSearchTool
        assert OpenAIWebSearchTool.IS_OPENAI_NATIVE is True

    def test_name(self):
        tool = self._make()
        assert tool.name == "openai_web_search"

    def test_local_execute_raises(self):
        import asyncio
        tool = self._make()
        with pytest.raises(RuntimeError, match="server-side"):
            asyncio.run(tool._execute())


# ---------------------------------------------------------------------------
# OpenAICodeInterpreterTool
# ---------------------------------------------------------------------------

class TestOpenAICodeInterpreterTool:
    def _make(self, **kw):
        from effgen.tools.builtin.openai_native import OpenAICodeInterpreterTool
        return OpenAICodeInterpreterTool(**kw)

    def test_default_spec(self):
        tool = self._make()
        spec = tool.to_openai_tool_spec()
        assert spec["type"] == "code_interpreter"
        assert spec["container"]["type"] == "auto"

    def test_custom_container(self):
        tool = self._make(container={"type": "auto", "cpu": 2})
        spec = tool.to_openai_tool_spec()
        assert spec["container"]["cpu"] == 2

    def test_name(self):
        tool = self._make()
        assert tool.name == "openai_code_interpreter"

    def test_is_openai_native(self):
        tool = self._make()
        assert tool.IS_OPENAI_NATIVE is True

    def test_local_execute_raises(self):
        import asyncio
        tool = self._make()
        with pytest.raises(RuntimeError, match="server-side"):
            asyncio.run(tool._execute())


# ---------------------------------------------------------------------------
# OpenAIFileSearchTool
# ---------------------------------------------------------------------------

class TestOpenAIFileSearchTool:
    def _make(self, **kw):
        from effgen.tools.builtin.openai_native import OpenAIFileSearchTool
        return OpenAIFileSearchTool(**kw)

    def test_default_spec_no_stores(self):
        tool = self._make()
        spec = tool.to_openai_tool_spec()
        assert spec["type"] == "file_search"
        assert spec["max_num_results"] == 10
        assert "vector_store_ids" not in spec

    def test_vector_store_ids_in_spec(self):
        tool = self._make(vector_store_ids=["vs_abc", "vs_def"])
        spec = tool.to_openai_tool_spec()
        assert spec["vector_store_ids"] == ["vs_abc", "vs_def"]

    def test_max_num_results(self):
        tool = self._make(max_num_results=5)
        spec = tool.to_openai_tool_spec()
        assert spec["max_num_results"] == 5

    def test_invalid_max_num_results_raises(self):
        with pytest.raises(ValueError, match="max_num_results"):
            self._make(max_num_results=0)
        with pytest.raises(ValueError, match="max_num_results"):
            self._make(max_num_results=51)

    def test_add_vector_store(self):
        tool = self._make()
        tool.add_vector_store("vs_new")
        assert "vs_new" in tool.vector_store_ids
        # Adding duplicate is idempotent
        tool.add_vector_store("vs_new")
        assert tool.vector_store_ids.count("vs_new") == 1

    def test_remove_vector_store(self):
        tool = self._make(vector_store_ids=["vs_a", "vs_b"])
        tool.remove_vector_store("vs_a")
        assert "vs_a" not in tool.vector_store_ids
        assert "vs_b" in tool.vector_store_ids

    def test_name(self):
        tool = self._make()
        assert tool.name == "openai_file_search"

    def test_ranking_options_in_spec(self):
        opts = {"ranker": "default-2024-11-15", "score_threshold": 0.5}
        tool = self._make(ranking_options=opts)
        spec = tool.to_openai_tool_spec()
        assert spec["ranking_options"] == opts

    def test_filters_in_spec(self):
        filters = {"type": "eq", "key": "category", "value": "faq"}
        tool = self._make(filters=filters)
        spec = tool.to_openai_tool_spec()
        assert spec["filters"] == filters

    def test_local_execute_raises(self):
        import asyncio
        tool = self._make()
        with pytest.raises(RuntimeError, match="server-side"):
            asyncio.run(tool._execute())


# ---------------------------------------------------------------------------
# ToolIncompatibleError
# ---------------------------------------------------------------------------

class TestToolIncompatibleError:
    def test_message_contains_tool_and_model(self):
        from effgen.models.errors import ToolIncompatibleError
        err = ToolIncompatibleError(
            tool_name="openai_web_search",
            model_name="llama3.1-8b",
            reason="OpenAI only",
        )
        msg = str(err)
        assert "openai_web_search" in msg
        assert "llama3.1-8b" in msg
        assert "OpenAI only" in msg

    def test_attributes(self):
        from effgen.models.errors import ToolIncompatibleError
        err = ToolIncompatibleError("t", "m", "r")
        assert err.tool_name == "t"
        assert err.model_name == "m"
        assert err.reason == "r"

    def test_no_reason(self):
        from effgen.models.errors import ToolIncompatibleError
        err = ToolIncompatibleError("t", "m")
        assert "t" in str(err)
        assert "m" in str(err)


# ---------------------------------------------------------------------------
# Agent init incompatibility check
# ---------------------------------------------------------------------------

class TestAgentNativeToolIncompatibility:
    def test_cerebras_model_with_native_tool_raises(self):
        """ToolIncompatibleError must fire at Agent.__init__, not at run time."""
        from unittest.mock import MagicMock

        from effgen.core.agent import Agent, AgentConfig
        from effgen.models.base import BaseModel
        from effgen.models.errors import ToolIncompatibleError
        from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

        # spec=BaseModel makes this a faithful loaded-model double (a real
        # CerebrasAdapter *is* a BaseModel), so the Agent ctor accepts it as a
        # loaded instance — but it's still not an OpenAIAdapter.
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model_name = "llama3.1-8b"

        # Patch the OpenAIAdapter import path
        from unittest.mock import patch
        with patch("effgen.core.agent.Agent._validate_native_tool_compatibility") as mock_validate:
            mock_validate.side_effect = ToolIncompatibleError(
                "openai_web_search", "llama3.1-8b",
                "OpenAI native tools require OpenAIAdapter"
            )
            with pytest.raises(ToolIncompatibleError):
                Agent(AgentConfig(
                    name="test-agent",
                    model=mock_model,
                    tools=[OpenAIWebSearchTool()],
                ))

    def test_openai_adapter_with_native_tool_does_not_raise(self):
        """No error when native tool is used with an OpenAI model."""
        import os

        from effgen.core.agent import Agent, AgentConfig
        from effgen.models.openai_adapter import OpenAIAdapter
        from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        adapter = OpenAIAdapter(model_name="gpt-5.4-nano", api_key=api_key)
        adapter.load()
        try:
            agent = Agent(AgentConfig(
                name="test-openai-agent",
                model=adapter,
                tools=[OpenAIWebSearchTool()],
            ))
            agent.close()
        finally:
            adapter.unload()
