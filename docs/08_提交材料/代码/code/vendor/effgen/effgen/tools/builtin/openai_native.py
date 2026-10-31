"""
OpenAI first-party tool wrappers for effGen.

These wrappers expose OpenAI's hosted tools — web_search, code_interpreter,
and file_search — as effGen BaseTool subclasses.  Unlike effGen's local tools,
these run server-side inside OpenAI's infrastructure.  The adapter routes them
through the OpenAI Responses API instead of executing them locally.

Cost note:
  web_search adds a per-call surcharge on top of token costs.  See
  https://platform.openai.com/docs/pricing for current rates.  As of 2026-04-24
  the search tool costs $30/1k calls for gpt-4o-class models.

Usage:
    from effgen.tools.builtin.openai_native import (
        OpenAIWebSearchTool,
        OpenAICodeInterpreterTool,
        OpenAIFileSearchTool,
    )
    tool = OpenAIWebSearchTool()
    spec = tool.to_openai_tool_spec()   # pass to the Responses API
"""

from __future__ import annotations

import logging
from typing import Any

from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


class OpenAINativeTool(BaseTool):
    """Base class for OpenAI server-side (first-party) tools.

    Subclasses must implement ``to_openai_tool_spec()`` to return the exact
    JSON dict that OpenAI's Responses API / Chat Completions tools array
    expects.

    Server-side tools are never executed locally — the OpenAI backend runs
    them and returns a structured result that the adapter surfaces as a normal
    ToolCallResult.
    """

    IS_OPENAI_NATIVE = True  # sentinel used by the adapter / Agent

    def to_openai_tool_spec(self) -> dict[str, Any]:
        """Return the OpenAI-format tool spec for this native tool."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_openai_tool_spec() to return "
            "the OpenAI Responses-API tool dict (e.g. {'type': 'web_search_preview'}). "
            "Use OpenAIWebSearchTool / OpenAICodeInterpreterTool / OpenAIFileSearchTool, "
            "or override this method in your subclass."
        )

    async def _execute(self, **kwargs) -> Any:
        """Local execution is intentionally unsupported for native tools.

        Execution happens server-side inside OpenAI's infrastructure.
        The adapter intercepts native tools before they reach this method.
        """
        raise RuntimeError(
            f"'{self.name}' is an OpenAI server-side tool and cannot be executed locally. "
            "Use an OpenAI model with the Native tool-calling strategy."
        )


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

class OpenAIWebSearchTool(OpenAINativeTool):
    """OpenAI hosted web-search tool.

    Lets the model search the live web and include cited results in its reply.
    Requires an OpenAI model — raises ToolIncompatibleError at Agent init if
    used with any other provider.

    Cost note (as of 2026-04-24):
        web_search_preview adds ~$30 / 1,000 calls on top of token costs.
        Budget accordingly.  See https://platform.openai.com/docs/pricing.

    Args:
        search_context_size: Amount of context returned per query.
            One of "low", "medium" (default), "high".  Higher = more tokens.
        user_location: Optional geographic context dict, e.g.
            {"type": "approximate", "country": "US", "city": "New York"}.
    """

    def __init__(
        self,
        search_context_size: str = "medium",
        user_location: dict[str, Any] | None = None,
    ) -> None:
        if search_context_size not in ("low", "medium", "high"):
            raise ValueError(
                f"search_context_size must be 'low', 'medium', or 'high', "
                f"got {search_context_size!r}"
            )
        super().__init__(
            metadata=ToolMetadata(
                name="openai_web_search",
                description=(
                    "Search the live web using OpenAI's hosted web_search tool. "
                    "Returns up-to-date information with citations. "
                    "SERVER-SIDE: executed by OpenAI, not locally. "
                    "Cost: ~$30/1k calls on top of token costs."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="The search query to execute.",
                        required=True,
                    ),
                ],
                requires_api_key=True,
                cost_estimate="medium",
                tags=["web", "search", "openai-native", "server-side"],
            )
        )
        self.search_context_size = search_context_size
        self.user_location = user_location

    def to_openai_tool_spec(self) -> dict[str, Any]:
        """Return the spec for OpenAI's web_search_preview tool."""
        spec: dict[str, Any] = {
            "type": "web_search_preview",
            "search_context_size": self.search_context_size,
        }
        if self.user_location:
            spec["user_location"] = self.user_location
        return spec


# ---------------------------------------------------------------------------
# code_interpreter
# ---------------------------------------------------------------------------

class OpenAICodeInterpreterTool(OpenAINativeTool):
    """OpenAI hosted code-interpreter tool.

    Runs Python code in a sandboxed environment on OpenAI's servers.
    Supports file I/O, matplotlib charts, and data analysis.  Results
    (stdout, generated files, error messages) come back as part of the
    model response.

    Requires an OpenAI model — raises ToolIncompatibleError at Agent init
    if used with any other provider.

    Args:
        container: Optional container configuration dict.  Defaults to
            ``{"type": "auto"}`` which lets OpenAI choose the runtime.
    """

    def __init__(self, container: dict[str, Any] | None = None) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="openai_code_interpreter",
                description=(
                    "Execute Python code in a secure OpenAI-hosted sandbox. "
                    "Supports data analysis, math, file I/O, and chart generation. "
                    "SERVER-SIDE: executed by OpenAI, not locally."
                ),
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="Python code to execute.",
                        required=True,
                    ),
                ],
                requires_api_key=True,
                cost_estimate="medium",
                tags=["code", "python", "interpreter", "openai-native", "server-side"],
            )
        )
        self.container = container or {"type": "auto"}

    def to_openai_tool_spec(self) -> dict[str, Any]:
        """Return the spec for OpenAI's code_interpreter tool."""
        return {
            "type": "code_interpreter",
            "container": self.container,
        }


# ---------------------------------------------------------------------------
# file_search
# ---------------------------------------------------------------------------

class OpenAIFileSearchTool(OpenAINativeTool):
    """OpenAI hosted file-search (RAG) tool.

    Performs vector search over files previously uploaded to an OpenAI
    vector store.  Returns relevant excerpts that the model uses to answer
    questions grounded in your documents.

    Requires an OpenAI model — raises ToolIncompatibleError at Agent init
    if used with any other provider.

    Args:
        vector_store_ids: List of OpenAI vector store IDs to search.
            Create a vector store via the Files API and pass the resulting ID.
        max_num_results: Maximum number of chunks to retrieve (1–50).
            Defaults to 10.
        ranking_options: Optional dict with ``ranker`` and
            ``score_threshold`` keys.
        filters: Optional metadata filter dict for scoped retrieval.
    """

    def __init__(
        self,
        vector_store_ids: list[str] | None = None,
        max_num_results: int = 10,
        ranking_options: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        if max_num_results < 1 or max_num_results > 50:
            raise ValueError(f"max_num_results must be between 1 and 50, got {max_num_results}")
        super().__init__(
            metadata=ToolMetadata(
                name="openai_file_search",
                description=(
                    "Search over uploaded files using OpenAI's hosted vector-search tool. "
                    "Returns relevant document excerpts for grounded answers (RAG). "
                    "SERVER-SIDE: executed by OpenAI, not locally."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Query to search the uploaded documents for.",
                        required=True,
                    ),
                ],
                requires_api_key=True,
                cost_estimate="low",
                tags=["files", "rag", "search", "openai-native", "server-side"],
            )
        )
        self.vector_store_ids: list[str] = vector_store_ids or []
        self.max_num_results = max_num_results
        self.ranking_options = ranking_options
        self.filters = filters

    def to_openai_tool_spec(self) -> dict[str, Any]:
        """Return the spec for OpenAI's file_search tool."""
        spec: dict[str, Any] = {
            "type": "file_search",
            "max_num_results": self.max_num_results,
        }
        if self.vector_store_ids:
            spec["vector_store_ids"] = self.vector_store_ids
        if self.ranking_options:
            spec["ranking_options"] = self.ranking_options
        if self.filters:
            spec["filters"] = self.filters
        return spec

    def add_vector_store(self, vector_store_id: str) -> None:
        """Register an additional vector store ID."""
        if vector_store_id not in self.vector_store_ids:
            self.vector_store_ids.append(vector_store_id)

    def remove_vector_store(self, vector_store_id: str) -> None:
        """Deregister a vector store ID."""
        self.vector_store_ids = [vid for vid in self.vector_store_ids if vid != vector_store_id]
