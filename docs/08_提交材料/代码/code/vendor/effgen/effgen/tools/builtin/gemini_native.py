"""
Gemini first-party tool wrappers for effGen.

These wrappers expose Gemini's server-side built-in tools — google_search,
url_context, and code_execution — as effGen BaseTool subclasses.  Unlike
effGen's local tools, these run inside Google's infrastructure.

Usage:
    from effgen.tools.builtin.gemini_native import (
        GoogleSearchTool,
        GeminiUrlContextTool,
        GeminiCodeExecutionTool,
    )
    agent = Agent(config=AgentConfig(model=gemini_model, tools=[GoogleSearchTool()]))
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


class GeminiNativeTool(BaseTool):
    """Base class for Gemini server-side (first-party) tools.

    These tools are injected into the Gemini request as SDK ``Tool`` objects.
    The model executes them server-side and returns results as part of the
    response.  Local execution is not supported.
    """

    IS_GEMINI_NATIVE = True  # sentinel used by the adapter / Agent

    def to_gemini_tool(self) -> Any:
        """Return a ``google.genai.types.Tool`` for this native tool."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_gemini_tool() to return a "
            "google.genai.types.Tool (e.g. types.Tool(google_search=types.GoogleSearch())). "
            "Use GoogleSearchTool / GeminiUrlContextTool / GeminiCodeExecutionTool, "
            "or override this method in your subclass."
        )

    async def _execute(self, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"'{self.name}' is a Gemini server-side tool and cannot be executed "
            "locally.  Use a GeminiAdapter model with this tool."
        )


# ---------------------------------------------------------------------------
# GoogleSearchTool
# ---------------------------------------------------------------------------

class GoogleSearchTool(GeminiNativeTool):
    """Gemini built-in Google Search tool.

    Activates Gemini's server-side web search, grounding the model's reply
    with live search results and citations.

    Requires a Gemini model — raises ToolIncompatibleError at Agent init if
    used with any other provider.

    Note:
        This is distinct from ``GenerationConfig(grounding=True)``.  Use this
        class when you want the Agent orchestration loop to manage search as an
        explicit tool call rather than as a transparent grounding pass.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="google_search",
                description=(
                    "Search the live web using Google Search.  "
                    "Returns up-to-date information with citations.  "
                    "SERVER-SIDE: executed by Google, not locally.  "
                    "Only compatible with Gemini models."
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
                cost_estimate="low",
                tags=["web", "search", "gemini-native", "server-side", "grounding"],
            )
        )

    def to_gemini_tool(self) -> Any:
        """Return the SDK ``Tool`` that enables server-side Google Search."""
        from google.genai import types  # type: ignore[import]
        return types.Tool(google_search=types.GoogleSearch())


# ---------------------------------------------------------------------------
# GeminiUrlContextTool
# ---------------------------------------------------------------------------

class GeminiUrlContextTool(GeminiNativeTool):
    """Gemini built-in URL context tool.

    Fetches the content of one or more URLs server-side and makes the
    retrieved text available to the model as context.

    Requires a Gemini model — raises ToolIncompatibleError at Agent init if
    used with any other provider.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="url_context",
                description=(
                    "Fetch and read the content of a URL server-side.  "
                    "The retrieved page text is provided to the model as context.  "
                    "SERVER-SIDE: executed by Google, not locally.  "
                    "Only compatible with Gemini models."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="url",
                        type=ParameterType.STRING,
                        description="The URL whose content should be fetched.",
                        required=True,
                    ),
                ],
                requires_api_key=True,
                cost_estimate="low",
                tags=["url", "fetch", "context", "gemini-native", "server-side"],
            )
        )

    def to_gemini_tool(self) -> Any:
        """Return the SDK ``Tool`` that enables server-side URL context fetching."""
        from google.genai import types  # type: ignore[import]
        return types.Tool(url_context=types.UrlContext())


# ---------------------------------------------------------------------------
# GeminiCodeExecutionTool
# ---------------------------------------------------------------------------

class GeminiCodeExecutionTool(GeminiNativeTool):
    """Gemini built-in code execution tool.

    Executes Python code in a secure, server-side sandbox.  Results (stdout,
    generated files, error messages) come back as part of the model response.

    Requires a Gemini model — raises ToolIncompatibleError at Agent init if
    used with any other provider.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="code_execution",
                description=(
                    "Execute Python code in a secure Google-hosted sandbox.  "
                    "Supports numeric computation, data analysis, and more.  "
                    "SERVER-SIDE: executed by Google, not locally.  "
                    "Only compatible with Gemini models."
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
                cost_estimate="low",
                tags=["code", "python", "execution", "gemini-native", "server-side"],
            )
        )

    def to_gemini_tool(self) -> Any:
        """Return the SDK ``Tool`` that enables server-side code execution."""
        from google.genai import types  # type: ignore[import]
        return types.Tool(code_execution=types.ToolCodeExecution())
