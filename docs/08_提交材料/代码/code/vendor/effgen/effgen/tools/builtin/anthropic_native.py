"""
Anthropic first-party computer-use tool wrappers for effGen.

**EXPERIMENTAL** — these tools require the ``computer-use-2025-11-24`` beta header
and are subject to change.  They are NOT registered in any default tool preset.

These wrappers expose Anthropic's server-side computer-use tools — bash, text_editor,
and computer — as effGen ``BaseTool`` subclasses.  Unlike effGen's local tools, these
run inside Anthropic's infrastructure when paired with a compatible Claude model.

Usage::

    from effgen.tools.builtin.anthropic_native import (
        AnthropicBashTool,
        AnthropicTextEditorTool,
        AnthropicComputerTool,
    )
    tool = AnthropicBashTool()
    spec = tool.to_anthropic_tool_spec()   # pass to AnthropicAdapter

Compatible models (as of 2026-04-28):
    claude-opus-4-7, claude-opus-4-6, claude-sonnet-4-6, claude-opus-4-5

The beta header ``"computer-use-2025-11-24"`` must be included in the request.
Pass it via ``AnthropicAdapter`` ``extra_headers`` kwarg::

    result = adapter.generate_with_tools(
        prompt, tools_specs,
        extra_headers={"anthropic-beta": "computer-use-2025-11-24"},
    )
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


class AnthropicNativeTool(BaseTool):
    """Base class for Anthropic server-side (computer-use) tools.

    **EXPERIMENTAL** — requires ``computer-use-2025-11-24`` beta header.

    These tools are executed by Anthropic's backend.  Local execution is not
    supported and raises ``RuntimeError`` if attempted.
    """

    IS_ANTHROPIC_NATIVE = True  # sentinel used by the adapter / Agent

    def to_anthropic_tool_spec(self) -> dict[str, Any]:
        """Return the Anthropic-format tool spec for this native tool."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_anthropic_tool_spec() to return "
            "the Anthropic computer-use tool dict (e.g. {'type': 'bash_20250124', 'name': 'bash'}). "
            "Use AnthropicBashTool / AnthropicTextEditorTool / AnthropicComputerTool, "
            "or override this method in your subclass."
        )

    async def _execute(self, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"'{self.name}' is an Anthropic server-side computer-use tool and cannot "
            "be executed locally.  Use an AnthropicAdapter model with this tool and "
            "include the 'computer-use-2025-11-24' beta header."
        )


# ---------------------------------------------------------------------------
# AnthropicBashTool
# ---------------------------------------------------------------------------

class AnthropicBashTool(AnthropicNativeTool):
    """Anthropic hosted bash execution tool.

    **EXPERIMENTAL** — requires ``computer-use-2025-11-24`` beta header.

    Executes bash commands in a sandboxed environment on Anthropic's servers.
    The model can run arbitrary shell commands and receive stdout/stderr back
    as part of its context.

    Requires an Anthropic model — raises ``ToolIncompatibleError`` at Agent init
    if used with any other provider.

    tool_type (Anthropic API): ``"bash_20250124"``

    Note:
        The beta header ``"computer-use-2025-11-24"`` must be included in every
        request that uses this tool.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="anthropic_bash",
                description=(
                    "[EXPERIMENTAL] Execute bash commands in an Anthropic-hosted sandbox. "
                    "Returns stdout, stderr, and exit code. "
                    "SERVER-SIDE: executed by Anthropic, not locally. "
                    "Requires 'computer-use-2025-11-24' beta header. "
                    "Only compatible with Anthropic models."
                ),
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="command",
                        type=ParameterType.STRING,
                        description="The bash command to execute.",
                        required=True,
                    ),
                    ParameterSpec(
                        name="restart",
                        type=ParameterType.BOOLEAN,
                        description="If true, restart the bash tool to clear state.",
                        required=False,
                    ),
                ],
                requires_api_key=True,
                cost_estimate="medium",
                tags=["bash", "shell", "exec", "anthropic-native", "server-side", "experimental", "computer-use"],
            )
        )

    def to_anthropic_tool_spec(self) -> dict[str, Any]:
        """Return the Anthropic computer-use bash tool spec (type ``bash_20250124``)."""
        return {"type": "bash_20250124", "name": "bash"}


# ---------------------------------------------------------------------------
# AnthropicTextEditorTool
# ---------------------------------------------------------------------------

class AnthropicTextEditorTool(AnthropicNativeTool):
    """Anthropic hosted text editor tool.

    **EXPERIMENTAL** — requires ``computer-use-2025-11-24`` beta header.

    Allows the model to view and edit files in a server-side environment.
    Supports operations: ``view``, ``create``, ``str_replace``, ``insert``,
    ``undo_edit``.

    Requires an Anthropic model — raises ``ToolIncompatibleError`` at Agent init
    if used with any other provider.

    tool_type (Anthropic API): ``"text_editor_20250728"``

    Note:
        The beta header ``"computer-use-2025-11-24"`` must be included in every
        request that uses this tool.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="anthropic_text_editor",
                description=(
                    "[EXPERIMENTAL] View and edit files in an Anthropic-hosted environment. "
                    "Supports view, create, str_replace, insert, undo_edit operations. "
                    "SERVER-SIDE: executed by Anthropic, not locally. "
                    "Requires 'computer-use-2025-11-24' beta header. "
                    "Only compatible with Anthropic models."
                ),
                category=ToolCategory.FILE_OPERATIONS,
                parameters=[
                    ParameterSpec(
                        name="command",
                        type=ParameterType.STRING,
                        description=(
                            "The editor command to run: 'view', 'create', "
                            "'str_replace', 'insert', or 'undo_edit'."
                        ),
                        required=True,
                    ),
                    ParameterSpec(
                        name="path",
                        type=ParameterType.STRING,
                        description="Absolute path to the file to operate on.",
                        required=True,
                    ),
                ],
                requires_api_key=True,
                cost_estimate="low",
                tags=["file", "editor", "text", "anthropic-native", "server-side", "experimental", "computer-use"],
            )
        )

    def to_anthropic_tool_spec(self) -> dict[str, Any]:
        """Return the Anthropic computer-use text editor tool spec (type ``text_editor_20250728``)."""
        return {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}


# ---------------------------------------------------------------------------
# AnthropicComputerTool
# ---------------------------------------------------------------------------

class AnthropicComputerTool(AnthropicNativeTool):
    """Anthropic hosted computer-use tool.

    **EXPERIMENTAL** — requires ``computer-use-2025-11-24`` beta header.

    Gives the model full computer control: mouse clicks, keyboard input,
    screenshots, and scrolling in a server-side virtual display.

    Requires an Anthropic model — raises ``ToolIncompatibleError`` at Agent init
    if used with any other provider.

    tool_type (Anthropic API): ``"computer_20251124"``

    Compatible models (as of 2026-04-28):
        claude-opus-4-7, claude-opus-4-6, claude-sonnet-4-6, claude-opus-4-5

    Args:
        display_width_px: Width of the virtual display in pixels (default 1024).
        display_height_px: Height of the virtual display in pixels (default 768).
        display_number: X display number (Linux only, default 1).

    Note:
        The beta header ``"computer-use-2025-11-24"`` must be included in every
        request that uses this tool.  Screenshot resolution directly affects
        token costs; keep resolution as low as practical.
    """

    def __init__(
        self,
        display_width_px: int = 1024,
        display_height_px: int = 768,
        display_number: int | None = 1,
    ) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="anthropic_computer",
                description=(
                    "[EXPERIMENTAL] Control a virtual computer: mouse, keyboard, screenshots. "
                    "SERVER-SIDE: executed by Anthropic, not locally. "
                    "Requires 'computer-use-2025-11-24' beta header. "
                    "Only compatible with Anthropic models (Opus 4.7/4.6, Sonnet 4.6, Opus 4.5)."
                ),
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="action",
                        type=ParameterType.STRING,
                        description=(
                            "Computer action: 'screenshot', 'left_click', 'right_click', "
                            "'double_click', 'type', 'key', 'scroll', 'mouse_move', "
                            "'left_click_drag', 'cursor_position'."
                        ),
                        required=True,
                    ),
                    ParameterSpec(
                        name="coordinate",
                        type=ParameterType.ARRAY,
                        description="[x, y] pixel coordinates for click/move actions.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="text",
                        type=ParameterType.STRING,
                        description="Text to type or key sequence to press.",
                        required=False,
                    ),
                ],
                requires_api_key=True,
                cost_estimate="high",
                tags=["computer", "mouse", "keyboard", "screenshot", "anthropic-native", "server-side", "experimental", "computer-use"],
            )
        )
        self.display_width_px = display_width_px
        self.display_height_px = display_height_px
        self.display_number = display_number

    def to_anthropic_tool_spec(self) -> dict[str, Any]:
        """Return the Anthropic computer-use computer tool spec (type ``computer_20251124``)."""
        spec: dict[str, Any] = {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": self.display_width_px,
            "display_height_px": self.display_height_px,
        }
        if self.display_number is not None:
            spec["display_number"] = self.display_number
        return spec
