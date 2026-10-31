# Anthropic Native Computer-Use Tools

> **EXPERIMENTAL**
>
> These tools wrap Anthropic's server-side computer-use capabilities.  They require
> the `computer-use-2025-11-24` beta header and are subject to breaking changes by
> Anthropic at any time.  They are **not** registered in any default tool preset and
> are not enabled unless you explicitly import and pass them.
>
> Anthropic features in effGen v0.2.2 are unit-tested but **not live-tested**
> (no Anthropic API key available in the dev environment).  Validate with your own
> key before deploying.

## Overview

effGen exposes three Anthropic computer-use tools as `BaseTool` subclasses:

| Class | Anthropic `type` | Purpose |
|---|---|---|
| `AnthropicBashTool` | `bash_20250124` | Execute bash commands server-side |
| `AnthropicTextEditorTool` | `text_editor_20250728` | View and edit files server-side |
| `AnthropicComputerTool` | `computer_20251124` | Full computer control (mouse, keyboard, screenshots) |

These tools run **inside Anthropic's infrastructure**.  Local execution raises
`RuntimeError`.  Each requires the `ToolIncompatibleError` guard — if you pass
them to an Agent backed by a non-Anthropic model, Agent init raises `ToolIncompatibleError`.

Compatible models (as of 2026-04-28):
- `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-5`

## Installation note

Computer-use tools require the `computer-use-2025-11-24` beta header on every request.
Pass it via `extra_headers`:

```python
result = adapter.generate_with_tools(
    prompt,
    tools=[spec],
    extra_headers={"anthropic-beta": "computer-use-2025-11-24"},
)
```

## Usage

```python
from effgen.tools.builtin.anthropic_native import (
    AnthropicBashTool,
    AnthropicTextEditorTool,
    AnthropicComputerTool,
)

bash_tool = AnthropicBashTool()
editor_tool = AnthropicTextEditorTool()
computer_tool = AnthropicComputerTool(display_width_px=1920, display_height_px=1080)

# Get the tool spec to pass to the API
bash_spec = bash_tool.to_anthropic_tool_spec()
# → {"type": "bash_20250124", "name": "bash"}
```

### With AnthropicAdapter

```python
from effgen.models import AnthropicAdapter
from effgen.tools.builtin.anthropic_native import AnthropicBashTool

tool = AnthropicBashTool()
spec = tool.to_anthropic_tool_spec()

with AnthropicAdapter(model_name="claude-sonnet-4-6") as adapter:
    result = adapter.generate_with_tools(
        "List the files in /tmp and show their sizes.",
        tools=[spec],
        extra_headers={"anthropic-beta": "computer-use-2025-11-24"},
    )
    for call in result.metadata["tool_uses"]:
        print(f"Tool: {call['name']}, Input: {call['input']}")
```

### ToolIncompatibleError guard

```python
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin.anthropic_native import AnthropicBashTool
from effgen.models import GeminiAdapter  # wrong provider

model = GeminiAdapter(model_name="gemini-2.5-flash")
model.load()

# Raises ToolIncompatibleError at init — not at call time
agent = Agent(config=AgentConfig(model=model, tools=[AnthropicBashTool()]))
#  ^ ToolIncompatibleError: 'anthropic_bash' is incompatible with 'gemini-2.5-flash'
```

## Tool reference

### AnthropicBashTool

Executes bash commands in a sandboxed server-side shell.

**Spec:** `{"type": "bash_20250124", "name": "bash"}`

**Input schema (Anthropic-side):**
- `command` (str): The bash command to run.
- `restart` (bool, optional): Restart the bash session (clears state).

### AnthropicTextEditorTool

Views and edits files in a server-side filesystem.

**Spec:** `{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}`

**Supported commands:** `view`, `create`, `str_replace`, `insert`, `undo_edit`

### AnthropicComputerTool

Provides full computer control via mouse, keyboard, and screenshots.

**Spec:**
```python
{
    "type": "computer_20251124",
    "name": "computer",
    "display_width_px": 1024,   # default
    "display_height_px": 768,   # default
    "display_number": 1,        # Linux X display; set to None to omit
}
```

**Supported actions:** `screenshot`, `left_click`, `right_click`, `double_click`,
`type`, `key`, `scroll`, `mouse_move`, `left_click_drag`, `cursor_position`

**Cost note:** Screenshots are billed as vision tokens.  Keep display resolution as
low as practical to control costs.

## Safety and limitations

- These tools give the model significant system access.  Review Anthropic's
  [computer use safety guidance](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)
  before deploying in production.
- The `computer-use-2025-11-24` beta header is required; requests without it will
  receive a 400 error from the Anthropic API.
- Tool types (e.g., `bash_20250124`) are versioned by Anthropic and may change.
  Check [Anthropic's model compatibility matrix](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)
  for the latest type strings.
