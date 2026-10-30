"""
effGen Agent Presets — Ready-to-use agent configurations.

Provides factory functions for common agent types:
- math: Calculator + PythonREPL for mathematical tasks
- research: WebSearch + URLFetch + Wikipedia + academic literature tools
- coding: CodeExecutor + PythonREPL + FileOperations + BashTool for coding tasks
- general: All available tools for general-purpose tasks
- minimal: No tools, direct model inference only

Usage:
    from effgen.presets import create_agent, list_presets
    from effgen import load_model

    model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
    agent = create_agent("math", model)
    result = agent.run("What is the square root of 144?")
"""

# Side-effect imports: register additional presets into PRESETS dict
from effgen.presets import media as _media_preset  # noqa: F401
from effgen.presets import multimodal as _multimodal_preset  # noqa: F401
from effgen.presets import notify as _notify_preset  # noqa: F401
from effgen.presets.registry import (
    PRESETS,
    UnknownPresetError,
    _refresh_create_agent_doc,
    create_agent,
    get_preset,
    list_presets,
)

# All bundled presets (including the side-effect ones above) are now registered;
# regenerate create_agent's docstring so it lists every preset.
_refresh_create_agent_doc()

__all__ = [
    "create_agent",
    "get_preset",
    "list_presets",
    "PRESETS",
    "UnknownPresetError",
]
