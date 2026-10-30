"""effGen CLI package — re-exports the main entry points."""

from effgen.cli._main import (
    CLIInterface,
    _handle_cost_command,
    _handle_prompts_command,
    agent_main,
    main,
    web_agent_main,
)

__all__ = [
    "main",
    "agent_main",
    "web_agent_main",
    "CLIInterface",
    "_handle_cost_command",
    "_handle_prompts_command",
]
