"""The ``effgen presets`` command: list the agent presets and their tool cost.

:mod:`effgen.cli._main` parses arguments and dispatches to this handler. Prints
each preset's name, description, tool count and approximate per-call tool-schema
size, as a table or as JSON.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from effgen.ui.render import json_ensure_ascii

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def handle_presets_command(args: argparse.Namespace, cli: "CLIInterface") -> int:
    """Handle the 'effgen presets' subcommand."""
    from effgen.presets import list_presets as _list_presets
    from effgen.presets.registry import preset_tool_overhead
    if getattr(args, 'output_json', False):
        rows = []
        for name, desc in _list_presets().items():
            try:
                n_tools, approx = preset_tool_overhead(name)
            except Exception:  # noqa: BLE001 - listing never fails on one preset
                n_tools, approx = 0, 0
            rows.append({
                "name": name,
                "description": desc,
                "tool_count": n_tools,
                "approx_tokens_per_call": approx,
            })
        print(json.dumps(rows, indent=2, ensure_ascii=json_ensure_ascii()))
        return 0

    cli.print_header("Available Agent Presets")
    for name, desc in _list_presets().items():
        try:
            n_tools, approx = preset_tool_overhead(name)
        except Exception:  # noqa: BLE001 - listing never fails on one preset
            n_tools, approx = 0, 0
        if n_tools:
            overhead = f"{n_tools} tool{'s' if n_tools != 1 else ''} · ~{approx} tok/call"
        else:
            overhead = "no tools"
        cli.print(f"  {name:12s}  {overhead}")
        cli.print(f"               {desc}")
    cli.print(
        "\n'~N tok/call' is the approximate tool-schema size sent on every "
        "request — a tool-heavy preset costs more per call and can exceed "
        "a small-context or rate-limited model."
    )
    cli.print("\nUsage: effgen run --preset <name> \"your task\"")
    return 0
