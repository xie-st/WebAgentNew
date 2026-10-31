"""The ``effgen tools`` command group: list, inspect, and test tools.

``_main`` parses arguments and dispatches; the ``CLIInterface._tools_*`` and
``tools_commands`` methods delegate here. Holds the registry listing, schema
rendering, example synthesis, and single-tool execution logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface

logger = logging.getLogger(__name__)


def tools_commands(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Tool management commands.

    Args:
        args: Parsed command-line arguments
    """
    from effgen.cli.commands._shared import _print_group_help

    if args.tool_command == 'list':
        return cli._tools_list(args) or 0
    elif args.tool_command == 'info':
        return cli._tools_info(args)
    elif args.tool_command == 'test':
        return cli._tools_test(args)
    elif args.tool_command is None:
        return _print_group_help(args)
    else:
        cli.print_error(f"Unknown tools command: {args.tool_command}")
        return 1


def suggest_tool(cli: "CLIInterface", name: str) -> None:
    """Print a 'tool not found' error with close-match suggestions."""
    import difflib

    cli.print_error(f"Tool not found: {name}")
    try:
        available = cli.tool_registry.list_tools()
    except Exception:
        available = []
    close = difflib.get_close_matches(name, available, n=3, cutoff=0.5)
    if close:
        cli.print(f"Did you mean: {', '.join(close)}?")
    cli.print("Run 'effgen tools list' to see all available tools.")


def tools_list(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """List available tools."""
    # Get tools (the registry auto-discovers built-ins on first access)
    tools = cli.tool_registry.list_tools()
    category_filter = getattr(args, "category", None)

    def _meta(name):
        try:
            return cli.tool_registry.get_metadata(name)
        except Exception as e:
            logger.debug(f"Error getting metadata for {name}: {e}")
            return None

    registered = len(tools)
    known_categories: list[str] = []
    if category_filter:
        kept = []
        for name in tools:
            m = _meta(name)
            if m is None:
                continue
            if m.category.value not in known_categories:
                known_categories.append(m.category.value)
            if m.category.value == category_filter:
                kept.append(name)
        tools = kept
        known_categories.sort()

    # JSON output — machine-readable, no decorative header/table.
    if getattr(args, "output_json", False):
        out = []
        for name in tools:
            m = _meta(name)
            if m is None:
                continue
            out.append({
                "name": m.name,
                "category": m.category.value,
                "description": m.description,
                "version": getattr(m, "version", None),
            })
        print(json.dumps(out, indent=2))
        return 0

    cli.print_header("Available Tools")

    if not tools:
        # An empty registry and a filter that matched nothing are different
        # problems. Reporting the first for the second sends the reader after a
        # broken install when the registry is full one line further down.
        if category_filter and registered:
            cli.print_warning(
                f"No tools in category '{category_filter}'. "
                f"Available categories: {', '.join(known_categories)}."
            )
        else:
            cli.print_warning("No tools registered")
        return 0

    if cli.console:
        from rich.table import Table
        table = Table(title=f"Registered Tools ({len(tools)})")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white")

        for tool_name in tools:
            try:
                metadata = cli.tool_registry.get_metadata(tool_name)
                table.add_row(
                    metadata.name,
                    metadata.category.value,
                    metadata.description[:50] + "..." if len(metadata.description) > 50 else metadata.description
                )
            except Exception as e:
                logger.debug(f"Error getting metadata for {tool_name}: {e}")

        cli.console.print(table)
    else:
        for tool_name in tools:
            print(f"- {tool_name}")
    return 0


def example_input(cli: "CLIInterface", metadata: Any, tool: Any = None) -> dict:
    """Build a runnable example input for a tool from its metadata.

    Args:
        cli: The interface used to resolve display settings.
        metadata: The tool's metadata, whose schema the example is built from.
        tool: The tool instance, when one is available for better defaults.

    Returns:
        Example arguments the tool would accept.
    """
    # Prefer a curated example (drop the illustrative 'output' field).
    for ex in metadata.examples or []:
        if isinstance(ex, dict):
            curated = {k: v for k, v in ex.items() if k != "output"}
            if curated:
                return curated
    # Otherwise synthesize from required parameters.
    from effgen.tools.base_tool import ParameterType

    sample = {
        ParameterType.STRING: "example",
        ParameterType.INTEGER: 1,
        ParameterType.FLOAT: 1.0,
        ParameterType.BOOLEAN: True,
        ParameterType.ARRAY: [],
        ParameterType.OBJECT: {},
        ParameterType.ANY: "example",
    }
    example: dict = {}
    # Mirror the printed schema (to_json_schema), which excludes developer-only
    # toggles, so the copy-paste example never references a hidden parameter.
    for p in metadata.model_facing_parameters:
        if p.required or p.default is not None:
            if p.enum:
                example[p.name] = p.enum[0]
            elif p.default is not None:
                example[p.name] = p.default
            else:
                example[p.name] = sample.get(p.type, "example")
    return example


def print_tool_usage(cli: "CLIInterface", metadata: Any, tool: Any = None) -> None:
    """Print a tool's input schema and a copy-paste runnable example.

    Args:
        cli: The interface the lines are written through.
        metadata: The tool's metadata, whose schema is printed.
        tool: The tool instance, when one is available for better defaults.
    """
    cli.print("\n[bold]Input schema:[/bold]" if cli.console else "\nInput schema:")
    schema = metadata.to_json_schema()
    if cli.console:
        from rich.syntax import Syntax

        from effgen.ui.theme import CODE_THEME
        cli.console.print(Syntax(json.dumps(schema, indent=2), "json", theme=CODE_THEME))
    else:
        print(json.dumps(schema, indent=2))

    example = cli._example_input(metadata, tool)
    if example:
        cmd = f"effgen tools test {metadata.name} -i '{json.dumps(example)}'"
        cli.print("\n[bold]Example:[/bold]" if cli.console else "\nExample:")
        cli.print(f"  {cmd}")


def tools_info(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Show detailed tool information."""
    if not args.name:
        cli.print_error("Tool name required")
        return 1

    try:
        # get_metadata auto-discovers built-ins, so info works standalone.
        metadata = cli.tool_registry.get_metadata(args.name)
    except KeyError:
        cli._suggest_tool(args.name)
        return 1
    except Exception as e:
        cli.print_error(f"Error getting tool info: {e}")
        return 1

    cli.print_header(f"Tool: {metadata.name}")
    cli.print(f"\n[bold]Description:[/bold] {metadata.description}" if cli.console else f"\nDescription: {metadata.description}")
    cli.print(f"[bold]Category:[/bold] {metadata.category.value}" if cli.console else f"Category: {metadata.category.value}")
    cli.print(f"[bold]Version:[/bold] {metadata.version}" if cli.console else f"Version: {metadata.version}")

    if metadata.tags:
        cli.print(f"[bold]Tags:[/bold] {', '.join(metadata.tags)}" if cli.console else f"Tags: {', '.join(metadata.tags)}")

    # Selector aliases, if this tool accepts natural operation names.
    tool = None
    try:
        tool = cli.tool_registry.get_tool_sync(args.name, initialize=False)
    except Exception:
        tool = None
    aliases = getattr(tool, "operation_aliases", {}) if tool else {}
    if aliases:
        alias_str = ", ".join(f"{a} -> {c}" for a, c in sorted(aliases.items()))
        cli.print(f"[bold]Operation aliases:[/bold] {alias_str}" if cli.console else f"Operation aliases: {alias_str}")

    # Show parameters
    if metadata.parameters:
        cli.print("\n[bold]Parameters:[/bold]" if cli.console else "\nParameters:")
        schema = metadata.to_json_schema()
        if cli.console:
            from rich.syntax import Syntax

            from effgen.ui.theme import CODE_THEME
            cli.console.print(Syntax(json.dumps(schema, indent=2), "json", theme=CODE_THEME))
        else:
            print(json.dumps(schema, indent=2))

    # Show a runnable example.
    example = cli._example_input(metadata, tool)
    if example:
        cmd = f"effgen tools test {metadata.name} -i '{json.dumps(example)}'"
        cli.print("\n[bold]Example:[/bold]" if cli.console else "\nExample:")
        cli.print(f"  {cmd}")

    return 0


def tools_test(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Test a tool with sample input."""
    if not args.name:
        cli.print_error("Tool name required")
        return 1

    try:
        # Synchronous accessor: no asyncio.run boilerplate, and it auto-
        # discovers built-ins so 'test' works without a prior 'list'.
        tool = cli.tool_registry.get_tool_sync(args.name)
    except KeyError:
        cli._suggest_tool(args.name)
        return 1
    except Exception as e:
        cli.print_error(f"Error loading tool: {e}")
        return 1

    metadata = tool.metadata

    # No input? Show the schema and a runnable example instead of guessing.
    if not args.input:
        cli.print_header(f"Tool: {metadata.name}")
        cli.print_warning("No input provided. Supply one with -i/--input as JSON.")
        cli._print_tool_usage(metadata, tool)
        return 1

    # Parse input — must be a JSON object of parameters.
    try:
        input_data = json.loads(args.input)
    except json.JSONDecodeError as e:
        cli.print_error(f"Input must be valid JSON: {e}")
        cli._print_tool_usage(metadata, tool)
        return 1
    if not isinstance(input_data, dict):
        cli.print_error("Input must be a JSON object of parameters, e.g. '{\"expression\": \"2+2\"}'.")
        cli._print_tool_usage(metadata, tool)
        return 1

    cli.print_header(f"Testing Tool: {metadata.name}")
    cli.print(f"Input: {input_data}\n")
    try:
        result = asyncio.run(tool.execute(**input_data))
    except Exception as e:
        cli.print_error(f"Error testing tool: {e}")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        return 1

    cli.print("[bold]Result:[/bold]" if cli.console else "Result:")
    border = "green" if result.success else "red"
    if cli.console:
        from rich.panel import Panel
        cli.console.print(Panel(str(result), border_style=border))
    else:
        print(result)
    return 0 if result.success else 1
