"""The ``effgen examples`` command group: discover and run bundled examples.

``_main`` parses arguments and dispatches; the ``CLIInterface.examples_*``
methods delegate here. Holds the example discovery, listing, and run logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def examples_commands(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Run example scripts.

    Args:
        args: Parsed command-line arguments
    """
    from effgen.cli.commands._shared import _print_group_help

    if args.example_command == 'list':
        return examples_list(cli, args) or 0
    elif args.example_command == 'run':
        return examples_run(cli, args) or 0
    elif args.example_command is None:
        return _print_group_help(args)
    else:
        cli.print_error(f"Unknown examples command: {args.example_command}")
        return 1


def find_examples_dir() -> "Path | None":
    """Locate the bundled `examples/` directory.

    Examples ship with the source tree (repo root), not inside the installed
    `effgen` package, so probe several real locations rather than the old
    package-relative path that was broken for every pip-installed user.
    """
    candidates = []
    env_dir = os.environ.get("EFFGEN_EXAMPLES_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    # repo root: effgen/cli/commands/examples.py -> <repo>/examples
    candidates.append(Path(__file__).resolve().parent.parent.parent.parent / "examples")
    # current working directory (running from a checkout)
    candidates.append(Path.cwd() / "examples")
    for c in candidates:
        if c.is_dir() and any(c.rglob("*.py")):
            return c
    return None


def examples_list(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """List available examples."""
    cli.print_header("Available Examples")

    examples_dir = cli._find_examples_dir()

    if examples_dir is None:
        cli.print_warning("No examples directory found.")
        cli.print(
            "Examples ship with the source repository, not the installed wheel.\n"
            "Run from a cloned checkout (repo root), or set "
            "EFFGEN_EXAMPLES_DIR=/path/to/examples."
        )
        return 0

    # Examples are grouped into subdirectories (basic/, advanced/, …), so
    # walk the tree and present each as its path relative to examples/.
    examples = []
    for file in examples_dir.rglob("*.py"):
        if file.name.startswith("_"):
            continue
        rel = file.relative_to(examples_dir).with_suffix("")
        examples.append(rel.as_posix())

    if cli.console:
        from rich.table import Table
        table = Table(title=f"Example Scripts ({len(examples)})")
        table.add_column("Name", style="cyan")
        table.add_column("Command", style="magenta")

        for example in sorted(examples):
            table.add_row(example, f"effgen examples run {example}")

        cli.console.print(table)
    else:
        for example in sorted(examples):
            print(f"- {example}")
    return None


def examples_run(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Run an example script."""
    if not args.name:
        cli.print_error("Example name required")
        return 1

    examples_dir = cli._find_examples_dir()
    if examples_dir is None:
        cli.print_error(
            "No examples directory found. Run from a source checkout or set "
            "EFFGEN_EXAMPLES_DIR=/path/to/examples."
        )
        return 1
    # Accept either a bare name or a subdir path (e.g. "basic/quickstart").
    name = args.name[:-3] if args.name.endswith(".py") else args.name
    example_path = (examples_dir / f"{name}.py").resolve()
    # Path-traversal guard: stay within the examples directory.
    try:
        example_path.relative_to(examples_dir.resolve())
    except ValueError:
        cli.print_error(f"Invalid example path: {args.name}")
        return 1

    if not example_path.exists():
        # Try to match by basename across subdirectories for convenience.
        matches = list(examples_dir.rglob(f"{Path(name).name}.py"))
        if len(matches) == 1:
            example_path = matches[0]
        else:
            cli.print_error(f"Example not found: {args.name}")
            if matches:
                cli.print("Did you mean one of:")
                for m in matches:
                    cli.print(f"  {m.relative_to(examples_dir).with_suffix('').as_posix()}")
            return 1

    cli.print_header(f"Running Example: {args.name}")
    cli.print()

    # Load and run example
    try:
        spec = importlib.util.spec_from_file_location("example", example_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Run main function if exists
        if hasattr(module, 'main'):
            module.main()
        else:
            cli.print_warning("Example does not have a main() function")

    except Exception as e:
        cli.print_error(f"Error running example: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1
    return 0
