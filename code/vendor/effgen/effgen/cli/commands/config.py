"""The ``effgen config`` command group: show, validate, init, and set config.

``_main`` parses arguments and dispatches; the ``CLIInterface._config_*``
methods delegate here. Holds the config-file round-trip and budget-key logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from effgen.cli import onboarding as _onboarding

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def config_commands(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Configuration management commands.

    Args:
        args: Parsed command-line arguments
    """
    from effgen.cli.commands._shared import _print_group_help

    if args.config_command == 'show':
        return cli._config_show(args) or 0
    if args.config_command == 'validate':
        return cli._config_validate(args) or 0
    if args.config_command == 'init':
        return cli._config_init(args) or 0
    if args.config_command == 'set':
        return cli._config_set(args) or 0
    if args.config_command is None:
        return _print_group_help(args)
    cli.print_error(f"Unknown config command: {args.config_command}")
    return 1


def config_set(cli: "CLIInterface", args: argparse.Namespace) -> int:
    """Handle 'effgen config set <key> <value>'; non-zero when the key is refused."""
    key: str = args.key
    value_str: str = args.value

    _supported_keys = ("budget.daily", "budget.monthly")
    # Route budget.* keys to the cost tracker's budget config.
    if key.startswith("budget."):
        budget_key = key[len("budget."):]
        if budget_key not in {"daily", "monthly"}:
            hint = _onboarding.did_you_mean(key, _supported_keys, n=1, cutoff=0.5)
            cli.print_error(
                f"Unknown budget key: {key!r}. {hint + ' ' if hint else ''}"
                f"Supported: {', '.join(_supported_keys)}."
            )
            return 1
        try:
            value = float(value_str)
        except ValueError:
            cli.print_error(f"Budget value must be a number, got: {value_str!r}")
            return 1
        from effgen.models._cost import _budget_config_path
        budget_path = _budget_config_path()
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if budget_path.exists():
            try:
                existing = json.loads(budget_path.read_text())
            except Exception:
                pass
        existing[budget_key] = value
        budget_path.write_text(json.dumps(existing, indent=2))
        cli.print_success(f"Set {key} = {value}")
        return 0
    hint = _onboarding.did_you_mean(key, _supported_keys, n=1, cutoff=0.5)
    cli.print_error(
        f"Unknown config key: {key!r}. {hint + ' ' if hint else ''}"
        f"Supported: {', '.join(_supported_keys)}."
    )
    return 1


def config_show(cli: "CLIInterface", args: argparse.Namespace) -> int:
    """Show current configuration; non-zero when the file cannot be read."""
    cli.print_header("Configuration")

    if args.file:
        try:
            config = cli.config_loader.load_config(args.file)

            if cli.console:
                from rich.syntax import Syntax

                from effgen.ui.theme import CODE_THEME
                syntax = Syntax(
                    json.dumps(config.to_dict(), indent=2),
                    "json",
                    theme=CODE_THEME,
                    line_numbers=True
                )
                cli.console.print(syntax)
            else:
                print(json.dumps(config.to_dict(), indent=2))

        except Exception as e:
            cli.print_error(f"Error loading config: {e}")
            return 1
        return 0

    cli.print_warning("No configuration file specified")
    cli.print("Use: effgen config show --file <path>")
    return 1


def config_validate(cli: "CLIInterface", args: argparse.Namespace) -> int:
    """Validate a configuration file, returning non-zero when it is not valid.

    The exit code is what a CI step gates on, so a file that is refused has to
    make the command fail, not print a message and succeed.
    """
    if not args.file:
        cli.print_error("Configuration file required")
        return 1

    try:
        cli.config_loader.load_config(args.file, validate=True)
    except Exception as e:
        cli.print_error(f"Configuration validation failed: {e}")
        return 1
    cli.print_success(f"Configuration is valid: {args.file}")
    return 0


def config_init(cli: "CLIInterface", args: argparse.Namespace) -> int:
    """Write a config file a run reads; non-zero when nothing was written.

    Emits the same document ``effgen quickstart --init`` scaffolds, so the keys
    in it are the keys ``effgen run -c`` applies rather than a sample no
    command reads.
    """
    from effgen.cli import scaffold as _scaffold

    output_path = Path(args.output or "config.yaml")

    if output_path.exists() and not args.force:
        cli.print_error(f"File already exists: {output_path}")
        cli.print("Use --force to overwrite")
        return 1

    model_id, _reason = _scaffold.resolve_model()
    body = _scaffold.render_project_config(model_id, config_name=output_path.name)
    try:
        output_path.write_text(body, encoding="utf-8")
    except OSError as e:
        cli.print_error(f"Could not write {output_path}: {e}")
        return 1

    cli.print_success(f"Configuration initialized: {output_path}")
    cli.print(f"Run it with: effgen run \"{_scaffold.SAMPLE_TASK}\" -c {output_path}")
    cli.print("A whole project — this file, a .env template and a runnable "
              "example — comes from: effgen quickstart --init")
    return 0
