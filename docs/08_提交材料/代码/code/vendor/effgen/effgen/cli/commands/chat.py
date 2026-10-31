"""The ``effgen chat`` command.

Validates the requested provider and hands off to the REPL implementation in
:mod:`effgen.cli.chat` (a different module: that one holds ``ChatREPL``, this
one holds the command body). :mod:`effgen.cli._main` parses the arguments and
exposes this through ``CLIInterface.chat_mode``.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from effgen.cli.commands._shared import resolve_provider_name

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def chat_mode(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Interactive chat REPL.

    Delegates to :class:`effgen.cli.chat.ChatREPL`, which provides streaming
    answers with a thinking spinner, a model/tool-aware prompt, slash
    commands (``/model``, ``/tools``, ``/cost``, ``/trace``, …), persistent
    ↑/↓ history, multiline input, and a per-turn Ctrl-C that cancels the
    current turn without exiting the session.
    """
    # Validate an explicit --provider up front (a typo like "grok" should
    # fail fast with a suggestion, exactly as ``run`` does).
    provider, prov_err = resolve_provider_name(getattr(args, "provider", None))
    if prov_err:
        cli.print_error(prov_err)
        return 1
    try:
        args._provider = provider
    except Exception:  # noqa: BLE001 - argparse Namespace always allows this
        pass

    from effgen.cli.chat import ChatREPL

    try:
        return ChatREPL(cli, args).run()
    except Exception as e:  # noqa: BLE001
        cli.print_error(f"Error in chat mode: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1
