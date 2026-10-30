"""The branded landing shown for a bare ``effgen`` at an interactive terminal.

Typing ``effgen`` with no command on a real terminal presents the product
identity (logo + version) and a short list of quick actions, then reads one
keypress and routes it to the matching command. Pressing Enter continues into
the interactive setup wizard, so the historical bare-command flow is one keypress
away and still reachable.

This surface is TTY-only. It renders only when stdout *and* stdin are terminals,
``rich`` is present, the terminal reports itself as such, and the user did not ask
for ``--quiet``. Piped/redirected output, CI, ``TERM=dumb``, and a rich-less
install all fall through to the current wizard path unchanged — the caller
(:func:`effgen.cli._main.main`) gates on :func:`should_show` and only then calls
:func:`run`. ``NO_COLOR`` still renders the landing, without color.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from effgen import __version__
from effgen.cli import onboarding as _onboarding
from effgen.ui import brand
from effgen.ui.palette import glyph
from effgen.ui.theme import resolve_theme_name, rich_available

__all__ = ["should_show", "run"]

# Sentinels for the two actions that are not a subcommand dispatch.
_WIZARD = "__wizard__"
_HELP = "__help__"
_EXIT = "__exit__"

# The quick actions, in display order: (key, one-line description, target). The
# target is a subcommand name, or a sentinel handled inline.
_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("Enter", "set up and run an agent (interactive wizard)", _WIZARD),
    ("c", "chat — open an interactive session", "chat"),
    ("e", "code — write, run and fix code in this directory", "code"),
    ("q", "quickstart — a 2-minute guided first run", "quickstart"),
    ("d", "doctor — check which providers are ready", "doctor"),
    ("m", "models — browse the model catalog", "models"),
    ("h", "help — all commands", _HELP),
    ("x", "exit", _EXIT),
)

# What a typed choice resolves to. Both the single-key shortcut and the full
# command word are accepted; an empty line (Enter) continues into the wizard.
_CHOICES: dict[str, str] = {
    "": _WIZARD,
    "c": "chat", "chat": "chat",
    "e": "code", "code": "code",
    "q": "quickstart", "quickstart": "quickstart",
    "d": "doctor", "doctor": "doctor",
    "m": "models", "models": "models",
    "h": _HELP, "help": _HELP, "?": _HELP,
    "x": _EXIT, "exit": _EXIT, "quit": _EXIT,
}

_MAX_ATTEMPTS = 3


def should_show(cli: Any, args: Any) -> bool:
    """True when the branded landing should replace the bare-command wizard drop.

    Requires: no subcommand, not ``--quiet``, ``rich`` importable, an interactive
    session (stdout and stdin both terminals, not CI), and a console that reports
    itself as a terminal (so ``TERM=dumb`` falls through).
    """
    if getattr(args, "command", None) is not None:
        return False
    if getattr(args, "quiet", False):
        return False
    if not rich_available():
        return False
    if not _onboarding.is_interactive():
        return False
    console = getattr(cli, "console", None)
    if console is None or not getattr(console, "is_terminal", False):
        return False
    return True


def _render(console: Any) -> None:
    """Draw the logo, version/environment lines, and the quick-action list.

    Every line is printed with value auto-highlighting off. Rich otherwise
    recolors numbers and brackets inside an already-styled line, which splits
    ``effGen v0.3.2`` into three differently coloured runs and bolds the
    parentheses in an action description.
    """
    import platform

    from effgen.ui.render import ascii_fold

    stream = getattr(console, "file", None)
    theme = resolve_theme_name()
    py = platform.python_version()

    console.print()
    console.print(brand.logo_renderable(console=console))
    console.print()
    console.print(
        ascii_fold(f"effGen v{__version__} · agents on small (and cloud) models", stream),
        style="effgen.accent",
        highlight=False,
    )
    console.print(
        ascii_fold(f"Python {py} · theme: {theme} · docs.effgen.org", stream),
        style="effgen.muted",
        highlight=False,
    )
    console.print()
    console.print("What next?", style="effgen.heading", highlight=False)
    for key, desc, _target in _ACTIONS:
        console.print(
            f"  [effgen.accent]\\[{key}][/effgen.accent]  {ascii_fold(desc, stream)}",
            highlight=False,
        )
    console.print()


def _prompt(console: Any) -> str:
    """Read one choice, echoing a themed prompt. Empty string on end-of-input."""
    arrow = glyph("prompt", getattr(console, "file", None)) or ">"
    try:
        return str(console.input(f"choice [effgen.prompt]{arrow}[/effgen.prompt] ")).strip().lower()
    except EOFError:
        console.print()
        return "exit"


def run(
    cli: Any,
    parser: Any,
    args: Any,
    dispatch: Callable[[Any, Any, Any], int],
) -> int:
    """Render the landing, read one action, and route it to a handler.

    *dispatch* is :func:`effgen.cli._main._dispatch`; the landing calls it so a
    quick action reaches the exact handler the typed command would. Returns the
    dispatched command's exit code (or 0 for help/exit).

    Args:
        cli: The CLI interface the landing draws on.
        parser: The argument parser a chosen action is re-parsed with.
        args: The parsed arguments of the bare invocation.
        dispatch: The command dispatcher a chosen action is routed to.
    """
    console = cli.console

    # On the very first bare run, mark the one-time welcome as seen so the plain
    # welcome text does not also fire on the next command; the landing itself is
    # the branded welcome.
    if _onboarding.first_run_pending():
        _onboarding.mark_welcomed()

    _render(console)

    for _attempt in range(_MAX_ATTEMPTS):
        choice = _prompt(console)
        target = _CHOICES.get(choice)
        if target is None:
            hint = _onboarding.did_you_mean(
                choice, [c for c in _CHOICES if c and len(c) > 1], n=1, cutoff=0.5
            )
            tail = f" {hint}" if hint else ""
            console.print(
                f"[effgen.warning]{glyph('warning')}[/effgen.warning] "
                f"Unknown choice '{choice}'.{tail} "
                "Press Enter for the wizard, or [effgen.accent]h[/effgen.accent] for help.",
                highlight=False,
            )
            continue
        if target == _WIZARD:
            return dispatch(args, cli, parser)
        if target == _HELP:
            parser.print_help()
            return 0
        if target == _EXIT:
            return 0
        sub_args = parser.parse_args([target])
        return dispatch(sub_args, cli, parser)

    # Too many unrecognized entries: show help and leave without side effects.
    parser.print_help()
    return 0
