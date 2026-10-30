"""How a library warning reaches the terminal.

``warnings.warn`` is the right mechanism for a library: it is filterable,
catchable, and ``-W`` / ``PYTHONWARNINGS`` work on it. Its *default rendering*
is the problem. Python prints the file and line that raised, plus the source
line that happened to be executing, so a one-line heads-up about a model's
price arrives mid-answer as::

    /usr/lib/python3.11/site-packages/effgen/core/agent_streaming.py:124:
    UserWarning: effGen budget: no published price for 'groq:allam-2-7b' ...
      for token in stream_iter:

which points the reader at internals they did not write. The message is fine;
only the frame around it is wrong, so the fix belongs here at the entry point
rather than at the several dozen call sites.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

#: Warnings raised from inside this package are the product talking to its
#: user. Anything else — a deprecation from a dependency, say — keeps Python's
#: rendering, because its file and line are what makes it actionable.
_OWN_PACKAGE = "effgen"


def _is_ours(filename: str) -> bool:
    """Whether *filename* belongs to this package."""
    parts = os.path.normpath(filename).split(os.sep)
    return _OWN_PACKAGE in parts


def _wants_plain_output() -> bool:
    """Whether the line should carry no colour."""
    if os.environ.get("NO_COLOR"):
        return True
    try:
        return not sys.stderr.isatty()
    except (AttributeError, ValueError):
        return True


def install(force: bool = False) -> bool:
    """Render this package's own warnings as one line, without the frame.

    Args:
        force: Install even when the user has asked for specific warning
            behaviour. Off by default so ``-W`` and ``PYTHONWARNINGS`` keep
            working: someone who set them wants Python's rendering.

    Returns:
        True when the hook was installed.
    """
    if not force and (sys.warnoptions or os.environ.get("PYTHONWARNINGS")):
        return False

    previous = warnings.showwarning

    def show(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        if not (issubclass(category, UserWarning) and _is_ours(filename)):
            previous(message, category, filename, lineno, file, line)
            return

        stream = file if file is not None else sys.stderr
        text = str(message)
        try:
            if _wants_plain_output():
                print(f"warning: {text}", file=stream)
            else:
                from effgen.ui.theme import get_console

                # stderr, so a piped stdout stays free of it; a bare
                # rich.Console has no effgen.* styles and would raise.
                get_console(stderr=True).print(
                    f"[effgen.warning]warning:[/effgen.warning] {text}",
                    highlight=False,
                )
        except Exception:  # noqa: BLE001 - a warning must never be the failure
            print(f"warning: {text}", file=sys.stderr)

    warnings.showwarning = show
    return True
