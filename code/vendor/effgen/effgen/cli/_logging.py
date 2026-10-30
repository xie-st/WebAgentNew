"""Logging configuration for the ``effgen`` CLI.

:mod:`effgen.cli._main` owns argument parsing and dispatch and re-exports these
names, so ``effgen.cli._main`` stays the import path callers use. Holds the
default-quiet level policy, the optional ``--log-file`` handler, and the filter
that keeps a library ``ERROR`` record from printing a second time beneath a
command that already renders its own failure.
"""

from __future__ import annotations

import logging
import sys

# Commands that render their own failures (a classified message pointing at the
# fix, a red error panel). For these, a raw ``<timestamp> - <logger> - ERROR``
# console line from the library beneath the framed message is duplicate noise at
# default verbosity — it is kept only under ``--verbose`` and in a ``--log-file``.
_SELF_RENDERING_ERROR_COMMANDS = frozenset({
    "run", "batch", "chat", "eval", "compare", "debug", "resume",
    "quickstart", "tutorial",
})


class _CLIEchoedErrorFilter(logging.Filter):
    """Drop console ERROR records the CLI already surfaces as a framed message.

    Applied to the console handler only (a ``--log-file`` still captures the full
    stream) and only for the commands that render their own errors, so a library
    ``ERROR`` log does not print a second time beneath the CLI's own message.
    ``effgen serve`` is excluded — an operator wants those server-side ERRORs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR and (
            record.name == "effgen" or record.name.startswith("effgen.")
        ):
            return False
        return True


def _suppress_echoed_error_logs() -> None:
    """Attach :class:`_CLIEchoedErrorFilter` to the root console handler."""
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.addFilter(_CLIEchoedErrorFilter())


# Configure logging
def setup_logging(
    verbose: bool = False,
    log_file: str | None = None,
    quiet: bool = False,
) -> None:
    """
    Configure logging for CLI.

    The CLI is quiet by default so command output (tables, answers) stays clean
    and copy-pasteable: library diagnostics are kept at WARNING and above, and
    informational chatter from tool discovery / config loading is hidden. Use
    ``--verbose`` to surface DEBUG/INFO, or ``--quiet`` to show errors only.

    Args:
        verbose: Show DEBUG/INFO diagnostics.
        log_file: Optional log file path (always captures full DEBUG detail).
        quiet: Show errors only (suppress warnings).
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    else:
        # Default: quiet, professional CLI output — warnings and errors only.
        level = logging.WARNING

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )
    # Route library logs to stderr so they never interleave with stdout tables,
    # and keep the console at the requested level even if a log file is attached.
    logging.getLogger().setLevel(min(level, logging.DEBUG) if log_file else level)
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)
