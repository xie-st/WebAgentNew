"""The ``effgen health`` command: infrastructure checks against external services.

:mod:`effgen.cli._main` parses arguments and dispatches to this handler. The
checks contact effgen.org, docs.effgen.org and PyPI, so they are opt-in: without
``--remote`` (or ``EFFGEN_HEALTH_REMOTE=1``) the command explains what it would
contact and exits 0 without reaching the network.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def handle_health_command(args: argparse.Namespace, cli: "CLIInterface") -> int:
    """Handle the 'effgen health' subcommand."""
    # Fail-closed on privacy: contacting effgen.org / PyPI is opt-in.
    remote = (
        getattr(args, 'health_remote', False)
        or os.environ.get("EFFGEN_HEALTH_REMOTE", "").strip().lower() in ("1", "true", "yes")
    )
    if not remote:
        print("effgen health performs network checks against external services "
              "(effgen.org, docs.effgen.org, PyPI).")
        print("These are not run without consent. Re-run with --remote (or set "
              "EFFGEN_HEALTH_REMOTE=1) to enable them.")
        return 0

    from effgen.utils.health import HealthChecker
    checker = HealthChecker()
    all_passed = checker.print_results()
    return 0 if all_passed else 1
