"""
Library logging-hygiene tests.

A library must never emit log records or configure logging on import. These
tests assert the package logger carries a NullHandler, that importing effGen and
listing tools produce no log records even when the *root* logger is configured
at INFO, and that third-party faiss INFO noise is pinned to WARNING.
"""

from __future__ import annotations

import logging
import subprocess
import sys


def test_package_logger_has_null_handler():
    import effgen  # noqa: F401

    handlers = logging.getLogger("effgen").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers), (
        "effgen package logger must carry a NullHandler so import is silent"
    )


def test_faiss_loggers_pinned_to_warning():
    import effgen  # noqa: F401

    for name in ("faiss", "faiss.loader"):
        assert logging.getLogger(name).level >= logging.WARNING, (
            f"{name} logger must be >= WARNING to avoid INFO noise on import"
        )


def _run_capture(code: str) -> str:
    """Run *code* in a fresh interpreter with root logging at INFO; return stderr."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.stdout + proc.stderr


def test_import_emits_no_log_records():
    # basicConfig(INFO) sends any leaked record to stderr — there must be none.
    out = _run_capture(
        "import logging; logging.basicConfig(level=logging.INFO); "
        "import effgen; print('OK')"
    )
    assert "OK" in out
    leaked = [
        ln
        for ln in out.splitlines()
        if ln.strip() and ln.strip() != "OK"
    ]
    assert not leaked, f"importing effgen leaked log lines: {leaked}"


def test_tool_listing_emits_no_info():
    out = _run_capture(
        "import logging; logging.basicConfig(level=logging.INFO); "
        "from effgen.tools import get_registry; "
        "n=len(get_registry().list_tools()); print('TOOLS', n)"
    )
    assert "TOOLS" in out
    leaked = [
        ln
        for ln in out.splitlines()
        if ln.strip() and not ln.startswith("TOOLS")
    ]
    assert not leaked, f"listing tools leaked log lines: {leaked}"
