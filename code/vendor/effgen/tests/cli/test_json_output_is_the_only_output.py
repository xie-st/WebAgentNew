"""A ``--json`` command writes its document and nothing else.

``effgen presets --json`` builds every preset's tool set to report how many tools
it carries and what their schemas cost per call. Building a tool whose optional
backend is absent logs a heads-up from the tool's own module — a real message on
a real install, and one that goes to stderr while the JSON document goes to
stdout. Anything reading the command as a single stream (``2>&1``, a CI log
collector, a subprocess that merges both) then gets a line of log text in front
of the payload and cannot parse it.

The listing helper reports nothing but two numbers to its caller, so it is quiet
for the duration of the inspection. What is pinned here is that the command's
stderr is empty even when a backend is missing, on every ``--json`` command that
builds tools to answer.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys

import pytest

JSON_COMMANDS = [
    ["presets", "--json"],
    ["tools", "list", "--json"],
    ["models", "list", "--json"],
]


@pytest.fixture
def no_sentence_transformers(monkeypatch):
    """Make the retrieval tool's optional embedding backend look uninstalled.

    It is the backend a default install is most likely to be without, and the
    tool it belongs to is in two presets, so building the preset list logs the
    fallback heads-up.
    """
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "sentence_transformers":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    rc = 0
    argv_before = sys.argv
    sys.argv = ["effgen", *argv]
    from effgen.cli._main import main as cli_main

    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli_main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = argv_before
    return rc, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("argv", JSON_COMMANDS, ids=lambda a: " ".join(a))
def test_nothing_is_written_beside_the_document(argv, no_sentence_transformers):
    rc, out, err = _run(argv)
    assert rc == 0, err[-600:]
    assert err == "", f"stderr carried: {err[:600]!r}"


@pytest.mark.parametrize("argv", JSON_COMMANDS, ids=lambda a: " ".join(a))
def test_the_merged_stream_still_parses(argv, no_sentence_transformers):
    """What ``effgen ... --json 2>&1`` hands a consumer is the document alone."""
    rc, out, err = _run(argv)
    assert rc == 0
    assert isinstance(json.loads(err + out), dict | list)


def test_the_preset_listing_reports_the_tools_it_could_build(no_sentence_transformers):
    """Quieting the log must not quiet the answer: the counts are still real."""
    rc, out, _ = _run(["presets", "--json"])
    assert rc == 0
    rows = json.loads(out)
    assert any(row["tool_count"] > 0 for row in rows)


def test_the_heads_up_still_reaches_a_caller_that_builds_the_tool(
    no_sentence_transformers, caplog
):
    """Only the listing helper is quiet — building the tool to use it still says so."""
    import logging

    from effgen.tools.builtin import Retrieval

    with caplog.at_level(logging.WARNING, logger="effgen.tools.builtin.retrieval"):
        Retrieval()
    assert any("sentence-transformers" in r.getMessage() for r in caplog.records)
