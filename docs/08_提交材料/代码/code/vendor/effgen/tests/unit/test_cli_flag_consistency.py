"""A flag spells the same concept the same way in every command that has it.

Read from the live argument parser, not from the source, so an option added
anywhere in the parser tree is covered without being registered here.

Three properties hold across the whole command surface:

* Every option documents itself, so ``--help`` never lists a bare flag.
* A short flag names one concept. Where two long names describe the same
  concept — ``--model``/``--models`` for one model or several — they may share
  a short form; that pairing is declared below.
* A concept that has a short form somewhere has the same short form
  everywhere it appears.

The declared exceptions each carry their reason. Repointing a short flag that
users already type is a breaking change, so the two pre-existing collisions are
recorded below rather than silently accepted.
"""

from __future__ import annotations

import argparse

from effgen.cli._main import create_parser

# Long names that describe one concept, and so may share a short form.
_SAME_CONCEPT: tuple[frozenset[str], ...] = (
    # One model, or several: `battle`/`compare` take a list, everything else one.
    frozenset({"--model", "--models"}),
    # Where output goes: a file for most commands, a directory for the
    # scaffolding command that writes several.
    frozenset({"--output", "--output-dir"}),
)

# Short flags bound to two genuinely different concepts before this check
# existed. Repointing either would break a command line users already type, so
# each is recorded with the commands it separates. Nothing may be added here.
# Frozen, not pending. Repointing either would break a documented command line
# (`effgen code -p`, `effgen run -c FILE`) that appears in shipped examples, for
# a cosmetic gain; the long forms are the portable spelling and
# `docs/dx/cli.md` says so under "Short flags are per-command". What this map
# buys is that the situation cannot get *worse*: a third concept reaching for
# `-c` or `-p`, or a new collision on any other short flag, fails the build.
_KNOWN_COLLISIONS: dict[str, frozenset[str]] = {
    "-c": frozenset({"--concurrency", "--config"}),
    "-p": frozenset({"--port", "--print"}),
}

# A long name that is an alias of another on the same option, and takes the
# short form of the name it is an alias of.
_ALIAS_LONG_NAMES: dict[str, str] = {
    # `run --input PATH` is a second spelling of `run --file PATH`; the option
    # is `--file`, and carries `-f`.
    "run:--input": "--file",
}


def _walk(parser: argparse.ArgumentParser, path: list[str]):
    yield path, parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield from _walk(sub, [*path, name])


def _options() -> list[dict]:
    """Every option on every command, with its long names, shorts and help."""
    rows: list[dict] = []
    for path, parser in _walk(create_parser(), []):
        command = " ".join(path) or "(root)"
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                continue
            if not action.option_strings:
                continue
            rows.append({
                "command": command,
                "longs": [o for o in action.option_strings if o.startswith("--")],
                "shorts": [o for o in action.option_strings if not o.startswith("--")],
                "help": (action.help or "").strip(),
            })
    return rows


OPTIONS = _options()


def test_the_parser_tree_was_walked() -> None:
    """The walk reaches the whole command surface, not just the root parser."""
    commands = {row["command"] for row in OPTIONS}
    assert len(commands) >= 60, f"only {len(commands)} commands walked"
    assert len(OPTIONS) >= 300, f"only {len(OPTIONS)} options walked"


def test_every_option_documents_itself() -> None:
    undocumented = [
        f"  {row['command']}  {row['longs'] or row['shorts']}"
        for row in OPTIONS
        if not row["help"]
    ]
    assert not undocumented, (
        f"{len(undocumented)} options would list as a bare flag in --help:\n"
        + "\n".join(sorted(undocumented))
    )


def test_a_short_flag_names_one_concept() -> None:
    bindings: dict[str, set[str]] = {}
    where: dict[str, list[str]] = {}
    for row in OPTIONS:
        primary = row["longs"][0] if row["longs"] else ""
        for short in row["shorts"]:
            if short == "-h":
                continue
            bindings.setdefault(short, set()).add(primary)
            where.setdefault(short, []).append(f"{row['command']} {primary}")

    problems = []
    for short, longs in sorted(bindings.items()):
        if len(longs) < 2:
            continue
        if any(longs <= concept for concept in _SAME_CONCEPT):
            continue
        if _KNOWN_COLLISIONS.get(short) == longs:
            continue
        problems.append(
            f"  {short} -> {sorted(longs)}\n"
            + "\n".join(f"      {site}" for site in sorted(where[short]))
        )
    assert not problems, (
        "a short flag means different things in different commands:\n"
        + "\n".join(problems)
    )


def test_a_concept_keeps_its_short_form_everywhere() -> None:
    forms: dict[str, dict[str, tuple[str, ...]]] = {}
    for row in OPTIONS:
        for long_name in row["longs"]:
            if long_name == "--help":
                continue
            resolved = _ALIAS_LONG_NAMES.get(f"{row['command']}:{long_name}", long_name)
            if resolved != long_name:
                continue
            forms.setdefault(long_name, {})[row["command"]] = tuple(
                sorted(s for s in row["shorts"] if s != "-h")
            )

    problems = []
    for long_name, by_command in sorted(forms.items()):
        if len(set(by_command.values())) < 2:
            continue
        problems.append(
            f"  {long_name}\n"
            + "\n".join(
                f"      {command:26s} {list(shorts) or '(no short form)'}"
                for command, shorts in sorted(by_command.items())
            )
        )
    assert not problems, (
        "a flag carries a short form in some commands and not others:\n"
        + "\n".join(problems)
    )


def test_the_per_command_short_flags_are_documented():
    """The decision is "leave it and document it", so the document must exist.

    Option 3 of the three the report offered: repointing `-p` or `-c` breaks a
    documented command line for a cosmetic gain, so the collision stays and the
    long form is named as the portable spelling. That is only a real answer if a
    reader can find it, so the doc is part of the contract.
    """
    from pathlib import Path

    doc = Path(__file__).resolve().parents[2] / "docs" / "dx" / "cli.md"
    text = doc.read_text(encoding="utf-8")
    assert "Short flags are per-command" in text
    for token in ("--concurrency", "--config", "--port", "--print"):
        assert token in text, f"{token} is not named in the short-flag section"
    # Every collision the gate knows about is described, so the two cannot drift.
    for short in _KNOWN_COLLISIONS:
        assert f"`{short}`" in text, f"{short} is pinned but undocumented"
