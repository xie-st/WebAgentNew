#!/usr/bin/env python
"""Static-check ratchet: the recorded mypy result may improve, never regress.

effGen's mypy configuration is deliberately permissive, so the package does not
type-check clean today. A plain "must be clean" gate would therefore have to be
turned off, and a lane that is off cannot notice anything. This script gates on
the *recorded* result instead, so annotations that disagree with the code cannot
accumulate:

* every error mypy reports is reduced to an **identity** — its file, its error
  code and its message, with the line and column dropped, so moving code around
  does not read as a new problem;
* the baseline (``scripts/mypy_baseline.json``) records each identity with how
  many times it occurs, the totals, and the toolchain that produced it;
* a run **fails** when a new identity appears, when a recorded identity occurs
  more often than recorded, or when the error total rises.

Identities disappearing, occurring less often, and a falling total are all fine
and are reported — that is the ratchet turning. ``--update`` re-records the
baseline and refuses to record a worse result unless ``--allow-increase`` says
so explicitly.

What this does *not* cover: the configuration sets ``disallow_untyped_defs =
false``, so mypy reports nothing for a def that carries no annotations at all
and a wholly untyped new module would leave the recorded result untouched.
Unannotated public signatures are gated separately, and at zero tolerance, by
``tests/unit/test_public_signature_hints.py``.

The baseline is toolchain-specific: a different mypy version, or a different set
of flags, reports a different error set. Both are recorded and checked before
anything is compared, and a mismatch is a hard, actionable failure rather than a
silent pass. What the baseline is deliberately *not* specific to is the
environment — see ``MYPY_ARGS`` below.

Usage::

    python scripts/mypy_ratchet.py              # gate (exit 1 if it regressed)
    python scripts/mypy_ratchet.py --update     # re-record after an improvement
    python scripts/mypy_ratchet.py --from-output run.txt   # gate a saved run

Exit codes: 0 pass, 1 the ratchet was violated, 2 the check could not run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "scripts" / "mypy_baseline.json"
DEFAULT_TARGET = "effgen"

# The result must not depend on which optional extras happen to be installed, or
# a developer with `[all]` and CI with a base install would disagree about
# whether the ratchet holds. `--ignore-missing-imports` alone does not achieve
# that — it silences the import error for an *absent* package, but an installed
# one still contributes its real types and the errors that come with them.
# `--no-site-packages` is what equalizes the two: every third-party import
# becomes Any either way, so the recorded set depends only on effGen's own
# source and the pinned mypy. The trade is that effGen's *use* of third-party
# APIs is not checked here; the public-surface lane above, which runs against
# the installed packages, is where that shows up.
MYPY_ARGS = [
    "--ignore-missing-imports",
    "--no-site-packages",
    "--no-pretty",
    "--no-error-summary",
    "--show-error-codes",
    "--no-color-output",
]

# "path:line:col: error: message  [code]" — the column is optional because it
# depends on `show_column_numbers`.
_ERROR_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):(?:(?P<col>\d+):)?\s+error:\s+(?P<msg>.*?)"
    r"(?:\s+\[(?P<code>[a-z-]+)\])?$"
)


def _identity(path: str, code: str, message: str) -> str:
    """Return the stable identity of one error: file, code and message."""
    rel = Path(path)
    if rel.is_absolute():
        try:
            rel = rel.relative_to(REPO_ROOT)
        except ValueError:
            pass
    return f"{rel.as_posix()}|{code}|{message.strip()}"


def parse_errors(text: str) -> Counter[str]:
    """Reduce raw mypy output to a count per error identity."""
    found: Counter[str] = Counter()
    for line in text.splitlines():
        match = _ERROR_RE.match(line.rstrip())
        if match is None:
            continue
        found[
            _identity(
                match.group("path"), match.group("code") or "no-code", match.group("msg")
            )
        ] += 1
    return found


def run_mypy(target: str) -> tuple[str, int]:
    """Run mypy over *target* and return its combined output and exit status."""
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", *MYPY_ARGS, target],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.stdout + proc.stderr, proc.returncode


def mypy_version() -> str:
    """Return the running mypy's version string, or ``"absent"``."""
    try:
        import mypy.version

        return str(mypy.version.__version__)
    except Exception:
        return "absent"


def load_baseline(path: Path) -> dict[str, Any]:
    """Read the recorded baseline, or exit 2 explaining how to create one."""
    if not path.exists():
        print(
            f"ERROR: no baseline at {path}.\n"
            f"       Create one with: python {Path(__file__).name} --update",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def build_baseline(counts: Counter[str], target: str) -> dict[str, Any]:
    """Assemble the on-disk baseline document for *counts*."""
    by_code: Counter[str] = Counter()
    for identity, n in counts.items():
        by_code[identity.split("|", 2)[1]] += n
    return {
        "_comment": (
            "Recorded mypy result for the ratchet in scripts/mypy_ratchet.py. "
            "It may only improve: run the script with --update after a change "
            "that removes errors. Regenerating it with a different mypy version "
            "changes the whole set, so the version is recorded and checked."
        ),
        "toolchain": {
            "mypy": mypy_version(),
            "target": target,
            "mypy_args": MYPY_ARGS,
            "config": "pyproject.toml [tool.mypy] (python_version = 3.11)",
        },
        "totals": {"errors": sum(counts.values()), "identities": len(counts)},
        "by_code": dict(sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))),
        "identities": dict(sorted(counts.items())),
    }


def compare(counts: Counter[str], baseline: dict[str, Any]) -> tuple[int, list[str]]:
    """Compare a run against *baseline*; return an exit status and the report."""
    recorded: dict[str, int] = baseline.get("identities", {})
    recorded_total = int(baseline.get("totals", {}).get("errors", 0))
    total = sum(counts.values())

    new = sorted(i for i in counts if i not in recorded)
    grew = sorted(i for i in counts if i in recorded and counts[i] > recorded[i])
    gone = sorted(i for i in recorded if i not in counts)
    shrank = sorted(i for i in counts if i in recorded and counts[i] < recorded[i])

    recorded_mypy = baseline.get("toolchain", {}).get("mypy", "?")
    lines = [
        (
            f"recorded: {recorded_total} errors across {len(recorded)} "
            f"identities (mypy {recorded_mypy})"
        ),
        (
            f"this run: {total} errors across {len(counts)} identities "
            f"(mypy {mypy_version()})"
        ),
    ]
    if gone or shrank:
        lines.append(f"improved: {len(gone)} identities gone, {len(shrank)} rarer")
        for identity in gone[:20]:
            lines.append(f"  - gone   {identity}")
        for identity in shrank[:20]:
            lines.append(
                f"  - fewer  {identity} ({recorded[identity]} -> {counts[identity]})"
            )
    status = 0
    if new or grew or total > recorded_total:
        status = 1
        lines.append("")
        lines.append("RATCHET VIOLATED — the recorded static-check result got worse.")
        for identity in new:
            lines.append(f"  + NEW    {identity} (x{counts[identity]})")
        for identity in grew:
            before, after = recorded[identity], counts[identity]
            lines.append(f"  + MORE   {identity} ({before} -> {after})")
        if total > recorded_total:
            lines.append(f"  + TOTAL  {recorded_total} -> {total}")
        lines.append("")
        lines.append(
            "Annotate the new code, or — if the increase is deliberate and "
            "reviewed — re-record with --update --allow-increase."
        )
    else:
        lines.append("ratchet holds: nothing new, nothing more frequent.")
    return status, lines


def check_toolchain(baseline: dict[str, Any]) -> None:
    """Exit 2 when this run cannot be compared against how the baseline was taken."""
    recorded = baseline.get("toolchain", {}).get("mypy")
    running = mypy_version()
    if running == "absent":
        print(
            "ERROR: mypy is not installed; the ratchet cannot run.\n"
            "       pip install -r requirements-dev.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    recorded_args = baseline.get("toolchain", {}).get("mypy_args")
    if recorded_args is not None and list(recorded_args) != MYPY_ARGS:
        print(
            f"ERROR: baseline was recorded with {recorded_args}, this script "
            f"now runs {MYPY_ARGS}.\n"
            "       Different flags report a different error set — in "
            "particular --no-site-packages is what makes the result the same "
            "with and without the optional extras installed.\n"
            "       Re-record the baseline with --update after reviewing the "
            "diff.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if recorded and recorded != running:
        print(
            f"ERROR: baseline was recorded with mypy {recorded}, this "
            f"environment has mypy {running}.\n"
            "       A different mypy reports a different error set, so the "
            "comparison would be meaningless.\n"
            f"       Install mypy=={recorded}, or re-record the baseline with "
            "--update after reviewing the diff.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    """Run, or gate, the recorded mypy result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--from-output",
        type=Path,
        help="gate a saved mypy transcript instead of running mypy",
    )
    parser.add_argument(
        "--update", action="store_true", help="re-record the baseline from this run"
    )
    parser.add_argument(
        "--allow-increase",
        action="store_true",
        help="with --update, permit recording a worse result",
    )
    parser.add_argument(
        "--save-output", type=Path, help="write the raw mypy transcript here"
    )
    args = parser.parse_args(argv)

    if args.from_output is not None:
        raw = args.from_output.read_text(encoding="utf-8")
        rc = 1
    else:
        raw, rc = run_mypy(args.target)
        if rc >= 2:
            print(raw, file=sys.stderr)
            print(f"ERROR: mypy failed to run (exit {rc})", file=sys.stderr)
            return 2
    if args.save_output is not None:
        args.save_output.write_text(raw, encoding="utf-8")

    counts = parse_errors(raw)

    if args.update:
        new_doc = build_baseline(counts, args.target)
        if args.baseline.exists():
            old = json.loads(args.baseline.read_text(encoding="utf-8"))
            old_total = int(old.get("totals", {}).get("errors", 0))
            if new_doc["totals"]["errors"] > old_total and not args.allow_increase:
                print(
                    f"ERROR: refusing to record {new_doc['totals']['errors']} "
                    f"errors over the recorded {old_total}. The ratchet only "
                    "turns one way; pass --allow-increase if this is deliberate.",
                    file=sys.stderr,
                )
                return 2
        args.baseline.write_text(
            json.dumps(new_doc, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(
            f"recorded {new_doc['totals']['errors']} errors across "
            f"{new_doc['totals']['identities']} identities -> {args.baseline}"
        )
        return 0

    baseline = load_baseline(args.baseline)
    if args.from_output is None:
        check_toolchain(baseline)
    status, lines = compare(counts, baseline)
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(main())
