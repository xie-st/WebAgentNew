#!/usr/bin/env python3
"""Regenerate the two dependency lockfiles from pyproject.toml.

``requirements-lock.txt`` is the hash-pinned lock for the base install;
``requirements-all-lock.txt`` is the constraints lock for the ``[all]`` extra.
Both are compiled with ``uv pip compile``, which resolves for a target Python
version instead of the interpreter that happens to be running.

``uv`` rewrites the whole file, so the explanatory header each lockfile carries
is re-applied here. Run this instead of calling ``uv`` by hand:

    python scripts/gen_locks.py

Pass ``--upgrade-package NAME`` (repeatable) to move a single package — for
example when raising a floor past an advisory — while leaving the rest of the
resolution on its current pins.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PYTHON_VERSION = "3.11"

CORE_LOCK = REPO_ROOT / "requirements-lock.txt"
ALL_LOCK = REPO_ROOT / "requirements-all-lock.txt"

CORE_HEADER = """\
# requirements-lock.txt — effGen hash-pinned dependency lockfile
# Regenerate: python scripts/gen_locks.py
# Usage (non-editable production install): pip install effgen -r requirements-lock.txt
# Note: editable installs (-e .) cannot use hash checking; use a built wheel for CI
#
"""

ALL_HEADER = """\
# Constraints lock for the `[all]` extra.
#
# Why this exists: `pip install -e ".[all]"` alone exceeds pip's resolver depth
# (vllm + every provider SDK + the google client stack under the protobuf>=5.29.5
# CVE floor) and fails with `resolution-too-deep`. Installing WITH this lock
# collapses the search to a single consistent, CVE-safe solution:
#
#     pip install -e ".[all]" -c requirements-all-lock.txt
#
# Regenerate after changing dependencies in pyproject.toml:
#
#     python scripts/gen_locks.py
#
"""


def _compile(output: Path, header: str, extra_args: list[str], upgrades: list[str]) -> None:
    cmd = [
        "uv",
        "pip",
        "compile",
        PYPROJECT.name,
        "--quiet",
        "--python-version",
        PYTHON_VERSION,
        "--output-file",
        output.name,
        *extra_args,
    ]
    for package in upgrades:
        cmd += ["--upgrade-package", package]

    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        raise SystemExit(f"uv pip compile failed for {output.name}")

    output.write_text(header + output.read_text())
    print(f"Wrote {output.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upgrade-package",
        action="append",
        default=[],
        metavar="NAME",
        help="move this package to its latest allowed version; repeatable",
    )
    args = parser.parse_args()

    if shutil.which("uv") is None:
        raise SystemExit("uv not found on PATH — install it from https://docs.astral.sh/uv/")

    _compile(CORE_LOCK, CORE_HEADER, ["--generate-hashes"], args.upgrade_package)
    _compile(ALL_LOCK, ALL_HEADER, ["--extra", "all"], args.upgrade_package)
    return 0


if __name__ == "__main__":
    sys.exit(main())
