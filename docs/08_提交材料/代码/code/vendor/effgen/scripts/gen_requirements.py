#!/usr/bin/env python3
"""Regenerate requirements.txt from pyproject.toml's [project.dependencies].

pyproject.toml is the single source of truth for effGen's dependencies. The flat
requirements.txt is a generated mirror consumed by tooling that cannot read
pyproject (SBOM generation, `safety`/`pip-audit` scans, plain
`pip install -r requirements.txt`). Run this after editing the base dependency
list so the two never drift:

    python scripts/gen_requirements.py
"""

from __future__ import annotations

import tomllib  # type: ignore[no-redef]
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

HEADER = """\
# ==============================================================================
# effGen Framework - core runtime requirements
#
# GENERATED FILE — do not edit by hand. This is an exact mirror of the
# [project.dependencies] table in pyproject.toml, which is the single source of
# truth for effGen's dependencies. It exists for tooling that consumes a flat
# requirements file (SBOM generation, `safety`/`pip-audit` scans, simple
# `pip install -r requirements.txt`).
#
# Regenerate after editing pyproject.toml:
#     python scripts/gen_requirements.py
#
# For the full optional stack use the extras instead:
#     pip install -e ".[all]" -c requirements-all-lock.txt
# Python version: >=3.11
# ==============================================================================

"""


def main() -> None:
    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps = cfg["project"]["dependencies"]
    REQUIREMENTS.write_text(HEADER + "".join(f"{d}\n" for d in deps), encoding="utf-8")
    print(f"Wrote {REQUIREMENTS.relative_to(REPO_ROOT)} with {len(deps)} dependencies.")


if __name__ == "__main__":
    main()
