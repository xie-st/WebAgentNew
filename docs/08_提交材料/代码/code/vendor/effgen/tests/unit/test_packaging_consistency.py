"""Guard the packaging single-source-of-truth invariants.

pyproject.toml is the one authoritative place for effGen's dependencies, extras,
and version. These tests fail loudly if a future change reintroduces the drift
the 0.3.0 packaging pass removed:

  * setup.py growing its own (divergent) dependency/extras declarations again,
  * requirements.txt falling out of sync with the base dependency table,
  * flash-attn sneaking back into [all] (its build breaks `pip install`),
  * fast-moving majors losing their tested upper bounds.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
SETUP_PY = REPO_ROOT / "setup.py"


def _cfg() -> dict:
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def _requirements_lines() -> list[str]:
    lines = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and not line.startswith("-r"):
            lines.append(line)
    return lines


def test_requirements_txt_mirrors_pyproject_base_deps():
    """requirements.txt must be an exact mirror of [project.dependencies].

    Regenerate with `python scripts/gen_requirements.py` if this fails.
    """
    base = _cfg()["project"]["dependencies"]
    assert _requirements_lines() == base, (
        "requirements.txt has drifted from pyproject.toml [project.dependencies]. "
        "Run: python scripts/gen_requirements.py"
    )


def test_setup_py_is_a_metadata_free_shim():
    """setup.py must not redeclare dependencies/extras (pyproject owns them)."""
    text = SETUP_PY.read_text(encoding="utf-8")
    for forbidden in ("install_requires", "extras_require", "flash-attn"):
        assert forbidden not in text, (
            f"setup.py reintroduced '{forbidden}'. All packaging metadata belongs "
            "in pyproject.toml; keep setup.py a bare `setup()` shim."
        )


def test_flash_attn_not_in_all_extra():
    """flash-attn must stay out of [all] — its setup.py breaks `pip install`."""
    all_extra = _cfg()["project"]["optional-dependencies"]["all"]
    assert not any(dep.lower().startswith("flash-attn") for dep in all_extra), (
        "flash-attn is back in [all]; it makes `pip install effgen[all]` fail "
        "(its setup.py imports torch before the isolated build env has it)."
    )


def test_fast_movers_have_upper_bounds():
    """Fast-moving majors must carry a tested `<` ceiling in the base deps.

    An uncapped one takes the whole test suite down the day it ships a major:
    `mcp` 2.x removed `mcp.server.fastmcp` and the `streamablehttp_client`
    alias, so every module importing `effgen.tools.protocols` failed at
    collection — including the thousands of tests that use no protocol at all.
    """
    base = _cfg()["project"]["dependencies"]
    by_name = {dep.split(">=")[0].split("[")[0].strip().lower(): dep for dep in base}
    for pkg in ("torch", "transformers", "numpy", "pandas", "fastapi", "starlette", "mcp"):
        spec = by_name.get(pkg, "")
        assert "<" in spec, f"{pkg} is missing a tested upper bound in pyproject.toml: {spec!r}"
