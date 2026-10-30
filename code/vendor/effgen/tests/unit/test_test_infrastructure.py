"""Regression tests for the test/CI infrastructure itself.

These guard the test-infrastructure fixes so they cannot silently regress:
  * coverage must NOT be wired into the default pytest addopts (otherwise
    `--collect-only` and parallel shards corrupt the coverage DB),
  * coverage.run must be parallel-safe,
  * the marker surface must stay complete and map to CI jobs, and
  * tests/security and tests/deploy must be auto-marked by directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

pytestmark = pytest.mark.unit


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def test_default_addopts_has_no_coverage():
    """Coverage must be opt-in, never in the default addopts."""
    addopts = _pyproject()["tool"]["pytest"]["ini_options"]["addopts"]
    joined = " ".join(addopts)
    assert "--cov" not in joined, (
        "coverage must not be in default addopts: it makes --collect-only and "
        "parallel shards corrupt the .coverage SQLite DB"
    )


def test_coverage_run_is_parallel_safe():
    cov_run = _pyproject()["tool"]["coverage"]["run"]
    assert cov_run.get("parallel") is True, "coverage.run.parallel must be true for sharded runs"


def test_marker_surface_is_complete():
    """All markers referenced by CI jobs must be declared (strict-markers is on)."""
    markers = _pyproject()["tool"]["pytest"]["ini_options"]["markers"]
    names = {m.split(":", 1)[0].strip() for m in markers}
    required = {
        "unit", "integration", "live", "api", "gpu", "docker",
        "deployment", "security", "expensive", "slow", "reliability", "fuzz",
    }
    missing = required - names
    assert not missing, f"missing pytest markers: {sorted(missing)}"


@pytest.mark.parametrize(
    "script",
    [
        "_install_matrix_common.sh",
        "check_install_cpu.sh",
        "check_install_cu124.sh",
        "check_install_vllm_cu124.sh",
        "check_install_server.sh",
        "gpu_shard_runner.sh",
        "run_coverage.sh",
        "release_artifacts.sh",
    ],
)
def test_release_scripts_exist_and_are_executable(script):
    path = SCRIPTS / script
    assert path.exists(), f"missing release script: {script}"
    assert os.access(path, os.X_OK), f"release script not executable: {script}"


def _count_collected(rel: str, marker_expr: str) -> int:
    """Run pytest --collect-only in an isolated subprocess and count node ids."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", rel, "-m", marker_expr,
         "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    return sum(1 for line in proc.stdout.splitlines() if "::" in line)


@pytest.mark.parametrize(
    "rel, marker",
    [("tests/security", "security"), ("tests/deploy", "deployment")],
)
def test_dir_tests_are_auto_marked(rel, marker):
    """Directory-based auto-marking must select the right suites for `-m <marker>`."""
    selected = _count_collected(rel, marker)
    deselected = _count_collected(rel, f"not {marker}")
    assert selected > 0, f"`-m {marker}` collected nothing under {rel}"
    assert deselected == 0, (
        f"`-m not {marker}` still collected {deselected} test(s) under {rel}: "
        f"directory auto-marking is not applied to every test"
    )
