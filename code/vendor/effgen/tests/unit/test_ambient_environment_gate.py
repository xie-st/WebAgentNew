"""A test may not depend on the state of the machine it happens to run on.

The suite can be run with that state removed — ``EFFGEN_TEST_HERMETIC=1`` points the
home directory at an empty temporary one, reduces the environment to a declared
allowlist and refuses any socket that is not loopback. These cases prove the removal
works by planting the three dependencies the gate exists to catch and running a real
pytest session over them twice: once with the machine's state present, where all of
them pass, and once with it removed, where every one of them fails.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests._harness import hermetic

REPO_ROOT = Path(__file__).resolve().parents[2]

PLANTED_CONFTEST = """
import sys
sys.path.insert(0, {repo!r})

from tests._harness import hermetic

hermetic.activate()
"""

PLANTED_TESTS = '''
"""Four dependencies on ambient state, planted so the gate can be shown to catch them."""

import os
import socket
from pathlib import Path

RECORDED_HOME = {home!r}


def test_reads_the_home_directory_of_whoever_started_the_run():
    assert str(Path.home()) == RECORDED_HOME


def test_reads_an_environment_variable_no_test_ever_set():
    assert os.environ["AMBIENT_PROBE_VARIABLE"] == "set-by-the-machine"


def test_assumes_the_home_directory_already_has_something_in_it():
    assert any(Path.home().iterdir())


def test_reaches_for_a_host_on_the_network():
    try:
        socket.create_connection(("example.com", 80), timeout=3).close()
    except OSError as exc:
        assert "ambient environment" not in str(exc), str(exc)
'''

PLANTED_IDS = (
    "test_reads_the_home_directory_of_whoever_started_the_run",
    "test_reads_an_environment_variable_no_test_ever_set",
    "test_assumes_the_home_directory_already_has_something_in_it",
    "test_reaches_for_a_host_on_the_network",
)


def _plant(tmp_path: Path) -> Path:
    plot = tmp_path / "planted"
    plot.mkdir()
    (plot / "conftest.py").write_text(
        PLANTED_CONFTEST.format(repo=str(REPO_ROOT)), encoding="utf-8"
    )
    recorded_home = os.environ.get("HOME") or str(Path.home())
    (plot / "test_planted.py").write_text(
        PLANTED_TESTS.format(home=recorded_home), encoding="utf-8"
    )
    return plot


def _run(plot: Path, *, scrubbed: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["AMBIENT_PROBE_VARIABLE"] = "set-by-the-machine"
    env.pop("EFFGEN_LANE_TIMING", None)
    if scrubbed:
        env["EFFGEN_TEST_HERMETIC"] = "1"
    else:
        env.pop("EFFGEN_TEST_HERMETIC", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(plot),
            "-o",
            "addopts=",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "--no-header",
            "--tb=no",
            "-q",
            "-rf",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        cwd=str(plot),
    )


# ---------------------------------------------------------------------------
# The gate, planted and proven
# ---------------------------------------------------------------------------


def test_planted_ambient_dependencies_pass_while_the_machine_state_is_present(tmp_path):
    result = _run(_plant(tmp_path), scrubbed=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout


@pytest.mark.parametrize("planted", PLANTED_IDS)
def test_the_gate_fails_each_planted_ambient_dependency(tmp_path, planted):
    result = _run(_plant(tmp_path), scrubbed=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"test_planted.py::{planted}" in result.stdout, result.stdout


def test_the_scrubbed_run_reports_what_it_removed(tmp_path):
    result = _run(_plant(tmp_path), scrubbed=True)
    assert "4 failed" in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# The pieces the gate is built from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.1.1", "localhost", "::1", "0.0.0.0", ""],
)
def test_loopback_destinations_are_reachable(host):
    assert hermetic._is_loopback(host) is True


@pytest.mark.parametrize("host", ["example.com", "8.8.8.8", "api.openai.com", "169.254.169.254"])
def test_every_other_destination_is_not(host):
    assert hermetic._is_loopback(host) is False


def test_a_unix_socket_is_never_treated_as_a_network_route():
    assert hermetic._address_allowed(socket.AF_UNIX, "/tmp/some.sock") is True


@pytest.mark.parametrize(
    "name", ["OPENAI_API_KEY", "HF_TOKEN", "GITHUB_TOKEN", "TERM", "USER", "CI", "NO_COLOR"]
)
def test_a_credential_or_a_terminal_setting_is_not_on_the_allowlist(name):
    assert hermetic._keeps(name) is False


@pytest.mark.parametrize("name", ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTEST_CURRENT_TEST"])
def test_the_interpreter_can_still_find_itself(name):
    assert hermetic._keeps(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_XET_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "TORCH_HOME",
    ],
)
def test_a_download_cache_survives_the_scrub(name):
    """A cache pointer is a place on disk, not a decision the suite reads.

    Dropping one turns "this model is already downloaded" into a fetch, which
    the scrub's own network guard then refuses — so every test that loads a
    cached model fails on a machine whose cache is somewhere the default layout
    does not cover.
    """
    assert name in hermetic.CACHE_NAMES


def test_the_cache_variable_huggingface_actually_reads_is_covered():
    """``HF_HUB_CACHE`` wins over ``HF_HOME``, so keeping only the latter is not enough."""
    from huggingface_hub import constants

    configured = constants.HF_HUB_CACHE
    assert "HF_HUB_CACHE" in hermetic.CACHE_NAMES, (
        f"huggingface_hub resolves its store to {configured}; the scrub has to "
        "keep the variable that decided it"
    )


def test_the_scrub_is_inert_unless_it_is_asked_for(monkeypatch):
    monkeypatch.delenv(hermetic.ENABLE_VAR, raising=False)
    assert hermetic.requested() is False


def test_the_suite_states_whether_it_is_running_scrubbed():
    line = hermetic.report_line()
    if hermetic.is_active():
        assert "ambient environment removed" in line
    else:
        assert line == ""
