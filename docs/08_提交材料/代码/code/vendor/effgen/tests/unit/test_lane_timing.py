"""Per-lane timing is produced on request and changes nothing when it is not.

A full run of the suite is one number, and a lane that has grown to a third of it is
invisible in that number. ``EFFGEN_LANE_TIMING`` names a file the run writes its
per-lane breakdown to. These cases check the lane mapping, the shape of the report and
— the part that matters for every other run — that a session without the variable
produces byte-identical output.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests._harness import lane_timing
from tests._harness.lanes import ROOT_LANE, lane_of_nodeid, lane_of_path, lanes_with_tests

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"

SAMPLE_TESTS = '''
import time


def test_quick():
    assert True


def test_slower():
    time.sleep(0.05)
    assert True


def test_skipped():
    import pytest

    pytest.skip("nothing to do here")
'''


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nodeid", "expected"),
    [
        ("tests/unit/test_x.py::test_y", "unit"),
        ("unit/test_x.py::test_y", "unit"),
        ("tests/server/sub/test_x.py::TestC::test_y", "server"),
        ("tests/conftest.py::test_y", ROOT_LANE),
        ("test_x.py::test_y", ROOT_LANE),
    ],
)
def test_a_node_id_resolves_to_its_lane(nodeid, expected):
    assert lane_of_nodeid(nodeid) == expected


def test_a_path_resolves_to_the_same_lane_as_its_node_id():
    path = TESTS_ROOT / "unit" / "test_lane_timing.py"
    assert lane_of_path(path) == "unit"
    assert lane_of_path(TESTS_ROOT / "conftest.py") == ROOT_LANE


def test_every_directory_holding_tests_is_a_lane():
    lanes = lanes_with_tests()
    assert "unit" in lanes
    assert "server" in lanes
    for lane in lanes:
        if lane == ROOT_LANE:
            continue
        assert (TESTS_ROOT / lane).is_dir(), lane


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _run(plot: Path, report: Path | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("EFFGEN_LANE_TIMING", None)
    env.pop("EFFGEN_LANE_TIMING_TABLE", None)
    # The nested session is a plain one whatever the outer run is doing, so the
    # report it writes describes itself rather than its parent.
    env.pop("EFFGEN_TEST_HERMETIC", None)
    env.pop("EFFGEN_TEST_REVERSE_ORDER", None)
    if report is not None:
        env["EFFGEN_LANE_TIMING"] = str(report)
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
            "-q",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        cwd=str(plot),
    )


@pytest.fixture
def plot(tmp_path):
    directory = tmp_path / "lane"
    directory.mkdir()
    (directory / "conftest.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from tests._harness import lane_timing\n\n\n"
        "def pytest_configure(config):\n"
        "    import time\n"
        "    config._effgen_session_start = time.time()\n"
        "    lane_timing.maybe_register(config)\n",
        encoding="utf-8",
    )
    (directory / "test_sample.py").write_text(SAMPLE_TESTS, encoding="utf-8")
    return directory


def test_the_report_names_every_lane_that_ran(plot, tmp_path):
    report = tmp_path / "timing.json"
    result = _run(plot, report)
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["lanes"], document
    lane = document["lanes"][0]
    assert lane["tests"] == 3
    assert lane["passed"] == 2
    assert lane["skipped"] == 1
    assert lane["failed"] == 0
    assert lane["duration_s"] > 0


def test_the_report_accounts_for_nearly_all_of_the_session(plot, tmp_path):
    report = tmp_path / "timing.json"
    _run(plot, report)
    document = json.loads(report.read_text(encoding="utf-8"))
    assert 0.0 < document["accounted_fraction"] <= 1.0
    assert document["session_duration_s"] >= document["accounted_s"]


def test_the_report_names_the_slowest_test(plot, tmp_path):
    report = tmp_path / "timing.json"
    _run(plot, report)
    document = json.loads(report.read_text(encoding="utf-8"))
    slowest = document["slowest_tests"][0]["nodeid"]
    assert slowest.endswith("test_slower")


def test_the_report_records_the_order_the_session_ran_in(plot, tmp_path):
    report = tmp_path / "timing.json"
    _run(plot, report)
    document = json.loads(report.read_text(encoding="utf-8"))
    assert "randomly_seed" in document["order"]
    assert document["order"]["reversed"] is False
    assert document["hermetic"] is False


def test_a_run_without_the_variable_writes_no_report_and_prints_the_same_bytes(plot, tmp_path):
    report = tmp_path / "timing.json"
    without = _run(plot, None)
    again = _run(plot, None)
    assert not report.exists()
    assert without.returncode == again.returncode
    assert _strip_timing(without.stdout) == _strip_timing(again.stdout)

    with_report = _run(plot, report)
    assert report.exists()
    assert _strip_timing(with_report.stdout) == _strip_timing(without.stdout)


_DURATION = re.compile(r"\bin \d+\.\d+s\b")


def _strip_timing(text: str) -> str:
    """Drop the wall-clock number pytest prints, which differs between any two runs."""
    return _DURATION.sub("in <duration>", text).replace("\r", "")


def test_the_plugin_is_not_registered_without_the_variable(monkeypatch):
    monkeypatch.delenv(lane_timing.REPORT_PATH_VAR, raising=False)

    class _Manager:
        def has_plugin(self, name):
            return False

        def register(self, plugin, name):  # pragma: no cover - must not be reached
            raise AssertionError("the timing plugin registered itself uninvited")

    class _Config:
        pluginmanager = _Manager()

    assert lane_timing.maybe_register(_Config()) is None
