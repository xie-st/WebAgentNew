"""The register of tests known to fail for a reason outside the tree stays accurate.

``tests/flake_register.toml`` is the alternative to re-running a suite until it goes
green. It only works if an entry cannot be added without saying what happened and who
has to deal with it, and cannot sit there after the test it names is gone or the cause
it describes has been addressed. These cases check the register the tree ships and
then check the checks, by planting one of each violation shape in a register of their
own.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from tests._harness import flake_register as fr
from tests._harness.lanes import lane_of_nodeid

REPO_ROOT = Path(__file__).resolve().parents[2]

SOUND_ENTRY = """
[[flake]]
id = "tests/unit/test_flake_register.py::test_the_shipped_register_has_no_violations"
cause = "external-service"
reason = "A placeholder reason long enough to explain what was actually observed."
owner = "tests/flake_register.toml"
first_seen = "2026-01-01"
expires = "2099-01-01"
"""


def _register(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "flake_register.toml"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The register this tree ships
# ---------------------------------------------------------------------------


def test_the_shipped_register_has_no_violations():
    found = fr.violations()
    assert found == [], "the flake register is not sound:\n  " + "\n  ".join(found)


def test_the_shipped_register_is_readable_and_every_entry_is_complete():
    for entry in fr.load():
        assert entry.id
        assert entry.cause in fr.CAUSE_CLASSES
        assert len(entry.reason.strip()) >= fr.MIN_REASON_CHARS
        assert (REPO_ROOT / entry.owner).exists()
        assert dt.date.fromisoformat(entry.expires) > dt.date.fromisoformat(entry.first_seen)


def test_every_registered_test_belongs_to_a_lane():
    for entry in fr.load():
        assert lane_of_nodeid(entry.id) != "", entry.id


def test_a_failure_is_matched_to_its_entry_by_node_id_or_by_module():
    entries = fr.load()
    if not entries:
        pytest.skip("the register is empty, so there is nothing to match")
    module_entries = [e for e in entries if "::" not in e.id]
    if module_entries:
        entry = module_entries[0]
        assert fr.entry_for(f"{entry.id}::test_anything", entries) is entry
    assert fr.entry_for("tests/unit/test_nothing_here.py::test_x", entries) is None


# ---------------------------------------------------------------------------
# The checks themselves, one planted violation per shape
# ---------------------------------------------------------------------------


def test_a_sound_entry_is_accepted(tmp_path):
    path = _register(tmp_path, SOUND_ENTRY)
    assert fr.violations(path, repo_root=REPO_ROOT) == []


def test_a_missing_field_is_caught(tmp_path):
    body = "\n".join(
        line for line in SOUND_ENTRY.splitlines() if not line.startswith("owner")
    )
    found = fr.violations(_register(tmp_path, body), repo_root=REPO_ROOT)
    assert any("missing required field 'owner'" in message for message in found), found


def test_a_reason_too_short_to_explain_anything_is_caught(tmp_path):
    body = SOUND_ENTRY.replace(
        '"A placeholder reason long enough to explain what was actually observed."',
        '"flaky"',
    )
    found = fr.violations(_register(tmp_path, body), repo_root=REPO_ROOT)
    assert any("reason is 5 characters" in message for message in found), found


def test_a_cause_outside_the_declared_classes_is_caught(tmp_path):
    body = SOUND_ENTRY.replace('cause = "external-service"', 'cause = "who knows"')
    found = fr.violations(_register(tmp_path, body), repo_root=REPO_ROOT)
    assert any("is not one of" in message for message in found), found


def test_an_owner_that_is_not_a_path_in_the_repository_is_caught(tmp_path):
    body = SOUND_ENTRY.replace(
        'owner = "tests/flake_register.toml"', 'owner = "somebody@example.com"'
    )
    found = fr.violations(_register(tmp_path, body), repo_root=REPO_ROOT)
    assert any("is not a path in the repository" in message for message in found), found


def test_an_entry_naming_a_test_that_no_longer_exists_is_caught(tmp_path):
    body = SOUND_ENTRY.replace(
        "::test_the_shipped_register_has_no_violations", "::test_removed_last_year"
    )
    found = fr.violations(_register(tmp_path, body), repo_root=REPO_ROOT)
    assert any("declares no test named" in message for message in found), found


def test_an_entry_naming_a_module_that_no_longer_exists_is_caught(tmp_path):
    body = SOUND_ENTRY.replace("tests/unit/test_flake_register.py", "tests/unit/test_gone.py")
    found = fr.violations(_register(tmp_path, body), repo_root=REPO_ROOT)
    assert any("is not a file in the tree" in message for message in found), found


def test_an_entry_past_its_expiry_date_is_caught(tmp_path):
    body = SOUND_ENTRY.replace('expires = "2099-01-01"', 'expires = "2026-02-01"')
    found = fr.violations(
        _register(tmp_path, body), repo_root=REPO_ROOT, today=dt.date(2026, 3, 1)
    )
    assert any("expired on 2026-02-01" in message for message in found), found


def test_an_expiry_before_the_first_sighting_is_caught(tmp_path):
    body = SOUND_ENTRY.replace('expires = "2099-01-01"', 'expires = "2025-01-01"')
    found = fr.violations(
        _register(tmp_path, body), repo_root=REPO_ROOT, today=dt.date(2024, 1, 1)
    )
    assert any("is not after first_seen" in message for message in found), found


def test_a_date_that_is_not_iso_8601_is_caught(tmp_path):
    body = SOUND_ENTRY.replace('first_seen = "2026-01-01"', 'first_seen = "January 2026"')
    found = fr.violations(_register(tmp_path, body), repo_root=REPO_ROOT)
    assert any("is not an ISO-8601 date" in message for message in found), found


def test_the_same_test_listed_twice_is_caught(tmp_path):
    found = fr.violations(_register(tmp_path, SOUND_ENTRY * 2), repo_root=REPO_ROOT)
    assert any("listed more than once" in message for message in found), found


def test_an_id_the_tree_does_not_collect_is_caught_when_collection_is_known(tmp_path):
    path = _register(tmp_path, SOUND_ENTRY)
    found = fr.violations(path, repo_root=REPO_ROOT, known_ids=set())
    assert any("no test with this id is collected" in message for message in found), found


def test_an_absent_register_reads_as_empty_rather_than_failing(tmp_path):
    assert fr.load(tmp_path / "nothing.toml") == []
    assert fr.violations(tmp_path / "nothing.toml", repo_root=REPO_ROOT) == []
