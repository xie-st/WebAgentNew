"""Shared pieces for the instrument's own tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .records import Cell, CellKey, Record

FIXTURES = Path(__file__).parent / "fixtures"

#: Where the recorded runs live. The checks that need them say loudly what they
#: could not measure when the path is absent, because a guard that skips quietly
#: reads exactly like a guard that passed.
RECORD_TREE_ENV = "AGENTLOOP_RECORDS"


def fixture_cell(name: str, bench: str = "arc_e", **kwargs) -> Cell:
    """Load one of the fixture record files as a cell."""
    records = [
        Record.from_json(json.loads(line))
        for line in (FIXTURES / f"{name}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary_path = FIXTURES / f"{name}.summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return Cell(CellKey("fixture", bench, name), records, summary, **kwargs)


def fixture_expected(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.expected.json").read_text())


def record_tree_path() -> Path | None:
    value = os.environ.get(RECORD_TREE_ENV)
    if not value:
        return None
    path = Path(value)
    return path if path.is_dir() else None


def require_record_tree() -> Path:
    path = record_tree_path()
    if path is None:
        pytest.skip(
            f"NOT MEASURED: no recorded runs. Set {RECORD_TREE_ENV} to the directory "
            "holding <model>/<benchmark>/<system>/records.jsonl."
        )
    return path


@pytest.fixture
def known_answer() -> Cell:
    return fixture_cell("known_answer")
