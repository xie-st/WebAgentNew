"""Property and regression tests for catalog snapshot loading.

A snapshot under ``effgen/models/_data/<provider>.json`` is written by a refresh, a
package build, or a hand edit, and read on the paths that price a call and answer
``effgen models``. It must never be able to break those paths: the catalog's own
contract is that a stale or damaged snapshot degrades to the in-package source and
says so.

Asserts that:
  1. ``load_snapshot`` returns a mapping or ``None`` for *any* file content —
     never raising, as its docstring promises.
  2. ``load_snapshot_records`` skips an unusable entry and keeps every usable one,
     reporting how many it dropped rather than losing the file or the models.
  3. ``save_snapshot`` → ``load_snapshot_records`` round-trips every field.
  4. A private ``ft:`` fine-tune id is never persisted.
  5. The snapshots effGen ships all load, and their counts match their contents.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.models import _catalog
from effgen.models._catalog import ModelRecord

pytestmark = pytest.mark.fuzz

_TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=30)

_ANY = st.recursive(
    st.none() | st.booleans() | st.integers(min_value=-(10**9), max_value=10**9)
    | st.floats(allow_nan=False, allow_infinity=False, width=32) | _TEXT,
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(_TEXT, children, max_size=3),
    max_leaves=8,
)


def _write(tmp_path, payload):
    path = tmp_path / "provider.json"
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Never raises
# ---------------------------------------------------------------------------

@settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(_ANY)
def test_loading_any_snapshot_never_raises(tmp_path, payload) -> None:
    path = _write(tmp_path, payload)
    data = _catalog.load_snapshot("openai", path=path)
    assert data is None or isinstance(data, dict)
    assert isinstance(_catalog.load_snapshot_records("openai", path=path), list)
    age = _catalog.snapshot_age_days("openai", path=path)
    assert age is None or isinstance(age, int)


@pytest.mark.parametrize(
    "raw",
    ["null", "[]", "[1, 2, 3]", '"text"', "12", "true", "{not json", "", "   "],
)
def test_a_snapshot_that_is_not_an_object_reads_as_absent(tmp_path, raw) -> None:
    path = tmp_path / "provider.json"
    path.write_text(raw, encoding="utf-8")
    assert _catalog.load_snapshot("openai", path=path) is None
    assert _catalog.load_snapshot_records("openai", path=path) == []
    assert _catalog.snapshot_age_days("openai", path=path) is None


@pytest.mark.parametrize(
    "models", [None, "abc", 5, {"a": 1}, True],
)
def test_a_models_block_that_is_not_a_list_reads_as_empty(tmp_path, models) -> None:
    path = _write(tmp_path, {"provider": "openai", "models": models})
    assert _catalog.load_snapshot_records("openai", path=path) == []


# ---------------------------------------------------------------------------
# One damaged entry does not cost the caller the rest of the file
# ---------------------------------------------------------------------------

def test_an_unusable_entry_is_skipped_and_the_rest_survive(tmp_path, caplog) -> None:
    path = _write(
        tmp_path,
        {
            "provider": "openai",
            "models": [
                {"id": "good-1", "provider": "openai", "context_window": 8192},
                None,
                "a string",
                5,
                {},
                {"id": "", "provider": "openai"},
                {"provider": "openai"},
                {"id": "no-provider"},
                {"id": "good-2", "provider": "openai"},
            ],
        },
    )
    with caplog.at_level("WARNING"):
        records = _catalog.load_snapshot_records("openai", path=path)
    assert [r.id for r in records] == ["good-1", "good-2"]
    assert "skipped 7" in caplog.text
    assert str(path) in caplog.text


@pytest.mark.parametrize("entry", [None, "text", 5, [], True])
def test_a_record_that_is_not_a_mapping_is_named(entry) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        ModelRecord.from_dict(entry)


@pytest.mark.parametrize(
    "entry",
    [{}, {"id": "m"}, {"provider": "openai"}, {"id": "", "provider": "openai"},
     {"id": "m", "provider": ""}],
)
def test_a_record_missing_id_or_provider_is_named(entry) -> None:
    with pytest.raises(ValueError, match="missing required field"):
        ModelRecord.from_dict(entry)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

_RECORDS = st.lists(
    st.builds(
        ModelRecord,
        id=st.text(alphabet="abcdefghij-0123456789", min_size=1, max_size=12),
        provider=st.sampled_from(["openai", "groq", "gemini"]),
        context_window=st.integers(min_value=0, max_value=10**7),
        max_output=st.integers(min_value=0, max_value=10**6),
        price_in_per_1m=st.none() | st.floats(min_value=0, max_value=1000, width=32),
        price_out_per_1m=st.none() | st.floats(min_value=0, max_value=1000, width=32),
        supports_tools=st.booleans(),
        supports_vision=st.booleans(),
        free_tier=st.booleans(),
        deprecated=st.booleans(),
        notes=_TEXT,
    ),
    max_size=6,
    unique_by=lambda r: r.id,
)


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(_RECORDS)
def test_records_survive_save_and_load(tmp_path, records) -> None:
    path = tmp_path / "provider.json"
    _catalog.save_snapshot("openai", records, verified_on="2026-01-01", path=path)
    loaded = _catalog.load_snapshot_records("openai", path=path)

    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in records]
    meta = _catalog.load_snapshot("openai", path=path)
    assert meta["count"] == len(records)
    assert meta["verified_on"] == "2026-01-01"
    assert _catalog.diff_records(records, loaded)["changed"] == {}


def test_a_private_finetune_id_is_never_persisted(tmp_path) -> None:
    path = tmp_path / "provider.json"
    _catalog.save_snapshot(
        "openai",
        [
            ModelRecord(id="ft:gpt-x:org:suffix", provider="openai"),
            ModelRecord(id="gpt-x", provider="openai"),
        ],
        path=path,
    )
    ids = [r.id for r in _catalog.load_snapshot_records("openai", path=path)]
    assert ids == ["gpt-x"]
    assert "ft:" not in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The snapshots effGen actually ships
# ---------------------------------------------------------------------------

def test_every_shipped_snapshot_loads_and_counts_itself_correctly() -> None:
    checked = 0
    for path in sorted(_catalog._DATA_DIR.glob("*.json")):
        provider = path.stem
        data = _catalog.load_snapshot(provider, path=path)
        if data is None or "models" not in data:
            continue
        records = _catalog.load_snapshot_records(provider, path=path)
        assert len(records) == len(data["models"]), path
        if isinstance(data.get("count"), int):
            assert data["count"] == len(records), path
        assert all(r.id and r.provider for r in records), path
        checked += 1
    assert checked >= 1, "no shipped snapshot was checked"
