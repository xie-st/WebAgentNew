"""Property and regression tests for session and checkpoint round-trips.

A session or checkpoint file is JSON that a previous effGen build, another process
or a hand edit produced, and every listing, resume and export path reads it back.

Asserts that:
  1. ``save`` → ``load`` returns the same field values, for arbitrary content —
     text with control characters, nested metadata, large numbers, non-ASCII.
  2. A file that is valid JSON but not a session/checkpoint/state document raises
     ``CorruptStateError`` naming the file, not an ``AttributeError`` from inside
     the loader.
  3. A store holding a damaged file still lists every readable one, and reports
     the damaged file rather than under-counting silently.
  4. A cost that is not a finite number is not summed into a reported total, and
     a per-run cost is counted once however the run ids were written.
  5. A record loaded from a file whose fields drifted carries the types its class
     declares, so the conversation view, the search and a resuming agent all read
     it rather than crashing on it.
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.core.checkpoint import Checkpoint, CheckpointManager
from effgen.core.session import Session, SessionManager
from effgen.core.state import AgentState
from effgen.errors import CorruptStateError

pytestmark = pytest.mark.fuzz

_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), max_size=60
)

_JSON = st.recursive(
    st.none() | st.booleans() | st.integers(min_value=-(10**12), max_value=10**12)
    | st.floats(allow_nan=False, allow_infinity=False, width=32) | _TEXT,
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(_TEXT, children, max_size=3),
    max_leaves=8,
)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(content=_TEXT, metadata=st.dictionaries(_TEXT, _JSON, max_size=3))
def test_a_session_survives_save_and_load(tmp_path, content, metadata) -> None:
    store = str(tmp_path / "sessions")
    session = Session(session_id="rt", agent_name="fuzz", metadata=metadata)
    session.add_user_message(content)
    session.add_assistant_message(content)
    session.save(store)

    loaded = Session.load("rt", store)
    assert loaded.session_id == session.session_id
    assert loaded.agent_name == session.agent_name
    assert loaded.metadata == metadata
    assert [m["content"] for m in loaded.messages] == [content, content]
    assert [m["role"] for m in loaded.messages] == ["user", "assistant"]


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(task=_TEXT, memory=st.dictionaries(_TEXT, _JSON, max_size=3),
       iteration=st.integers(min_value=0, max_value=10**6))
def test_a_checkpoint_survives_save_and_load(
    tmp_path, task, memory, iteration
) -> None:
    manager = CheckpointManager(str(tmp_path / "cp"))
    checkpoint = Checkpoint(
        checkpoint_id="rt",
        agent_name="fuzz",
        task=task,
        iteration=iteration,
        memory=memory,
    )
    manager.save(checkpoint)

    loaded = manager.load("rt")
    assert loaded.task == task
    assert loaded.iteration == iteration
    assert loaded.memory == memory
    assert manager.load_latest().checkpoint_id == "rt"


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(task=_TEXT, memory=st.dictionaries(_TEXT, _JSON, max_size=3))
def test_a_sqlite_checkpoint_survives_save_and_load(tmp_path, task, memory) -> None:
    manager = CheckpointManager(str(tmp_path / "cpdb"), backend="sqlite")
    manager.save(
        Checkpoint(
            checkpoint_id="rt", agent_name="fuzz", task=task, iteration=3,
            memory=memory,
        )
    )
    loaded = manager.load("rt")
    assert loaded.task == task
    assert loaded.memory == memory


# ---------------------------------------------------------------------------
# A file that is not a session document
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    ["null", "[]", "[1, 2, 3]", '"a string"', "12", "true", '[{"session_id": "x"}]'],
)
def test_a_non_object_session_file_names_the_file(tmp_path, payload) -> None:
    store = tmp_path / "sessions"
    store.mkdir()
    (store / "broken.json").write_text(payload, encoding="utf-8")
    with pytest.raises(CorruptStateError) as excinfo:
        Session.load("broken", str(store))
    assert str(store / "broken.json") in str(excinfo.value)


@pytest.mark.parametrize("payload", ["null", "[]", "[1, 2]", '"text"', "5"])
def test_a_non_object_checkpoint_file_names_the_file(tmp_path, payload) -> None:
    store = tmp_path / "cp"
    store.mkdir()
    (store / "broken.json").write_text(payload, encoding="utf-8")
    manager = CheckpointManager(str(store))
    with pytest.raises(CorruptStateError) as excinfo:
        manager.load("broken")
    assert "broken.json" in str(excinfo.value)


def test_a_damaged_file_is_reported_not_hidden(tmp_path) -> None:
    """A listing never under-counts without saying which file it could not read."""
    store = tmp_path / "sessions"
    store.mkdir()
    Session(session_id="good", agent_name="a").save(str(store))
    (store / "bad.json").write_text("[1, 2, 3]", encoding="utf-8")
    (store / "worse.json").write_text("{not json", encoding="utf-8")

    manager = SessionManager(str(store))
    sessions, unreadable = manager.scan()
    assert [s["session_id"] for s in sessions] == ["good"]
    assert sorted(u["file"] for u in unreadable) == ["bad.json", "worse.json"]
    assert all(u["reason"] for u in unreadable)


def test_a_listing_survives_timestamps_of_mixed_types(tmp_path) -> None:
    """Two files whose ``updated_at`` are different JSON types still both list."""
    store = tmp_path / "sessions"
    store.mkdir()
    for name, updated in (("a", "2026-01-01T00:00:00"), ("b", 5), ("c", None)):
        (store / f"{name}.json").write_text(
            json.dumps({"session_id": name, "messages": [], "updated_at": updated}),
            encoding="utf-8",
        )
    manager = SessionManager(str(store))
    assert sorted(s["session_id"] for s in manager.list_sessions()) == ["a", "b", "c"]
    # A timestamp that is not text is not a timestamp — nothing is deleted on a guess.
    manager.cleanup(older_than_days=0)
    assert (store / "b.json").exists()
    assert (store / "c.json").exists()
    assert not (store / "a.json").exists()


def test_a_checkpoint_listing_survives_a_non_object_file(tmp_path) -> None:
    store = tmp_path / "cp"
    manager = CheckpointManager(str(store))
    manager.save(Checkpoint(checkpoint_id="good", agent_name="a", task="t", iteration=1))
    (store / "bad.json").write_text("[1, 2, 3]", encoding="utf-8")
    (store / "mixed.json").write_text(
        json.dumps({"checkpoint_id": "mixed", "created_at": 5}), encoding="utf-8"
    )
    ids = [c["checkpoint_id"] for c in manager.list_checkpoints()]
    assert "good" in ids and "mixed" in ids
    assert "bad" not in ids


# ---------------------------------------------------------------------------
# Message metadata read off disk
# ---------------------------------------------------------------------------

def _session_file(store, messages) -> None:
    store.mkdir(exist_ok=True)
    (store / "s.json").write_text(
        json.dumps({"session_id": "s", "messages": messages}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "messages",
    [
        3,
        "abc",
        None,
        [None],
        ["plain"],
        [{"role": "assistant", "metadata": None}],
        [{"role": "assistant", "metadata": "x"}],
        [{"role": "assistant", "metadata": []}],
    ],
)
def test_a_drifted_messages_field_still_lists(tmp_path, messages) -> None:
    store = tmp_path / "sessions"
    _session_file(store, messages)
    summaries = SessionManager(str(store)).list_sessions()
    assert len(summaries) == 1
    assert isinstance(summaries[0]["messages"], int)
    assert SessionManager(str(store)).export("s", "text")


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (0.25, 0.25),
        (True, None),          # a flag is not a cost
        (float("nan"), None),  # not a finite number
        (float("inf"), None),
        ("free", None),
        (None, None),
    ],
)
def test_only_a_finite_number_counts_as_spend(tmp_path, cost, expected) -> None:
    store = tmp_path / "sessions"
    _session_file(
        store,
        [{"role": "assistant", "content": "a", "metadata": {"cost_usd": cost}}],
    )
    total = SessionManager(str(store)).list_sessions()[0]["cost_usd"]
    if expected is None:
        assert total is None
    else:
        assert total is not None and math.isclose(total, expected)


def test_a_run_is_counted_once_however_its_id_was_written(tmp_path) -> None:
    """An unhashable run id read off disk de-duplicates on its text, not a crash."""
    store = tmp_path / "sessions"
    _session_file(
        store,
        [
            {"role": "assistant", "metadata": {"run_id": ["r"], "cost_usd": 1.0}},
            {"role": "assistant", "metadata": {"run_id": ["r"], "cost_usd": 1.0}},
            {"role": "assistant", "metadata": {"run_id": "other", "cost_usd": 2.0}},
        ],
    )
    assert SessionManager(str(store)).list_sessions()[0]["cost_usd"] == 3.0


def test_from_dict_refuses_a_document_that_is_not_a_mapping() -> None:
    for payload in (None, [], [1, 2], "text", 5):
        with pytest.raises(ValueError, match="expected a JSON object"):
            Session.from_dict(payload)
        with pytest.raises(ValueError, match="expected a JSON object"):
            Checkpoint.from_dict(payload)


# ---------------------------------------------------------------------------
# A loaded record is usable by every reader, not only by the listing
# ---------------------------------------------------------------------------

def test_a_loaded_session_carries_the_field_types_it_declares(tmp_path) -> None:
    """A drifted file loads as a session a reader can render, not one that crashes.

    Every consumer — the conversation view, the search, an agent hydrating its
    memory — iterates ``messages`` and reads ``metadata``. A file whose fields
    hold another type used to hand each of them an ``AttributeError`` from deep
    inside itself.
    """
    store = tmp_path / "sessions"
    store.mkdir()
    (store / "d.json").write_text(
        json.dumps({
            "session_id": 5,
            "agent_name": ["a"],
            "messages": "abc",
            "memory": "nope",
            "metadata": [1, 2],
            "created_at": 5,
            "updated_at": 7,
        }),
        encoding="utf-8",
    )
    session = Session.load("d", str(store))

    assert isinstance(session.session_id, str)
    assert isinstance(session.agent_name, str)
    assert session.messages == []
    assert session.memory == {}
    assert session.metadata == {}
    assert isinstance(session.created_at, str)
    # What the conversation view and the search do with it.
    assert [str(m.get("content", "")) for m in session.messages] == []
    assert session.metadata.get("model") is None


def test_a_message_that_is_not_a_message_is_dropped_not_rendered(tmp_path) -> None:
    store = tmp_path / "sessions"
    store.mkdir()
    (store / "d.json").write_text(
        json.dumps({
            "session_id": "d",
            "messages": ["plain", 5, None, {"role": "user", "content": "hi"}],
        }),
        encoding="utf-8",
    )
    session = Session.load("d", str(store))
    assert [m["content"] for m in session.messages] == ["hi"]


def test_a_loaded_checkpoint_carries_the_field_types_it_declares(tmp_path) -> None:
    store = tmp_path / "cp"
    manager = CheckpointManager(str(store))
    (store / "d.json").write_text(
        json.dumps({
            "checkpoint_id": 1,
            "agent_name": None,
            "task": "t",
            "iteration": "many",
            "memory": "x",
            "tool_states": 5,
            "metadata": None,
            "tokens_used": "lots",
        }),
        encoding="utf-8",
    )
    checkpoint = manager.load("d")
    assert checkpoint.checkpoint_id == "1"
    assert checkpoint.agent_name == ""
    assert checkpoint.iteration == 0
    assert checkpoint.memory == {}
    assert checkpoint.tool_states == {}
    assert checkpoint.metadata == {}
    assert checkpoint.tokens_used == 0


def test_a_repair_is_reported_not_silent(tmp_path, caplog) -> None:
    store = tmp_path / "sessions"
    store.mkdir()
    (store / "d.json").write_text(
        json.dumps({"session_id": "d", "messages": "abc"}), encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        Session.load("d", str(store))
    assert "messages" in caplog.text


# ---------------------------------------------------------------------------
# Agent state — the third document a run persists
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", ["null", "[]", "[1, 2]", '"text"', "5", "{oops"])
def test_a_state_file_that_is_not_a_state_document_names_the_file(
    tmp_path, payload
) -> None:
    path = tmp_path / "state.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(CorruptStateError) as excinfo:
        AgentState.load(str(path))
    assert str(path) in str(excinfo.value)


def test_a_drifted_state_file_loads_as_usable_state(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({
            "agent_id": 7,
            "conversation_history": "abc",
            "tool_history": None,
            "memory": [1],
            "metadata": "x",
            "created_at": 5,
            "updated_at": "not-a-timestamp",
        }),
        encoding="utf-8",
    )
    state = AgentState.load(str(path))
    assert state.agent_id == "7"
    assert state.conversation_history == []
    assert state.tool_history == []
    assert state.memory == {} and state.metadata == {}
    assert isinstance(state.created_at, datetime)
    assert isinstance(state.updated_at, datetime)
    # The reader an agent is: hydrate from history, then keep going.
    state.add_message("user", "hi")
    assert state.conversation_history[-1]["content"] == "hi"


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(agent_id=_TEXT, metadata=st.dictionaries(_TEXT, _JSON, max_size=3))
def test_agent_state_survives_save_and_load(tmp_path, agent_id, metadata) -> None:
    path = tmp_path / "state.json"
    state = AgentState(agent_id=agent_id, metadata=metadata)
    state.add_message("user", agent_id)
    state.save(str(path))

    loaded = AgentState.load(str(path))
    assert loaded.agent_id == agent_id
    assert loaded.metadata == metadata
    assert [m["content"] for m in loaded.conversation_history] == [agent_id]
