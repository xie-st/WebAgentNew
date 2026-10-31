"""v0.2.x behavioral compatibility: documented entry points, additive aliases, and
forgiving migration of serialized session/state/memory files.

No live calls — these exercise the offline public contract only.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- imports
def test_documented_imports_resolve():
    """The headline v0.2.x entry points import from the top level."""
    from effgen import (  # noqa: F401
        Agent,
        AgentConfig,
        BaseTool,
        ConfigLoader,
        Message,
        ShortTermMemory,
        create_agent,
        list_presets,
        list_providers,
        load_model,
    )

    # Construction must work without downloading a model, so this import-surface
    # smoke stays fast and offline (no network in CI). A model id string would
    # eagerly load weights here; ``model=None`` + ``require_model=False`` defers
    # loading while still exercising the documented AgentConfig/Agent surface.
    cfg = AgentConfig(name="demo", model=None, require_model=False)
    assert Agent(config=cfg) is not None


# ------------------------------------------------------------- additive aliases
def test_configloader_load_alias_matches_load_config():
    from effgen import ConfigLoader

    loader = ConfigLoader()
    assert hasattr(loader, "load")
    assert hasattr(loader, "load_config")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.yaml"
        p.write_text("models:\n  demo:\n    temperature: 0.5\n")
        cfg = loader.load(str(p), validate=False)
        assert cfg.get("models", {}).get("demo", {}).get("temperature") == 0.5


def test_short_term_memory_get_messages_alias():
    from effgen import ShortTermMemory
    from effgen.memory.short_term import MessageRole

    mem = ShortTermMemory(max_tokens=1000)
    mem.add_message(MessageRole.USER, "hello")
    mem.add_message(MessageRole.ASSISTANT, "hi there")
    # get_messages is the additive alias; get_recent_messages is the original name.
    assert len(mem.get_messages()) == 2
    assert len(mem.get_recent_messages()) == 2
    assert mem.get_messages(1)[-1].content == "hi there"


def test_rag_text_chunker_alias_present():
    import effgen.rag as rag

    assert hasattr(rag, "TextChunker")
    chunker = rag.TextChunker(chunk_size=20, overlap=0)
    chunks = chunker.split_text("word " * 40)
    assert len(chunks) >= 1


def test_eval_kwargs_aliases():
    from effgen import TestCase, TestSuite

    # TestSuite(test_cases=) and TestCase(input=) are the documented-ergonomic forms.
    tc = TestCase(input="2+2?", expected_output="4")
    suite = TestSuite(test_cases=[tc])
    assert len(suite.test_cases) == 1
    # The original `query` field still reflects the value supplied via input=.
    assert suite.test_cases[0].query == "2+2?"


def test_template_manager_populated_by_default():
    from effgen import TemplateManager

    assert len(TemplateManager().list_templates()) > 0


def test_tool_registry_aliases():
    """Both top-level get_tool_registry and effgen.tools.get_registry resolve a registry."""
    import effgen
    from effgen.tools import get_registry

    r1 = effgen.get_tool_registry()
    r2 = get_registry()
    assert r1.list_tools()
    assert r2.list_tools()


# --------------------------------------------------------- session serialization compat
def test_session_round_trip_clean_no_warning(recwarn):
    from effgen.core.session import Session

    s = Session(session_id="s1", agent_name="demo")
    s.add_user_message("hi")
    s.add_assistant_message("hello")
    restored = Session.from_dict(s.to_dict())
    assert restored.session_id == "s1"
    assert len(restored.messages) == 2
    # A same-version round-trip must not nag the user.
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_session_loads_legacy_unknown_fields_with_warning():
    from effgen.core.session import Session

    legacy = {
        "session_id": "old",
        "agent_name": "demo",
        "messages": [],
        "memory": {},
        "metadata": {},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "title": "a removed field",  # field that no longer exists
        "schema_version": 1,
    }
    with pytest.warns(DeprecationWarning):
        s = Session.from_dict(legacy)
    assert s.session_id == "old"


def test_session_disk_round_trip():
    from effgen.core.session import Session

    with tempfile.TemporaryDirectory() as d:
        s = Session(session_id="disk1")
        s.add_user_message("q")
        s.save(sessions_dir=d)
        loaded = Session.load("disk1", sessions_dir=d)
        assert loaded.messages[0]["content"] == "q"


# ----------------------------------------------------------- agent-state migration
def test_agent_state_json_round_trip():
    from effgen.core.state import AgentState

    st = AgentState(agent_id="ag1")
    st.add_message("user", "hi")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "state.json"
        st.save(str(p))
        loaded = AgentState.load(str(p))
        assert loaded.agent_id == "ag1"
        assert loaded.conversation_history[0]["content"] == "hi"


def test_agent_state_legacy_unknown_field_loads_with_warning():
    from effgen.core.state import AgentState

    data = {
        "agent_id": "ag2",
        "conversation_history": [],
        "tool_history": [],
        "memory": {},
        "metadata": {},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "legacy_field": "x",
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        p.write_text(json.dumps(data))
        with pytest.warns(DeprecationWarning):
            st = AgentState.load(str(p))
    assert st.agent_id == "ag2"


def test_agent_state_missing_required_field_clear_error():
    from effgen.core.state import AgentState

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.json"
        p.write_text(json.dumps({"conversation_history": []}))  # no agent_id
        with pytest.raises(ValueError):
            AgentState.load(str(p))


# ----------------------------------------------------------- memory migration
def test_short_term_memory_round_trip():
    from effgen.memory.short_term import MessageRole, ShortTermMemory

    mem = ShortTermMemory(max_tokens=500)
    mem.add_message(MessageRole.USER, "remember this")
    restored = ShortTermMemory.from_dict(mem.to_dict())
    assert restored.max_tokens == 500
    assert len(restored.get_messages()) == 1


def test_short_term_memory_legacy_config_key_loads_with_warning():
    from effgen.memory.short_term import ShortTermMemory

    data = {
        "config": {"max_tokens": 800, "legacy_opt": True},  # legacy_opt no longer exists
        "messages": [],
        "summaries": [],
        "statistics": {},
    }
    with pytest.warns(DeprecationWarning):
        mem = ShortTermMemory.from_dict(data)
    assert mem.max_tokens == 800


# ----------------------------------------------------------- checkpoint migration
def test_checkpoint_round_trip_clean_no_warning(recwarn):
    from effgen.core.checkpoint import Checkpoint

    cp = Checkpoint(checkpoint_id="c1", agent_name="demo", task="t", iteration=2)
    restored = Checkpoint.from_dict(cp.to_dict())
    assert restored.checkpoint_id == "c1"
    assert restored.iteration == 2
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_checkpoint_legacy_field_loads_with_warning():
    from effgen.core.checkpoint import Checkpoint

    legacy = {
        "checkpoint_id": "c2",
        "agent_name": "demo",
        "task": "t",
        "iteration": 1,
        "removed_field": "x",  # field a different build no longer has
    }
    with pytest.warns(DeprecationWarning):
        cp = Checkpoint.from_dict(legacy)
    assert cp.checkpoint_id == "c2"


# ----------------------------------------------- long-term memory session migration
def test_long_term_session_legacy_field_loads_with_warning():
    from effgen.memory.long_term import Session as LTSession

    with pytest.warns(DeprecationWarning):
        s = LTSession.from_dict({"id": "lt1", "legacy_field": "y"})
    assert s.id == "lt1"
