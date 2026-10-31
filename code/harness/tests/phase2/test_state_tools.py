"""State tools use the harness's existing BaseTool registration and never verify."""

from __future__ import annotations

import json
from pathlib import Path

from state_helpers import goal, new_store, ref_for

from effgen.tools.base_tool import BaseTool
from webagent.effgen_bridge import build_effgen_agent
from webagent.models.fake import ScriptedModel
from webagent.state import (
    STATE_TOOL_NAMES,
    StateStore,
    StateToolRuntime,
    StateUpdate,
    build_state_tools,
)


def _tools(store: StateStore):
    return build_state_tools(StateToolRuntime(store))


def test_state_tools_register_through_existing_base_tool_path(tmp_path: Path) -> None:
    store = new_store(tmp_path / "runtime")
    tools = _tools(store)
    assert [tool.name for tool in tools] == ["state_read", "state_update", "plan_update"]
    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert set(tool.name for tool in tools) == STATE_TOOL_NAMES

    agent = build_effgen_agent(ScriptedModel([]), extra_tools=tools)
    assert STATE_TOOL_NAMES.issubset(set(agent.tools))


def test_state_update_tool_records_but_does_not_verify(tmp_path: Path) -> None:
    store = new_store(tmp_path / "runtime")
    read_tool, update_tool, _ = _tools(store)
    result = update_tool.execute(
        base_revision=0,
        plan_version=0,
        goals=[
            {
                "id": "g1",
                "requirement": "collect the requested objects",
                "acceptance_criteria": ["each object is present"],
                "object_scope": ["obj-1"],
                "status": "pending",
            }
        ],
        note="first requirement",
    )
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["recorded"] is True
    assert payload["revision"] == 1
    assert "does not verify" in payload["message"]
    assert store.state.goals["g1"].status.value == "pending"
    assert store.state.verified_goal_ids == ()

    read = read_tool.execute(goal_id="g1")
    assert read.success is True
    view = json.loads(read.output)
    assert view["revision"] == 1
    assert view["completion_status"] == "incomplete"
    assert view["goals"]["g1"]["status"] == "pending"


def test_state_update_tool_returns_current_state_on_stale_revision(tmp_path: Path) -> None:
    store = new_store(tmp_path / "runtime")
    _, update_tool, _ = _tools(store)
    first = update_tool.execute(
        base_revision=0,
        plan_version=0,
        goals=[
            {
                "id": "g1",
                "requirement": "collect the requested objects",
                "acceptance_criteria": ["object present"],
            }
        ],
    )
    assert first.success is True

    stale = update_tool.execute(
        base_revision=0,
        plan_version=0,
        goals=[
            {
                "id": "g2",
                "requirement": "second requirement",
                "acceptance_criteria": ["second present"],
            }
        ],
    )
    assert stale.success is False
    error = json.loads(stale.error)
    assert error["reason"] == "stale_base_revision"
    assert error["current"]["revision"] == 1
    assert stale.metadata["current_revision"] == 1


def test_plan_tool_refuses_to_delete_unfinished_requirement(tmp_path: Path) -> None:
    store = new_store(tmp_path / "runtime")
    store.apply_update(
        StateUpdate(
            base_revision=0,
            plan_version=0,
            goals=(goal(),),
        )
    )
    _, _, plan_tool = _tools(store)
    result = plan_tool.execute(
        base_revision=1,
        plan_version=0,
        plan=[],
        remove_goal_ids=["g1"],
    )
    assert result.success is False
    assert "cannot remove unfinished goal" in (result.error or "")
    assert set(store.state.goals) == {"g1"}


def test_model_tool_cannot_mark_goal_verified(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    ref = ref_for(runtime)
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(evidence=(ref,)),))
    )
    _, update_tool, _ = _tools(store)
    result = update_tool.execute(
        base_revision=1,
        plan_version=0,
        goals=[
            {
                "id": "g1",
                "requirement": "collect the requested objects",
                "acceptance_criteria": ["each object is present"],
                "status": "verified",
                "evidence": ["ev-1"],
            }
        ],
    )
    assert result.success is False
    assert "cannot become verified" in (result.error or "")
    assert store.state.verified_goal_ids == ()
    assert store.state.revision == 1
