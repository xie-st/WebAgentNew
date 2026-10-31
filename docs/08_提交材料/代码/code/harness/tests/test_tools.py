from __future__ import annotations

import json

import pytest

from webagent.errors import ActionRejected, RejectReason
from webagent.protocol import ActionOp, ElementRef, ToolRequest, WebAction
from webagent.tools import (
    TOOL_SPECS,
    build_web_action,
    guard_requests,
    render_catalog,
    resolve_target,
    same_origin,
    tool_schema_catalog,
)


def test_canonical_tool_catalog_is_single_and_complete() -> None:
    names = [spec.name for spec in TOOL_SPECS]
    assert len(names) == len(set(names))
    assert set(names) == {
        "web_click",
        "web_type",
        "web_press",
        "web_scroll",
        "web_goto",
        "web_back",
        "web_forward",
        "web_observe",
        "task_note",
        "evidence_save",
    }
    web_ops = {spec.op for spec in TOOL_SPECS if spec.is_web_action}
    assert web_ops == {
        ActionOp.CLICK,
        ActionOp.TYPE,
        ActionOp.PRESS,
        ActionOp.SCROLL,
        ActionOp.GOTO,
        ActionOp.GO_BACK,
        ActionOp.GO_FORWARD,
    }
    for spec in TOOL_SPECS:
        assert spec.parameters["additionalProperties"] is False
    catalog = tool_schema_catalog()
    json.dumps(catalog)
    assert [item["name"] for item in catalog] == names
    rendered = render_catalog()
    assert "one action per step" in rendered
    assert "web_observe (internal)" in rendered


def test_build_action_binds_current_observation(make_observation) -> None:
    observation = make_observation()
    action = build_web_action(
        ToolRequest("web_click", {"target": "1"}), observation, action_id="act-1"
    )
    assert action.run_id == observation.run_id
    assert action.page_id == observation.page_id
    assert action.observation_id == observation.observation_id
    assert action.op is ActionOp.CLICK
    assert resolve_target(action, observation).role == "textbox"


def test_same_number_in_new_observation_is_a_new_element(make_observation) -> None:
    first = make_observation(observation_id="obs-1", elements={"1": ("textbox", "Query")})
    second = make_observation(observation_id="obs-2", elements={"1": ("link", "Result")})
    action = build_web_action(ToolRequest("web_click", {"target": "1"}), second, action_id="act-2")
    assert action.observation_id == "obs-2"
    assert resolve_target(action, second).role == "link"
    old_action = build_web_action(
        ToolRequest("web_click", {"target": "1"}), first, action_id="act-1"
    )
    with pytest.raises(ActionRejected) as excinfo:
        resolve_target(old_action, second)
    assert excinfo.value.reason is RejectReason.STALE_OBSERVATION
    assert "obs-2" in excinfo.value.correction


def test_action_from_wrong_page_or_run_is_rejected(make_observation) -> None:
    observation = make_observation()
    wrong_page = WebAction(
        action_id="act-1",
        run_id="run-a",
        page_id="page-9",
        observation_id=observation.observation_id,
        op="click",
        args={"target": "1"},
    )
    with pytest.raises(ActionRejected) as excinfo:
        resolve_target(wrong_page, observation)
    assert excinfo.value.reason is RejectReason.WRONG_PAGE

    wrong_run = WebAction(
        action_id="act-1",
        run_id="run-b",
        page_id=observation.page_id,
        observation_id=observation.observation_id,
        op="click",
        args={"target": "1"},
    )
    with pytest.raises(ActionRejected) as excinfo:
        resolve_target(wrong_run, observation)
    assert excinfo.value.reason is RejectReason.WRONG_RUN


def test_multiple_web_actions_are_rejected_with_correction(make_observation) -> None:
    observation = make_observation()
    requests = [
        ToolRequest("web_click", {"target": "1"}),
        ToolRequest("web_scroll", {"direction": "down"}),
    ]
    with pytest.raises(ActionRejected) as excinfo:
        guard_requests(requests, observation)
    rejected = excinfo.value
    assert rejected.reason is RejectReason.MULTIPLE_ACTIONS
    assert "2 web actions" in rejected.correction
    assert "no action from this batch was queued" in rejected.correction
    assert "obs-1" in rejected.correction
    assert "1, 2" in rejected.correction


def test_internal_tools_do_not_become_page_actions(make_observation) -> None:
    observation = make_observation()
    result = guard_requests(
        [
            ToolRequest("web_observe", {}),
            ToolRequest("task_note", {"note": "price is 10"}),
            ToolRequest("evidence_save", {"kind": "screenshot"}),
        ],
        observation,
    )
    assert result.action is None
    assert [request.tool for request in result.internal] == [
        "web_observe",
        "task_note",
        "evidence_save",
    ]
    assert result.requires_environment is False
    with pytest.raises(ActionRejected) as excinfo:
        build_web_action(ToolRequest("web_observe", {}), observation)
    assert excinfo.value.reason is RejectReason.INVALID_ARGUMENTS


def test_internal_tool_arguments_are_validated(make_observation) -> None:
    observation = make_observation()
    with pytest.raises(ActionRejected) as excinfo:
        guard_requests([ToolRequest("task_note", {})], observation)
    assert excinfo.value.reason is RejectReason.INVALID_ARGUMENTS
    assert "task_note requires: note" in excinfo.value.correction


def test_unknown_tool_and_unknown_element(make_observation) -> None:
    observation = make_observation()
    with pytest.raises(ActionRejected) as excinfo:
        guard_requests([ToolRequest("browser_eval", {"js": "1"})], observation)
    assert excinfo.value.reason is RejectReason.UNKNOWN_TOOL
    with pytest.raises(ActionRejected) as excinfo:
        build_web_action(ToolRequest("web_click", {"target": "99"}), observation)
    assert excinfo.value.reason is RejectReason.UNKNOWN_ELEMENT
    assert "element '99'" in excinfo.value.details[0]
    assert "1, 2" in excinfo.value.correction


def test_type_has_no_hidden_clear_flag(make_observation) -> None:
    observation = make_observation()
    action = build_web_action(
        ToolRequest("web_type", {"target": "1", "text": "abc"}), observation
    )
    assert action.submit is False
    with pytest.raises(ActionRejected) as excinfo:
        build_web_action(
            ToolRequest("web_type", {"target": "1", "text": "abc", "clear": True}), observation
        )
    assert excinfo.value.reason is RejectReason.INVALID_ARGUMENTS
    assert "does not accept: clear" in excinfo.value.details[0]


def test_goto_is_limited_to_same_origin(make_observation) -> None:
    observation = make_observation(url="http://fixture.local/start")
    assert same_origin("http://fixture.local/results", observation.url)
    assert not same_origin("https://fixture.local/results", observation.url)
    with pytest.raises(ActionRejected) as excinfo:
        build_web_action(
            ToolRequest("web_goto", {"url": "https://evil.example/collect"}), observation
        )
    assert excinfo.value.reason is RejectReason.EXTERNAL_ORIGIN
    action = build_web_action(
        ToolRequest("web_goto", {"url": "http://fixture.local/results"}), observation
    )
    assert action.op is ActionOp.GOTO
    resolve_target(action, observation)


def test_restricted_capability_rejects_action(make_observation) -> None:
    from webagent.protocol import Capabilities

    observation = make_observation()
    restricted = type(observation)(
        run_id=observation.run_id,
        page_id=observation.page_id,
        observation_id=observation.observation_id,
        url=observation.url,
        ax_text=observation.ax_text,
        elements=observation.elements,
        capabilities=Capabilities(actions={ActionOp.CLICK}),
    )
    with pytest.raises(ActionRejected) as excinfo:
        build_web_action(ToolRequest("web_scroll", {"direction": "down"}), restricted)
    assert excinfo.value.reason is RejectReason.INVALID_ARGUMENTS
    build_web_action(ToolRequest("web_click", {"target": "1"}), restricted)
