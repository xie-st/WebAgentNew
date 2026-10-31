from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from webagent.testing.cases import (
    CaseValidationError,
    agent_payload,
    case_ids,
    list_cases,
    load_case,
)
from webagent.testing.site import generate_dataset
from webagent.testing.site.faults import FaultDefinition, FaultPlanError


def test_all_declared_cases_load_with_required_fields():
    ids = case_ids()
    assert ids[:3] == ("L01", "L02", "L03")
    assert set(ids) == {"L01", "L02", "L03", "L04", "L05", "L06", "L07"}
    cases = list_cases()
    assert [case.case_id for case in cases] == list(ids)
    for case in cases:
        assert case.case_version == "1"
        assert case.title
        assert case.start_path.startswith("/")
        assert case.seeds == (101, 202, 303)
        assert set(case.support["reserved_seeds"]) == {404, 505}
        assert case.budget_seconds in (600, 1800)
        assert case.setup["page_size"] == case.generator_params["page_size"]
        assert case.completion["method"] == "local_judge"
        assert case.completion["criteria"]
        assert case.reset["method"] == "rebuild_database"
        for fault in case.fault_plan:
            assert fault["fault_id"]
            assert fault["type"]
            assert "trigger_event" in fault and "trigger_after" in fault
            assert isinstance(fault["params"], dict)


def test_task_templates_render_and_have_no_placeholders():
    for case in list_cases():
        task = case.render_task()
        assert task.strip()
        assert "{" not in task and "}" not in task
        assert case.agent_view()["task"] == task


def test_agent_view_and_hidden_view_are_separate():
    case = load_case("L02")
    view = case.agent_view()
    assert set(view) == {
        "case_id",
        "case_version",
        "task",
        "start_path",
        "task_params",
        "budget_seconds",
        "modes",
    }
    serialized = agent_payload(case)
    assert "l02_match" not in serialized
    assert "max_submissions" not in serialized
    assert "require_evidence_each" not in serialized
    assert "role" not in view and "judge" not in view
    assert "data_generator" not in view
    hidden = case.hidden_view()
    assert hidden["hidden"]["judge"]["role"] == "l02_match"
    assert hidden["fault_plan"]


def test_generator_produces_expected_hidden_truth_per_seed():
    cases = {case.case_id: case for case in list_cases()}
    for case_id, expected_role, expected_count in (
        ("L01", "l01_target", 1),
        ("L02", "l02_match", 5),
        ("L03", "l03_target", 20),
        ("L04", "l01_target", 1),
        ("L05", "l02_match", 3),
        ("L06", "l06_target", 1),
        ("L07", "l07_target", 1),
    ):
        case = cases[case_id]
        for seed in case.seeds:
            dataset = generate_dataset(seed, case.generator_params)
            records = [record for record in dataset.hidden if record.role == expected_role]
            assert len(records) == expected_count
            assert len({record.item_id for record in records}) == expected_count
            assert len(dataset.products) == 32
            assert len(dataset.posts) == 20
            assert all(record.item_type == case.judge_config["object_type"] for record in records)


def _base_case_dict(case_id: str = "L01") -> dict:
    return load_case(case_id).to_dict()


def test_l02_match_positions_span_multiple_pages():
    case = load_case("L02")
    for seed in case.seeds:
        dataset = generate_dataset(seed, case.generator_params)
        matches = [record.item_id for record in dataset.hidden if record.role == "l02_match"]
        positions = [
            product.position
            for product in dataset.products
            if product.id in set(matches)
        ]
        pages = {position // case.setup["page_size"] for position in positions}
        assert len(matches) == 5
        assert len(pages) >= 3


def test_missing_required_field_is_rejected(tmp_path: Path):
    raw = _base_case_dict()
    raw.pop("support")
    path = tmp_path / "L01.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(CaseValidationError, match="missing required fields"):
        load_case("L01", path=path)


def test_unknown_fault_type_is_rejected(tmp_path: Path):
    raw = _base_case_dict()
    raw["fault_plan"] = [
        {
            "fault_id": "weird",
            "type": "not_a_real_fault",
            "enabled": False,
            "trigger": {"event": "page_view", "after": 1},
            "params": {},
        }
    ]
    path = tmp_path / "L01.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(FaultPlanError, match="unknown fault type"):
        load_case("L01", path=path)


def test_reserved_fault_fields_are_accepted_but_cannot_be_enabled():
    disabled = FaultDefinition.from_mapping(
        {
            "fault_id": "missing",
            "type": "missing_control",
            "enabled": False,
            "trigger": {"event": "view_item", "after": 1},
            "params": {},
        }
    )
    assert disabled.enabled is False
    with pytest.raises(FaultPlanError, match="reserved"):
        FaultDefinition.from_mapping(
            {
                "fault_id": "missing",
                "type": "missing_control",
                "enabled": True,
                "trigger": {"event": "view_item", "after": 1},
                "params": {},
            }
        )


def test_duplicate_button_and_delay_fields_are_reserved_in_the_base_cases():
    base_cases = {"L01", "L02", "L03"}
    for case in list_cases():
        if case.case_id not in base_cases:
            continue
        types = {fault["type"] for fault in case.fault_plan}
        assert "duplicate_button" in types
        assert "slow_response" in types
        assert all(fault["enabled"] is False for fault in case.fault_plan)


def test_only_visible_overlay_faults_may_be_enabled_for_control_cases():
    for case in list_cases():
        for fault in case.fault_plan:
            if fault["enabled"]:
                assert case.case_id in {"L06", "L07"}
                assert fault["type"] in {"login_overlay", "verification_prompt"}


def test_manifest_json_is_round_trippable():
    case = load_case("L03")
    payload = case.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert "\"post_count\": 20" in text
    assert payload["hidden"]["judge"]["max_submissions"] == 0
    assert payload["support"]["budget_seconds"] == 1800
