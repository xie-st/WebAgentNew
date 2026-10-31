from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pytest
import yaml

from webagent.testing import judge as judge_entry
from webagent.testing.cases import load_case
from webagent.testing.evaluation import (
    HiddenStateReader,
    JudgeDispatchError,
    LocalJudge,
    judge_checker_name,
    resolve_judge_checker,
)

from d1_helpers import fetch, screenshot


def _target(prepared) -> str:
    role = str(prepared.case.judge_config["role"])
    return str(prepared.hidden_truth(role=role)[0]["item_id"])


def _evidence(object_type: str, object_id: str, path: Path) -> dict:
    return {
        "object_type": object_type,
        "object_id": object_id,
        "kind": "screenshot",
        "reference": str(path),
    }


def _save(prepared, target_id: str, text: str) -> None:
    """Use the site's normal draft save endpoint (no test-only route)."""

    body = urlencode({"action": "save", "text": text}).encode("utf-8")
    html = fetch(f"{prepared.base_url}/product/{target_id}/draft", data=body)
    assert "Review draft" in html


def _checks(result) -> dict[str, object]:
    return {check.name: check for check in result.checks}


def test_new_case_is_dispatched_through_the_unified_judge_entry(make_case, tmp_path: Path):
    prepared = make_case("L15", 101)
    case = prepared.case
    assert judge_checker_name(case) == "field_replacement"
    checker = resolve_judge_checker(case)
    assert callable(checker)
    # The new case must be judged by the shared dispatch table, not by a
    # case-specific method on LocalJudge.
    assert not hasattr(LocalJudge, "_judge_l15")

    serialized = json.dumps(case.agent_view(), ensure_ascii=False, sort_keys=True)
    for token in ("field_replacement", "l01_target", "replaced_text", "expected_count"):
        assert token not in serialized
    assert case.hidden_view()["hidden"]["judge"]["checker"] == "field_replacement"

    # An unknown checker fails closed instead of silently passing.
    raw = case.to_dict()
    raw["hidden"]["judge"]["checker"] = "not_a_registered_checker"
    path = tmp_path / "L15.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    bad_case = load_case("L15", path=path)
    with pytest.raises(JudgeDispatchError, match="not_a_registered_checker"):
        resolve_judge_checker(bad_case)


def test_correct_replacement_passes_the_unified_judge(make_case, browser):
    prepared = make_case("L15", 101)
    target_id = _target(prepared)
    initial = str(prepared.case.task_params["l15_initial_text"])
    expected = str(prepared.case.task_params["l15_expected_text"])

    _save(prepared, target_id, initial)

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/product/{target_id}")
        field = page.get_by_label("Review draft")
        assert field.input_value() == initial
        field.fill(expected)
        page.get_by_role("button", name="Save draft").click()
        page.wait_for_load_state("domcontentloaded")
        assert field.input_value() == expected
        shot = screenshot(page, prepared.run_dir / "evidence" / "l15-replaced.png")
    finally:
        context.close()

    result = prepared.judge(
        evidence=[_evidence("product", target_id, shot)],
        answer={"passed": True, "product_ids": [target_id]},
    )
    assert result.passed, result.to_dict()
    checks = _checks(result)
    assert checks["field_value_exact"].passed
    assert checks["field_replaced"].passed
    assert checks["baseline_saved"].passed
    assert checks["screenshot"].passed
    reader = HiddenStateReader(prepared.db_path)
    assert reader.current_draft("product", target_id) == expected


def test_wrong_append_fails_with_a_concrete_gap(make_case, browser):
    prepared = make_case("L15", 202)
    target_id = _target(prepared)
    initial = str(prepared.case.task_params["l15_initial_text"])
    expected = str(prepared.case.task_params["l15_expected_text"])
    appended = f"{initial}{expected}"

    _save(prepared, target_id, initial)

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/product/{target_id}")
        field = page.get_by_label("Review draft")
        field.fill(appended)
        page.get_by_role("button", name="Save draft").click()
        page.wait_for_load_state("domcontentloaded")
        shot = screenshot(page, prepared.run_dir / "evidence" / "l15-appended.png")
    finally:
        context.close()

    result = prepared.judge(evidence=[_evidence("product", target_id, shot)])
    assert not result.passed, result.to_dict()
    checks = _checks(result)
    assert not checks["field_value_exact"].passed
    assert not checks["field_replaced"].passed
    gap = checks["field_replaced"].gap
    assert "appended, not replaced" in gap
    assert initial in gap
    assert checks["field_value_exact"].gap == gap
    assert checks["field_replaced"].to_dict()["gap"] == gap
    reader = HiddenStateReader(prepared.db_path)
    assert reader.current_draft("product", target_id) == appended


def test_detected_append_can_be_corrected_in_the_same_run(make_case, browser):
    prepared = make_case("L15", 303)
    target_id = _target(prepared)
    initial = str(prepared.case.task_params["l15_initial_text"])
    expected = str(prepared.case.task_params["l15_expected_text"])
    appended = f"{initial}{expected}"

    _save(prepared, target_id, initial)

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/product/{target_id}")
        field = page.get_by_label("Review draft")
        field.fill(appended)
        page.get_by_role("button", name="Save draft").click()
        page.wait_for_load_state("domcontentloaded")
    finally:
        context.close()

    rejected = prepared.judge(evidence=[])
    assert not rejected.passed
    assert not _checks(rejected)["field_replaced"].passed

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/product/{target_id}")
        field = page.get_by_label("Review draft")
        assert field.input_value() == appended
        field.fill(expected)
        page.get_by_role("button", name="Save draft").click()
        page.wait_for_load_state("domcontentloaded")
        shot = screenshot(page, prepared.run_dir / "evidence" / "l15-corrected.png")
    finally:
        context.close()

    corrected = prepared.judge(evidence=[_evidence("product", target_id, shot)])
    assert corrected.passed, corrected.to_dict()
    assert _checks(corrected)["field_replaced"].gap == ""
    assert HiddenStateReader(prepared.db_path).current_draft("product", target_id) == expected


def test_judge_reads_only_observable_site_state(make_case, browser):
    prepared = make_case("L15", 202)
    target_id = _target(prepared)
    initial = str(prepared.case.task_params["l15_initial_text"])
    expected = str(prepared.case.task_params["l15_expected_text"])
    _save(prepared, target_id, initial)
    _save(prepared, target_id, f"{initial}{expected}")

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/product/{target_id}")
        shot = screenshot(page, prepared.run_dir / "evidence" / "l15-observable.png")
    finally:
        context.close()

    evidence = [_evidence("product", target_id, shot)]
    plain = judge_entry(prepared, evidence=evidence)
    with_claims = prepared.judge(
        evidence=evidence,
        answer={
            "passed": True,
            "verdict": "pass",
            "agent_state": {"goals": {"g1": {"status": "verified"}}},
            "field_value_exact": True,
        },
    )
    assert plain.to_dict()["checks"] == with_claims.to_dict()["checks"]
    assert not with_claims.passed

    # The judge must not import Agent/session/ledger internals.
    import importlib

    judge_module = importlib.import_module("webagent.testing.evaluation.judge")
    source = Path(judge_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "webagent.session",
        "webagent.state",
        "webagent.official_agent",
        "webagent.effgen_bridge",
        "webagent.verification",
    ):
        assert forbidden not in source

    # Judging still works after the site stops: the observable DB is enough.
    prepared.site.stop()
    offline = prepared.judge(evidence=evidence)
    assert offline.to_dict()["checks"] == plain.to_dict()["checks"]
    assert not offline.passed
