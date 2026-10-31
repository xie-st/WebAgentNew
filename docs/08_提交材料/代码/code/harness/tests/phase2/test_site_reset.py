from __future__ import annotations

import json

import pytest

from webagent.testing import judge as judge_run
from webagent.testing import reset as reset_run
from webagent.testing import start
from webagent.testing.cases import load_case
from webagent.testing.evaluation import HiddenStateReader

from d1_helpers import fetch


def _display(p):
    products = p.site.store.all_products()
    posts = p.site.store.list_posts(page=1, page_size=100)[0]
    return {
        "products": [dict(row) for row in products],
        "posts": [dict(row) for row in posts],
    }


def test_same_seed_reset_restores_identical_hidden_state(make_case):
    prepared = make_case("L02", 101)
    before = prepared.hidden_snapshot()
    before_view = _display(prepared)

    prepared.reset()
    after = prepared.hidden_snapshot()
    after_view = _display(prepared)

    assert before == after
    assert before_view == after_view
    assert before["meta"]["seed"] == "101"
    assert len(prepared.hidden_truth(role="l02_match")) == 5


def test_reset_clears_activity_drafts_and_submissions(make_case):
    prepared = make_case("L01", 101)
    target_id = prepared.hidden_truth(role="l01_target")[0]["item_id"]
    prepared.site.store.save_draft("product", target_id, "temporary draft")
    assert prepared.site.store.get_draft("product", target_id) == "temporary draft"
    prepared.reset()
    assert prepared.site.store.get_draft("product", target_id) == ""
    assert prepared.site.store.activities() == []
    assert prepared.submission_count() == 0


def test_different_seeds_change_names_values_and_order(make_case):
    first = make_case("L02", 101)
    second = make_case("L02", 202)
    first_view = _display(first)
    second_view = _display(second)

    assert first.hidden_snapshot() != second.hidden_snapshot()
    first_names = [row["name"] for row in first_view["products"]]
    second_names = [row["name"] for row in second_view["products"]]
    assert first_names != second_names
    assert set(first_names) != set(second_names)
    assert [row["price_cents"] for row in first_view["products"]] != [
        row["price_cents"] for row in second_view["products"]
    ]
    assert [row["id"] for row in first_view["products"]] != [
        row["id"] for row in second_view["products"]
    ]
    assert [row["id"] for row in first_view["posts"]] != [
        row["id"] for row in second_view["posts"]
    ]


def test_each_run_has_an_independent_database_copy(make_case):
    first = make_case("L03", 303, run_id="independent-a")
    second = make_case("L03", 303, run_id="independent-b")
    assert first.db_path != second.db_path
    assert first.hidden_snapshot() == second.hidden_snapshot()

    post_id = first.hidden_truth(role="l03_target")[0]["item_id"]
    first.site.store.send_comment("post", post_id, "sent only in copy a")
    assert first.submission_count() == 1
    assert second.submission_count() == 0
    assert second.site.store.get_draft("post", post_id) == ""


def test_reset_to_another_seed_updates_seed_and_data(make_case):
    prepared = make_case("L01", 101)
    before = prepared.hidden_snapshot()
    prepared.reset(202)
    after = prepared.hidden_snapshot()
    assert before != after
    assert prepared.seed == 202
    assert after["meta"]["seed"] == "202"
    assert len(prepared.hidden_truth(role="l01_target")) == 1


def test_agent_view_contains_no_hidden_truth_or_fault_plan(make_case):
    prepared = make_case("L03", 101)
    payload = prepared.agent_view()
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for token in (
        "l03_target",
        "hidden",
        "fault_plan",
        "fault_id",
        "duplicate-send",
        "expected_count",
        "require_emoji",
        "hidden_truth",
    ):
        assert token not in serialized
    assert "task" in payload and "start_url" in payload
    assert payload["task_params"]["post_count"] == 20


def test_seed_validation_and_reserved_seed(tmp_path):
    case = load_case("L01")
    assert case.seeds == (101, 202, 303)
    assert tuple(case.support["reserved_seeds"]) == (404, 505)
    with pytest.raises(ValueError):
        start("L01", 999, runtime_dir=tmp_path / "runtime")
    reserved = start(
        "L01", 505, runtime_dir=tmp_path / "runtime", run_id="reserved-505"
    )
    try:
        assert reserved.seed == 505
        assert reserved.hidden_snapshot()["meta"]["seed"] == "505"
    finally:
        reserved.stop()


def test_site_has_no_truth_tables_in_rendered_pages(make_case):
    prepared = make_case("L01", 101)
    html = "\n".join(
        [
            fetch(prepared.base_url + "/"),
            fetch(prepared.base_url + "/products"),
            fetch(prepared.base_url + "/posts"),
        ]
    )
    assert "hidden_truth" not in html
    assert "hidden_config" not in html
    assert "l01_target" not in html


def test_programmatic_reset_and_judge_entry_points(make_case):
    prepared = make_case("L02", 303)
    reset_run(prepared)
    result = judge_run(prepared, evidence=[], answer={"product_ids": []})
    assert not result.passed
    assert result.case_id == "L02"
    assert result.seed == 303


def test_external_hidden_reader_matches_prepared_snapshot(make_case):
    prepared = make_case("L01", 101)
    reader = HiddenStateReader(prepared.db_path)
    assert reader.snapshot() == prepared.hidden_snapshot()
