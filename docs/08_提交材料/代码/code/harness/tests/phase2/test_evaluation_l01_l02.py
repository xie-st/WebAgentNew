from __future__ import annotations

import json
from pathlib import Path

from webagent.testing.evaluation import HiddenStateReader

from d1_helpers import screenshot


def _evidence(object_type: str, object_id: str, path: Path) -> dict:
    return {
        "object_type": object_type,
        "object_id": object_id,
        "kind": "screenshot",
        "reference": str(path),
    }


def test_l01_cross_page_draft_and_cancel_passes(make_case, browser):
    prepared = make_case("L01", 101)
    target_id = prepared.hidden_truth(role="l01_target")[0]["item_id"]
    product = dict(prepared.site.store.get_product(target_id))
    draft = prepared.case.judge_config["expected_draft_text"]
    page_number = product["position"] // prepared.site.page_size + 1

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/products?page={page_number}")
        link = page.get_by_role("link", name=product["name"], exact=True)
        assert link.is_visible()
        link.click()
        page.wait_for_load_state("domcontentloaded")
        assert target_id in page.url

        page.get_by_label("Review draft").fill(draft)
        shot = screenshot(page, prepared.run_dir / "evidence" / "l01-target.png")
        assert shot.read_bytes().startswith(b"\x89PNG")
        page.get_by_role("button", name="Cancel draft").click()
        page.wait_for_load_state("domcontentloaded")
    finally:
        context.close()

    result = prepared.judge(
        evidence=[_evidence("product", target_id, shot)],
        answer={"product_ids": [target_id]},
    )
    assert result.passed, result.to_dict()
    assert result.seed == 101
    assert len(result.hidden_state_hash) == 64
    assert prepared.submission_count() == 0
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["verdict"] == "pass"


def test_l01_fails_without_screenshot(make_case, browser):
    prepared = make_case("L01", 202)
    target_id = prepared.hidden_truth(role="l01_target")[0]["item_id"]
    draft = prepared.case.judge_config["expected_draft_text"]
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/product/{target_id}")
        page.get_by_label("Review draft").fill(draft)
        page.get_by_role("button", name="Cancel draft").click()
        page.wait_for_load_state("domcontentloaded")
    finally:
        context.close()
    result = prepared.judge(evidence=[])
    assert not result.passed
    assert any(check.name == "screenshot" and not check.passed for check in result.checks)


def test_l01_fails_when_draft_is_saved_but_not_cancelled(make_case, browser):
    prepared = make_case("L01", 303)
    target_id = prepared.hidden_truth(role="l01_target")[0]["item_id"]
    draft = prepared.case.judge_config["expected_draft_text"]
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/product/{target_id}")
        page.get_by_label("Review draft").fill(draft)
        shot = screenshot(page, prepared.run_dir / "evidence" / "l01-saved.png")
        page.get_by_role("button", name="Save draft").click()
        page.wait_for_load_state("domcontentloaded")
    finally:
        context.close()
    result = prepared.judge(evidence=[_evidence("product", target_id, shot)])
    assert not result.passed
    assert any(check.name == "draft_cancelled" and not check.passed for check in result.checks)


def test_l02_five_distinct_objects_pass(make_case, browser):
    prepared = make_case("L02", 101)
    expected = [
        record["item_id"] for record in prepared.hidden_truth(role="l02_match")
    ]
    evidence = []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        for index, product_id in enumerate(expected):
            page.goto(f"{prepared.base_url}/product/{product_id}")
            path = screenshot(
                page, prepared.run_dir / "evidence" / f"l02-{index}.png"
            )
            evidence.append(_evidence("product", product_id, path))
    finally:
        context.close()

    result = prepared.judge(
        evidence=evidence, answer={"product_ids": expected}
    )
    assert result.passed, result.to_dict()
    assert result.to_dict()["evidence_count"] == 5


def test_l02_fails_with_duplicate_or_incomplete_answer(make_case, browser):
    prepared = make_case("L02", 202)
    expected = [
        record["item_id"] for record in prepared.hidden_truth(role="l02_match")
    ]
    evidence = []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        for index, product_id in enumerate(expected):
            page.goto(f"{prepared.base_url}/product/{product_id}")
            path = screenshot(
                page, prepared.run_dir / "evidence" / f"l02-neg-{index}.png"
            )
            evidence.append(_evidence("product", product_id, path))
    finally:
        context.close()

    incomplete = prepared.judge(
        evidence=evidence, answer={"product_ids": expected[:-1]}
    )
    assert not incomplete.passed
    assert any(check.name == "answer_exact" and not check.passed for check in incomplete.checks)

    duplicate = prepared.judge(
        evidence=evidence, answer={"product_ids": [*expected, expected[0]]}
    )
    assert not duplicate.passed
    assert any(
        check.name == "answer_distinct" and not check.passed for check in duplicate.checks
    )

    missing_evidence = prepared.judge(
        evidence=evidence[:-1], answer={"product_ids": expected}
    )
    assert not missing_evidence.passed
    assert any(
        check.name == "screenshot" and not check.passed
        for check in missing_evidence.checks
    )


def test_hidden_state_reader_is_read_only(make_case):
    prepared = make_case("L01", 101)
    reader = HiddenStateReader(prepared.db_path)
    assert len(reader.products()) == 32
    assert reader.meta()["seed"] == "101"
    connection = reader._connect()
    try:
        try:
            connection.execute("DELETE FROM products")
        except Exception as exc:  # sqlite3.OperationalError
            assert "readonly" in str(exc).lower() or "query_only" in str(exc).lower()
        else:
            raise AssertionError("judge connection unexpectedly allowed a write")
    finally:
        connection.close()


def test_judge_reads_external_database_after_site_stops(make_case):
    prepared = make_case("L01", 101)
    prepared.site.stop()
    result = prepared.judge(evidence=[])
    assert not result.passed
    assert any(check.name == "target_viewed" and not check.passed for check in result.checks)
    assert len(result.hidden_state_hash) == 64
