from __future__ import annotations

import pytest

from d1_helpers import screenshot

DRAFT = "Useful field note, thanks for sharing the details! 😊"


def _evidence(post_id: str, path) -> dict:
    return {
        "object_type": "post",
        "object_id": post_id,
        "kind": "screenshot",
        "reference": str(path),
    }


@pytest.mark.timeout(240)
def test_l03_twenty_edit_screenshot_cancel_cycles_pass(make_case, browser):
    prepared = make_case("L03", 101)
    posts = [record["item_id"] for record in prepared.hidden_truth(role="l03_target")]
    assert len(posts) == 20
    evidence = []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        for index, post_id in enumerate(posts):
            page.goto(f"{prepared.base_url}/post/{post_id}")
            page.get_by_label("Comment draft").fill(DRAFT)
            path = screenshot(
                page, prepared.run_dir / "evidence" / f"l03-{index:02d}.png"
            )
            evidence.append(_evidence(post_id, path))
            page.get_by_role("button", name="Cancel draft").click()
            page.wait_for_load_state("domcontentloaded")
    finally:
        context.close()

    assert prepared.submission_count() == 0
    result = prepared.judge(evidence=evidence)
    assert result.passed, result.to_dict()
    assert result.to_dict()["evidence_count"] == 20
    assert prepared.submission_count() == 0
    assert len(prepared.site.store.activities(kinds=["draft_cancelled"])) == 20


@pytest.mark.timeout(120)
def test_l03_fails_after_one_send_click(make_case, browser):
    prepared = make_case("L03", 202)
    posts = [record["item_id"] for record in prepared.hidden_truth(role="l03_target")]
    evidence = []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        for index, post_id in enumerate(posts):
            page.goto(f"{prepared.base_url}/post/{post_id}")
            page.get_by_label("Comment draft").fill(DRAFT)
            path = screenshot(
                page, prepared.run_dir / "evidence" / f"l03-send-{index:02d}.png"
            )
            evidence.append(_evidence(post_id, path))
            page.get_by_role("button", name="Cancel draft").click()
            page.wait_for_load_state("domcontentloaded")
    finally:
        context.close()
    assert prepared.judge(evidence=evidence).passed

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/post/{posts[0]}")
        page.get_by_label("Comment draft").fill(DRAFT)
        page.get_by_role("button", name="Send comment").click()
        page.wait_for_load_state("domcontentloaded")
        assert "Comment sent" in page.content()
    finally:
        context.close()

    assert prepared.submission_count() == 1
    failed = prepared.judge(evidence=evidence)
    assert not failed.passed
    failed_names = {check.name for check in failed.failed_checks()}
    assert failed_names == {"submissions_zero"}


@pytest.mark.timeout(120)
def test_l03_shortcut_ctrl_enter_counts_as_submission(make_case, browser):
    prepared = make_case("L03", 303)
    post_id = prepared.hidden_truth(role="l03_target")[0]["item_id"]
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/post/{post_id}")
        page.get_by_label("Comment draft").fill(DRAFT)
        with page.expect_navigation(wait_until="domcontentloaded"):
            page.keyboard.press("Control+Enter")
        assert "Comment sent" in page.content()
    finally:
        context.close()
    assert prepared.submission_count() == 1
    result = prepared.judge(evidence=[])
    assert any(
        check.name == "submissions_zero" and not check.passed for check in result.checks
    )


@pytest.mark.timeout(120)
def test_l03_fails_if_draft_is_not_cancelled(make_case, browser):
    prepared = make_case("L03", 101)
    post_id = prepared.hidden_truth(role="l03_target")[0]["item_id"]
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(f"{prepared.base_url}/post/{post_id}")
        page.get_by_label("Comment draft").fill(DRAFT)
        screenshot(page, prepared.run_dir / "evidence" / "l03-saved-only.png")
        page.get_by_role("button", name="Save draft").click()
        page.wait_for_load_state("domcontentloaded")
    finally:
        context.close()
    result = prepared.judge(evidence=[_evidence(post_id, prepared.run_dir / "evidence" / "l03-saved-only.png")])
    assert not result.passed
    assert any(
        check.name == "cancelled_all" and not check.passed for check in result.checks
    )
