from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urlencode

from webagent.testing import start

from d1_helpers import fetch, fetch_status


def test_pages_render_and_draft_state_is_shared(make_case):
    prepared = make_case("L01", 101)
    base = prepared.base_url
    target_id = prepared.hidden_truth(role="l01_target")[0]["item_id"]
    product = prepared.site.store.get_product(target_id)

    home = fetch(base + "/")
    assert "<title>Harbor Market &amp; Forum</title>" in home
    assert "Search products" in home

    products = fetch(base + "/products")
    assert "<title>Harbor Market — Products</title>" in products
    assert "Apply filters" in products
    assert "Next page" in products
    assert f"/product/{target_id}" in products or "Page" in products

    posts = fetch(base + "/posts")
    assert "<title>Harbor Forum — Posts</title>" in posts
    assert "Search posts" in posts
    assert "Previous" not in posts and "Page 1" in posts

    post_search = fetch(base + "/posts?q=trail")
    assert "Search posts" in post_search
    assert "Page 1" in post_search or "No posts match" in post_search

    detail = fetch(base + f"/product/{target_id}")
    assert f"<title>Harbor Market — Product</title>" in detail
    assert product["name"] in detail
    assert "Review draft" in detail
    assert "Cancel draft" in detail
    assert f"${product['price_cents'] / 100:.2f}" in detail

    draft_text = "Shared tab draft note 42"
    body = urlencode({"text": draft_text, "action": "save"}).encode("utf-8")
    saved_page = fetch(base + f"/product/{target_id}/draft", body)
    assert "Draft saved" in saved_page

    # A second request is a second tab: it must read the same SQLite draft.
    second_tab = fetch(base + f"/product/{target_id}")
    assert draft_text in second_tab

    body = urlencode({"text": draft_text, "action": "cancel"}).encode("utf-8")
    cancel_page = fetch(base + f"/product/{target_id}/draft", body)
    assert "Draft cancelled" in cancel_page
    cleared = fetch(base + f"/product/{target_id}")
    assert draft_text not in cleared
    assert prepared.db_path.is_file()


def test_search_filters_and_pagination(make_case):
    prepared = make_case("L02", 101)
    base = prepared.base_url

    page_two = fetch(base + "/products?page=2")
    assert "Page 2 of 4" in page_two
    assert "Previous page" in page_two
    assert "Next page" in page_two

    filtered = fetch(
        base
        + "/products?category=Camping&max_price=65&min_rating=4.5&in_stock=1"
    )
    assert "of 5 products" in filtered or "of 5 products" in fetch(
        base + "/products?category=Camping&max_price=65&min_rating=4.5&in_stock=1&page=1"
    )
    # The five matches are spread across pages, so the first filtered page
    # cannot contain all of them.
    assert filtered.count('class="card"') <= prepared.site.page_size
    assert "Search products" in filtered


def test_no_judge_or_hidden_routes_are_exposed(make_case):
    prepared = make_case("L03", 101)
    base = prepared.base_url
    for path in (
        "/judge",
        "/hidden",
        "/hidden_truth",
        "/api/truth",
        "/api/state",
        "/api/judge",
        "/debug",
        "/state",
    ):
        assert fetch_status(base + path) == 404, path

    rendered = "\n".join(
        [
            fetch(base + "/"),
            fetch(base + "/products"),
            fetch(base + "/posts"),
        ]
    )
    for token in (
        "hidden_truth",
        "hidden_config",
        "fault_plan",
        "l01_target",
        "l02_match",
        "l03_target",
        "expected_draft_text",
    ):
        assert token not in rendered
    assert prepared.hidden_truth(role="l03_target")


def test_post_send_records_submission_and_reset_clears_it(make_case):
    prepared = make_case("L03", 101)
    base = prepared.base_url
    post_id = prepared.hidden_truth(role="l03_target")[0]["item_id"]
    body = urlencode({"text": "A short note 😊", "action": "send"}).encode("utf-8")
    page = fetch(base + f"/post/{post_id}/draft", body)
    assert "Comment sent" in page
    assert prepared.submission_count() == 1
    assert prepared.submission_count(post_id) == 1

    prepared.reset()
    assert prepared.submission_count() == 0
    assert prepared.hidden_snapshot()["meta"]["seed"] == "101"


def test_login_overlay_blocks_page_until_sign_in(make_case, browser):
    prepared = make_case(
        "L01",
        101,
        faults=[
            {
                "fault_id": "login-on-load",
                "type": "login_overlay",
                "enabled": True,
                "trigger": {"event": "page_view", "after": 1},
                "params": {},
            }
        ],
    )
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(prepared.base_url + "/products")
        assert page.get_by_role("heading", name="Sign in required").is_visible()
        assert page.get_by_role("button", name="Sign in").is_visible()
        assert "Apply filters" not in page.content()

        page.get_by_role("button", name="Sign in").click()
        page.wait_for_load_state("domcontentloaded")
        assert page.url.endswith("/products")
        assert page.get_by_label("Search products").is_visible()
        assert prepared.fault_events()[0]["fault_type"] == "login_overlay"
    finally:
        context.close()


def test_verification_prompt_blocks_page_until_completed(make_case, browser):
    prepared = make_case(
        "L03",
        202,
        faults=[
            {
                "fault_id": "verify-on-load",
                "type": "verification_prompt",
                "enabled": True,
                "trigger": {"event": "page_view", "after": 1},
                "params": {},
            }
        ],
    )
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(prepared.base_url + "/posts")
        assert page.get_by_role("heading", name="Verification required").is_visible()
        page.get_by_role("button", name="Complete verification").click()
        page.wait_for_load_state("domcontentloaded")
        assert "Forum posts" in page.content()
        assert prepared.fault_events()[0]["fault_type"] == "verification_prompt"
    finally:
        context.close()


def test_slow_response_fault_is_activated_by_activity_not_agent(make_case):
    prepared = make_case("L01", 101)
    prepared.inject_fault(
        {
            "fault_id": "slow",
            "type": "slow_response",
            "enabled": True,
            "trigger": {"event": "page_view", "after": 1},
            "params": {"delay_ms": 250, "times": 1},
        }
    )
    started = time.perf_counter()
    fetch(prepared.base_url + "/")
    first = time.perf_counter() - started
    started = time.perf_counter()
    fetch(prepared.base_url + "/products")
    second = time.perf_counter() - started
    assert max(first, second) >= 0.20
    events = prepared.fault_events()
    assert len(events) == 1
    assert events[0]["fault_id"] == "slow"
    assert events[0]["truth"]["type"] == "slow_response"
    # The fault truth is not exposed on any rendered page.
    assert "slow" not in fetch(prepared.base_url + "/products")


def test_feedback_delay_fault_delays_after_save(make_case):
    prepared = make_case("L01", 101)
    prepared.inject_fault(
        {
            "fault_id": "delay-feedback",
            "type": "feedback_delay",
            "enabled": True,
            "trigger": {"event": "draft_saved", "after": 1},
            "params": {"delay_ms": 250, "times": 1},
        }
    )
    target_id = prepared.hidden_truth(role="l01_target")[0]["item_id"]
    body = urlencode({"text": "delayed note", "action": "save"}).encode("utf-8")
    started = time.perf_counter()
    page = fetch(prepared.base_url + f"/product/{target_id}/draft", body)
    elapsed = time.perf_counter() - started
    assert elapsed >= 0.20
    assert "Draft saved" in page
    assert prepared.fault_events()[0]["fault_type"] == "feedback_delay"
