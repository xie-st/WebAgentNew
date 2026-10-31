from __future__ import annotations

from d1_helpers import screenshot


def test_playwright_renders_pages_and_buttons_and_shares_sqlite_state(
    make_case, browser
):
    prepared = make_case("L01", 101)
    target_id = prepared.hidden_truth(role="l01_target")[0]["item_id"]
    product = dict(prepared.site.store.get_product(target_id))
    draft = "Cross-tab note kept in SQLite"

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    first = context.new_page()
    second = context.new_page()
    try:
        first.goto(prepared.base_url + "/products")
        assert first.title() == "Harbor Market — Products"
        assert first.get_by_role("heading", name="Products").is_visible()
        search = first.get_by_label("Search products")
        search.fill("lantern")
        with first.expect_navigation():
            first.get_by_role("button", name="Apply filters").click()
        assert "q=lantern" in first.url

        first.goto(prepared.base_url + "/products?page=1")
        with first.expect_navigation():
            first.get_by_role("link", name="Next page").click()
        assert "page=2" in first.url

        page_number = product["position"] // prepared.site.page_size + 1
        first.goto(f"{prepared.base_url}/products?page={page_number}")
        first.get_by_role("link", name=product["name"], exact=True).click()
        first.wait_for_load_state("domcontentloaded")
        assert first.title() == "Harbor Market — Product"
        area = first.get_by_label("Review draft")
        area.fill(draft)
        shot = screenshot(first, prepared.run_dir / "render" / "product-draft.png")
        with first.expect_navigation():
            first.get_by_role("button", name="Save draft").click()
        assert "Draft saved" in first.content()

        # Second tab, same browser context, independent page load: same SQLite row.
        second.goto(prepared.base_url + f"/product/{target_id}")
        assert second.get_by_label("Review draft").input_value() == draft
        second_shot = screenshot(
            second, prepared.run_dir / "render" / "second-tab-draft.png"
        )

        with second.expect_navigation():
            second.get_by_role("button", name="Cancel draft").click()
        first.reload()
        assert first.get_by_label("Review draft").input_value() == ""
        assert shot.read_bytes().startswith(b"\x89PNG")
        assert second_shot.read_bytes().startswith(b"\x89PNG")
    finally:
        context.close()


def test_posts_page_has_send_and_cancel_controls(make_case, browser):
    prepared = make_case("L03", 101)
    post_id = prepared.hidden_truth(role="l03_target")[0]["item_id"]
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        page.goto(prepared.base_url + f"/post/{post_id}")
        assert page.title() == "Harbor Forum — Post"
        assert page.get_by_role("button", name="Save draft").is_visible()
        assert page.get_by_role("button", name="Cancel draft").is_visible()
        assert page.get_by_role("button", name="Send comment").is_visible()
        page.get_by_label("Comment draft").fill("A note with an emoji 😊")
        assert page.get_by_label("Comment draft").input_value().endswith("😊")
    finally:
        context.close()
    assert prepared.submission_count() == 0
