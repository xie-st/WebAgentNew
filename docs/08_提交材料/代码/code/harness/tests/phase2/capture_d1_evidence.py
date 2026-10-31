"""Capture D1 evidence for the local test environment (offline, headless).

Run from the repository root:

    .venv\\Scripts\\python.exe code\\harness\\tests\\phase2\\capture_d1_evidence.py

It starts real local sites on 127.0.0.1, drives headless Chromium, and writes
screenshots plus machine-readable check results under runtime/phase2_d1/.
No network, official platform or model call is used.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from d1_helpers import screenshot
from webagent.testing import start
from webagent.testing.evaluation import HiddenStateReader

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT = REPO_ROOT / "runtime" / "phase2_d1"
EVIDENCE = OUTPUT / "evidence"
DRAFT_L03 = "Useful field note, thanks for sharing the details! 😊"


def write_json(name: str, payload: dict) -> Path:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def display_snapshot(prepared):
    products = prepared.site.store.all_products()
    posts, _ = prepared.site.store.list_posts(page=1, page_size=100)
    return {
        "product_names": [row["name"] for row in products],
        "product_prices": [row["price_cents"] for row in products],
        "product_ids": [row["id"] for row in products],
        "post_titles": [row["title"] for row in posts],
        "post_ids": [row["id"] for row in posts],
    }


def render_and_state_checks() -> dict:
    prepared = start("L01", 101, runtime_dir=OUTPUT)
    checks: dict = {"case_id": "L01", "seed": 101, "pages": [], "state": {}}
    try:
        target_id = prepared.hidden_truth(role="l01_target")[0]["item_id"]
        product = dict(prepared.site.store.get_product(target_id))
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 800})
            first = context.new_page()
            second = context.new_page()
            try:
                first.goto(prepared.base_url + "/")
                checks["pages"].append(
                    {
                        "path": "/",
                        "title": first.title(),
                        "screenshot": str(
                            screenshot(first, EVIDENCE / "render" / "home.png")
                        ),
                    }
                )

                first.goto(prepared.base_url + "/products")
                title = first.title()
                first.get_by_label("Search products").fill("lantern")
                with first.expect_navigation():
                    first.get_by_role("button", name="Apply filters").click()
                search_ok = "q=lantern" in first.url
                first.goto(prepared.base_url + "/products?page=1")
                with first.expect_navigation():
                    first.get_by_role("link", name="Next page").click()
                pager_ok = "page=2" in first.url
                checks["pages"].append(
                    {
                        "path": "/products",
                        "title": title,
                        "screenshot": str(
                            screenshot(first, EVIDENCE / "render" / "products.png")
                        ),
                        "search_submitted": search_ok,
                        "next_page_clicked": pager_ok,
                    }
                )

                first.goto(prepared.base_url + "/posts")
                checks["pages"].append(
                    {
                        "path": "/posts",
                        "title": first.title(),
                        "screenshot": str(
                            screenshot(first, EVIDENCE / "render" / "posts.png")
                        ),
                    }
                )

                page_number = product["position"] // prepared.site.page_size + 1
                first.goto(f"{prepared.base_url}/products?page={page_number}")
                first.get_by_role("link", name=product["name"], exact=True).click()
                first.wait_for_load_state("domcontentloaded")
                first.get_by_label("Review draft").fill(
                    prepared.case.judge_config["expected_draft_text"]
                )
                detail_shot = screenshot(
                    first, EVIDENCE / "render" / "product-draft.png"
                )
                with first.expect_navigation():
                    first.get_by_role("button", name="Save draft").click()
                saved_notice = "Draft saved" in first.content()

                second.goto(prepared.base_url + f"/product/{target_id}")
                cross_tab_value = second.get_by_label("Review draft").input_value()
                second_shot = screenshot(
                    second, EVIDENCE / "render" / "second-tab.png"
                )
                with second.expect_navigation():
                    second.get_by_role("button", name="Cancel draft").click()
                first.reload()
                cleared_value = first.get_by_label("Review draft").input_value()

                checks["pages"].append(
                    {
                        "path": f"/product/{target_id}",
                        "title": first.title(),
                        "screenshot": str(detail_shot),
                        "second_tab_screenshot": str(second_shot),
                    }
                )
                checks["state"] = {
                    "draft_save_visible": saved_notice,
                    "second_tab_value": cross_tab_value,
                    "value_after_cancel": cleared_value,
                    "same_sqlite_row": cross_tab_value
                    == prepared.case.judge_config["expected_draft_text"],
                    "cleared_after_cancel": cleared_value == "",
                    "db_path": str(prepared.db_path),
                }

                # Login overlay / verification prompt are real rendered states.
                prepared.inject_fault(
                    {
                        "fault_id": "evidence-login",
                        "type": "login_overlay",
                        "enabled": True,
                        "trigger": {"event": "page_view", "after": 1},
                        "params": {},
                    }
                )
                first.goto(prepared.base_url + "/products")
                login_visible = first.get_by_role(
                    "heading", name="Sign in required"
                ).is_visible()
                login_shot = screenshot(
                    first, EVIDENCE / "render" / "login-overlay.png"
                )
                first.get_by_role("button", name="Sign in").click()
                first.wait_for_load_state("domcontentloaded")
                login_cleared = first.get_by_label("Search products").is_visible()
                checks["login_overlay"] = {
                    "visible": login_visible,
                    "screenshot": str(login_shot),
                    "cleared_after_sign_in": login_cleared,
                }

                prepared.inject_fault(
                    {
                        "fault_id": "evidence-verify",
                        "type": "verification_prompt",
                        "enabled": True,
                        "trigger": {"event": "page_view", "after": 1},
                        "params": {},
                    }
                )
                first.goto(prepared.base_url + "/posts")
                verification_visible = first.get_by_role(
                    "heading", name="Verification required"
                ).is_visible()
                verification_shot = screenshot(
                    first, EVIDENCE / "render" / "verification-prompt.png"
                )
                first.get_by_role("button", name="Complete verification").click()
                first.wait_for_load_state("domcontentloaded")
                checks["verification_prompt"] = {
                    "visible": verification_visible,
                    "screenshot": str(verification_shot),
                    "cleared_after_verification": "Forum posts" in first.content(),
                }

                l01_judge = prepared.judge(
                    evidence=[
                        {
                            "object_type": "product",
                            "object_id": target_id,
                            "kind": "screenshot",
                            "reference": str(detail_shot),
                        }
                    ],
                    answer={"product_ids": [target_id]},
                )
                write_json("l01_judge.json", l01_judge.to_dict())
                checks["l01_judge"] = l01_judge.verdict
            finally:
                context.close()
                browser.close()
        return checks
    finally:
        prepared.stop()


def determinism_checks() -> dict:
    prepared = start("L02", 101, runtime_dir=OUTPUT)
    other = start("L02", 202, runtime_dir=OUTPUT)
    try:
        first = display_snapshot(prepared)
        hash_before = HiddenStateReader(prepared.db_path).snapshot_hash()
        prepared.reset()
        second = display_snapshot(prepared)
        hash_after = HiddenStateReader(prepared.db_path).snapshot_hash()
        other_snapshot = display_snapshot(other)
        other_hash = HiddenStateReader(other.db_path).snapshot_hash()
        payload = {
            "case_id": "L02",
            "seed_101_hash_before_reset": hash_before,
            "seed_101_hash_after_reset": hash_after,
            "same_seed_identical": hash_before == hash_after and first == second,
            "seed_202_hash": other_hash,
            "different_seed_hash_differs": hash_before != other_hash,
            "names_differ": first["product_names"] != other_snapshot["product_names"],
            "prices_differ": first["product_prices"] != other_snapshot["product_prices"],
            "order_differ": first["product_ids"] != other_snapshot["product_ids"],
            "sample_names_seed_101": first["product_names"][:3],
            "sample_names_seed_202": other_snapshot["product_names"][:3],
            "sample_prices_seed_101": first["product_prices"][:5],
            "sample_prices_seed_202": other_snapshot["product_prices"][:5],
        }
        write_json("determinism.json", payload)
        return payload
    finally:
        prepared.stop()
        other.stop()


def l02_judge_checks() -> dict:
    prepared = start("L02", 202, runtime_dir=OUTPUT)
    evidence = []
    try:
        expected = [
            record["item_id"] for record in prepared.hidden_truth(role="l02_match")
        ]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                for index, product_id in enumerate(expected):
                    page.goto(f"{prepared.base_url}/product/{product_id}")
                    path = screenshot(
                        page, EVIDENCE / "l02" / f"product-{index:02d}.png"
                    )
                    evidence.append(
                        {
                            "object_type": "product",
                            "object_id": product_id,
                            "kind": "screenshot",
                            "reference": str(path),
                        }
                    )
            finally:
                browser.close()
        result = prepared.judge(
            evidence=evidence, answer={"product_ids": expected}
        )
        write_json("l02_judge.json", result.to_dict())
        return result.to_dict()
    finally:
        prepared.stop()


def l03_judge_checks() -> dict:
    prepared = start("L03", 101, runtime_dir=OUTPUT)
    evidence = []
    try:
        posts = [record["item_id"] for record in prepared.hidden_truth(role="l03_target")]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                for index, post_id in enumerate(posts):
                    page.goto(f"{prepared.base_url}/post/{post_id}")
                    page.get_by_label("Comment draft").fill(DRAFT_L03)
                    path = screenshot(
                        page, EVIDENCE / "l03" / f"post-{index:02d}.png"
                    )
                    page.get_by_role("button", name="Cancel draft").click()
                    page.wait_for_load_state("domcontentloaded")
                    evidence.append(
                        {
                            "object_type": "post",
                            "object_id": post_id,
                            "kind": "screenshot",
                            "reference": str(path),
                        }
                    )
            finally:
                browser.close()
        before = prepared.judge(evidence=evidence)
        submission_count_after_flow = prepared.submission_count()
        write_json("l03_judge_pass.json", before.to_dict())

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(f"{prepared.base_url}/post/{posts[0]}")
                page.get_by_label("Comment draft").fill(DRAFT_L03)
                page.get_by_role("button", name="Send comment").click()
                page.wait_for_load_state("domcontentloaded")
            finally:
                browser.close()
        after = prepared.judge(evidence=evidence)
        write_json("l03_judge_after_send_fail.json", after.to_dict())
        return {
            "posts": len(posts),
            "screenshots": len(evidence),
            "submission_count_after_complete_flow": submission_count_after_flow,
            "pass_verdict": before.verdict,
            "submission_count_after_send": prepared.submission_count(),
            "fail_verdict": after.verdict,
            "failed_checks_after_send": [
                check.name for check in after.failed_checks()
            ],
        }
    finally:
        prepared.stop()


def main() -> int:
    for name in ("evidence", "runs"):
        shutil.rmtree(OUTPUT / name, ignore_errors=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    summary = {
        "render": render_and_state_checks(),
        "determinism": determinism_checks(),
        "l02_judge": l02_judge_checks(),
        "l03": l03_judge_checks(),
    }
    path = write_json("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
