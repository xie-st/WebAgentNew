"""Local dynamic website bound to 127.0.0.1 on a random free port.

The site is a normal server-rendered application: list, search, pagination,
detail, draft save/cancel/send, login overlay and verification prompt. State is
kept in SQLite, so two tabs see the same drafts and submissions.

No evaluation route exists. The hidden truth, fault plan and judge are Python
APIs used by the test preparation/evaluation code, not HTTP endpoints.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

from .data import Dataset, generate_dataset
from .faults import FaultController, FaultDefinition, normalize_fault_plan
from .pages import (
    STATIC_CSS,
    STATIC_JS,
    auth_page,
    home_page,
    layout,
    not_found_page,
    post_page,
    posts_page,
    product_page,
    products_page,
)
from .store import SiteStore


def _first(values: Mapping[str, list[str]], key: str, default: str = "") -> str:
    items = values.get(key)
    return default if not items else str(items[0])


def _int_query(value: str, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_query(value: str, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _price_cents(value: str) -> int | None:
    number = _float_query(value)
    if number is None:
        return None
    return max(0, int(round(number * 100)))


class LocalSite:
    """One isolated case/run instance of the local site."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        case_id: str,
        case_version: str,
        seed: int = 101,
        generator_params: Mapping[str, Any] | None = None,
        hidden_config: Mapping[str, Any] | None = None,
        fault_plan: Iterable[Mapping[str, Any] | FaultDefinition] | None = None,
        setup: Mapping[str, Any] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.case_id = case_id
        self.case_version = case_version
        self.seed = int(seed)
        self.generator_params = dict(generator_params or {})
        self.hidden_config = dict(hidden_config or {})
        self.setup = dict(setup or {})
        self.page_size = int(self.setup.get("page_size", 8))
        raw_faults = [
            fault.to_store_row() if isinstance(fault, FaultDefinition) else fault
            for fault in (fault_plan or ())
        ]
        self.fault_definitions = normalize_fault_plan(raw_faults)
        self.store = SiteStore(self.db_path)
        self.controller = FaultController(self.store)
        self.dataset: Dataset | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._base_url: str | None = None
        self.reset(seed=self.seed)

    # ------------------------------------------------------------- lifecycle
    def reset(self, seed: int | None = None) -> Dataset:
        if seed is not None:
            self.seed = int(seed)
        dataset = generate_dataset(self.seed, self.generator_params)
        self.store.reset(
            dataset,
            case_id=self.case_id,
            case_version=self.case_version,
            hidden_config=self.hidden_config,
            faults=[fault.to_store_row() for fault in self.fault_definitions],
            page_size=self.page_size,
            setup=self.setup,
        )
        self.dataset = dataset
        if self.setup.get("login_overlay"):
            self.controller.force_auth("login")
        elif self.setup.get("verification_prompt"):
            self.controller.force_auth("verification")
        return dataset

    def start(self) -> "LocalSite":
        if self._server is not None:
            return self
        handler = functools.partial(_SiteRequestHandler, site=self)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        self._base_url = f"http://127.0.0.1:{server.server_address[1]}"
        return self

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        self._base_url = None

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("local site has not been started")
        return self._base_url

    def url(self, path: str = "/") -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def __enter__(self) -> "LocalSite":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ---------------------------------------------------------- test-side API
    def inject_fault(self, fault: Mapping[str, Any]) -> FaultDefinition:
        return self.controller.inject(fault)

    def hidden_truth(self, *, role: str | None = None, item_type: str | None = None):
        return self.store.hidden_truth(role=role, item_type=item_type)

    def hidden_snapshot(self) -> dict[str, Any]:
        return self.store.hidden_snapshot()

    def submission_count(self, item_id: str | None = None) -> int:
        return self.store.submission_count(item_id)

    def fault_events(self) -> list[dict[str, Any]]:
        return self.store.fault_events()


class _SiteRequestHandler(BaseHTTPRequestHandler):
    server_version = "WebAgentLocalSite/2.0"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, site: LocalSite, **kwargs: Any) -> None:
        self.site = site
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    # ------------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if path == "/static/site.css":
            self._send(200, STATIC_CSS.encode("utf-8"), "text/css; charset=utf-8")
            return
        if path == "/static/site.js":
            self._send(200, STATIC_JS.encode("utf-8"), "text/javascript; charset=utf-8")
            return

        self._begin_page(path)
        if self._blocked_by_auth(path):
            return
        try:
            if path == "/":
                self._home()
            elif path == "/products":
                self._products(query)
            elif path == "/posts":
                self._posts(query)
            elif path.startswith("/product/"):
                self._product_detail(path[len("/product/") :], query)
            elif path.startswith("/post/"):
                self._post_detail(path[len("/post/") :], query)
            else:
                self._send_html(404, not_found_page(path))
        except BrokenPipeError:
            return

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        length = _int_query(self.headers.get("Content-Length", "0"), 0) or 0
        if length > 200_000:
            self._send_html(413, layout("Harbor Market — Too large", "<h1>Request too large</h1>"))
            return
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(body, keep_blank_values=True)
        action = _first(form, "action")
        if path.startswith("/product/") and path.endswith("/draft"):
            self._draft_post(
                path[len("/product/") : -len("/draft")],
                form,
                action,
                item_type="product",
            )
        elif path.startswith("/post/") and path.endswith("/draft"):
            self._draft_post(
                path[len("/post/") : -len("/draft")],
                form,
                action,
                item_type="post",
            )
        elif path == "/auth":
            self._auth_post(form, action)
        else:
            self._send_html(404, not_found_page(path))

    # -------------------------------------------------------------- page flow
    def _begin_page(self, path: str) -> None:
        self.site.store.record_activity("page_view", path=path)
        self.site.controller.after_event("page_view")
        delay_ms = self.site.controller.consume_delay_ms("page")
        if delay_ms:
            time.sleep(delay_ms / 1000)

    def _blocked_by_auth(self, path: str) -> bool:
        kind = self.site.controller.auth_kind()
        if kind is None:
            return False
        self.site.store.record_activity("auth_blocked", path=path, payload={"kind": kind})
        self.site.controller.after_event("auth_blocked")
        self._send_html(200, auth_page(kind, next_path=path))
        return True

    def _after_item_event(self, kind: str) -> bool:
        self.site.controller.after_event(kind)
        return self.site.controller.auth_kind() is not None

    def _sleep_feedback_delay(self) -> None:
        delay_ms = self.site.controller.consume_delay_ms("feedback")
        if delay_ms:
            time.sleep(delay_ms / 1000)

    def _home(self) -> None:
        products, product_total = self.site.store.list_products(
            page=1, page_size=1
        )
        posts, post_total = self.site.store.list_posts(page=1, page_size=1)
        self._send_html(200, home_page(product_total, post_total))

    def _products(self, query: Mapping[str, list[str]]) -> None:
        q = _first(query, "q")
        category = _first(query, "category")
        min_price = _first(query, "min_price")
        max_price = _first(query, "max_price")
        min_rating = _first(query, "min_rating")
        in_stock = _first(query, "in_stock") in ("1", "true", "on", "yes")
        page = max(1, _int_query(_first(query, "page", "1"), 1) or 1)
        filters = {
            "q": q,
            "category": category,
            "min_price": min_price,
            "max_price": max_price,
            "min_rating": min_rating,
            "in_stock": in_stock,
        }
        self.site.store.record_activity("list_view", path="/products", payload=filters)
        if self._after_item_event("list_view"):
            self._send_html(200, auth_page(self.site.controller.auth_kind() or "login", "/products"))
            return
        rows, total = self.site.store.list_products(
            query=q,
            category=category,
            min_price_cents=_price_cents(min_price),
            max_price_cents=_price_cents(max_price),
            min_rating=_float_query(min_rating),
            in_stock=in_stock,
            page=page,
            page_size=self.site.page_size,
        )
        self._send_html(
            200,
            products_page(
                rows,
                total=total,
                page=page,
                page_size=self.site.page_size,
                filters={**filters, "notice": _first(query, "notice")},
                categories=self.site.store.categories(),
            ),
        )

    def _posts(self, query: Mapping[str, list[str]]) -> None:
        page = max(1, _int_query(_first(query, "page", "1"), 1) or 1)
        q = _first(query, "q")
        self.site.store.record_activity(
            "list_view", path="/posts", payload={"page": page, "q": q}
        )
        if self._after_item_event("list_view"):
            self._send_html(200, auth_page(self.site.controller.auth_kind() or "login", "/posts"))
            return
        rows, total = self.site.store.list_posts(
            query=q, page=page, page_size=self.site.page_size
        )
        self._send_html(
            200,
            posts_page(
                rows,
                total=total,
                page=page,
                page_size=self.site.page_size,
                query=q,
            ),
        )

    def _product_detail(self, product_id: str, query: Mapping[str, list[str]]) -> None:
        product = self.site.store.get_product(product_id)
        if product is None:
            self._send_html(404, not_found_page(self.path))
            return
        self.site.store.record_activity(
            "view_item", item_type="product", item_id=product_id
        )
        if self._after_item_event("view_item"):
            self._send_html(
                200,
                auth_page(
                    self.site.controller.auth_kind() or "login",
                    f"/product/{product_id}",
                ),
            )
            return
        draft = self.site.store.get_draft("product", product_id)
        self._send_html(
            200,
            product_page(
                product,
                draft=draft,
                notice=_first(query, "notice") or None,
            ),
        )

    def _post_detail(self, post_id: str, query: Mapping[str, list[str]]) -> None:
        post = self.site.store.get_post(post_id)
        if post is None:
            self._send_html(404, not_found_page(self.path))
            return
        self.site.store.record_activity("view_item", item_type="post", item_id=post_id)
        if self._after_item_event("view_item"):
            self._send_html(
                200,
                auth_page(
                    self.site.controller.auth_kind() or "login",
                    f"/post/{post_id}",
                ),
            )
            return
        body = dict(post)
        body["tags"] = json.loads(post["tags"])
        draft = self.site.store.get_draft("post", post_id)
        self._send_html(
            200,
            post_page(
                body,
                draft=draft,
                notice=_first(query, "notice") or None,
                duplicate_send_buttons=self.site.controller.duplicate_button_count(),
            ),
        )

    # ------------------------------------------------------------------ POST
    def _draft_post(
        self,
        item_id: str,
        form: Mapping[str, list[str]],
        action: str,
        *,
        item_type: str,
    ) -> None:
        item = (
            self.site.store.get_product(item_id)
            if item_type == "product"
            else self.site.store.get_post(item_id)
        )
        if item is None:
            self._send_html(404, not_found_page(self.path))
            return
        text = _first(form, "text")
        if action == "save":
            self.site.store.save_draft(item_type, item_id, text)
            self.site.controller.after_event("draft_saved")
            self._sleep_feedback_delay()
            self._redirect(f"/{item_type}/{item_id}?notice=Draft+saved")
            return
        if action == "cancel":
            self.site.store.cancel_draft(item_type, item_id, text)
            self.site.controller.after_event("draft_cancelled")
            self._sleep_feedback_delay()
            self._redirect(f"/{item_type}/{item_id}?notice=Draft+cancelled")
            return
        if action == "send" and item_type == "post":
            self.site.store.send_comment(item_type, item_id, text)
            self.site.controller.after_event("comment_sent")
            self._sleep_feedback_delay()
            self._redirect(f"/{item_type}/{item_id}?notice=Comment+sent")
            return
        pages = {
            "product": product_page,
            "post": post_page,
        }
        page = pages[item_type]
        body = dict(item)
        if item_type == "post":
            body["tags"] = json.loads(item["tags"])
        content = page(
            body,
            draft=self.site.store.get_draft(item_type, item_id),
            notice="That action is not available for this item.",
        )
        self._send_html(400, content)

    def _auth_post(self, form: Mapping[str, list[str]], action: str) -> None:
        if action not in ("sign_in", "verify"):
            self._send_html(400, layout("Harbor Market — Bad request", "<h1>Unknown sign-in action</h1>"))
            return
        self.site.controller.clear_auth()
        self.site.store.record_activity("auth_completed", payload={"action": action})
        next_path = _first(form, "next", "/")
        if not next_path.startswith("/") or next_path.startswith("//") or next_path.startswith("/auth"):
            next_path = "/"
        self._redirect(next_path)

    # --------------------------------------------------------------- responses
    def _redirect(self, location: str) -> None:
        body = b""
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_html(self, status: int, content: str) -> None:
        self._send(status, content.encode("utf-8"), "text/html; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass


__all__ = ["LocalSite"]
