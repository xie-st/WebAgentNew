"""Server-rendered pages for the local dynamic website.

Only public product/post data and draft state are rendered. Hidden truth,
fault definitions and evaluation configuration never reach these functions.
"""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode

BRAND = "Harbor Market & Forum"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _notice(notice: str | None) -> str:
    if not notice:
        return ""
    return f'<p class="notice" role="status">{_esc(notice)}</p>'


def _query_url(path: str, params: Mapping[str, Any], page: int) -> str:
    cleaned = {
        key: value
        for key, value in params.items()
        if value not in (None, "") and key != "page"
    }
    cleaned["page"] = str(page)
    return f"{path}?{urlencode(cleaned)}"


def layout(
    title: str,
    content: str,
    *,
    notice: str | None = None,
    nav: bool = True,
) -> str:
    navigation = (
        '<nav class="topnav" aria-label="Site">'
        '<a href="/products">Products</a>'
        '<a href="/posts">Forum</a>'
        "</nav>"
        if nav
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <link rel="stylesheet" href="/static/site.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="/">{_esc(BRAND)}</a>
    {navigation}
  </header>
  <main>
    {_notice(notice)}
    {content}
  </main>
  <footer>Local offline evaluation site. All data is generated; no external requests.</footer>
  <script src="/static/site.js"></script>
</body>
</html>
"""


def home_page(product_count: int, post_count: int) -> str:
    content = f"""
<section class="hero">
  <h1>{_esc(BRAND)}</h1>
  <p>Browse the catalog and the community forum. Every item has an editable draft form.</p>
</section>
<section class="panel">
  <h2>Search products</h2>
  <form class="search-form" method="get" action="/products">
    <div class="field">
      <label for="home-query">Search products</label>
      <input id="home-query" name="q" type="search" autocomplete="off" placeholder="lantern, kettle, shelf">
    </div>
    <button type="submit">Search</button>
  </form>
</section>
<section class="tiles">
  <article class="tile">
    <h2><a href="/products">Product catalog</a></h2>
    <p>{product_count} generated products across several categories and pages.</p>
  </article>
  <article class="tile">
    <h2><a href="/posts">Community forum</a></h2>
    <p>{post_count} generated posts with comment drafts and a send control.</p>
  </article>
</section>
"""
    return layout(BRAND, content)


def products_page(
    products: Sequence[Mapping[str, Any]],
    *,
    total: int,
    page: int,
    page_size: int,
    filters: Mapping[str, Any],
    categories: Sequence[str],
) -> str:
    category_options = ['<option value="">All categories</option>']
    for category in categories:
        selected = " selected" if filters.get("category") == category else ""
        category_options.append(
            f'<option value="{_esc(category)}"{selected}>{_esc(category)}</option>'
        )
    rating_options = ['<option value="">Any rating</option>']
    for rating in ("4.0", "4.3", "4.5", "4.7"):
        selected = " selected" if str(filters.get("min_rating") or "") == rating else ""
        rating_options.append(
            f'<option value="{rating}"{selected}>{rating} and up</option>'
        )

    cards = []
    for product in products:
        cards.append(
            f"""
<li class="card">
  <h2><a href="/product/{quote(str(product['id']))}">{_esc(product['name'])}</a></h2>
  <p class="meta">{_esc(product['category'])} · {_esc(product['sku'])}</p>
  <p>{_esc(product['summary'])}</p>
  <dl class="facts">
    <div><dt>Price</dt><dd>${product['price_cents'] / 100:.2f}</dd></div>
    <div><dt>Rating</dt><dd>{product['rating']:.1f}</dd></div>
    <div><dt>Stock</dt><dd>{product['stock']}</dd></div>
  </dl>
</li>
"""
        )
    if not cards:
        cards.append('<li class="empty">No products match these filters.</li>')

    pages = max(1, (total + page_size - 1) // page_size)
    pager = []
    if page > 1:
        pager.append(
            f'<a rel="prev" href="{_esc(_query_url("/products", filters, page - 1))}">Previous page</a>'
        )
    pager.append(f"<span>Page {page} of {pages}</span>")
    if page < pages:
        pager.append(
            f'<a rel="next" href="{_esc(_query_url("/products", filters, page + 1))}">Next page</a>'
        )

    first = 0 if total == 0 else (page - 1) * page_size + 1
    last = min(total, page * page_size)
    content = f"""
<h1>Products</h1>
<section class="panel">
  <h2>Filter the catalog</h2>
  <form class="filter-form" method="get" action="/products">
    <div class="field">
      <label for="products-query">Search products</label>
      <input id="products-query" name="q" type="search" autocomplete="off" value="{_esc(filters.get('q') or '')}">
    </div>
    <div class="field">
      <label for="category">Category</label>
      <select id="category" name="category">{''.join(category_options)}</select>
    </div>
    <div class="field">
      <label for="min-price">Minimum price</label>
      <input id="min-price" name="min_price" type="number" min="0" step="1" value="{_esc(filters.get('min_price') or '')}">
    </div>
    <div class="field">
      <label for="max-price">Maximum price</label>
      <input id="max-price" name="max_price" type="number" min="0" step="1" value="{_esc(filters.get('max_price') or '')}">
    </div>
    <div class="field">
      <label for="min-rating">Minimum rating</label>
      <select id="min-rating" name="min_rating">{''.join(rating_options)}</select>
    </div>
    <div class="field checkbox">
      <input id="in-stock" name="in_stock" type="checkbox" value="1"{" checked" if filters.get("in_stock") else ""}>
      <label for="in-stock">In stock only</label>
    </div>
    <button type="submit">Apply filters</button>
  </form>
</section>
<p class="meta">Showing {first}–{last} of {total} products.</p>
<ul class="cards">
  {''.join(cards)}
</ul>
<nav class="pager" aria-label="Product pages">{''.join(pager)}</nav>
"""
    return layout("Harbor Market — Products", content, notice=filters.get("notice"))


def product_page(
    product: Mapping[str, Any],
    *,
    draft: str,
    notice: str | None = None,
) -> str:
    content = f"""
<p><a href="/products">Back to products</a></p>
<article class="detail">
  <h1>{_esc(product['name'])}</h1>
  <p class="meta">{_esc(product['category'])} · {_esc(product['sku'])}</p>
  <dl class="facts">
    <div><dt>Price</dt><dd>${product['price_cents'] / 100:.2f}</dd></div>
    <div><dt>Rating</dt><dd>{product['rating']:.1f}</dd></div>
    <div><dt>Stock</dt><dd>{product['stock']}</dd></div>
  </dl>
  <p>{_esc(product['description'])}</p>
  <section class="panel">
    <h2>Review draft</h2>
    <form id="draft-form" method="post" action="/product/{quote(str(product['id']))}/draft">
      <div class="field">
        <label for="draft-text">Review draft</label>
        <textarea id="draft-text" name="text" rows="5" placeholder="Write a short review note before deciding">{_esc(draft)}</textarea>
      </div>
      <div class="actions">
        <button type="submit" name="action" value="save">Save draft</button>
        <button type="submit" name="action" value="cancel">Cancel draft</button>
      </div>
    </form>
    <p class="hint">Saving keeps the draft for other tabs. Cancelling clears it. This page has no submit action.</p>
  </section>
</article>
"""
    return layout("Harbor Market — Product", content, notice=notice)


def posts_page(
    posts: Sequence[Mapping[str, Any]],
    *,
    total: int,
    page: int,
    page_size: int,
    query: str = "",
) -> str:
    cards = []
    for post in posts:
        cards.append(
            f"""
<li class="card">
  <h2><a href="/post/{quote(str(post['id']))}">{_esc(post['title'])}</a></h2>
  <p class="meta">{_esc(post['author'])} · {_esc(post['created_label'])} · {post['comments']} comments</p>
  <p>{_esc(str(post['body'])[:160])}…</p>
</li>
"""
        )
    if not cards:
        cards.append('<li class="empty">No posts match this search.</li>')
    pages = max(1, (total + page_size - 1) // page_size)
    pager = []
    if page > 1:
        pager.append(
            f'<a rel="prev" href="{_esc(_query_url("/posts", {"q": query}, page - 1))}">Previous page</a>'
        )
    pager.append(f"<span>Page {page} of {pages}</span>")
    if page < pages:
        pager.append(
            f'<a rel="next" href="{_esc(_query_url("/posts", {"q": query}, page + 1))}">Next page</a>'
        )
    content = f"""
<h1>Forum posts</h1>
<section class="panel">
  <h2>Search posts</h2>
  <form class="search-form" method="get" action="/posts">
    <div class="field">
      <label for="posts-query">Search posts</label>
      <input id="posts-query" name="q" type="search" autocomplete="off" value="{_esc(query)}" placeholder="trail, shelf, repair">
    </div>
    <button type="submit">Search posts</button>
  </form>
</section>
<p class="meta">Showing {len(posts)} of {total} posts on page {page} of {pages}.</p>
<ul class="cards">
  {''.join(cards)}
</ul>
<nav class="pager" aria-label="Post pages">{''.join(pager)}</nav>
"""
    return layout("Harbor Forum — Posts", content)


def post_page(
    post: Mapping[str, Any],
    *,
    draft: str,
    notice: str | None = None,
    duplicate_send_buttons: int = 0,
) -> str:
    tags = " ".join(f'<span class="tag">{_esc(tag)}</span>' for tag in post.get("tags", ()))
    send_controls = ['<button id="send-comment" type="submit" name="action" value="send">Send comment</button>']
    for index in range(max(0, duplicate_send_buttons)):
        send_controls.append(
            f'<button type="submit" name="action" value="send">Send comment</button>'
        )
    content = f"""
<p><a href="/posts">Back to posts</a></p>
<article class="detail">
  <h1>{_esc(post['title'])}</h1>
  <p class="meta">{_esc(post['author'])} · {_esc(post['created_label'])} · {post['likes']} likes</p>
  <p>{_esc(post['body'])}</p>
  <p class="tags">{tags}</p>
  <section class="panel">
    <h2>Comment draft</h2>
    <form id="draft-form" method="post" action="/post/{quote(str(post['id']))}/draft" data-shortcut-submit>
      <div class="field">
        <label for="draft-text">Comment draft</label>
        <textarea id="draft-text" name="text" rows="5" placeholder="Write a comment with text and at least one emoji">{_esc(draft)}</textarea>
      </div>
      <div class="actions">
        <button type="submit" name="action" value="save">Save draft</button>
        <button type="submit" name="action" value="cancel">Cancel draft</button>
        {''.join(send_controls)}
      </div>
    </form>
    <p class="hint">Ctrl+Enter or Cmd+Enter sends the comment. Cancel leaves it unsent.</p>
  </section>
</article>
"""
    return layout("Harbor Forum — Post", content, notice=notice)


def auth_page(kind: str, next_path: str) -> str:
    if kind == "verification":
        heading = "Verification required"
        message = (
            "The site needs a human verification step before continuing. "
            "Complete verification in this browser to restore the page."
        )
        button = "Complete verification"
        action = "verify"
    else:
        heading = "Sign in required"
        message = (
            "Your session has expired. Sign in in this browser to restore the page."
        )
        button = "Sign in"
        action = "sign_in"
    content = f"""
<div class="page-dim" aria-hidden="true">
  <h1>{_esc(BRAND)}</h1>
  <p>Page content is paused until the browser session is restored.</p>
</div>
<div class="overlay" role="dialog" aria-modal="true" aria-labelledby="auth-heading">
  <div class="overlay-card">
    <h1 id="auth-heading">{_esc(heading)}</h1>
    <p>{_esc(message)}</p>
    <form method="post" action="/auth">
      <input type="hidden" name="next" value="{_esc(next_path)}">
      <button type="submit" name="action" value="{action}">{_esc(button)}</button>
    </form>
  </div>
</div>
"""
    return layout(f"Harbor Market — {heading}", content, nav=False)


def not_found_page(path: str) -> str:
    content = f"""
<h1>Page not found</h1>
<p>No page exists at <code>{_esc(path)}</code>.</p>
<p><a href="/">Return to the home page</a></p>
"""
    return layout("Harbor Market — Not found", content)


STATIC_CSS = """
:root { color-scheme: light; --ink: #1f2933; --muted: #5f6b76; --line: #d8dee4;
  --accent: #155e75; --panel: #f6f8fa; --warn: #fff7e6; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink); background: #fff; line-height: 1.5; }
a { color: var(--accent); }
.topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem;
  padding: .8rem 1.2rem; border-bottom: 1px solid var(--line); background: var(--panel); }
.brand { font-weight: 700; text-decoration: none; }
.topnav { display: flex; gap: 1rem; }
main { max-width: 960px; margin: 0 auto; padding: 1.2rem; }
footer { max-width: 960px; margin: 0 auto; padding: 1.2rem; color: var(--muted);
  border-top: 1px solid var(--line); font-size: .9rem; }
h1 { line-height: 1.2; }
.hero { padding: 1rem 0; }
.panel { border: 1px solid var(--line); border-radius: .6rem; padding: 1rem;
  margin: 1rem 0; background: var(--panel); }
.field { display: flex; flex-direction: column; gap: .3rem; margin-bottom: .8rem; }
.field.checkbox { flex-direction: row; align-items: center; gap: .5rem; }
label { font-weight: 600; }
input, select, textarea { font: inherit; padding: .45rem .55rem; border: 1px solid var(--line);
  border-radius: .35rem; background: #fff; }
input[type="checkbox"] { width: auto; }
button { font: inherit; padding: .5rem .85rem; border-radius: .35rem; border: 1px solid var(--accent);
  background: var(--accent); color: #fff; cursor: pointer; }
button[value="cancel"] { background: #fff; color: var(--accent); }
button[value="send"] { background: #8a2c2c; border-color: #8a2c2c; }
.actions { display: flex; flex-wrap: wrap; gap: .6rem; }
.search-form, .filter-form { display: grid; gap: .6rem; }
@media (min-width: 720px) {
  .filter-form { grid-template-columns: repeat(3, minmax(0, 1fr)); align-items: end; }
}
.cards { list-style: none; padding: 0; display: grid; gap: 1rem; }
@media (min-width: 720px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.card { border: 1px solid var(--line); border-radius: .6rem; padding: 1rem; }
.card h2 { margin-top: 0; font-size: 1.1rem; }
.meta { color: var(--muted); font-size: .92rem; }
.facts { display: flex; flex-wrap: wrap; gap: 1rem; margin: .6rem 0; }
.facts div { display: flex; gap: .35rem; }
.facts dt { color: var(--muted); }
.facts dd { margin: 0; font-weight: 600; }
.pager { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem; margin: 1rem 0; }
.notice { border: 1px solid #b7dfc1; background: #eefaf1; padding: .6rem .8rem;
  border-radius: .35rem; }
.hint { color: var(--muted); font-size: .9rem; }
.tags { display: flex; gap: .4rem; flex-wrap: wrap; }
.tag { border: 1px solid var(--line); border-radius: 999px; padding: .1rem .5rem;
  font-size: .85rem; }
.overlay { position: fixed; inset: 0; display: grid; place-items: center;
  background: rgba(22, 30, 38, .55); z-index: 10; }
.overlay-card { background: #fff; border-radius: .7rem; padding: 1.4rem; max-width: 30rem;
  width: calc(100% - 2rem); box-shadow: 0 18px 50px rgba(0,0,0,.25); }
.page-dim { filter: blur(2px); opacity: .45; pointer-events: none; padding: 2rem; }
.empty { color: var(--muted); padding: 1rem; }
code { background: var(--panel); padding: .1rem .3rem; border-radius: .25rem; }
"""

STATIC_JS = """
(function () {
  "use strict";
  document.querySelectorAll("form[data-shortcut-submit]").forEach(function (form) {
    var area = form.querySelector("textarea");
    var send = form.querySelector('button[value="send"]');
    if (!area || !send) return;
    area.addEventListener("keydown", function (event) {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        send.click();
      }
    });
  });
})();
"""

__all__ = [
    "STATIC_CSS",
    "STATIC_JS",
    "auth_page",
    "home_page",
    "layout",
    "not_found_page",
    "post_page",
    "posts_page",
    "product_page",
    "products_page",
]
