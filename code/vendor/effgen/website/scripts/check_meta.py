"""Read the built export and report what each route tells a search engine.

A title and a description are the two things a result list shows, and both are
cut off if they run long — about 60 characters for a title and about 160 for a
description. Two routes that describe themselves identically are worse than
either: a reader cannot tell them apart, and a crawler has to guess which one
answers the query.

So this walks `out/`, reads the head of every page, and reports:

  * the title and the description, with their lengths;
  * anything missing, anything over the length that gets cut off, and any two
    routes sharing a title or a description;
  * the canonical address, and whether it matches the route it is on;
  * the Open Graph and Twitter cards, and whether the image they name is on disk;
  * `sitemap.xml`, `robots.txt` and the web manifest, and whether every address
    in the sitemap resolves to something the export actually serves.

    npm run build
    python scripts/check_meta.py            report; exit 1 on a problem
    python scripts/check_meta.py --quiet    problems only

The documentation is a single-page app, so its 72 routes are not files here and
are not walked: each one sets its own head as it renders, which needs a browser
to observe.
"""

from __future__ import annotations

import argparse
import html.parser
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SITE_URL = "https://effgen.org"

TITLE_LIMIT = 60
DESCRIPTION_LIMIT = 160
DESCRIPTION_FLOOR = 70


class Head(html.parser.HTMLParser):
    """The metadata in one page's `<head>`, in the order it is written."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta: dict[str, str] = {}
        self.canonical = ""
        self.manifest = ""
        self.icons: list[str] = []
        self.ld_json: list[str] = []
        self._in_title = False
        self._in_ld = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = a.get("name") or a.get("property")
            if key:
                self.meta[key] = a.get("content", "")
        elif tag == "link":
            rel = (a.get("rel") or "").lower()
            if rel == "canonical":
                self.canonical = a.get("href", "")
            elif rel == "manifest":
                self.manifest = a.get("href", "")
            elif "icon" in rel:
                self.icons.append(a.get("href", ""))
        elif tag == "script" and a.get("type") == "application/ld+json":
            self._in_ld = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_ld = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._in_ld:
            self.ld_json.append(data)


def routes() -> list[tuple[str, Path]]:
    """Every page the export serves, as (route, file), shallowest first."""
    found: list[tuple[str, Path]] = []
    for path in sorted(OUT.rglob("index.html")):
        rel = path.relative_to(OUT).parent.as_posix()
        # The documentation bundle is one file for 72 client-side routes.
        if rel == "docs" or rel.startswith("docs/"):
            continue
        found.append(("/" if rel == "." else f"/{rel}", path))
    # The export writes the 404 twice — as `404.html` for a host that serves it
    # by name, and as `404/index.html` for one that serves it by path. They are
    # the same page, so it is checked once.
    if not any(route == "/404" for route, _ in found) and (OUT / "404.html").exists():
        found.append(("/404", OUT / "404.html"))
    return sorted(found, key=lambda r: (r[0].count("/"), r[0]))


def local_path(url: str) -> Path | None:
    """Where an address the page names would be served from, inside `out/`."""
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != urlparse(SITE_URL).netloc:
        return None
    return OUT / parsed.path.lstrip("/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="print problems only")
    args = ap.parse_args()

    if not OUT.exists():
        print("out/ is not built — run npm run build first")
        return 1

    problems: list[str] = []
    seen_titles: dict[str, str] = {}
    seen_descriptions: dict[str, str] = {}

    for route, path in routes():
        head = Head()
        head.feed(path.read_text(encoding="utf-8", errors="replace"))
        title = head.title.strip()
        description = head.meta.get("description", "").strip()
        noindex = "noindex" in head.meta.get("robots", "")

        if not args.quiet:
            print(f"\n{route}")
            print(f"  title       {len(title):>3}  {title}")
            print(f"  description {len(description):>3}  {description}")
            print(f"  canonical        {head.canonical or '—'}")
            print(f"  og:image         {head.meta.get('og:image', '—')}")
            print(f"  twitter:card     {head.meta.get('twitter:card', '—')}")
            if noindex:
                print(f"  robots           {head.meta['robots']}")

        if not title:
            problems.append(f"{route}: no title")
        elif len(title) > TITLE_LIMIT:
            problems.append(f"{route}: title is {len(title)} characters, cut off after {TITLE_LIMIT}")
        if not description:
            problems.append(f"{route}: no description")
        elif len(description) > DESCRIPTION_LIMIT:
            problems.append(
                f"{route}: description is {len(description)} characters, "
                f"cut off after {DESCRIPTION_LIMIT}"
            )
        elif len(description) < DESCRIPTION_FLOOR:
            problems.append(
                f"{route}: description is only {len(description)} characters — "
                f"too short to say what is on the page"
            )

        if title and title in seen_titles:
            problems.append(f"{route}: same title as {seen_titles[title]}")
        seen_titles.setdefault(title, route)
        if description and description in seen_descriptions:
            problems.append(f"{route}: same description as {seen_descriptions[description]}")
        seen_descriptions.setdefault(description, route)

        if not head.canonical:
            problems.append(f"{route}: no canonical address")
        else:
            want = f"{SITE_URL}/" if route == "/" else f"{SITE_URL}{route}/"
            if route != "/404" and head.canonical != want:
                problems.append(f"{route}: canonical is {head.canonical}, expected {want}")

        for key in ("og:title", "og:description", "og:url", "og:image", "og:type",
                    "twitter:card", "twitter:title", "twitter:description", "twitter:image"):
            if not head.meta.get(key):
                problems.append(f"{route}: no {key}")

        for key in ("og:image", "twitter:image"):
            url = head.meta.get(key, "")
            if not url:
                continue
            if not url.startswith("http"):
                problems.append(f"{route}: {key} is relative ({url}) — a crawler cannot resolve it")
            target = local_path(url)
            if target is not None and not target.exists():
                problems.append(f"{route}: {key} names {url}, which is not in the export")

        if not head.icons:
            problems.append(f"{route}: no icon")
        if not head.manifest:
            problems.append(f"{route}: no web manifest")
        for href in head.icons + ([head.manifest] if head.manifest else []):
            target = local_path(href)
            if target is not None and not target.exists():
                problems.append(f"{route}: {href} is not in the export")

    # --- the files a crawler asks for by name -----------------------------
    for name in ("sitemap.xml", "robots.txt", "manifest.webmanifest"):
        if not (OUT / name).exists():
            problems.append(f"out/{name} is missing")

    sitemap = OUT / "sitemap.xml"
    if sitemap.exists():
        urls = re.findall(r"<loc>([^<]+)</loc>", sitemap.read_text())
        listed = {u.rstrip("/") or "/" for u in urls}
        if not args.quiet:
            print(f"\nsitemap.xml  {len(urls)} addresses")
        for url in urls:
            target = local_path(url)
            if target is None:
                continue
            # A documentation address is served by the SPA's own index.html.
            if "/docs/" in url or url.rstrip("/").endswith("/docs"):
                if not (OUT / "docs" / "index.html").exists():
                    problems.append(f"sitemap: {url} has no documentation bundle to serve it")
                continue
            if not (target / "index.html").exists() and not target.exists():
                problems.append(f"sitemap: {url} is not in the export")
        for route, _ in routes():
            if route == "/404":
                continue
            want = f"{SITE_URL}{route}".rstrip("/") or SITE_URL
            if want.rstrip("/") not in listed:
                problems.append(f"sitemap: {route} is served but not listed")

    robots = OUT / "robots.txt"
    if robots.exists():
        text = robots.read_text()
        if "Sitemap:" not in text:
            problems.append("robots.txt does not name the sitemap")
        if not args.quiet:
            print(f"\nrobots.txt\n{text.strip()}")

    print(f"\n{len(routes())} routes checked, {len(problems)} problems")
    for line in problems:
        print(f"  {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
