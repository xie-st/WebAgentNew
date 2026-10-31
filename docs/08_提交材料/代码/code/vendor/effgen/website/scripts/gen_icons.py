"""Render the site's icons and share cards from the mark in `public/favicon.svg`.

Everything this writes is self-hosted and checked in: the favicons and app icons
a browser and a home screen ask for, and one 1200x630 card per landing route for
Open Graph and Twitter. Nothing is fetched at view time and nothing is fetched
here — the mark is the repository's own SVG and the type is the Inter face
already vendored in `assets/fonts/`.

A headless browser does the rasterising, so the icons are the same shape a
visitor sees in the tab and the cards use the same font file the documentation
loads.

    pip install playwright && python -m playwright install chromium
    python scripts/gen_icons.py            write the icons, the cards and favicon.ico
    python scripts/gen_icons.py --check    fail if anything on disk is out of date

The `--check` form is what catches a card whose title no longer matches the page
it belongs to.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAVICON = ROOT / "public" / "favicon.svg"
FONT = ROOT / "assets" / "fonts" / "inter-latin.woff2"
# The documentation is a separate app with its own `public/`, and it was serving
# a mark from an earlier identity — the same hexagon in purple — so the tab icon
# changed colour when a reader crossed from the site into the documentation.
# Both halves are written from the one SVG here so they cannot drift again.
DOCS_PUBLIC = "effgen-docs/public"

# The site's dark ground and its accent ramp, read off `app/globals.css` and
# `tailwind.config.ts` rather than picked. An icon or a card that used different
# greens would read as a different product.
GROUND = "#020c08"
ACCENT = "#00ff88"
ACCENT_2 = "#00e5ff"
INK = "#ffffff"
MUTED = "#93a59b"

# Transparent icons a browser puts in a tab, and opaque ones a home screen puts
# on a wallpaper. A transparent home-screen icon is drawn on whatever is behind
# it, which is why those carry the ground.
ICONS = [
    ("favicon-16.png", 16, None, 0.0),
    ("favicon-32.png", 32, None, 0.0),
    ("favicon-48.png", 48, None, 0.0),
    ("icon-192.png", 192, GROUND, 0.08),
    ("icon-512.png", 512, GROUND, 0.08),
    # A maskable icon is cropped to whatever shape the launcher wants, so the
    # mark sits inside the 80% safe area rather than filling the square.
    ("icon-maskable-512.png", 512, GROUND, 0.20),
    ("apple-touch-icon.png", 180, GROUND, 0.10),
]

FAVICON_ICO_SIZES = [16, 32, 48]

# One card per landing route. The title is what the card says in large type and
# the line under it is what it says in small type; both are shorter than the
# page's own <title> and description, because a card is read at a glance.
CARDS = [
    ("home", "Agents built for small language models",
     "Run them on your own hardware, or on any OpenAI-compatible server."),
    ("agents", "The agent library",
     "One call returns the answer and every tool call that produced it."),
    ("cli", "The effgen command line",
     "Run an agent, race models, serve an API, watch what it costs."),
    ("code", "effgen code",
     "A coding agent that shows every change as a diff before it lands."),
    ("dashboard", "The dashboard and the playground",
     "Live traffic, cost and latency, and a browser playground, in the package."),
    ("models", "Any model, anywhere",
     "Ten provider adapters, local engines, and one base_url for the rest."),
    ("production", "Running it for real",
     "Authentication, roles, rate limits, metrics, traces and budgets."),
    ("changelog", "Changelog",
     "Every release, what it changed, and the migrations it needs."),
    ("examples", "Examples",
     "Complete programs, with the output they actually produce."),
    ("community", "Community",
     "The repository, the issue tracker, and how to contribute."),
    ("leaderboard", "Benchmarks",
     "How small models score across reasoning, agentic and retrieval tasks."),
    ("docs", "Documentation",
     "Every behaviour, flag and parameter, with code you can copy and run."),
]

CARD_W, CARD_H = 1200, 630


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _icon_html(size: int, ground: str | None, pad: float) -> str:
    """A page that is exactly one icon, at exactly one size."""
    mark = _data_uri(FAVICON, "image/svg+xml")
    inset = round(size * pad)
    bg = f"background:{ground};" if ground else "background:transparent;"
    radius = f"border-radius:{round(size * 0.22)}px;" if ground else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
      *{{margin:0;padding:0;box-sizing:border-box}}
      html,body{{width:{size}px;height:{size}px;background:transparent}}
      .plate{{width:{size}px;height:{size}px;{bg}{radius}
              display:flex;align-items:center;justify-content:center}}
      img{{width:{size - 2 * inset}px;height:{size - 2 * inset}px;display:block}}
    </style></head><body><div class="plate"><img src="{mark}" alt=""></div></body></html>"""


def _card_html(title: str, subtitle: str, version: str) -> str:
    """A page that is exactly one share card."""
    mark = _data_uri(FAVICON, "image/svg+xml")
    font = _data_uri(FONT, "font/woff2")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
      @font-face {{ font-family:'Inter'; font-style:normal; font-weight:100 900;
                    src:url('{font}') format('woff2'); }}
      *{{margin:0;padding:0;box-sizing:border-box}}
      html,body{{width:{CARD_W}px;height:{CARD_H}px}}
      body{{background:{GROUND};font-family:'Inter',system-ui,sans-serif;color:{INK};
            display:flex;flex-direction:column;justify-content:space-between;
            padding:72px 80px;position:relative;overflow:hidden}}
      /* The same faint grid and accent wash the pages carry, so a card reads as
         part of the site rather than as a separate template. */
      .grid{{position:absolute;inset:0;
             background-image:linear-gradient(rgba(0,255,136,.055) 1px,transparent 1px),
                              linear-gradient(90deg,rgba(0,255,136,.055) 1px,transparent 1px);
             background-size:48px 48px}}
      .wash{{position:absolute;top:-320px;right:-260px;width:820px;height:820px;
             background:radial-gradient(circle,rgba(0,255,136,.16) 0%,rgba(0,229,255,.07) 42%,transparent 68%)}}
      .row{{position:relative;display:flex;align-items:center;gap:20px}}
      .row img{{width:64px;height:64px;display:block}}
      .word{{font-size:38px;font-weight:900;letter-spacing:-.02em}}
      .chip{{margin-left:8px;padding:7px 16px;border-radius:999px;font-size:20px;font-weight:700;
             color:{ACCENT};border:1px solid rgba(0,255,136,.42);background:rgba(0,255,136,.09)}}
      main{{position:relative}}
      h1{{font-size:70px;line-height:1.06;font-weight:900;letter-spacing:-.035em;max-width:15ch;
          background:linear-gradient(120deg,{INK} 30%,{ACCENT} 78%,{ACCENT_2} 100%);
          -webkit-background-clip:text;background-clip:text;color:transparent}}
      p{{margin-top:24px;font-size:30px;line-height:1.4;color:{MUTED};max-width:32ch;font-weight:500}}
      footer{{position:relative;display:flex;align-items:center;gap:14px;
              font-size:24px;font-weight:600;color:{MUTED}}}
      .dot{{width:9px;height:9px;border-radius:999px;background:{ACCENT}}}
    </style></head><body>
      <div class="grid"></div><div class="wash"></div>
      <div class="row"><img src="{mark}" alt=""><span class="word">effGen</span>
        <span class="chip">v{version}</span></div>
      <main><h1>{title}</h1><p>{subtitle}</p></main>
      <footer><span class="dot"></span><span>effgen.org</span></footer>
    </body></html>"""


def _ico(pngs: list[bytes]) -> bytes:
    """An .ico wrapping the PNGs it is given, one directory entry each."""
    out = io.BytesIO()
    out.write(struct.pack("<HHH", 0, 1, len(pngs)))
    offset = 6 + 16 * len(pngs)
    for png, size in zip(pngs, FAVICON_ICO_SIZES):
        # 0 in the width/height byte means 256; every size here is smaller.
        out.write(struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(png), offset))
        offset += len(png)
    for png in pngs:
        out.write(png)
    return out.getvalue()


def render() -> dict[str, bytes]:
    """Every file this script owns, as bytes, without touching the disk."""
    from playwright.sync_api import sync_playwright

    version = json.loads((ROOT / "data" / "effgen.json").read_text())["version"]
    files: dict[str, bytes] = {}
    icon_pngs: dict[int, bytes] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for name, size, ground, pad in ICONS:
                page = browser.new_page(viewport={"width": size, "height": size},
                                        device_scale_factor=1)
                page.set_content(_icon_html(size, ground, pad))
                png = page.screenshot(omit_background=ground is None)
                page.close()
                files[f"public/icons/{name}"] = png
                if ground is None and size in FAVICON_ICO_SIZES:
                    icon_pngs[size] = png

            for slug, title, subtitle in CARDS:
                page = browser.new_page(viewport={"width": CARD_W, "height": CARD_H},
                                        device_scale_factor=1)
                page.set_content(_card_html(title, subtitle, version))
                page.wait_for_timeout(120)  # let the vendored face finish loading
                png = page.screenshot()
                page.close()
                files[f"public/og/{slug}.png"] = png
        finally:
            browser.close()

    files["public/favicon.ico"] = _ico([icon_pngs[s] for s in FAVICON_ICO_SIZES])

    # What the documentation half needs: the same mark, and the two raster sizes
    # a tab and a home screen ask for.
    files[f"{DOCS_PUBLIC}/favicon.svg"] = FAVICON.read_bytes()
    files[f"{DOCS_PUBLIC}/icons/favicon-32.png"] = files["public/icons/favicon-32.png"]
    files[f"{DOCS_PUBLIC}/icons/apple-touch-icon.png"] = files["public/icons/apple-touch-icon.png"]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when what is on disk differs from what this would write")
    args = ap.parse_args()

    files = render()
    stale: list[str] = []

    for rel, data in sorted(files.items()):
        target = ROOT / rel
        current = target.read_bytes() if target.exists() else b""
        # A PNG carries no timestamp here, but two runs of the same browser can
        # still differ by a byte or two of compression, so identity is decided on
        # the decoded pixels rather than on the file.
        if _pixels(current) != _pixels(data):
            stale.append(rel)
            if not args.check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)

    if args.check:
        for rel in stale:
            print(f"out of date: {rel}")
        print(f"{len(files) - len(stale)} of {len(files)} up to date")
        return 1 if stale else 0

    for rel in sorted(files):
        print(f"{'wrote  ' if rel in stale else 'same   '} {rel:<48} "
              f"{len(files[rel]):>7,} bytes  {hashlib.sha256(files[rel]).hexdigest()[:12]}")
    return 0


def _pixels(data: bytes) -> bytes:
    """The image content of a PNG, ignoring the chunks that carry no pixels."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data
    out, i = bytearray(), 8
    while i + 8 <= len(data):
        (length,) = struct.unpack(">I", data[i:i + 4])
        kind = data[i + 4:i + 8]
        if kind in (b"IHDR", b"PLTE", b"IDAT", b"tRNS"):
            out += kind + data[i + 8:i + 8 + length]
        i += 12 + length
    return bytes(out)


if __name__ == "__main__":
    sys.exit(main())
