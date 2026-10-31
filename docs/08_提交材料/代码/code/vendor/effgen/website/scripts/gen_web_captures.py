#!/usr/bin/env python3
"""Turn the screenshots of effGen's web surfaces into the images `/dashboard`
and `/models` show, and write ``data/captures.web.json``.

    python scripts/gen_web_captures.py --from DIR   # convert and write both
    python scripts/gen_web_captures.py --check      # exit 1 if either is stale

Every entry in ``SHOTS`` is a screenshot of a **running** effGen server, driven
by real traffic — successful calls, a call naming a model that does not exist,
a malformed body, a probe of an unknown route, and a two-agent team run — taken
with a headless browser. Nothing is composed, retouched or drawn.

The two lossless-enough transformations are the only ones that happen: a shot
taken at a device pixel ratio of 2 is scaled by ``scale`` so the shipped file is
about 1.5x, and the result is written as WebP at a quality that leaves UI text
sharp. The manifest records each file's ``sha256``, its dimensions, the surface
and panel it shows, the theme it was taken in, and how it was produced, so a
figure on a page can always be traced back to the bytes behind it.

Re-taking them needs three things: an ``effgen serve`` with traffic through it, a
headless browser, and this script pointed at the directory the browser wrote to.
``--check`` needs none of them, so it can run anywhere: it re-hashes the
checked-in images and compares them against the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHIPPED = REPO / "public" / "captures" / "web"
OUT = REPO / "data" / "captures.web.json"

# How each shot was taken. `dpr` is the device pixel ratio the browser used;
# `scale` is what the shipped file is multiplied by, so a 2x shot ships at 1.5x
# and a 1x shot ships unchanged.
QUALITY = 82

#: slug -> (source file, surface, theme, alt text, caption source, scale)
SHOTS: dict[str, dict] = {
    "dashboard-full": {
        "sources": {"dark": "dashboard-full-dark.png", "light": "dashboard-full-light.png"},
        "surface": "dashboard",
        "scale": 1.0,
        "produced_by": "the whole page at 1440px, after the traffic below went through the server",
        "alt": (
            "The effGen dashboard from top to bottom: five summary cards, SLO burn "
            "rates and a latency chart, a per-model table, HTTP status chips, a "
            "per-route table, recent agent runs, stored history, the span stream, a "
            "run timeline, a two-agent topology graph, the model catalog and the raw "
            "Prometheus metrics."
        ),
    },
    "dashboard-summary": {
        "sources": {"dark": "dashboard-summary-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the summary row",
        "alt": (
            "Five cards: total requests 7; model-call errors 1, with HTTP 4 4xx "
            "and 0 5xx printed under it; average latency 1.480s; session cost "
            "$0.000400 across 6 priced and 1 unpriced run; 998 tokens used."
        ),
    },
    "dashboard-slo": {
        "sources": {"dark": "dashboard-slo-dark.png", "light": "dashboard-slo-light.png"},
        "surface": "dashboard",
        "scale": 1.0,
        "produced_by": "the SLO panel",
        "alt": (
            "SLO burn rates: p99 latency at 100.0 percent of its budget, error "
            "rate at 100.0 percent, availability at 85.7 percent, with p50 1.56s, "
            "p95 4.13s, p99 4.83s and a two-second target printed underneath."
        ),
    },
    "dashboard-latency": {
        "sources": {"dark": "dashboard-latency-dark.png"},
        "surface": "dashboard",
        "scale": 1.0,
        "produced_by": "the latency chart, drawn on a canvas",
        "alt": (
            "A filled line chart headed Request latency (recent), its vertical "
            "axis labelled 0.00s to 1.70s, with the measured average sitting just "
            "under 1.5s across the recorded intervals. It is drawn on a canvas by "
            "the page itself."
        ),
    },
    "dashboard-by-model": {
        "sources": {"dark": "dashboard-by-model-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the By model panel",
        "alt": (
            "A table with one row per model and provider. gemini-3.1-flash-lite "
            "on gemini: 4 calls, 0.0 percent errors, p95 2.400s, outcome ok, 234 "
            "tokens in and 50 out, $0.000133. gpt-5-nano on openai: 2 calls, 0.0 "
            "percent, p95 4.750s, outcome ok, 54 and 660, $0.000267. "
            "gpt-9-does-not-exist on openai: 1 call, 100.0 percent errors, p95 "
            "0.098s, outcome not_found once, 0 tokens either way, and a dash "
            "where the cost would be."
        ),
    },
    "dashboard-by-status": {
        "sources": {"dark": "dashboard-by-status-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the HTTP responses by status panel",
        "alt": (
            "Three chips under the heading HTTP responses by status, each naming "
            "the code and its count as text: 200 twenty-one times, 404 three "
            "times, 422 once."
        ),
    },
    "dashboard-by-route": {
        "sources": {"dark": "dashboard-by-route-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the Responses by route panel",
        "alt": (
            "Requests, failures and error rate per route and method, worst first. "
            "POST /v1/chat/completions: 9 requests, 3 errors, 33.3 percent, by "
            "class 2xx six and 4xx three. A GET row labelled other: 14 requests, "
            "1 error, 7.1 percent — the note above the table says it collects the "
            "dashboard's own polling and any unrouted request. GET /health and "
            "GET /v1/models: one request each, 0.0 percent."
        ),
    },
    "dashboard-agent-runs": {
        "sources": {"dark": "dashboard-agent-runs-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the Recent agent runs panel",
        "alt": (
            "The last seven runs in time order with model, tokens in and out, "
            "cost, duration and a status badge. The failed run against "
            "gpt-9-does-not-exist shows a dash for tokens and a dash for cost "
            "rather than zeros, and carries a red error badge; the rest are green "
            "ok."
        ),
    },
    "dashboard-history": {
        "sources": {"dark": "dashboard-history-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the History panel",
        "alt": (
            "The History panel: 9 runs, 0 sessions, with the directory they are "
            "stored in named above a text search and a status filter. Each row "
            "carries the time, the model, the task as a link that opens a detail "
            "pane, the cost, the duration and a status badge. A saved-sessions "
            "table underneath reads: no saved sessions."
        ),
    },
    "dashboard-waterfall": {
        "sources": {"dark": "dashboard-waterfall-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the Run timeline panel",
        "alt": (
            "Spans grouped by run under the heading Run timeline. Each run is a "
            "labelled block with its total duration and span count on the right, "
            "and inside it a run bar above a model bar, positioned by start "
            "offset and sized by duration; one run also carries a short tool bar. "
            "The last run, against gpt-9-does-not-exist, is drawn in the error "
            "colour."
        ),
    },
    "dashboard-topology": {
        "sources": {"dark": "dashboard-topology-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the Agent topology panel, after a two-agent team run",
        "alt": (
            "A node-link graph headed Agent topology, captioned: team newsroom, 2 "
            "nodes, 1 edge, $0.000061, 88 tokens. Two nodes, writer and editor, "
            "each showing its model and an ok glyph with the word ok beside it, "
            "joined by a dashed arrow. Above the graph, an execution picker and a "
            "legend naming ok, running, skipped, failed, agent and tool in words "
            "as well as in glyphs."
        ),
    },
    "dashboard-catalog": {
        "sources": {"dark": "dashboard-catalog-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the Model catalog panel",
        "alt": (
            "The Model catalog panel, headed: 417 models across 9 providers, "
            "pricing from catalog snapshot verified 2026-06-08, run effgen models "
            "refresh to update. Under it a search box, a provider picker, a sort "
            "picker, tools, vision, audio and free checkboxes and a "
            "minimum-context field, then a table of provider, model id, context, "
            "max output, input and output price per million tokens, and tool and "
            "vision ticks. The pager reads 1 to 25 of 417."
        ),
    },
    "dashboard-catalog-filtered": {
        "sources": {"dark": "dashboard-catalog-filtered-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the same panel with gemini typed into its search box",
        "alt": (
            "The same panel with gemini typed into its search box: eleven rows, "
            "six Gemini models and two Gemma models from the gemini provider plus "
            "three google/gemini ids from replicate, each with its context "
            "window, output limit, prices and capability ticks. The pager reads 1 "
            "to 11 of 11, of 417."
        ),
    },
    "dashboard-palette": {
        "sources": {
            "dark": "dashboard-palette-dark.png",
            "light": "dashboard-palette-light.png",
        },
        "surface": "dashboard",
        "scale": 0.6,
        "produced_by": "Ctrl-K, then typing cost",
        "alt": (
            "The command palette open over the dashboard with gemini typed into "
            "it. The results are grouped: a Models heading over catalogue ids "
            "with their provider on the right, then a Runs heading over stored "
            "runs matched on their task text, each showing the model that "
            "answered. The footer reads: up and down move, Enter run, Esc close, "
            "question mark shortcuts."
        ),
    },
    "dashboard-shortcuts": {
        "sources": {"dark": "dashboard-shortcuts-dark.png"},
        "surface": "dashboard",
        "scale": 0.6,
        "produced_by": "pressing ?",
        "alt": (
            "The keyboard shortcut list: Ctrl-K opens the command palette; "
            "question mark shows this list; Escape closes the palette, this list, "
            "or an open detail; the up and down arrows move through palette "
            "results; Enter runs the highlighted command; and the four arrow keys "
            "move between topology nodes once one has focus. Escape closes the "
            "list."
        ),
    },
    "dashboard-auth": {
        "sources": {"dark": "dashboard-auth-dark.png"},
        "surface": "dashboard",
        "scale": 0.75,
        "produced_by": "the same page against a server started with no key configured",
        "alt": (
            "The dashboard against a server started with no key configured. A "
            "banner across the top reads: dashboard data requires authentication, "
            "restart the server with EFFGEN_PUBLIC_DASHBOARD=1 for local viewing, "
            "or supply an API key. The status in the header reads Offline, and "
            "every card and burn-rate figure below shows a dash rather than a "
            "zero."
        ),
    },
    "playground-run": {
        "sources": {
            "dark": "playground-run-dark.png",
            "light": "playground-run-light.png",
        },
        "surface": "playground",
        "scale": 0.7,
        "produced_by": (
            "one run of the prompt shown, with the calculator attached and streaming off"
        ),
        "alt": (
            "The playground after a run. On the left the compose column: an API "
            "key field noted as held in memory for this tab only, a preset "
            "picker, a single-run or battle mode choice, a model picker reading "
            "gemini-3.1-flash-lite with free, 1000K context, tools, vision and "
            "its verification date under it, the prompt, and advanced controls "
            "with the calculator tool ticked, temperature 0.7, max tokens 512 and "
            "streaming off. On the right the answer 1827993, then model, tokens "
            "163 in and 22 out, total 185, cost $0.000074 and latency 0.78s; a "
            "tool trace row reading calculator, the expression it was given and "
            "what it returned; and the run offered back as curl, CLI or Python, "
            "with the curl form shown."
        ),
    },
    "playground-battle": {
        "sources": {"dark": "playground-battle-dark.png"},
        "surface": "playground",
        "scale": 0.75,
        "produced_by": "battle mode with two contenders on one prompt",
        "alt": (
            "Battle mode: two columns, gemini:gemini-3.1-flash-lite and "
            "openai:gpt-5-nano, each carrying that model's full answer to the "
            "same prompt with its own time to first token, total time, token "
            "count and cost. A verdict panel underneath names the fastest, the "
            "cheapest and the longest answer, and states that 2 of 2 answered in "
            "4.74s for a total of $0.000423."
        ),
    },
    "eval-report": {
        "sources": {
            "dark": "eval-report-dark.png",
            "light": "eval-report-light.png",
        },
        "surface": "report",
        "scale": 0.85,
        "produced_by": (
            "the HTML file `effgen eval --suite math -m openai:gpt-5-nano "
            "--max-cases 5 --fail-under 0.8 --report math-eval.html` wrote, "
            "opened from disk with the network off"
        ),
        "alt": (
            "The evaluation report the command wrote: the heading Evaluation "
            "Report — math, the line math, 5 of 5 passed, scoring contains, the "
            "time it was generated, the effGen version, and the exact command "
            "that produced it. An exit-gate panel reads PASS — accuracy 100.0% "
            "is at or above the required 80%. Four cards follow: pass rate "
            "100.0% over 5 of 5 cases, average latency 1.5807s, 1,350 total "
            "tokens, and $0.000399 total cost on openai:gpt-5-nano. Then "
            "accuracy by difficulty, easy 5 of 5 at 100%, and a table of the "
            "five cases, each with the query, the expected answer, what the "
            "model returned, a PASS chip, the latency, the cost and the "
            "difficulty. The footer says every style, script and chart is "
            "contained in this file."
        ),
    },
    "playground-trace": {
        "sources": {"dark": "playground-trace-dark.png"},
        "surface": "playground",
        "scale": 0.6,
        "produced_by": "the tool trace of the run above",
        "alt": (
            "One trace row: the tool name calculator, the arguments it was called "
            "with, an arrow, the result 1827993, and the time it took on the "
            "right."
        ),
    },
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(source_dir: Path, only: set[str] | None = None) -> None:
    """Scale and re-encode every shot, and write it under ``public/``.

    ``only`` narrows the pass to the named slugs, so one figure can be re-taken
    without re-taking the whole set — the raw screenshots for the others live
    with whoever captured them, and the shipped WebP files are already on disk.
    """
    from PIL import Image

    unknown = (only or set()) - set(SHOTS)
    if unknown:
        raise SystemExit(f"no such shot: {', '.join(sorted(unknown))}")

    SHIPPED.mkdir(parents=True, exist_ok=True)
    for slug, shot in SHOTS.items():
        if only is not None and slug not in only:
            continue
        for theme, filename in shot["sources"].items():
            src = source_dir / filename
            if not src.exists():
                raise SystemExit(f"missing screenshot: {src}")
            image = Image.open(src).convert("RGB")
            scale = shot["scale"]
            if scale != 1.0:
                size = (round(image.width * scale), round(image.height * scale))
                image = image.resize(size, Image.LANCZOS)
            target = SHIPPED / f"{slug}-{theme}.webp"
            image.save(target, "WEBP", quality=QUALITY, method=6)
            print(f"  {target.name:38s} {image.width:5d}x{image.height:<5d} "
                  f"{target.stat().st_size / 1024:7.1f} kB")


def collect() -> dict:
    """The manifest, read back off the files that will ship."""
    from PIL import Image

    entries: dict[str, dict] = {}
    for slug, shot in SHOTS.items():
        for theme in shot["sources"]:
            path = SHIPPED / f"{slug}-{theme}.webp"
            if not path.exists():
                raise SystemExit(f"not converted yet: {path.relative_to(REPO)}")
            with Image.open(path) as image:
                width, height = image.size
            entries[f"{slug}-{theme}"] = {
                "slug": slug,
                "theme": theme,
                "surface": shot["surface"],
                "src": f"/captures/web/{path.name}",
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
                "produced_by": shot["produced_by"],
                "alt": shot["alt"],
            }
    return {"captures": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", help="directory holding the raw screenshots")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the manifest no longer matches the shipped images")
    parser.add_argument("--only", help="comma-separated slugs to convert, instead of all of them")
    args = parser.parse_args()

    if args.source:
        only = {slug.strip() for slug in args.only.split(",")} if args.only else None
        convert(Path(args.source), only)

    fresh = collect()
    if args.check:
        if not OUT.exists():
            print(f"{OUT.relative_to(REPO)} does not exist", file=sys.stderr)
            return 1
        current = json.loads(OUT.read_text(encoding="utf-8"))
        current.pop("generated_at", None)
        if current != fresh:
            print(f"{OUT.relative_to(REPO)} is stale", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(REPO)} matches {len(fresh['captures'])} shipped images")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {"generated_at": stamp, **fresh}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(entry["bytes"] for entry in fresh["captures"].values())
    print(f"wrote {OUT.relative_to(REPO)} — {len(fresh['captures'])} images, {total / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
