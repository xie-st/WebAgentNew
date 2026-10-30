#!/usr/bin/env python3
"""Serve a live web view of a test run started by ``scripts/run_tests.sh``.

    scripts/watch_tests_web.py                 http://127.0.0.1:8787
    scripts/watch_tests_web.py --port 9000
    scripts/watch_tests_web.py --open          also open a browser
    scripts/watch_tests_web.py --vendor        keep a local copy of React, then
                                               never reach the network again

It reads ``.test-run/`` and never writes there, so starting it, stopping it and
restarting it cannot affect the run. Stopping the server does not stop the tests.

The interface is React, written with ``React.createElement`` rather than JSX, so
there is no build step, no bundler and no ``node_modules`` — one Python file
serves one page. React itself comes from a CDN unless ``--vendor`` has stored a
copy under ``scripts/_watch_assets/``, after which the page loads with no network
at all. Everything else — styles, markup, polling — is inline.

The page refreshes itself every two seconds. When nothing is running it says so
and shows the command that starts a run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_DIR = REPO / ".test-run"

PCT = re.compile(rb"\[ *(\d+)%\]")
TOTALS = re.compile(rb"\d+ (?:passed|failed)[a-z0-9, ]*")
PROGRESS = re.compile(rb"^[.sFExX]+", re.MULTILINE)


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _int(path: Path) -> int | None:
    raw = _read(path).strip()
    try:
        return int(raw)
    except ValueError:
        return None


def collect() -> dict:
    """Everything the page needs, derived from the files the driver writes."""
    manifest = RUN_DIR / "manifest.txt"
    if not manifest.is_file():
        return {"active": False, "lanes": [], "totals": {}}

    now = int(time.time())
    started = _int(RUN_DIR / "run.start") or now
    lanes: list[dict] = []

    for line in _read(manifest).decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        lane_id, stream, est, label, state = parts[0], parts[1], parts[2], parts[3], parts[4]
        est_s = int(est) * 60 if est.isdigit() else 600
        row = {
            "id": lane_id, "stream": stream, "label": label.strip(),
            "est": est_s, "pct": 0, "elapsed": 0, "left": est_s,
            "info": "", "rc": None, "estimated": True,
        }

        if state.startswith("skipped"):
            row["status"] = "skipped"
            row["reason"] = state.partition(":")[2] or "not selected"
            lanes.append(row)
            continue
        if state.startswith("unavailable"):
            row["status"] = "unavailable"
            row["reason"] = state.partition(":")[2]
            lanes.append(row)
            continue

        log = RUN_DIR / f"{lane_id}.txt"
        rc = _int(RUN_DIR / f"{lane_id}.rc")
        start = _int(RUN_DIR / f"{lane_id}.start")

        if rc is not None:
            body = _read(log)
            hit = TOTALS.findall(body)
            info = hit[-1].decode("utf-8", "replace") if hit else ""
            if not info:
                tail = [ln for ln in body.splitlines() if ln.strip()]
                info = tail[-1].decode("utf-8", "replace")[:70] if tail else ""
            row.update(
                status="done", rc=rc, pct=100, left=0, info=info,
                elapsed=(_int(RUN_DIR / f"{lane_id}.min") or 0) * 60,
                ok=(rc == 0), timedout=(rc in (124, 137)),
            )
        elif start is not None:
            body = _read(log)
            elapsed = max(0, now - start)
            marks = PCT.findall(body)
            if marks:
                pct = max(1, min(99, int(marks[-1])))
                left = int(elapsed * (100 - pct) / pct)
                estimated = False
            else:
                pct = min(99, int(elapsed * 100 / est_s)) if est_s else 0
                left = max(60, est_s - elapsed)
                estimated = True
            chars = b"".join(PROGRESS.findall(body))
            reds = sum(chars.count(c) for c in (b"F", b"E"))
            info = f"{len(chars)} run" if chars else "running"
            if reds:
                info += f", {reds}F"
            row.update(
                status="running", pct=pct, elapsed=elapsed, left=left,
                info=info, estimated=estimated, reds=reds,
            )
        else:
            row["status"] = "queued"
        lanes.append(row)

    # Streams run at the same time, so what is left overall is the slowest of
    # them. Stream A's parts also run at the same time, so it contributes its
    # slowest part rather than the sum.
    per_stream: dict[str, int] = {}
    for row in lanes:
        if row["status"] not in ("running", "queued"):
            continue
        s = row["stream"]
        if s == "A":
            per_stream[s] = max(per_stream.get(s, 0), row["left"])
        else:
            per_stream[s] = per_stream.get(s, 0) + row["left"]
    slowest = max(per_stream, key=lambda k: per_stream[k]) if per_stream else None

    counts = {k: sum(1 for r in lanes if r["status"] == k)
              for k in ("done", "running", "queued", "skipped", "unavailable")}
    selected = counts["done"] + counts["running"] + counts["queued"]
    failing = sum(1 for r in lanes if r["status"] == "done" and r.get("rc"))

    verdict = ""
    summary = RUN_DIR / "summary.txt"
    if summary.is_file():
        for ln in _read(summary).decode("utf-8", "replace").splitlines():
            if ln.startswith("RESULT:"):
                verdict = ln
    return {
        "active": True,
        "finished": counts["running"] == 0 and counts["queued"] == 0 and counts["done"] > 0,
        "elapsed": now - started,
        "eta": per_stream.get(slowest, 0) if slowest else 0,
        "slowest": slowest,
        "lanes": lanes,
        "verdict": verdict,
        "totals": {
            **counts,
            "selected": selected,
            "failing": failing,
            "pct": int(counts["done"] * 100 / selected) if selected else 0,
        },
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>effGen — test run</title>
<style>
:root{
  --bg:#05070d; --panel:#0b1020; --line:#1b2540; --dim:#5b6b90;
  --fg:#d8e3ff; --accent:#38e8ff; --accent2:#7c5cff;
  --ok:#3ddc97; --bad:#ff4d6d; --warn:#ffc857;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--fg); min-height:100vh;
  font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background-image:
    radial-gradient(1200px 600px at 80% -10%, rgba(124,92,255,.16), transparent 60%),
    radial-gradient(900px 500px at 0% 110%, rgba(56,232,255,.10), transparent 60%),
    linear-gradient(rgba(27,37,64,.35) 1px, transparent 1px),
    linear-gradient(90deg, rgba(27,37,64,.35) 1px, transparent 1px);
  background-size:auto,auto,44px 44px,44px 44px;
}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;margin-bottom:22px}
h1{font-size:19px;letter-spacing:.16em;text-transform:uppercase;margin:0;
   background:linear-gradient(90deg,var(--accent),var(--accent2));
   -webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--dim);font-size:12px;letter-spacing:.08em}
.pill{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12px;
      color:var(--dim);border:1px solid var(--line);border-radius:99px;padding:5px 12px;
      background:rgba(11,16,32,.7)}
.dot{width:7px;height:7px;border-radius:99px;background:var(--ok);
     box-shadow:0 0 10px var(--ok);animation:pulse 1.8s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35}}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}
.tile{border:1px solid var(--line);border-radius:12px;padding:14px 16px;
      background:linear-gradient(180deg,rgba(19,26,48,.85),rgba(11,16,32,.85))}
.tile .k{color:var(--dim);font-size:11px;letter-spacing:.14em;text-transform:uppercase}
.tile .v{font-size:26px;margin-top:6px;font-variant-numeric:tabular-nums}
.stream{margin:22px 0 8px;color:var(--accent);font-size:11px;letter-spacing:.22em;
        text-transform:uppercase;display:flex;align-items:center;gap:10px}
.stream:after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}
.lane{display:grid;grid-template-columns:170px 1fr 92px 76px;gap:14px;align-items:center;
      padding:9px 12px;border:1px solid transparent;border-radius:10px}
.lane+.lane{margin-top:3px}
.lane.running{border-color:rgba(56,232,255,.28);background:rgba(56,232,255,.05)}
.lane.muted{opacity:.42}
.name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.name small{display:block;color:var(--dim);font-size:11px}
.track{height:9px;border-radius:99px;background:#111a30;overflow:hidden;position:relative}
.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--accent2));
      transition:width .6s ease}
.fill.ok{background:linear-gradient(90deg,#2fb37a,var(--ok))}
.fill.bad{background:linear-gradient(90deg,#c22b47,var(--bad))}
.fill.est{background:linear-gradient(90deg,#3a4a72,#5b6b90)}
.lane.running .track:after{content:"";position:absolute;inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.32),transparent);
  transform:translateX(-100%);animation:sweep 2.1s linear infinite}
@keyframes sweep{to{transform:translateX(100%)}}
.meta{color:var(--dim);font-size:12px;text-align:right;font-variant-numeric:tabular-nums}
.state{font-size:11px;letter-spacing:.1em;text-transform:uppercase;text-align:right}
.state.done{color:var(--ok)} .state.bad{color:var(--bad)} .state.run{color:var(--accent)}
.state.idle{color:var(--dim)}
.foot{margin-top:26px;border-top:1px solid var(--line);padding-top:16px;color:var(--dim);font-size:12px}
.verdict{margin-top:16px;padding:14px 16px;border-radius:12px;border:1px solid var(--line);
         background:rgba(11,16,32,.8);letter-spacing:.04em}
.verdict.bad{border-color:rgba(255,77,109,.45);color:#ffd7de}
.verdict.ok{border-color:rgba(61,220,151,.45);color:#ccffe8}
.idlebox{max-width:640px;margin:14vh auto 0;text-align:center}
.idlebox h2{font-size:15px;letter-spacing:.28em;text-transform:uppercase;color:var(--dim);margin:0 0 18px}
.idlebox .big{font-size:30px;margin-bottom:26px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;background-clip:text;color:transparent}
.cmd{display:flex;align-items:center;gap:12px;justify-content:space-between;
     border:1px solid var(--line);border-radius:12px;padding:14px 16px;
     background:rgba(11,16,32,.9);text-align:left}
.cmd code{color:var(--accent)}
.cmd button{background:transparent;border:1px solid var(--line);color:var(--dim);
     border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit;font-size:12px}
.cmd button:hover{color:var(--fg);border-color:var(--accent)}
.hint{color:var(--dim);margin-top:20px;font-size:12px;line-height:1.9}
</style></head><body><div class="wrap" id="root"></div>
__REACT_TAGS__
<script>
// React without JSX, so the page needs no build step and no transpiler.
const {createElement: h, useState, useEffect, Fragment} = React;

const hms = (s) => s >= 3600 ? `${Math.floor(s/3600)}h${String(Math.floor(s%3600/60)).padStart(2,'0')}m`
                 : s >= 60 ? `${Math.floor(s/60)}m` : `${s|0}s`;

function CopyButton({text}) {
  const [label, setLabel] = useState('copy');
  return h('button', {
    onClick: () => {
      navigator.clipboard?.writeText(text);
      setLabel('copied');
      setTimeout(() => setLabel('copy'), 1400);
    }
  }, label);
}

function Idle() {
  const cmd = 'scripts/run_tests.sh';
  return h('div', {className: 'idlebox'},
    h('h2', null, 'effGen'),
    h('div', {className: 'big'}, 'No test run in progress'),
    h('div', {className: 'cmd'},
      h('code', null, cmd),
      h(CopyButton, {text: cmd})),
    h('div', {className: 'hint'},
      'It asks which lanes to include, then runs them.', h('br'),
      h('code', null, 'scripts/run_tests.sh --all'), ' takes every lane this machine can run.', h('br'),
      h('code', null, 'scripts/run_tests.sh --list'), ' shows the lanes without running anything.', h('br'),
      'This page picks the run up on its own once it starts.'));
}

function Lane({lane}) {
  const muted = ['skipped', 'unavailable', 'queued'].includes(lane.status);
  let fill = '', pct = lane.pct, meta = '', stateCls = 'idle', stateText = '';

  if (lane.status === 'done') {
    fill = lane.rc === 0 ? 'ok' : 'bad';
    stateCls = lane.rc === 0 ? 'done' : 'bad';
    stateText = lane.timedout ? 'timeout' : (lane.rc === 0 ? 'passed' : 'failed');
    meta = `${Math.round(lane.elapsed / 60)}m`;
  } else if (lane.status === 'running') {
    fill = lane.estimated ? 'est' : '';
    stateCls = 'run';
    stateText = `${lane.pct}%`;
    meta = `~${hms(lane.left)}${lane.estimated ? ' est' : ''}`;
  } else if (lane.status === 'queued') {
    pct = 0; stateText = 'queued'; meta = `~${hms(lane.est)} est`;
  } else {
    pct = 0;
    stateText = lane.status === 'skipped' ? 'skipped by you' : 'unavailable';
  }

  return h('div', {
      className: ['lane', lane.status === 'running' ? 'running' : '', muted ? 'muted' : ''].join(' ').trim()
    },
    h('div', {className: 'name'}, lane.id,
      h('small', null, lane.reason || lane.info || lane.label)),
    h('div', {className: 'track'},
      h('div', {className: `fill ${fill}`, style: {width: `${pct}%`}})),
    h('div', {className: 'meta'}, meta),
    h('span', {className: `state ${stateCls}`}, stateText));
}

function Tile({label, value, sub, tone}) {
  return h('div', {className: 'tile'},
    h('div', {className: 'k'}, label),
    h('div', {className: 'v', style: tone ? {color: tone} : null},
      value, sub ? h('span', {className: 'sub'}, sub) : null));
}

function Dash({data}) {
  const t = data.totals;
  const streams = [...new Set(data.lanes.map(l => l.stream))];
  const pillText = data.finished
    ? 'finished'
    : `~${hms(data.eta)} remaining${data.slowest ? ` · stream ${data.slowest} is the long pole` : ''}`;

  return h(Fragment, null,
    h('header', null,
      h('h1', null, 'effGen · test run'),
      h('span', {className: 'sub'},
        `${data.finished ? 'complete' : 'live'} · elapsed ${hms(data.elapsed)}`),
      h('span', {className: 'pill'},
        data.finished ? null : h('span', {className: 'dot'}), pillText)),

    h('div', {className: 'tiles'},
      h(Tile, {label: 'lanes done', value: t.done, sub: ` / ${t.selected}`}),
      h(Tile, {label: 'running', value: t.running}),
      h(Tile, {label: 'queued', value: t.queued}),
      h(Tile, {label: 'skipped by you', value: t.skipped}),
      h(Tile, {label: 'unavailable', value: t.unavailable}),
      h(Tile, {label: 'lanes failing', value: t.failing,
               tone: t.failing ? 'var(--bad)' : 'var(--ok)'})),

    streams.map(s => h(Fragment, {key: s},
      h('div', {className: 'stream'}, `stream ${s}`),
      data.lanes.filter(l => l.stream === s)
                .map(l => h(Lane, {key: l.id, lane: l})))),

    data.verdict
      ? h('div', {className: `verdict ${/FAILURES/.test(data.verdict) ? 'bad' : 'ok'}`}, data.verdict)
      : null,

    h('div', {className: 'foot'},
      'Reads .test-run/ only — closing this page does not stop the run. Logs are in ',
      h('code', null, '.test-run/<lane>.txt'), '.'));
}

function App() {
  const [data, setData] = useState(null);
  useEffect(() => {
    let live = true;
    const tick = async () => {
      try {
        const r = await fetch('/api/status', {cache: 'no-store'});
        const d = await r.json();
        if (live) setData(d);
      } catch (e) {
        if (live) setData(null);
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => { live = false; clearInterval(id); };
  }, []);
  return (data && data.active) ? h(Dash, {data}) : h(Idle);
}

ReactDOM.createRoot(document.getElementById('root')).render(h(App));
</script></body></html>
"""


VENDOR_DIR = Path(__file__).resolve().parent / "_watch_assets"
CDN = {
    "react.js": "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom.js": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
}


def vendored() -> bool:
    """True when a local copy of React is present, so the page needs no network."""
    return all((VENDOR_DIR / name).is_file() for name in CDN)


def vendor() -> None:
    """Store React locally. Run once; after that the page works offline."""
    from urllib.request import urlopen

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in CDN.items():
        print(f"fetching {name} ...", flush=True)
        with urlopen(url, timeout=60) as response:  # noqa: S310 - a fixed https URL
            (VENDOR_DIR / name).write_bytes(response.read())
    print(f"stored in {VENDOR_DIR}")


def page() -> str:
    if vendored():
        tags = ('<script src="/vendor/react.js"></script>\n'
                '<script src="/vendor/react-dom.js"></script>')
    else:
        tags = "\n".join(f'<script src="{url}" crossorigin></script>' for url in CDN.values())
    return PAGE.replace("__REACT_TAGS__", tags)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - the base class names it
        if self.path.startswith("/api/status"):
            body = json.dumps(collect()).encode()
            ctype = "application/json"
        elif self.path.startswith("/vendor/"):
            name = self.path.rsplit("/", 1)[-1]
            target = VENDOR_DIR / name
            if name not in CDN or not target.is_file():
                self.send_error(404)
                return
            body = target.read_bytes()
            ctype = "application/javascript"
        elif self.path in ("/", "/index.html"):
            body = page().encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        """Stay quiet; a request per two seconds is not worth printing."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    ap.add_argument("--vendor", action="store_true",
                    help="store React locally so the page loads with no network")
    args = ap.parse_args()

    if args.vendor:
        vendor()

    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"watching {RUN_DIR}")
    print("React: local copy" if vendored() else "React: from unpkg (--vendor stores it locally)")
    print(f"open {url}   (Ctrl-C stops this page, not the tests)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped watching.")


if __name__ == "__main__":
    main()
