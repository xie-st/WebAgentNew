# effgen top — live terminal monitor

`effgen top` (alias `effgen monitor`) puts everything effGen already records on
one screen: what ran, what the server is serving, what it cost, and what the
GPUs are doing. On a terminal it refreshes in place; piped or with `--json` it
prints a single snapshot and exits.

```bash
effgen top                          # live view, refreshing in place
effgen top --interval 1             # faster refresh
effgen top --url http://host:8000   # watch a remote server
effgen top --once                   # one static snapshot
effgen top --json | jq .spend       # machine-readable snapshot
```

## Panels and where each number comes from

| Panel | Source | Window |
|---|---|---|
| Activity | the run history (`~/.effgen/runs`) | completed runs, all processes on the host |
| Traffic | the server's `/dashboard/data.json` | that server process, since it started |
| Per-model | the same endpoint's `by_model` block | that server process, since it started |
| Spend | the local cost ledger | last 24 hours, plus a trailing-hour burn rate |
| GPU | the GPU layer (`effgen models status` uses the same one) | physical devices, all processes |

These sources measure different things over different windows — a server's
in-memory request count and the durable spend ledger routinely differ by orders
of magnitude — so each panel states its own scope and the figures are never
added together.

The same care applies inside the Traffic panel: the generation count and the
HTTP status histogram measure different populations. Generations counts model
calls; the histogram counts responses on **every** route, health checks and
dashboard reads included, so it is normally the larger number. Both are
labelled, and the histogram's own scope is on the JSON panel as
`by_status_scope`.

Every Per-model row is scoped to one `(model, provider)` pair: its p95 comes
from that provider's own observations and its cost from the runs recorded
against it, so a model name served by two providers reports two rows that do not
borrow each other's latency tail or spend.

Activity lists **completed** runs. A run that is still executing is not in the
history yet and does not appear until it finishes.

A model with no published price shows `—`, never `$0`, and is excluded from the
spend total; the Spend panel counts how many such models it left out.

## Running without a server

Activity, Spend and GPU read local files and devices, so the view is useful on
a host that only runs local agents. When no server answers, the Traffic and
Per-model panels read as unavailable and name the URL that was tried, rather
than showing zeros that would look like "no traffic, no errors".

## Finding the server

The base URL is taken from `--url`, then `--port` (on `127.0.0.1`), then the
`EFFGEN_SERVER_URL` environment variable, then `http://127.0.0.1:8000` — the
port `effgen serve` binds by default.

Reading a server's data requires its API key unless it was started with
`EFFGEN_PUBLIC_DASHBOARD=1`. Pass `--api-key` or set `EFFGEN_API_KEY`. A
rejected key is reported in the panel with the reason, so a missing key is not
mistaken for an idle server.

```bash
export EFFGEN_SERVER_URL=http://10.0.0.4:8000
export EFFGEN_API_KEY=…
effgen top
```

## Terminal behavior

- `q`, `Escape` or `Ctrl-C` quits and restores the terminal.
- `--interval SECONDS` sets the refresh cadence (default 2, range 0.5–300).
  GPUs are sampled every 5 seconds regardless, since that is the slowest source.
- `--count N` stops after N refreshes.
- `--limit N` sets how many runs the Activity panel shows.
- `--theme` and `NO_COLOR` are honored, as everywhere else in the CLI.

The full-screen view is used only on an interactive terminal that has not opted
out. Piped output, `--json`, `--once`, `--no-animation`, `NO_COLOR`, `CI=1` and
`EFFGEN_NO_ANIM=1` each print one static snapshot and exit — no cursor control
reaches a pipe, and no invocation loops forever in a script.

## JSON snapshot

`--json` writes one document with a `header` plus the `activity`, `traffic`,
`by_model`, `spend` and `gpu` panels. Every panel carries `scope`, `available`,
and `unavailable_reason`, so an automated consumer can tell "no data" from
"zero".

```bash
effgen top --json | jq '{
  reachable: .header.server_reachable,
  errors:    .traffic.errors,
  p95:       .traffic.p95_latency_s,
  burn:      .spend.burn_rate_usd_per_hour,
  gpu_util:  [.gpu.devices[].utilization_pct]
}'
```

A cron job can watch the burn rate without a terminal:

```bash
effgen top --json | jq -e '.spend.burn_rate_usd_per_hour < 0.5' >/dev/null \
  || echo "spend velocity above threshold"
```
