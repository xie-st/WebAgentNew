// The grouping and the prose `/dashboard` renders.
//
// The panel list, the summary cards, the data endpoints, the snippet kinds, the
// static-file sizes, the count of references to an outside host and the `serve`
// flag and environment tables all come from `data/effgen.json`, which
// `scripts/gen_site_data.py` reads off the files the `effgen` package ships and
// out of the real `--help`. Nothing here restates any of them.
//
// What is written here is the one thing the package does not state: which
// panels answer the same question. `assertGroupsCoverEveryPanel` below fails the
// build if that grouping ever stops matching the panel list, so a panel added to
// the dashboard cannot quietly go missing from this page.

import { siteData } from "@/components/siteData";

const FRAMEWORK_DOCS = "https://github.com/ctrl-gaurav/effGen/blob/main";

export const DOCS_DASHBOARD_URL = `${FRAMEWORK_DOCS}/docs/dx/dashboard.md`;
export const DOCS_SERVER_URL = `${FRAMEWORK_DOCS}/docs/server/openai-compat.md`;
export const DOCS_AUTH_URL = `${FRAMEWORK_DOCS}/docs/server/auth.md`;

export interface PanelGroup {
  id: string;
  title: string;
  lede: string;
  accent: string;
  panels: string[];
}

/** The twelve panels, grouped by the question they answer. */
export const panelGroups: PanelGroup[] = [
  {
    id: "health",
    title: "Is it healthy",
    lede:
      "The five cards and the two panels beside them: how much went through, how " +
      "much of it failed, how close the latency and error budgets are to being " +
      "spent, and how the last few intervals compare.",
    accent: "#00ff88",
    panels: ["panel-slo", "panel-latency-chart"],
  },
  {
    id: "cost",
    title: "What did it cost, and which model spent it",
    lede:
      "One row per model and provider, so a model served by two providers reports " +
      "each provider's own latency tail and each provider's own spend.",
    accent: "#00e5ff",
    panels: ["panel-by-model"],
  },
  {
    id: "errors",
    title: "What failed, and where",
    lede:
      "A status code on its own says a request failed; it does not say which route " +
      "failed or why. These two panels carry the denominator.",
    accent: "#ff6b6b",
    panels: ["panel-by-status", "panel-by-route"],
  },
  {
    id: "runs",
    title: "What ran",
    lede:
      "The runs this process has seen, the durable history behind them, and the " +
      "spans each run produced — live, and laid out on a timeline.",
    accent: "#a78bfa",
    panels: ["panel-agent-runs", "panel-history", "panel-spans", "panel-waterfall"],
  },
  {
    id: "shape",
    title: "What the work looks like",
    lede:
      "A team or a workflow as a graph, and the catalog every model id on the page " +
      "resolves against.",
    accent: "#ffd700",
    panels: ["panel-topology", "panel-catalog"],
  },
  {
    id: "raw",
    title: "The numbers underneath",
    lede: "Every registered Prometheus metric and its current value, unsummarised.",
    accent: "#5eead4",
    panels: ["panel-metrics"],
  },
];

/**
 * Fail loudly if the grouping above and the derived panel list disagree.
 *
 * The panel list is read off the shipped `index.html`; this grouping is not.
 * Without this check a panel added to the dashboard would simply be absent from
 * the page, and nothing would say so.
 */
function assertGroupsCoverEveryPanel(): void {
  const grouped = panelGroups.flatMap((group) => group.panels);
  const derived = siteData.web.dashboard.panels.map((panel) => panel.id);

  const missing = derived.filter((id) => !grouped.includes(id));
  const unknown = grouped.filter((id) => !derived.includes(id));
  const duplicated = grouped.filter((id, i) => grouped.indexOf(id) !== i);

  if (missing.length || unknown.length || duplicated.length) {
    throw new Error(
      "app/dashboard/dashboardData.ts no longer matches the panel list in " +
        "data/effgen.json. " +
        [
          missing.length ? `Not in any group: ${missing.join(", ")}.` : "",
          unknown.length ? `Grouped but not a panel: ${unknown.join(", ")}.` : "",
          duplicated.length ? `In two groups: ${duplicated.join(", ")}.` : "",
        ]
          .filter(Boolean)
          .join(" "),
    );
  }
}

assertGroupsCoverEveryPanel();

/** A panel's own heading, from the shipped `index.html`. */
export function panelTitle(id: string): string {
  return siteData.web.dashboard.panels.find((panel) => panel.id === id)?.title ?? "";
}

/** How the jump row above the panels labels it. */
export function panelNav(id: string): string {
  return siteData.web.dashboard.panels.find((panel) => panel.id === id)?.nav ?? "";
}

/* ── What each data route answers ── */

export interface EndpointNote {
  path: string;
  note: string;
}

// One line per route the page reads. Every one is same-origin; the paths come
// from `data/effgen.json`, the notes from what each one returned when it was
// called for this page.
export const endpointNotes: Record<string, string> = {
  "/dashboard/data.json":
    "The five cards, the SLO burn, the per-model rows, the status and route " +
    "breakdowns, the recent runs and the buffered spans. Polled every five seconds.",
  "/dashboard/spans":
    "Server-sent events, one JSON object per span, pushed as they are recorded.",
  "/dashboard/catalog.json": "Every model the catalog knows, for the catalog panel.",
  "/dashboard/history.json": "Stored runs and saved sessions, for the history panel.",
  "/dashboard/topology.json":
    "Recent team and workflow executions as node-link graphs. `?limit=` bounds it.",
};

export function endpoints(): EndpointNote[] {
  return siteData.web.dashboard.data_endpoints.map((path) => ({
    path,
    note: endpointNotes[path] ?? "",
  }));
}

/* ── The three forms a finished run can be copied as ── */

export interface SnippetForm {
  kind: string;
  label: string;
  language: "bash" | "python";
  accent: string;
  /** Exactly what the playground put on the clipboard. */
  code: string;
  /** What that produced when it was run. */
  output: string;
  outputLabel: string;
}

// Captured from the playground after the run the figure above shows: the Copy
// button was pressed, the clipboard text was read back out of the page, and
// each of the three forms was then run. The outputs below are what they
// printed — none of them is written by hand.
export const snippetForms: SnippetForm[] = [
  {
    kind: "curl",
    label: "curl",
    language: "bash",
    accent: "#00e5ff",
    code: `curl http://127.0.0.1:8000/v1/chat/completions \\
  -H 'Authorization: Bearer $EFFGEN_API_KEY' \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"gemini:gemini-3.1-flash-lite","messages":[{"role":"user","content":"What is 8347 * 219? Use the calculator."}],"temperature":0.7,"max_tokens":512,"tools":[{"type":"function","function":{"name":"calculator"}}]}'`,
    output: `1827993
usage: {'prompt_tokens': 163, 'completion_tokens': 22, 'total_tokens': 185}`,
    outputLabel: "the answer and the usage the server returned",
  },
  {
    kind: "cli",
    label: "CLI",
    language: "bash",
    accent: "#00ff88",
    code: `effgen run 'What is 8347 * 219? Use the calculator.' -m 'gemini:gemini-3.1-flash-lite' -t calculator --temperature 0.7 --max-tokens 512`,
    output: `╭─────────────────────────────── Agent Response ───────────────────────────────╮
│ 1827993                                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
✓ Done in 2.7s · 1 tool · 183 tokens · $0.000073
1 tool step — run with --trace to see the timeline`,
    outputLabel: "what that command printed",
  },
  {
    kind: "python",
    label: "Python",
    language: "python",
    accent: "#a78bfa",
    code: `from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools import get_registry

reg = get_registry()

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    tools=[reg.get_tool_sync("calculator")],
    temperature=0.7,
    max_tokens=512
))
print(agent.run("What is 8347 * 219? Use the calculator."))`,
    output: `1827993`,
    outputLabel: "what that program printed",
  },
];

/* ── What the keyboard layer does ── */

export interface PaletteGroup {
  name: string;
  what: string;
}

// The four groups the palette searches, from `docs/dx/dashboard.md` and
// confirmed by opening it: typing `cost` matched entries in more than one.
export const paletteGroups: PaletteGroup[] = [
  {
    name: "Navigate",
    what: "every panel on this surface, plus the other surface",
  },
  {
    name: "Actions",
    what: "switch theme, refresh, clear or pause the span stream, focus a search box",
  },
  {
    name: "Runs",
    what: "stored runs, matched on task text, model, status or run id — selecting one opens its detail",
  },
  {
    name: "Models",
    what: "the catalog, matched on id, provider, family or capability — selecting one filters the table",
  },
];
