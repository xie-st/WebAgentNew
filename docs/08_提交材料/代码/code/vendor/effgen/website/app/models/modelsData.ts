// The grouping and the prose `/models` renders.
//
// The adapter list, the per-provider model counts, capability counts, context
// windows, default models, catalog verification dates and environment variable
// names all come from `data/effgen.json`, which `scripts/gen_site_data.py`
// reads off the installed package. The `effgen models` flag tables come from the
// real `--help`. Nothing here restates any of them.
//
// What is written here is what the package does not state: which servers speak
// the OpenAI protocol, and one line per adapter saying what its catalog is for.
// `assertNotesCoverEveryAdapter` below fails the build if that list ever stops
// matching the registered adapters.

import { siteData } from "@/components/siteData";

const FRAMEWORK_DOCS = "https://github.com/ctrl-gaurav/effGen/blob/main";

export const DOCS_COMPATIBLE_URL = `${FRAMEWORK_DOCS}/docs/models/openai-compatible.md`;
export const DOCS_REGISTRY_URL = `${FRAMEWORK_DOCS}/docs/models/registry.md`;
export const DOCS_ROUTER_URL = `${FRAMEWORK_DOCS}/docs/models/router.md`;

/* ── Servers that speak the protocol ── */

export interface CompatibleServer {
  name: string;
  what: string;
}

// The eight named in `docs/models/openai-compatible.md`, plus the gateway case
// it also covers. Each one exposes the OpenAI chat-completions API, which is
// the only thing that matters here.
export const compatibleServers: CompatibleServer[] = [
  { name: "vLLM", what: "continuous batching on your own GPUs" },
  { name: "SGLang", what: "structured generation and prefix caching" },
  { name: "TGI", what: "Hugging Face's inference server" },
  { name: "llama.cpp", what: "its bundled server, on CPU or a laptop GPU" },
  { name: "Ollama", what: "the local runner, at /v1" },
  { name: "LM Studio", what: "a desktop server on a workstation" },
  { name: "LiteLLM", what: "a proxy in front of several providers" },
  { name: "A gateway", what: "your company's, with its own key and quota" },
];

/** The provider spellings the adapter answers to, from the framework docs. */
export const providerAliases = [
  "openai_compatible",
  "openai-compatible",
  "openai_compat",
  "compatible",
  "server",
  "vllm_server",
  "local_server",
];

/* ── One line per adapter ── */

export interface AdapterNote {
  /** Must match a registered adapter name. */
  name: string;
  label: string;
  accent: string;
  /** What its bundled catalog is, in one clause. Facts only. */
  note: string;
}

// Every clause below is a statement about the bundled catalog, and every number
// in it is read from `data/effgen.json` at render time rather than written here.
export const adapterNotes: AdapterNote[] = [
  {
    name: "openai",
    label: "OpenAI",
    accent: "#00ff88",
    note: "Every model in the catalog calls tools, and most of them take images.",
  },
  {
    name: "anthropic",
    label: "Anthropic",
    accent: "#ff9500",
    note: "Every model in the catalog calls tools and takes images — no exceptions either way.",
  },
  {
    name: "gemini",
    label: "Gemini",
    accent: "#00e5ff",
    note: "The only bundled catalog carrying audio input, and the one with the most free-tier models.",
  },
  {
    name: "groq",
    label: "Groq",
    accent: "#ff6b6b",
    note: "Small open models served fast, on a short catalog; about half of them call tools.",
  },
  {
    name: "cerebras",
    label: "Cerebras",
    accent: "#a78bfa",
    note: "Two models, both free-tier and rate-limited, at a 64K context window.",
  },
  {
    name: "together",
    label: "Together",
    accent: "#ffd700",
    note: "The largest bundled catalog, and the widest range of open weights.",
  },
  {
    name: "fireworks",
    label: "Fireworks",
    accent: "#5eead4",
    note: "Open models with a serverless tier, several of them vision-capable.",
  },
  {
    name: "hf",
    label: "Hugging Face",
    accent: "#f472b6",
    note: "Hosted inference over the Hub, with the second-largest catalog here.",
  },
  {
    name: "replicate",
    label: "Replicate",
    accent: "#60a5fa",
    note: "Hosted open models, many of them billed by hardware time rather than by token.",
  },
  {
    name: "openai_compatible",
    label: "Any OpenAI-protocol server",
    accent: "#00ff88",
    note: "No bundled catalog at all: it serves whatever the endpoint you point it at serves.",
  },
];

/**
 * Fail loudly if the notes above and the registered adapters disagree.
 *
 * The adapter list is generated; these notes are not. Without this check an
 * adapter added to effGen would simply be absent from the page.
 */
function assertNotesCoverEveryAdapter(): void {
  const noted = adapterNotes.map((adapter) => adapter.name);
  const derived = siteData.models.adapters;

  const missing = derived.filter((name) => !noted.includes(name));
  const unknown = noted.filter((name) => !derived.includes(name));

  if (missing.length || unknown.length) {
    throw new Error(
      "app/models/modelsData.ts no longer matches the adapter list in " +
        "data/effgen.json. " +
        [
          missing.length ? `Not described: ${missing.join(", ")}.` : "",
          unknown.length ? `Described but not registered: ${unknown.join(", ")}.` : "",
        ]
          .filter(Boolean)
          .join(" "),
    );
  }
}

assertNotesCoverEveryAdapter();

/** The generated row for one adapter. */
export function providerRow(name: string) {
  const row = siteData.models.providers.find((provider) => provider.name === name);
  if (!row) {
    throw new Error(`No provider row for "${name}" in data/effgen.json.`);
  }
  return row;
}

/* ── The local engines ── */

export interface EngineNote {
  name: string;
  what: string;
}

// The four local engines, from `data/effgen.json`, with what each one runs.
// Sources: `docs/architecture/overview.md`'s engine table.
export const engineNotes: Record<string, string> = {
  transformers: "The default. Runs the weights in-process on a local GPU, or on the CPU.",
  vllm: "Higher throughput on the same weights, for a machine serving many requests.",
  gguf: "Quantised GGUF weights through llama.cpp — a laptop, or a box with no GPU.",
  mlx: "Apple Silicon, through MLX. A vision-language variant runs multimodal models.",
};

export function engines(): EngineNote[] {
  return siteData.models.local_engines.map((name) => ({
    name,
    what: engineNotes[name] ?? "",
  }));
}

/* ── Where the endpoint comes from ── */

export interface Source {
  order: string;
  what: string;
}

// The resolution order in `docs/models/openai-compatible.md`, confirmed against
// the adapter: `resolve_base_url()` consults them in this order.
export const baseUrlSources: Source[] = [
  {
    order: "1",
    what: "base_url= passed to load_model(), AgentConfig or the adapter",
  },
  { order: "2", what: "EFFGEN_BASE_URL" },
  { order: "3", what: "OPENAI_BASE_URL" },
  { order: "4", what: "OPENAI_API_BASE" },
];
