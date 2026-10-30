// The shape of `data/effgen.json`, which `scripts/gen_site_data.py` writes from
// the installed framework. Both codebases import this file, so a field can only
// mean one thing across the site.
//
// Regenerate the data — never edit it — with:
//
//   export PATH=/path/to/your/effgen/env/bin:$PATH   # or activate it however you do
//   python scripts/gen_site_data.py
//
// `python scripts/gen_site_data.py --check` exits non-zero when the checked-in
// file no longer matches the installed framework.

export interface ParamInfo {
  name: string;
  /** `string`, `integer`, `number`, `boolean`, `array` or `object`. */
  type: string;
  required: boolean;
  default: string | number | boolean | null;
  /** The allowed values, when the parameter is constrained to a set. */
  enum: string[] | null;
  description: string;
}

export interface ToolInfo {
  name: string;
  /** The Python class, e.g. `ArXivTool` — what a snippet instantiates. */
  class_name: string;
  /** The module it is imported from, e.g. `effgen.tools.builtin.arxiv`. */
  module: string;
  category: string;
  description: string;
  tags: string[];
  requires_api_key: boolean;
  requires_auth: boolean;
  /** Whether a call has to be confirmed before it runs. */
  requires_approval: boolean;
  cost_estimate: string | null;
  timeout_seconds: number;
  /** How many parameters the tool declares. */
  parameters: number;
  params: ParamInfo[];
  /** The arguments the gallery's snippet passes. */
  example_arguments: Record<string, unknown>;
  /**
   * `tool` when those arguments are the tool's own worked example, `schema`
   * when they are placeholders filled from its required parameters.
   */
  example_source: "tool" | "schema";
}

export interface PresetInfo {
  name: string;
  description: string;
  tools: string[];
  tool_count: number;
  /**
   * The size of the tool schemas this preset sends on every request, which is
   * what `effgen presets` prints as `~N tok/call`. It decides whether a preset
   * fits a small-context or rate-limited model.
   */
  approx_tokens_per_call: number;
  temperature: number;
  max_iterations: number;
  enable_memory: boolean;
  enable_sub_agents: boolean;
  tags: string[];
}

export interface PromptInfo {
  name: string;
  domain: string;
  /** The prompting technique the template uses: `zero_shot`, `few_shot`, `cot`, … */
  variant: string;
  description: string;
  tags: string[];
  /** The inputs `render()` validates against the template's JSON schema. */
  variables: PromptVariable[];
  /** The keys of the worked example the template ships, for `effgen prompts eval`. */
  fixture_keys: string[];
  /** How the output is checked: `json`, `regex`, `function`, or `none`. */
  check: string;
}

export interface PromptVariable {
  name: string;
  type: string;
  required: boolean;
  description: string;
  enum: string[] | null;
}

export interface ProviderInfo {
  name: string;
  /**
   * The environment variables this adapter reads a credential (or, for
   * `openai_compatible`, an endpoint) from. A fact about effGen, not about the
   * machine the site was built on.
   */
  env_keys: string[];
  /** How many models the bundled catalog carries. `openai_compatible` carries none. */
  models: number;
  /** The model this adapter reaches for when a call names none. */
  default: string | null;
  /** When the catalog was last checked against the provider's live API. */
  verified_on: string | null;
  /** `live-verified` or `bundled-catalog`. */
  source: string | null;
  supports_tools: number;
  supports_vision: number;
  supports_audio: number;
  free_tier: number;
  priced: number;
  max_context: number | null;
}

export interface CommandInfo {
  name: string;
  summary: string;
}

export interface OptionInfo {
  /** The flag exactly as `--help` spells it, e.g. `-w DIR, --workspace DIR`. */
  name: string;
  /** Its help text, verbatim. Never a paraphrase. */
  description: string;
}

export interface SlashCommandInfo {
  /** The command as it is typed, including the leading slash. */
  name: string;
  /** Its one-line description, exactly as `/help` prints it. */
  summary: string;
}

export interface SiteData {
  derived_at: string;
  version: string;
  python_versions: string[];
  python_requires: string;
  public_names: number;
  tools: {
    count: number;
    categories: Record<string, string[]>;
    category_counts: Record<string, number>;
    items: ToolInfo[];
  };
  presets: { count: number; items: PresetInfo[] };
  prompts: {
    templates: number;
    library: number;
    domains: string[];
    domain_counts: Record<string, number>;
    items: PromptInfo[];
    /** Every registered template name, including the ones outside the domain library. */
    template_names: string[];
  };
  models: {
    adapters: string[];
    adapter_count: number;
    /** The adapters that ship a bundled catalog — every one except `openai_compatible`. */
    with_catalog: string[];
    with_catalog_count: number;
    models: number;
    models_by_provider: Record<string, number>;
    /** One row per adapter, for the provider table. */
    providers: ProviderInfo[];
    /** How many catalogued models carry each capability, across every provider. */
    capability_totals: {
      supports_tools: number;
      supports_vision: number;
      supports_audio: number;
      free_tier: number;
      priced: number;
    };
    local_engines: string[];
  };
  /** What a coding or chat session offers, read off the modules behind it. */
  code: {
    /** Every option `effgen code --help` declares, in its order. */
    options: OptionInfo[];
    slash_commands: SlashCommandInfo[];
    slash_command_count: number;
    chat_slash_commands: SlashCommandInfo[];
    chat_slash_command_count: number;
    /** How many applied edits the per-workspace undo journal keeps. */
    undo_journal_entries: number;
    /** The git sub-commands a coding session may run. */
    git_allowed: string[];
    /** The ones it refuses, including through the shell. */
    git_refused: string[];
    /** Flags it refuses wherever they appear. */
    git_refused_flags: string[];
  };
  cli: {
    commands: CommandInfo[];
    command_count: number;
    subcommands: Record<string, string[]>;
    subcommand_count: number;
    /** The options `effgen --help` declares before a command is named. */
    global_options: OptionInfo[];
    /** Per-command flag tables, keyed by the command line that prints them. */
    command_options: Record<string, OptionInfo[]>;
    /**
     * The environment variables `effgen serve --help` documents in its epilog.
     * They configure auth, rate limiting, CORS and the model pool, so they are
     * part of the command's surface even though they are not flags.
     */
    serve_env: OptionInfo[];
    themes: string[];
    completion_shells: string[];
  };
  /** The dashboard and the playground, read off the files the package ships. */
  web: WebSurfaces;
  /** The example programs in the framework repository. */
  examples: ExampleCatalogue;
  /** The library surface, introspected from the classes themselves. */
  api: ApiSurface;
  /** The operational surface, read from the server and observability modules. */
  production: ProductionSurface;
}

/**
 * The `examples/` directory of the framework repository, as it stands.
 *
 * The scripts are not part of the wheel — the build excludes them — so this is
 * read from a checkout, through the same discovery function `effgen examples`
 * uses to find them.
 */
export interface ExampleCatalogue {
  count: number;
  groups: { id: string; what: string; count: number }[];
  /** How many scripts parse their own command-line arguments. */
  parses_arguments: number;
  items: ExampleInfo[];
}

export interface ExampleInfo {
  /** Path under `examples/`, without the extension — what `effgen examples run` takes. */
  name: string;
  /** The directory it sits in. */
  group: string;
  /** Path in the framework repository. */
  file: string;
  /** The first line of the script's docstring. */
  summary: string;
  /**
   * True when the script builds its own `argparse` parser. Those cannot be
   * started with `effgen examples run`, which leaves `sys.argv` alone.
   */
  parses_arguments: boolean;
  /** True when the script needs a provider key to do anything. */
  needs_key: boolean;
}

/** One field of a dataclass, as the class declares it. */
export interface FieldInfo {
  name: string;
  /** The annotation, as written in the source. */
  type: string;
  /** The default as a Python literal, or null when there is none written. */
  default: string | null;
  /** True when the caller has to supply it. */
  required: boolean;
}

export interface HookInfo {
  name: string;
  /** The hook's signature with `self` removed. */
  signature: string;
  /** Its docstring summary, one line. */
  what: string;
}

export interface NamedSummary {
  name: string;
  what: string;
}

export interface CompactionInfo extends NamedSummary {
  parameters: { name: string; default: string | null }[];
}

export interface ApiSurface {
  /** Every `AgentConfig` field, with its annotation and written default. */
  agent_config: FieldInfo[];
  /** Every `AgentResponse` field. */
  agent_response: FieldInfo[];
  /** Every `ToolCall` field. */
  tool_call: FieldInfo[];
  /** The reading surface `ToolCallList` adds on top of `list`. */
  tool_call_list: string[];
  /** Every `ToolResult` field. There is no `data` field and it is not indexable. */
  tool_result: FieldInfo[];
  /** The six middleware points, in the order they fire. */
  middleware_hooks: HookInfo[];
  /** The middleware the framework ships. */
  middleware_shipped: NamedSummary[];
  /** The four compaction strategies, with their constructor defaults. */
  compaction: CompactionInfo[];
  /** Where a workflow checkpoint can be kept. */
  checkpoint_stores: NamedSummary[];
  /** The orchestration patterns a team can be built with. */
  orchestration_patterns: string[];
}

export interface RoleInfo {
  name: string;
  /** `all` or `none` — whether the role may call tools at all. */
  tools: string;
  /** `all`, or the model ids the role is restricted to. */
  models: string | string[];
  /** USD per UTC day. `0` is unlimited. */
  max_cost_per_day: number;
}

export interface MetricInfo {
  /** The Prometheus name, as `/metrics` prints it. */
  name: string;
  /** `histogram`, `counter` or `gauge`. */
  kind: string;
  /** The instrument's own help text. */
  help: string;
  /** Declared label names, where the instrument declares them. */
  labels: string[];
}

export interface ErrorClassInfo {
  /** The exception class name. */
  name: string;
  /** What the classifier calls it. */
  category: string;
  /** Whether retrying that call could plausibly succeed. */
  should_retry: boolean;
}

export interface ProductionSurface {
  /** The roles the server ships, as the policy registry holds them. */
  rbac_roles: RoleInfo[];
  /** Every instrument `/metrics` exposes. */
  metrics: MetricInfo[];
  /** The guardrail presets and what each chain contains. */
  guardrail_presets: { name: string; guardrails: string[] }[];
  /** Where in a run a guardrail can sit. */
  guardrail_positions: string[];
  /** Each typed error, classified by the real classifier. */
  error_classes: ErrorClassInfo[];
  /** Every category the remediation table carries a next step for. */
  remediation_categories: string[];
  /** The endpoints that answer without a credential. */
  public_endpoints: string[];
}

export interface PanelInfo {
  /** The element id the jump row and the command palette link to. */
  id: string;
  /** The panel's own heading. */
  title: string;
  /** How the jump row above the panels labels it. */
  nav: string;
}

export interface StaticFileInfo {
  /** Path inside the installed `effgen` package. */
  file: string;
  bytes: number;
}

export interface WebSurfaces {
  dashboard: {
    panels: PanelInfo[];
    summary_cards: string[];
    /** The routes the page polls. Every one is same-origin. */
    data_endpoints: string[];
  };
  playground: {
    /** `curl`, `cli`, `python` — the forms a finished run can be copied as. */
    snippet_kinds: string[];
    modes: string[];
  };
  static_files: StaticFileInfo[];
  static_bytes: number;
  /**
   * References to a host that is not this machine, found in the shipped static
   * files by the same expression the framework's own test uses. It is 0, which
   * is what lets the page say the surfaces fetch nothing at view time.
   */
  external_references: number;
  external_reference_list: string[];
  /**
   * Loopback addresses in the same files. The one that exists is the example
   * endpoint printed inside the playground's copy-as-curl snippet — text the
   * page shows, not something the browser requests.
   */
  loopback_references: string[];
}
