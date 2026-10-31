"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useCallback, useId, useMemo, useRef, useState } from "react";
import { FiCheck, FiChevronDown, FiCopy, FiKey, FiSearch, FiTool, FiX } from "react-icons/fi";
import Container from "./Container";
import { useFocusTrap } from "./useFocusTrap";
import ParamTable from "./ui/ParamTable";
import { highlightCode } from "./syntaxHighlight";
import { siteData } from "./siteData";
import type { ToolInfo } from "@/data/siteData.types";

/* ── The tool gallery ──
 *
 * Every built-in tool, browsable by the categories the registry files them
 * under. Every field on this section comes from `data/effgen.json`, which
 * `scripts/gen_site_data.py` writes out of the installed package: the names,
 * the categories and their counts, the descriptions, the parameter tables and
 * the arguments in each snippet, which are the tool's own worked example where
 * it ships one.
 *
 * The call shape is the same for every one of them and it is the shape the
 * framework accepts: `await tool.execute(**kwargs)`, returning a `ToolResult`
 * whose fields are `success / output / error / execution_time / metadata /
 * timestamp`. It has no `data` field and it is not subscriptable. Every snippet
 * below was generated from this data and run — see the section note.
 *
 * Only the accent colour per category is written here.
 */

/* How much of a preset a tool is.
 *
 * Counting how many presets carry a tool ranks a tool in the 31-tool `general`
 * preset level with one of the two tools `math` is made of, which is backwards:
 * the small preset's tools are the ones that define it. So each preset a tool
 * belongs to contributes 1/(that preset's size), and `retrieval` — the whole of
 * the `rag` preset — comes out top. Everything here is read from
 * `data/effgen.json`; no tool is promoted by hand. */
const presetWeight = new Map<string, number>();
for (const preset of siteData.presets.items) {
  if (preset.tools.length === 0) continue;
  for (const name of preset.tools) {
    presetWeight.set(name, (presetWeight.get(name) ?? 0) + 1 / preset.tools.length);
  }
}

const tools = [...siteData.tools.items].sort((a, b) => {
  const weight = (presetWeight.get(b.name) ?? 0) - (presetWeight.get(a.name) ?? 0);
  if (weight !== 0) return weight;
  // A tool that runs with nothing configured is the better one to meet first.
  const keys = Number(a.requires_api_key) - Number(b.requires_api_key);
  if (keys !== 0) return keys;
  return a.name.localeCompare(b.name);
});

/** Per category, how many the section shows before the reader asks for the rest. */
const PER_CATEGORY = 2;

/** The eight categories the registry files tools under, in size order. */
const CATEGORY_LABELS: Record<string, string> = {
  information_retrieval: "Information retrieval",
  data_processing: "Data processing",
  external_api: "External APIs",
  communication: "Communication",
  code_execution: "Code execution",
  system: "System",
  computation: "Computation",
  file_operations: "File operations",
};

/** Accent per category. Presentation only — no fact lives here. */
const CATEGORY_ACCENTS: Record<string, string> = {
  information_retrieval: "#00e5ff",
  data_processing: "#a78bfa",
  external_api: "#ff6b6b",
  communication: "#ec4899",
  code_execution: "#00ff88",
  system: "#ff9500",
  computation: "#ffd700",
  file_operations: "#10b981",
};

const ALL = "ALL";

const accentOf = (category: string) => CATEGORY_ACCENTS[category] ?? "#00ff88";
const labelOf = (category: string) => CATEGORY_LABELS[category] ?? category.replace(/_/g, " ");

/** How many tools each category holds, straight from the generated counts. */
const categories = Object.entries(siteData.tools.category_counts)
  .sort((a, b) => b[1] - a[1])
  .map(([key, count]) => ({ key, count, label: labelOf(key), accent: accentOf(key) }));

/* What the section opens on: the top `PER_CATEGORY` of every category, in the
 * order the tabs are in. Sixty-six cards of the same shape is a wall, and a
 * plain "first sixteen" would be sixteen cards from three categories. */
const featured = categories.flatMap((category) =>
  tools.filter((tool) => tool.category === category.key).slice(0, PER_CATEGORY),
);

/** A Python literal for one argument value, so the snippet reads as Python. */
function pythonLiteral(value: unknown): string {
  if (value === null) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return `[${value.map(pythonLiteral).join(", ")}]`;
  if (typeof value === "object") {
    const body = Object.entries(value as Record<string, unknown>)
      .map(([k, v]) => `${JSON.stringify(k)}: ${pythonLiteral(v)}`)
      .join(", ");
    return `{${body}}`;
  }
  return JSON.stringify(value);
}

/**
 * The program that calls one tool.
 *
 * Identical for all 66 apart from the name and the arguments, because the
 * framework's tool interface is identical for all 66: look the tool up in the
 * registry, await `execute` with keyword arguments, and read `success` and
 * `output` off the `ToolResult` that comes back.
 */
function snippetFor(tool: ToolInfo): string {
  const args = Object.entries(tool.example_arguments)
    .map(([key, value]) => `${key}=${pythonLiteral(value)}`)
    .join(", ");

  return [
    "import asyncio",
    "",
    "from effgen.tools import get_registry",
    "",
    `tool = get_registry().get_tool_sync("${tool.name}")`,
    `result = asyncio.run(tool.execute(${args}))`,
    "",
    "print(result.success, result.output)",
  ].join("\n");
}

/* ── One tool card ── */

function ToolCard({ tool, onOpen }: { tool: ToolInfo; onOpen: () => void }) {
  const accent = accentOf(tool.category);

  return (
    <motion.button
      type="button"
      variants={{
        hidden: { y: 20, opacity: 0, scale: 0.97 },
        visible: { y: 0, opacity: 1, scale: 1, transition: { duration: 0.35, ease: "easeOut" as const } },
      }}
      whileHover={{ y: -6, scale: 1.02, transition: { type: "spring", stiffness: 400, damping: 25 } }}
      onClick={onOpen}
      className="group relative p-5 rounded-xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden cursor-pointer text-left flex flex-col"
    >
      {/* The card already shows the tool's name and what it does. An
          `aria-label` would replace those words rather than contain them, which
          is what WCAG 2.5.3 asks for on a control with visible text; this adds
          only the category and what pressing it does. */}
      <span className="sr-only">
        {labelOf(tool.category)} — open details
      </span>
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"
        style={{ background: `radial-gradient(circle at 50% 30%, ${accent}12 0%, transparent 70%)` }}
      />

      <div className="flex items-start justify-between gap-2 mb-3 relative z-10">
        <code className="text-sm font-mono font-bold text-gray-900 dark:text-white break-all">
          {tool.name}
        </code>
        {tool.requires_api_key && (
          <span
            className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider flex-shrink-0 text-orange-700 dark:text-orange-300 bg-orange-500/10 border border-orange-500/25"
            title="Needs an API key"
          >
            <FiKey size={9} aria-hidden="true" />
            key
          </span>
        )}
      </div>

      <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed mb-3 line-clamp-3 relative z-10">
        {tool.description}
      </p>

      <div className="flex flex-wrap gap-1 mt-auto relative z-10">
        {tool.params.slice(0, 4).map((param) => (
          <span
            key={param.name}
            className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700"
          >
            {param.name}
          </span>
        ))}
        {tool.params.length > 4 && (
          <span className="px-1.5 py-0.5 text-[9px] font-mono text-gray-600 dark:text-gray-400">
            +{tool.params.length - 4}
          </span>
        )}
      </div>

      <span
        className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
      />
    </motion.button>
  );
}

/* ── Detail dialog ── */

function ToolDetail({ tool, onClose }: { tool: ToolInfo; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);
  const titleId = useId();
  const accent = accentOf(tool.category);
  const snippet = snippetFor(tool);

  // Escape closes, Tab stays inside, focus moves into the panel when it opens
  // and returns to the card that opened it when it closes. `useFocusTrap` is the
  // shared copy — the local one this replaced never moved focus in, so opening
  // the dialog left a keyboard reader on the card behind it.
  useFocusTrap(panelRef, true, onClose);

  const copy = async () => {
    await navigator.clipboard.writeText(snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const related = tools.filter((t) => t.category === tool.category && t.name !== tool.name);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      />

      <motion.div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl"
        style={{ boxShadow: `0 0 50px ${accent}15` }}
        initial={{ scale: 0.94, opacity: 0, y: 24 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.94, opacity: 0, y: 24 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        <div
          className="h-1 rounded-t-2xl"
          style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
        />

        <div className="p-6">
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="absolute top-4 right-4 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors z-10"
          >
            <FiX size={16} />
          </button>

          <div className="mb-4 pr-10">
            <h3 id={titleId} className="text-xl font-black text-gray-900 dark:text-white font-mono break-all">
              {tool.name}
            </h3>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span
                className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider"
                style={{ background: `${accent}15`, color: accent, border: `1px solid ${accent}30` }}
              >
                {labelOf(tool.category)}
              </span>
              {tool.requires_api_key && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider text-orange-700 dark:text-orange-300 bg-orange-500/10 border border-orange-500/25">
                  <FiKey size={10} aria-hidden="true" />
                  needs a key
                </span>
              )}
              {tool.requires_approval && (
                <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider text-orange-700 dark:text-orange-300 bg-orange-500/10 border border-orange-500/25">
                  asks first
                </span>
              )}
              <span className="text-[10px] font-mono text-gray-600 dark:text-gray-400">
                timeout {tool.timeout_seconds}s
              </span>
            </div>
          </div>

          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
            {tool.description}
          </p>

          {/* Call it */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest">
                Call it
              </h4>
              <button
                type="button"
                onClick={copy}
                className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors"
              >
                {copied ? <><FiCheck size={10} className="text-green-400" /> Copied</> : <><FiCopy size={10} /> Copy</>}
              </button>
            </div>
            <div className="p-3 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 overflow-x-auto">
              <pre className="text-xs font-mono leading-relaxed text-gray-700 dark:text-gray-300">
                <code
                  className="syntax-code"
                  dangerouslySetInnerHTML={{ __html: highlightCode(snippet, "python") }}
                />
              </pre>
            </div>
            <p className="mt-2 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
              {tool.example_source === "tool"
                ? "The arguments are the tool's own worked example."
                : "The arguments are placeholders — this tool declares no worked example, so the required parameters are filled from its schema."}{" "}
              An agent calls it the same way, by name, once you pass the tool in{" "}
              <code className="font-mono">AgentConfig(tools=[...])</code>.
            </p>
          </div>

          {/* Parameters */}
          {tool.params.length > 0 && (
            <div className="mb-6">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-2">
                Parameters
              </h4>
              <ParamTable
                nameLabel="Parameter"
                params={tool.params.map((param) => ({
                  name: param.name,
                  type: param.type,
                  required: param.required,
                  default: param.default === null || param.default === undefined
                    ? undefined
                    : String(param.default),
                  description: param.enum
                    ? `${param.description} One of: ${param.enum.join(", ")}.`
                    : param.description,
                }))}
              />
            </div>
          )}

          {related.length > 0 && (
            <div>
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-2">
                Also filed under {labelOf(tool.category)}
              </h4>
              <div className="flex flex-wrap gap-2">
                {related.map((other) => (
                  <span
                    key={other.name}
                    className="px-2.5 py-1 rounded-full text-xs font-mono bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700"
                  >
                    {other.name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ── The section ── */

export default function ToolShowcase() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [activeCategory, setActiveCategory] = useState<string>(ALL);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<ToolInfo | null>(null);
  const [showAll, setShowAll] = useState(false);
  const closeDetail = useCallback(() => setSelected(null), []);
  const searchId = useId();

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return tools.filter((tool) => {
      if (activeCategory !== ALL && tool.category !== activeCategory) return false;
      if (!needle) return true;
      return (
        tool.name.includes(needle) ||
        tool.description.toLowerCase().includes(needle) ||
        tool.tags.some((tag) => tag.toLowerCase().includes(needle)) ||
        tool.params.some((param) => param.name.includes(needle))
      );
    });
  }, [activeCategory, query]);

  // A search or a category is a reader narrowing the list themselves, and that
  // result is shown whole. It is only the unfiltered view that opens on a
  // sample, because sixty-six cards of the same shape is a wall rather than a
  // list. Either way the search and the tabs run over all sixty-six.
  const browsing = activeCategory === ALL && query.trim() === "";
  const collapsed = browsing && !showAll;
  const visible = collapsed ? featured : filtered;

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.02 } },
  };

  return (
    <>
      <section id="tools" className="py-24 bg-white dark:bg-[#030f07] relative overflow-hidden" ref={ref}>
        <div className="absolute inset-0 grid-pattern" />
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

        <Container className="relative z-10">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="text-center mb-10"
          >
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
              <FiTool size={14} />
              Built-in tools
            </span>
            <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
              <span className="gradient-text">{siteData.tools.count} tools</span>, in{" "}
              {categories.length} categories
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              Every one is registered under the name below and called the same way — awaited, with
              keyword arguments, returning a <code className="font-mono">ToolResult</code>. Open one
              for its parameters and the line that runs it, search for the one you need, or show the
              whole set.
            </p>
          </motion.div>

          {/* Search */}
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.15 }}
            className="max-w-md mx-auto mb-6"
          >
            <label htmlFor={searchId} className="sr-only">
              Search the tools by name, description, tag or parameter
            </label>
            <div className="relative">
              <FiSearch
                size={16}
                aria-hidden="true"
                className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-400"
              />
              <input
                id={searchId}
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={`Search ${siteData.tools.count} tools — pdf, weather, sandbox, slack…`}
                className="w-full pl-11 pr-4 py-3 rounded-full bg-gray-50 dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 text-sm text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
              />
            </div>
          </motion.div>

          {/* Category tabs */}
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-wrap justify-center gap-2 mb-8"
            role="group"
            aria-label="Filter tools by category"
          >
            {[{ key: ALL, label: "All", count: siteData.tools.count, accent: "#00ff88" }, ...categories].map(
              (category) => {
                const isActive = activeCategory === category.key;
                return (
                  <button
                    key={category.key}
                    type="button"
                    onClick={() => setActiveCategory(category.key)}
                    aria-pressed={isActive}
                    className={`px-4 py-2 rounded-full text-sm font-semibold transition-all ${
                      isActive
                        ? "text-black"
                        : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/50"
                    }`}
                    style={
                      isActive
                        ? {
                            background: `linear-gradient(135deg, ${category.accent}, ${category.accent}cc)`,
                            boxShadow: `0 0 20px ${category.accent}30`,
                          }
                        : {}
                    }
                  >
                    {category.label}
                    <span className={`ml-1.5 text-[10px] ${isActive ? "text-black/60" : "text-gray-400 dark:text-gray-400"}`}>
                      {category.count}
                    </span>
                  </button>
                );
              },
            )}
          </motion.div>

          {/* Result count, announced */}
          <p className="text-center text-sm text-gray-600 dark:text-gray-400 mb-6" aria-live="polite">
            {collapsed
              ? `The ${PER_CATEGORY} each category leans on most — ${visible.length} of ${siteData.tools.count}`
              : filtered.length === siteData.tools.count
                ? `All ${siteData.tools.count} tools`
                : `${filtered.length} of ${siteData.tools.count} tools`}
          </p>

          {/* Cards */}
          {filtered.length > 0 ? (
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
            >
              {visible.map((tool) => (
                <ToolCard key={tool.name} tool={tool} onOpen={() => setSelected(tool)} />
              ))}
            </motion.div>
          ) : (
            <p className="text-center text-gray-600 dark:text-gray-400 py-12">
              Nothing matches <span className="font-mono">{query}</span>. Try a tool name, a
              parameter or a word from a description.
            </p>
          )}

          {browsing && (
            <div className="text-center mt-8">
              <button
                type="button"
                onClick={() => setShowAll((open) => !open)}
                aria-expanded={showAll}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/60 text-sm font-semibold text-gray-700 dark:text-gray-300 hover:border-green-500/40 transition-colors"
              >
                {showAll
                  ? `Show ${PER_CATEGORY} per category`
                  : `Show all ${siteData.tools.count} tools`}
                <FiChevronDown
                  size={14}
                  aria-hidden="true"
                  className="transition-transform"
                  style={{ transform: showAll ? "rotate(180deg)" : "none" }}
                />
              </button>
            </div>
          )}

          {/* What a call returns */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-12 max-w-3xl mx-auto rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-6"
          >
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-2">
              What comes back
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
              A <code className="font-mono">ToolResult</code>, whatever happened. Its fields are{" "}
              <code className="font-mono">success</code>, <code className="font-mono">output</code>,{" "}
              <code className="font-mono">error</code>,{" "}
              <code className="font-mono">execution_time</code>,{" "}
              <code className="font-mono">metadata</code> and{" "}
              <code className="font-mono">timestamp</code>. It is not a dictionary and it has no{" "}
              <code className="font-mono">data</code> field, so read{" "}
              <code className="font-mono">result.output</code> after checking{" "}
              <code className="font-mono">result.success</code>. A tool that could not do its job
              says why in <code className="font-mono">result.error</code> — a missing key names the
              environment variable, a missing file names the path.
            </p>
          </motion.div>
        </Container>
      </section>

      <AnimatePresence>
        {selected && <ToolDetail tool={selected} onClose={closeDetail} />}
      </AnimatePresence>
    </>
  );
}
