"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useCallback, useId, useRef, useState } from "react";
import { FiX, FiCopy, FiCheck, FiZap } from "react-icons/fi";
import Container from "./Container";
import { useFocusTrap } from "./useFocusTrap";
import CodeSample from "./ui/CodeSample";
import { siteData } from "./siteData";
import type { PresetInfo } from "@/data/siteData.types";
import { accentTextStyle } from "./accentText";

/* ── The presets ──
 *
 * Every figure on this section — the nine names, each preset's tool list, its
 * tool count, its per-call tool-schema cost, its temperature, its iteration cap
 * and its description — is read from `data/effgen.json`, which
 * `scripts/gen_site_data.py` writes out of the installed package. Nothing here
 * is typed by hand, so a preset that gains a tool changes this section by
 * regenerating one file.
 *
 * The only things written here are presentational: the accent colour and the
 * glyph each card carries.
 */

const presets = siteData.presets.items;

/** Accent and glyph per preset. Presentation only — no fact lives here. */
const LOOK: Record<string, { accent: string; glyph: string }> = {
  math: { accent: "#00ff88", glyph: "\u{1F9EE}" },
  research: { accent: "#00e5ff", glyph: "\u{1F52C}" },
  coding: { accent: "#a78bfa", glyph: "\u{1F4BB}" },
  general: { accent: "#ffd700", glyph: "\u{1F680}" },
  rag: { accent: "#00c896", glyph: "\u{1F4D6}" },
  minimal: { accent: "#ff9500", glyph: "⚡" },
  multimodal: { accent: "#f472b6", glyph: "\u{1F5BC}️" },
  notify: { accent: "#ec4899", glyph: "\u{1F4E2}" },
  media: { accent: "#f59e0b", glyph: "\u{1F39E}️" },
};

const FALLBACK = { accent: "#00ff88", glyph: "⚙️" };

const look = (name: string) => LOOK[name] ?? FALLBACK;

/**
 * The line that creates this preset, for the card's code strip.
 *
 * `rag` is the one that differs: `create_agent("rag", model)` raises, because
 * the preset refuses to build a retrieval agent over zero documents. It names
 * the argument it needs, and so does this.
 */
function creationLineFlat(preset: PresetInfo): string {
  return preset.name === "rag"
    ? 'create_agent("rag", model, knowledge_base=…)'
    : `create_agent("${preset.name}", model)`;
}

/** The whole program the modal offers to copy, and the one this page ran. */
function fullExample(preset: PresetInfo): string {
  const kb = preset.name === "rag" ? ', knowledge_base="./docs/"' : "";
  return [
    "from effgen.presets import create_agent",
    "",
    `agent = create_agent("${preset.name}", "gemini:gemini-3.1-flash-lite"${kb})`,
    'print(agent.run("What is 24344 * 334?").text)',
  ].join("\n");
}

/** `~3374` reads better as `3.4k` on a card. */
function shortTokens(tokens: number): string {
  if (tokens === 0) return "0";
  if (tokens < 1000) return String(tokens);
  return `${(tokens / 1000).toFixed(1)}k`;
}

/* ── Temperature Indicator ── */

function TempIndicator({ temp }: { temp: number }) {
  const tempColor = temp <= 0.3 ? "#00e5ff" : temp <= 0.5 ? "#00ff88" : temp <= 0.7 ? "#ffd700" : "#ff6b6b";
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[9px] text-gray-600 dark:text-gray-400 font-mono">temp</span>
      <div className="w-12 h-1.5 rounded-full bg-gray-200 dark:bg-gray-800 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, #00e5ff, ${tempColor})`, width: `${temp * 100}%` }}
          initial={{ width: 0 }}
          animate={{ width: `${temp * 100}%` }}
          transition={{ duration: 0.8, delay: 0.3 }}
        />
      </div>
      <span className="text-[9px] font-mono" style={accentTextStyle(tempColor)}>{temp}</span>
    </div>
  );
}

/* ── Detail dialog ── */

function PresetDetail({ preset, onClose }: { preset: PresetInfo; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);
  const titleId = useId();
  const { accent, glyph } = look(preset.name);
  const example = fullExample(preset);

  // Escape closes, Tab stays inside, focus moves into the panel when it opens
  // and returns to the card that opened it when it closes. `useFocusTrap` is the
  // shared copy — the local one this replaced never moved focus in, so opening
  // the dialog left a keyboard reader on the card behind it.
  useFocusTrap(panelRef, true, onClose);

  const copy = async () => {
    await navigator.clipboard.writeText(example);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="absolute inset-0 bg-black/80 backdrop-blur-md"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        onClick={onClose}
      />

      <motion.div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl"
        style={{ boxShadow: `0 0 60px ${accent}20` }}
        initial={{ scale: 0.96, opacity: 0, y: 16 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.97, opacity: 0, y: 10 }}
        transition={{ duration: 0.22, ease: "easeOut" }}
      >
        <div
          className="h-1 rounded-t-2xl"
          style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
        />

        <div className="p-8">
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="absolute top-5 right-5 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors z-10"
          >
            <FiX size={18} />
          </button>

          <div className="flex items-center gap-4 mb-5">
            <div
              className="w-14 h-14 rounded-xl flex items-center justify-center text-2xl"
              style={{ background: `${accent}15`, border: `1px solid ${accent}30` }}
              aria-hidden="true"
            >
              {glyph}
            </div>
            <div>
              <h3 id={titleId} className="text-2xl font-black text-gray-900 dark:text-white">
                {preset.name}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm font-mono">
                effgen run --preset {preset.name} &quot;your task&quot;
              </p>
            </div>
          </div>

          {/* What it costs to use, from the package */}
          <div className="flex flex-wrap gap-3 mb-5">
            {[
              { label: "Tools", value: String(preset.tool_count) },
              { label: "Tok / call", value: shortTokens(preset.approx_tokens_per_call) },
              { label: "Temp", value: String(preset.temperature) },
              { label: "Max itr", value: String(preset.max_iterations) },
            ].map((stat) => (
              <div
                key={stat.label}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800"
              >
                <span className="text-[10px] text-gray-600 dark:text-gray-400 uppercase font-bold">{stat.label}</span>
                <span className="text-sm font-mono font-bold" style={accentTextStyle(accent)}>{stat.value}</span>
              </div>
            ))}
          </div>

          <p className="text-gray-600 dark:text-gray-400 mb-6 text-sm leading-relaxed">
            {preset.description}
          </p>

          {preset.tool_count > 0 ? (
            <div className="mb-6">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-3">
                The {preset.tool_count} tools it wires in
              </h4>
              <div className="flex flex-wrap gap-2">
                {preset.tools.map((toolName) => (
                  <span
                    key={toolName}
                    className="px-2.5 py-1 rounded-full text-[11px] font-mono font-semibold text-black"
                    style={{ background: accent }}
                  >
                    {toolName}
                  </span>
                ))}
              </div>
              <p className="mt-3 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                Those schemas are sent on every request, which is the{" "}
                <span className="font-mono">~{preset.approx_tokens_per_call.toLocaleString()}</span>{" "}
                tokens above. A tool-heavy preset costs more per call and can overrun a
                small-context or rate-limited model.
              </p>
            </div>
          ) : (
            <div className="mb-6">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest mb-3">
                Tools
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                None. The model answers directly, so nothing is added to the request and the
                iteration cap is 1.
              </p>
            </div>
          )}

          <div>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-xs font-bold text-gray-600 dark:text-gray-400 uppercase tracking-widest">
                Create it
              </h4>
              <button
                type="button"
                onClick={copy}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors border border-gray-200 dark:border-gray-700"
              >
                {copied ? <><FiCheck className="text-green-400" size={12} /> Copied</> : <><FiCopy size={12} /> Copy</>}
              </button>
            </div>
            <pre className="p-4 rounded-xl bg-gray-50 dark:bg-[#0a1a0f] border border-gray-300 dark:border-gray-800 overflow-x-auto text-xs font-mono text-gray-700 dark:text-gray-300 leading-relaxed">
              {example}
            </pre>
            <p className="mt-3 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
              The second argument is any model — a provider id like{" "}
              <code className="font-mono">gemini:gemini-3.1-flash-lite</code>, a local engine like{" "}
              <code className="font-mono">vllm:Qwen/Qwen2.5-7B-Instruct</code>, or a loaded model
              object.
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

export default function PresetShowcase() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [selected, setSelected] = useState<PresetInfo | null>(null);
  const closeDetail = useCallback(() => setSelected(null), []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.5, ease: "easeOut" as const } },
  };

  return (
    <>
      <section id="presets" className="py-24 bg-gray-50 dark:bg-[#030f07] relative overflow-hidden" ref={ref}>
        <div className="absolute inset-0 grid-pattern" />
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="text-center mb-14"
          >
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
              <FiZap size={14} />
              Agent presets
            </span>
            <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
              <span className="gradient-text">{siteData.presets.count} presets</span>, one line each
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              A preset is a tool set, a system prompt, a temperature and an iteration cap under one
              name. Each card carries what that preset costs you on every request: the tools it wires
              in, and the tokens their schemas take up.
            </p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
            className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-3 xl:grid-cols-3 gap-3 sm:gap-4"
          >
            {presets.map((preset) => {
              const { accent, glyph } = look(preset.name);
              return (
                <motion.button
                  key={preset.name}
                  type="button"
                  variants={itemVariants}
                  whileHover={{ y: -8, scale: 1.03 }}
                  transition={{ type: "spring", stiffness: 400, damping: 25 }}
                  onClick={() => setSelected(preset)}
                  aria-label={`${preset.name} preset — ${preset.tool_count} tools`}
                  className="group relative p-4 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none backdrop-blur-sm overflow-hidden cursor-pointer flex flex-col h-full text-left"
                >
                  {/* Hover glow */}
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                    style={{ background: `radial-gradient(circle at 30% 30%, ${accent}10 0%, transparent 70%)` }}
                  />

                  <div className="relative z-10 text-center">
                    <div
                      className="w-11 h-11 rounded-xl flex items-center justify-center mb-2 text-xl overflow-hidden mx-auto"
                      style={{ background: `${accent}15`, border: `1px solid ${accent}30` }}
                      aria-hidden="true"
                    >
                      {glyph}
                    </div>

                    <h3 className="text-sm font-bold mb-1 text-gray-900 dark:text-white">{preset.name}</h3>

                    <div className="flex items-center justify-center gap-1.5 mb-1.5 text-[10px] font-mono">
                      <span style={accentTextStyle(accent)}>
                        {preset.tool_count === 0 ? "no tools" : `${preset.tool_count} tools`}
                      </span>
                      {preset.approx_tokens_per_call > 0 && (
                        <>
                          <span className="text-gray-400 dark:text-gray-600">·</span>
                          <span className="text-gray-600 dark:text-gray-400">
                            ~{shortTokens(preset.approx_tokens_per_call)} tok/call
                          </span>
                        </>
                      )}
                    </div>

                    <p className="text-[11px] text-gray-600 dark:text-gray-400 leading-snug line-clamp-2">
                      {preset.description}
                    </p>
                  </div>

                  <div className="mt-auto pt-3 relative z-10">
                    <div className="flex items-center justify-between">
                      <TempIndicator temp={preset.temperature} />
                      <span className="text-[9px] font-mono text-gray-600 dark:text-gray-400">
                        <span className="text-gray-400 dark:text-gray-400">itr:</span> {preset.max_iterations}
                      </span>
                    </div>

                    <div className="px-2 py-1.5 mt-2 rounded-md bg-gray-50 dark:bg-[#0a1a0f] border border-gray-200 dark:border-gray-800 overflow-hidden">
                      <code
                        className="text-[10px] font-mono whitespace-nowrap block overflow-hidden text-ellipsis"
                        style={accentTextStyle(accent)}
                      >
                        {creationLineFlat(preset)}
                      </code>
                    </div>
                  </div>

                  <div
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity z-10"
                    style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
                  />
                </motion.button>
              );
            })}
          </motion.div>

          {/* One preset, end to end, with what the run printed. */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-12 max-w-3xl mx-auto"
          >
            <CodeSample
              code={`from effgen.presets import create_agent

agent = create_agent("math", "gemini:gemini-3.1-flash-lite")
response = agent.run("What is 24344 * 334?")

print(response.text)
print("tool calls:", response.tool_calls.total)`}
              language="python"
              output={`8130896
tool calls: 1`}
            />
            <p className="mt-4 text-sm text-center text-gray-600 dark:text-gray-400">
              The answer came back from the calculator the <code className="font-mono">math</code>{" "}
              preset wired in, not from the model doing arithmetic in its head — which is what the
              call count is there to tell you.
            </p>
          </motion.div>
        </Container>
      </section>

      <AnimatePresence>
        {selected && <PresetDetail preset={selected} onClose={closeDetail} />}
      </AnimatePresence>
    </>
  );
}
