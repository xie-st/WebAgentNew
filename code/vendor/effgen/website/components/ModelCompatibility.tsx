"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useCallback, useState } from "react";
import { FiChevronDown, FiCpu, FiHardDrive, FiServer } from "react-icons/fi";
import Container from "./Container";
import CodeSample from "./ui/CodeSample";
import DetailDialog from "./ui/DetailDialog";
import RouteLink from "./ui/RouteLink";
import { siteData } from "./siteData";
import { accentTextStyle } from "./accentText";

/* ── Models and providers ──
 *
 * The adapter list, the model counts, each adapter's default model, the date
 * its catalog was last checked against the provider's live API and what those
 * models can do all come from `data/effgen.json`, which
 * `scripts/gen_site_data.py` reads out of the installed package's bundled
 * catalog. Nothing on this section is typed in.
 *
 * Two numbers are true at once and the section says which it means: ten adapters
 * are registered, and nine of them carry a bundled catalog. The tenth,
 * `openai_compatible`, carries none, because it serves whatever the endpoint you
 * point it at serves.
 *
 * What is deliberately NOT shown: whether a provider is reachable from any
 * particular machine. That is a fact about a machine's keys, not about effGen.
 */

const models = siteData.models;

/** Presentation only. */
const ACCENT = {
  adapters: "#00ff88",
  catalog: "#00e5ff",
  models: "#a78bfa",
  local: "#ffd700",
};

const num = (value: number) => value.toLocaleString("en-US");

function contextLabel(tokens: number | null): string {
  if (!tokens) return "—";
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(tokens % 1_000_000 === 0 ? 0 : 1)}M`;
  return `${Math.round(tokens / 1000)}k`;
}

/* ── The two ways to bring your own model ──
 *
 * Both were long cards with their code on the section face. They are the same
 * content, with the code and the caveats behind the card, so the section is a
 * screen rather than three.
 */
interface Way {
  id: string;
  icon: typeof FiServer;
  accent: string;
  title: string;
  summary: string;
  chips: string[];
  chipLabel: string;
}

export default function ModelCompatibility() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [tableOpen, setTableOpen] = useState(false);
  const [openWay, setOpenWay] = useState<string | null>(null);
  const closeWay = useCallback(() => setOpenWay(null), []);

  const withCatalog = models.providers.filter((provider) => provider.models > 0);
  const compatible = models.providers.find((provider) => provider.name === "openai_compatible");

  const stats = [
    { value: num(models.adapter_count), label: "Provider adapters", accent: ACCENT.adapters, icon: FiServer },
    { value: num(models.with_catalog_count), label: "Ship a bundled catalog", accent: ACCENT.catalog, icon: FiCpu },
    { value: num(models.models), label: "Catalogued models", accent: ACCENT.models, icon: FiCpu },
    { value: num(models.local_engines.length), label: "Local engines", accent: ACCENT.local, icon: FiHardDrive },
  ];

  const ways: Way[] = [
    {
      id: "served",
      icon: FiServer,
      accent: ACCENT.catalog,
      title: "A server you already run",
      summary: `The tenth adapter, ${compatible?.name}, ships no catalog because it serves whatever your endpoint serves. One base_url points effGen at it.`,
      chipLabel: "Works with",
      chips: ["vLLM", "SGLang", "TGI", "llama.cpp", "Ollama", "LM Studio", "LiteLLM"],
    },
    {
      id: "local",
      icon: FiHardDrive,
      accent: ACCENT.local,
      title: "Or no server at all",
      summary: `${models.local_engines.length} local engines load the weights in the agent's own process. Name one as a prefix and it is used; leave it off and effGen picks from what is installed.`,
      chipLabel: "Local engines",
      chips: models.local_engines,
    },
  ];

  return (
    <section id="models" className="py-24 bg-white dark:bg-[#020c08] relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      <Container className="relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
            <FiCpu size={14} />
            Models and providers
          </span>
          <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
            <span className="gradient-text">Any model</span>, anywhere it runs
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            {models.adapter_count} provider adapters are registered.{" "}
            {models.with_catalog_count} of them ship a bundled catalog of{" "}
            {num(models.models)} models with context windows, prices and capabilities. The tenth is
            the one that matters if you already serve a model yourself.
          </p>
        </motion.div>

        {/* Headline figures */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
        >
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="rounded-2xl p-5 bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-center"
            >
              <stat.icon className="mx-auto mb-2" style={accentTextStyle(stat.accent)} size={20} />
              <div className="text-3xl font-black mb-0.5" style={accentTextStyle(stat.accent)}>
                {stat.value}
              </div>
              <div className="text-[10px] text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider">
                {stat.label}
              </div>
            </div>
          ))}
        </motion.div>

        {/* The two ways to bring your own model */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-8">
          {ways.map((way, index) => (
            <motion.button
              key={way.id}
              type="button"
              onClick={() => setOpenWay(way.id)}
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: 0.25 + index * 0.05 }}
              className="group rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-6 text-center flex flex-col hover:border-green-500/40 transition-colors"
            >
              {/* The card's own words name this button. An `aria-label` would
                  replace them rather than contain them, which is what WCAG
                  2.5.3 asks for when a control carries visible text; this adds
                  only what pressing it does. */}
              <span className="sr-only"> — open details</span>
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center mb-4 mx-auto"
                style={{ background: `${way.accent}15`, border: `1px solid ${way.accent}30` }}
              >
                <way.icon style={accentTextStyle(way.accent)} size={22} />
              </div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-2">{way.title}</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">{way.summary}</p>
              <div className="flex flex-wrap justify-center gap-1.5 mt-4">
                {way.chips.map((chip) => (
                  <span
                    key={chip}
                    className="px-2 py-1 rounded-md text-[10px] font-mono font-semibold"
                    style={{
                      ...accentTextStyle(way.accent),
                      background: `${way.accent}15`,
                      border: `1px solid ${way.accent}30`,
                    }}
                  >
                    {chip}
                  </span>
                ))}
              </div>
              <div className="mt-auto pt-4 text-[10px] text-gray-400 dark:text-gray-600 group-hover:text-gray-600 dark:group-hover:text-gray-400 transition-colors font-medium tracking-wide uppercase">
                Click for details
              </div>
            </motion.button>
          ))}
        </div>

        {/* The provider table, on request */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.35 }}
          className="text-center"
        >
          <button
            type="button"
            onClick={() => setTableOpen((open) => !open)}
            aria-expanded={tableOpen}
            aria-controls="provider-table"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/60 text-sm font-semibold text-gray-700 dark:text-gray-300 hover:border-green-500/40 transition-colors"
          >
            {tableOpen ? "Hide" : "Show"} all {models.with_catalog_count} catalogued providers
            <FiChevronDown
              size={14}
              className="transition-transform"
              style={{ transform: tableOpen ? "rotate(180deg)" : "none" }}
            />
          </button>

          <div id="provider-table" hidden={!tableOpen} className="mt-6 text-left">
            <div className="overflow-x-auto rounded-2xl border border-gray-200 dark:border-gray-800">
              <table className="w-full text-left text-sm border-collapse">
                <caption className="sr-only">
                  The {models.with_catalog_count} provider adapters that ship a bundled catalog, with
                  the models each carries.
                </caption>
                <thead>
                  <tr className="bg-gray-50 dark:bg-gray-900/80">
                    <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 whitespace-nowrap">
                      Provider
                    </th>
                    <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 text-right whitespace-nowrap">
                      Models
                    </th>
                    <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 whitespace-nowrap">
                      Default
                    </th>
                    <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 text-right whitespace-nowrap">
                      Tool calling
                    </th>
                    <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 text-right whitespace-nowrap">
                      Vision
                    </th>
                    <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 text-right whitespace-nowrap">
                      Largest window
                    </th>
                    <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 whitespace-nowrap">
                      Catalog checked
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {withCatalog.map((provider) => (
                    <tr
                      key={provider.name}
                      className="border-t border-gray-200 dark:border-gray-800/70 align-top"
                    >
                      <th scope="row" className="px-4 py-3 whitespace-nowrap font-normal">
                        <code className="font-mono text-[13px] font-bold text-green-700 dark:text-green-400">
                          {provider.name}
                        </code>
                      </th>
                      <td className="px-4 py-3 text-right font-mono text-[13px] text-gray-900 dark:text-white tabular-nums">
                        {num(provider.models)}
                      </td>
                      <td className="px-4 py-3">
                        <code className="font-mono text-[12px] text-gray-600 dark:text-gray-400 break-all">
                          {provider.default ?? "—"}
                        </code>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[13px] text-gray-600 dark:text-gray-400 tabular-nums">
                        {num(provider.supports_tools)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[13px] text-gray-600 dark:text-gray-400 tabular-nums">
                        {provider.supports_vision > 0 ? num(provider.supports_vision) : "—"}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[13px] text-gray-600 dark:text-gray-400 tabular-nums">
                        {contextLabel(provider.max_context)}
                      </td>
                      <td className="px-4 py-3 font-mono text-[12px] text-gray-600 dark:text-gray-400 whitespace-nowrap">
                        {provider.verified_on ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
              Model counts and capabilities are the bundled catalog&rsquo;s;{" "}
              <code className="font-mono">effgen models refresh</code> updates it against each
              provider&rsquo;s live API, and the last column is when that was last done.{" "}
              <code className="font-mono">
                {models.capability_totals.supports_tools} of {num(models.models)}
              </code>{" "}
              catalogued models call tools, {models.capability_totals.supports_vision} take images and{" "}
              {models.capability_totals.supports_audio} take audio. A model the catalog has never seen
              is still callable — it simply reports no price rather than a made-up one.
            </p>
          </div>

          <div className="mt-8">
            <RouteLink
              to="/models"
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
            >
              Everything about models and providers
            </RouteLink>
          </div>
        </motion.div>
      </Container>

      <AnimatePresence>
        {openWay === "served" && (
          <DetailDialog
            title="A server you already run"
            accent={ACCENT.catalog}
            onClose={closeWay}
          >
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-4">
              The tenth adapter, <code className="font-mono">{compatible?.name}</code>, ships no
              catalog because it serves whatever your endpoint serves. Point it at vLLM, SGLang,
              TGI, llama.cpp, Ollama, LM Studio, LiteLLM or a gateway with a{" "}
              <code className="font-mono">base_url</code> and effGen drives that, instead of loading
              a second copy of the weights inside the agent process.
            </p>
            <CodeSample
              code={`import os

from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    base_url="http://127.0.0.1:8123/v1",
    api_key=os.environ["EFFGEN_API_KEY"],
))
print(agent.run("Reply with the single word: ready").text)`}
              language="python"
              accent={ACCENT.catalog}
              output="ready"
            />
            <p className="mt-4 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
              The ids come from the server, so no catalog is consulted and no price is invented.
              Ask it what it has with <code className="font-mono">list_served_models()</code>, and
              pass <code className="font-mono">context_length=</code> when its window is not the
              32,768 tokens effGen otherwise assumes — it warns when it is assuming rather than
              failing later at a size nobody chose.
            </p>
          </DetailDialog>
        )}

        {openWay === "local" && (
          <DetailDialog
            title="Or no server at all"
            accent={ACCENT.local}
            onClose={closeWay}
          >
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-4">
              {models.local_engines.length} local engines load weights in the agent&rsquo;s own
              process. Name one as a prefix and it is used; leave the prefix off and effGen picks
              from what is installed.
            </p>

            <div className="flex flex-wrap gap-2 mb-4">
              {models.local_engines.map((engine) => (
                <span
                  key={engine}
                  className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold"
                  style={{
                    ...accentTextStyle(ACCENT.local),
                    background: `${ACCENT.local}15`,
                    border: `1px solid ${ACCENT.local}30`,
                  }}
                >
                  {engine}
                </span>
              ))}
            </div>

            <CodeSample
              code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="transformers:Qwen/Qwen2.5-1.5B-Instruct"))
print(agent.run("Reply with the single word: ready").text)`}
              language="python"
              accent={ACCENT.local}
              output="Ready"
            />
          </DetailDialog>
        )}
      </AnimatePresence>
    </section>
  );
}
