"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiCpu, FiServer, FiCloud, FiTarget } from "react-icons/fi";
import Container from "./Container";
import CodeSample from "./ui/CodeSample";
import {
  commandCount,
  modelCount,
  presetCount,
  providerCount,
  providersWithCatalog,
  publicNameCount,
  pythonVersions,
  siteData,
  toolCount,
  version,
} from "./siteData";
import { accentTextStyle } from "./accentText";

/* ── Where the model runs ── */

const places = [
  {
    icon: FiCpu,
    accent: "#00ff88",
    title: "In your own process",
    description:
      `The weights load where your code runs. Four local engines — ${siteData.models.local_engines.join(", ")} — and no key, no network and no provider between the agent and the model.`,
    code: `from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="Qwen/Qwen2.5-1.5B-Instruct"))
print(agent.run("Name the three primary colours of light, comma separated.").text)`,
    output: "Red, Green, Blue",
  },
  {
    icon: FiServer,
    accent: "#00e5ff",
    title: "On a server you already run",
    description:
      "One base_url points effGen at anything that speaks the OpenAI protocol — vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, LiteLLM, a company gateway. The weights load once and every caller shares them.",
    code: `import os

from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["EFFGEN_API_KEY"],
))
print(agent.run("Reply with the single word: ready").text)`,
    output: "ready",
  },
  {
    icon: FiCloud,
    accent: "#a78bfa",
    title: "Or a hosted provider",
    description:
      `${providerCount} provider adapters, ${providersWithCatalog} of them carrying a bundled catalog of ${modelCount} priced models. Changing provider is changing one string; the agent, the tools and the results object do not move.`,
    code: `from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="openai:gpt-5-nano"))
print(agent.run("Name the three primary colours of light, comma separated.").text)`,
    output: "red, green, blue",
  },
];

/* ── The derived index ── */

const index = [
  { value: version, label: "version" },
  { value: String(toolCount), label: "built-in tools" },
  { value: String(presetCount), label: "presets" },
  { value: String(providerCount), label: "provider adapters" },
  { value: String(modelCount), label: "catalogued models" },
  { value: String(commandCount), label: "CLI commands" },
  { value: String(publicNameCount), label: "public names" },
  {
    value: `${pythonVersions[0]}–${pythonVersions[pythonVersions.length - 1]}`,
    label: "python",
  },
];

export default function WhatItIs() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section
      id="what-it-is"
      className="py-24 bg-white dark:bg-[#020c08] relative overflow-hidden noise-overlay"
      ref={ref}
    >
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      <Container className="relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <motion.span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-6">
            <FiTarget size={14} />
            What effGen is
          </motion.span>
          <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
            Agents on small language models,
            <br />
            <span className="gradient-text">wherever you run them</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            A Python framework for building agents that reason, call tools and finish
            work — on a model you load yourself, on a server you already run, or on a
            hosted provider. The same agent, the same tools, the same result object in
            all three, with the server, the operations surface and the command line
            around them.
          </p>
        </motion.div>

        {/* Three places the model can live */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-14">
          {places.map((place, i) => (
            <motion.div
              key={place.title}
              initial={{ opacity: 0, y: 30 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="group relative p-6 rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden shadow-sm dark:shadow-none flex flex-col"
            >
              <motion.div
                className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                style={{ boxShadow: `inset 0 0 0 1px ${place.accent}30` }}
              />

              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ background: `${place.accent}15`, border: `1px solid ${place.accent}30` }}
              >
                <place.icon style={accentTextStyle(place.accent)} size={22} />
              </div>

              <h3 className="mt-4 text-lg font-bold text-gray-900 dark:text-white">
                {place.title}
              </h3>
              <p className="mt-2 mb-5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                {place.description}
              </p>

              <CodeSample
                className="mt-auto"
                code={place.code}
                accent={place.accent}
                output={place.output}
              />
            </motion.div>
          ))}
        </div>

        {/* The derived index */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.35 }}
          className="rounded-2xl border border-green-500/20 bg-gray-50 dark:bg-gray-900/60 px-6 py-7"
        >
          <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-7 text-center">
            {index.map((item) => (
              <div key={item.label}>
                <dt className="sr-only">{item.label}</dt>
                <dd>
                  <span className="block text-2xl md:text-3xl font-black font-mono text-gray-900 dark:text-white">
                    {item.value}
                  </span>
                  <span className="mt-1 block text-[10px] font-mono uppercase tracking-widest text-gray-600 dark:text-gray-400">
                    {item.label}
                  </span>
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-6 text-center text-xs text-gray-600 dark:text-gray-400">
            Every figure on this page is read from the installed package when the site
            is built, not written into it.
          </p>
        </motion.div>
      </Container>
    </section>
  );
}
