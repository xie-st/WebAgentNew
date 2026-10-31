"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiArrowRight, FiBarChart2, FiBookOpen, FiCloud, FiCode, FiSearch, FiUsers, FiZap,
} from "react-icons/fi";
import type { IconType } from "react-icons";
import Link from "next/link";
import Container from "./Container";
import { examples } from "@/app/examples/[id]/examplesData";
import { accentTextStyle } from "./accentText";

/* ── The examples teaser ──
 *
 * The id, the title, the accent, the one-line summary and the tools each
 * example uses all come from `examplesData`, which is also what the detail
 * routes render, so this teaser cannot drift from the pages it links to.
 *
 * Only the icon per example is written here.
 */

const ICONS: Record<string, IconType> = {
  "code-assistant": FiCode,
  "research-agent": FiSearch,
  "data-analysis": FiBarChart2,
  "multi-agent": FiUsers,
  "weather-json-pipeline": FiCloud,
  "rag-knowledge-base": FiBookOpen,
};

export default function Examples() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section id="examples" className="py-24 bg-gray-50 dark:bg-[#020c08] relative overflow-hidden" ref={ref}>
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
            Examples
          </span>
          <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
            {examples.length} agents you can <span className="gradient-text">run today</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Each one is built from a script that ships with the framework, and each page carries
            the program and what that program printed when it was run.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5 mb-12">
          {examples.map((example, index) => {
            const accent = example.accent;
            const Icon = ICONS[example.id] ?? FiCode;

            return (
              <motion.div
                key={example.id}
                initial={{ opacity: 0, y: 40 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: index * 0.08 }}
              >
                <Link
                  href={`/examples/${example.id}`}
                  className="group relative flex flex-col h-full overflow-hidden p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm shadow-sm dark:shadow-none transition-transform hover:-translate-y-1"
                >
                  <div
                    className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ boxShadow: `inset 0 0 0 1px ${accent}30` }}
                  />
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                    style={{ background: `radial-gradient(circle at 30% 30%, ${accent}10 0%, transparent 70%)` }}
                  />

                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center relative z-10"
                    style={{ background: `${accent}15`, border: `1px solid ${accent}25` }}
                  >
                    <Icon style={accentTextStyle(accent)} size={22} />
                  </div>

                  <h3 className="mt-4 text-lg font-bold text-gray-900 dark:text-white relative z-10">
                    {example.title}
                  </h3>
                  <p className="mt-2 mb-5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed relative z-10">
                    {example.subtitle}
                  </p>

                  <div className="flex flex-wrap gap-1.5 mb-5 relative z-10">
                    {example.tools.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-0.5 rounded text-[10px] font-mono bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>

                  <span
                    className="mt-auto inline-flex items-center gap-1.5 text-sm font-semibold relative z-10"
                    style={accentTextStyle(accent)}
                  >
                    Read the walkthrough
                    <FiArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
                  </span>

                  <span
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
                  />
                </Link>
              </motion.div>
            );
          })}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="text-center"
        >
          <Link
            href="/examples"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
            style={{
              background: "linear-gradient(135deg, #00ff88, #00c96e)",
              boxShadow: "0 0 30px rgba(0,255,136,0.3)",
            }}
          >
            All examples
            <FiArrowRight size={16} />
          </Link>
        </motion.div>
      </Container>
    </section>
  );
}
