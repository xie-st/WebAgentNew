"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiArrowRight, FiGithub, FiZap } from "react-icons/fi";
import Link from "next/link";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { examples } from "./[id]/examplesData";
import { accentTextStyle } from "@/components/accentText";

// Every card here reads its title, accent, tools and script path from the same
// module the detail pages render, so the two cannot disagree about an example.

const EXAMPLES_DIR = "https://github.com/ctrl-gaurav/effGen/tree/main/examples";

export default function ExamplesView() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">
        {/* Hero */}
        <section className="relative pt-32 pb-16 overflow-hidden">
          <div className="absolute inset-0 grid-pattern" />
          <Container className="relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="text-center max-w-3xl mx-auto"
            >
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8">
                <FiZap size={14} />
                Examples
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                <span className="gradient-text">{examples.length} agents</span>, and what they
                printed
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed">
                Each one is a short program built from a script that ships with the framework. Every
                one on this page was run against the released package, and each page carries that
                run&rsquo;s output under its code — including where the answer is a live API&rsquo;s
                and yours will read differently.
              </p>
            </motion.div>
          </Container>
        </section>

        {/* Cards */}
        <section className="py-12 pb-24 relative">
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <Container className="relative z-10">
            <motion.div
              ref={ref}
              initial="hidden"
              animate={inView ? "visible" : "hidden"}
              variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.08 } } }}
              className="grid grid-cols-1 lg:grid-cols-2 gap-6"
            >
              {examples.map((example) => (
                <motion.article
                  key={example.id}
                  variants={{ hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0 } }}
                  className="group relative overflow-hidden rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none backdrop-blur-sm transition-transform hover:-translate-y-1"
                >
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl pointer-events-none"
                    style={{ background: `radial-gradient(circle at 30% 30%, ${example.accent}0d 0%, transparent 70%)` }}
                  />

                  <div className="p-8 relative z-10 flex flex-col h-full">
                    <div className="flex items-center gap-4 mb-5">
                      <div
                        className="w-14 h-14 rounded-xl flex items-center justify-center text-2xl flex-shrink-0"
                        style={{ background: `${example.accent}15`, border: `1px solid ${example.accent}25` }}
                        aria-hidden="true"
                      >
                        {example.icon}
                      </div>
                      <span
                        className="px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider"
                        style={{
                          background: `${example.accent}15`,
                          ...accentTextStyle(example.accent),
                          border: `1px solid ${example.accent}30`,
                        }}
                      >
                        {example.badge}
                      </span>
                    </div>

                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      {example.title}
                    </h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-5 leading-relaxed">
                      {example.subtitle}
                    </p>

                    {/* The first line of what it printed, so the card is not a promise */}
                    <div className="mb-5 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3 overflow-hidden">
                      <div className="text-[10px] font-mono uppercase tracking-widest text-gray-600 dark:text-gray-400 mb-1.5">
                        what it printed
                      </div>
                      <pre className="text-xs font-mono text-gray-700 dark:text-gray-300 whitespace-pre-wrap line-clamp-3">
                        {example.run.output.trim().split("\n").slice(0, 3).join("\n")}
                      </pre>
                    </div>

                    <div className="flex flex-wrap gap-1.5 mb-6">
                      {example.tools.map((tool) => (
                        <span
                          key={tool}
                          className="px-2 py-0.5 rounded text-[10px] font-mono bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700"
                        >
                          {tool}
                        </span>
                      ))}
                    </div>

                    <div className="flex gap-3 mt-auto">
                      <Link
                        href={`/examples/${example.id}`}
                        className="flex-1 flex items-center justify-center gap-2 px-5 py-3 rounded-xl font-bold text-sm text-black"
                        style={{ background: `linear-gradient(135deg, ${example.accent}, ${example.accent}bb)` }}
                      >
                        Read the walkthrough
                        <FiArrowRight size={14} />
                      </Link>
                      <a
                        href={example.githubUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-5 py-3 rounded-xl font-semibold text-sm border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800/60 text-gray-700 dark:text-gray-300 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all flex items-center gap-2"
                        aria-label={`${example.title} — the full script on GitHub`}
                      >
                        <FiGithub size={14} aria-hidden="true" />
                        Script
                      </a>
                    </div>
                  </div>

                  <div
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${example.accent}, transparent)` }}
                  />
                </motion.article>
              ))}
            </motion.div>

            {/* Everything else the package ships */}
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="mt-12 rounded-2xl bg-gray-50 dark:bg-gray-950/80 border border-gray-200 dark:border-green-500/15 p-8 md:p-10 text-center"
            >
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-3">
                The rest ship with the package
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-2xl mx-auto text-sm leading-relaxed">
                The six above are the ones with a walkthrough. Every example script in the
                repository is also installed alongside the package, so you can list them and run one
                by name without cloning anything.
              </p>
              <pre className="inline-block text-left px-5 py-4 rounded-xl bg-white dark:bg-black/40 border border-gray-200 dark:border-gray-800 text-sm font-mono text-gray-700 dark:text-gray-300 overflow-x-auto max-w-full">
{`effgen examples list
effgen examples run <name>`}
              </pre>
              <p className="mt-4 text-xs text-gray-600 dark:text-gray-400 max-w-2xl mx-auto leading-relaxed">
                Many of the scripts name a local model, so they need a GPU with room for it. A
                script that cannot fit the device says so and names what to change.
              </p>
              <div className="mt-8">
                <a
                  href={EXAMPLES_DIR}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
                  style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)" }}
                >
                  Browse them on GitHub
                  <FiArrowRight size={14} />
                </a>
              </div>
            </motion.div>
          </Container>
        </section>
      </main>
      <Footer />
    </div>
  );
}
