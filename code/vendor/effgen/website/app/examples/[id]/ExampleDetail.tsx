"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { FiArrowLeft, FiArrowRight, FiCheck, FiGithub, FiTerminal } from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import CodeSample from "@/components/ui/CodeSample";
import { examples, examplesData } from "./examplesData";
import { accentTextStyle } from "@/components/accentText";

// The id arrives already resolved from the route above, so this renders the
// whole page on the first pass — the export carries the example's text rather
// than a spinner waiting for a promise the browser has to settle.
export default function ExampleDetailPage({ id }: { id: string }) {
  const example = examplesData[id];

  if (!example) {
    return (
      <div className="min-h-screen bg-white dark:bg-[#020c08]">
        <Navbar />
        <main id="main" className="pt-32 pb-24">
          <Container>
            <h1 className="text-3xl font-black text-gray-900 dark:text-white mb-4">
              No example with that name
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mb-8">
              There are {examples.length} examples, and this is not one of them.
            </p>
            <Link
              href="/examples"
              className="inline-flex items-center gap-2 text-green-700 dark:text-green-400 font-semibold"
            >
              <FiArrowLeft size={16} />
              All examples
            </Link>
          </Container>
        </main>
        <Footer />
      </div>
    );
  }

  const index = examples.findIndex((item) => item.id === id);
  const next = examples[(index + 1) % examples.length];
  const accent = example.accent;

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">
        {/* Hero */}
        <section className="relative pt-32 pb-12 overflow-hidden">
          <div className="absolute inset-0 grid-pattern" />
          <Container className="relative z-10">
            <Link
              href="/examples"
              className="inline-flex items-center gap-2 text-sm font-semibold text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 transition-colors mb-8"
            >
              <FiArrowLeft size={14} />
              All examples
            </Link>

            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="max-w-3xl"
            >
              <div className="flex items-center gap-4 mb-5">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl flex-shrink-0"
                  style={{ background: `${accent}15`, border: `1px solid ${accent}30` }}
                  aria-hidden="true"
                >
                  {example.icon}
                </div>
                <span
                  className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider"
                  style={{ background: `${accent}15`, color: accent, border: `1px solid ${accent}30` }}
                >
                  {example.badge}
                </span>
              </div>

              <h1 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white leading-tight">
                {example.title}
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-400 leading-relaxed">
                {example.subtitle}
              </p>

              <div className="mt-6 flex flex-wrap gap-2">
                {example.tools.map((tool) => (
                  <span
                    key={tool}
                    className="px-2.5 py-1 rounded-lg text-xs font-mono bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </motion.div>
          </Container>
        </section>

        {/* The run */}
        <section className="py-12 relative">
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
          <Container className="relative z-10">
            <div className="grid lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)] gap-8 items-start">
              <div>
                <h2 className="text-2xl font-black text-gray-900 dark:text-white mb-3">
                  The program, and what it printed
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
                  This ran against effGen with{" "}
                  <code className="font-mono">{example.run.model}</code>. The pane under the code is
                  that run&rsquo;s output, pasted — so where the answer depends on a live API or on
                  the model&rsquo;s wording, yours will differ. The pane is the shape that run
                  actually had, not a tidied one.
                </p>
                <CodeSample
                  code={example.run.code}
                  language="python"
                  accent={accent}
                  output={example.run.output}
                  outputLabel="what it printed"
                />
              </div>

              <div className="space-y-6">
                <div className="rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-6">
                  <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                    What it does
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {example.description}
                  </p>
                </div>

                <div className="rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-6">
                  <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                    What the run shows
                  </h3>
                  <ul className="space-y-3">
                    {example.observations.map((observation) => (
                      <li key={observation} className="flex gap-2.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                        <FiCheck
                          size={14}
                          className="mt-1 flex-shrink-0"
                          style={accentTextStyle(accent)}
                          aria-hidden="true"
                        />
                        <span>{observation}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-6">
                  <h3 className="flex items-center gap-2 text-base font-bold text-gray-900 dark:text-white mb-3">
                    <FiTerminal size={16} style={accentTextStyle(accent)} aria-hidden="true" />
                    The full script
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-4">
                    The program above is the short version. The one in the repository at{" "}
                    <code className="font-mono text-[12px] break-all">{example.script}</code> covers
                    more cases and takes a <code className="font-mono">--model</code> flag. It ships
                    with the package, so it runs from the command line without cloning anything:
                  </p>
                  <pre className="p-3 rounded-lg bg-white dark:bg-black/40 border border-gray-200 dark:border-gray-800 overflow-x-auto text-xs font-mono text-gray-700 dark:text-gray-300">
                    {example.command}
                  </pre>
                  <a
                    href={example.githubUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-4 inline-flex items-center gap-2 text-sm font-semibold"
                    style={accentTextStyle(accent)}
                  >
                    <FiGithub size={14} aria-hidden="true" />
                    Read it on GitHub
                  </a>
                </div>
              </div>
            </div>
          </Container>
        </section>

        {/* Next */}
        <section className="py-16 relative">
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
          <Container className="relative z-10">
            <div className="flex flex-wrap items-center justify-between gap-6">
              <Link
                href="/examples"
                className="inline-flex items-center gap-2 text-sm font-semibold text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 transition-colors"
              >
                <FiArrowLeft size={14} />
                All {examples.length} examples
              </Link>
              <Link
                href={`/examples/${next.id}`}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-black"
                style={{ background: `linear-gradient(135deg, ${next.accent}, ${next.accent}bb)` }}
              >
                Next: {next.title}
                <FiArrowRight size={14} />
              </Link>
            </div>
          </Container>
        </section>
      </main>
      <Footer />
    </div>
  );
}
