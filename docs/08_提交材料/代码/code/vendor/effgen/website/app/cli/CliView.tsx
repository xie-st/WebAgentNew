"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiActivity,
  FiArrowRight,
  FiCommand,
  FiDroplet,
  FiExternalLink,
  FiFileText,
  FiTerminal,
} from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Terminal from "@/components/ui/Terminal";
import TerminalReplay from "@/components/ui/TerminalReplay";
import ParamTable from "@/components/ui/ParamTable";
import CodeSample from "@/components/ui/CodeSample";
import { cliCapture, cliDocument } from "@/components/captures";
import { siteData, version } from "@/components/siteData";
import {
  DOCS_CLI_URL,
  DOCS_TOP_URL,
  commandGroups,
  jsonExamples,
  subcommandsOf,
  summaryOf,
  themeNotes,
} from "./cliData";
import { accentTextStyle } from "@/components/accentText";

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

const cli = siteData.cli;

function Band({
  id,
  eyebrow,
  title,
  lede,
  tinted = false,
  children,
}: {
  id: string;
  eyebrow: string;
  title: React.ReactNode;
  lede?: React.ReactNode;
  tinted?: boolean;
  children: React.ReactNode;
}) {
  const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section
      id={id}
      className={`py-16 relative scroll-mt-24 ${tinted ? "bg-gray-50 dark:bg-[#030f07]" : ""}`}
      aria-labelledby={`${id}-heading`}
    >
      {SECTION_DIVIDER}
      <Container className="relative z-10">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="mb-10 max-w-3xl"
        >
          <span className="text-[10px] font-mono uppercase tracking-widest text-green-700 dark:text-green-400">
            {eyebrow}
          </span>
          <h2
            id={`${id}-heading`}
            className="mt-2 text-3xl md:text-4xl font-black text-gray-900 dark:text-white leading-tight"
          >
            {title}
          </h2>
          {lede && <p className="mt-4 text-gray-600 dark:text-gray-400 leading-relaxed">{lede}</p>}
        </motion.div>
        {children}
      </Container>
    </section>
  );
}

export default function CliView() {
  const { ref: heroRef, inView: heroInView } = useInView({ triggerOnce: true, threshold: 0.05 });

  const top = cliCapture("cli-top");
  const battle = cliCapture("cli-battle");
  const completion = cliCapture("cli-completion-bash");
  const runsList = cliCapture("cli-runs-list");
  const runsShow = cliCapture("cli-runs-show");
  const sessionsList = cliCapture("cli-sessions-list");
  const costToday = cliCapture("cli-cost-today");
  const quickstart = cliCapture("cli-quickstart-init");
  const doctor = cliCapture("cli-doctor");
  const modelsInfo = cliCapture("cli-models-info");
  const chatSession = cliCapture("cli-chat-session");
  const reportRefusal = cliCapture("cli-report-refusal");
  const topJson = cliDocument("cli-top-json");

  const topPanels = Object.keys(topJson.document).filter(
    (key) => !["ts", "header"].includes(key),
  );

  const headline = [
    { value: String(cli.command_count), label: "Commands", accent: "#00ff88", icon: FiCommand },
    { value: String(cli.subcommand_count), label: "Sub-commands", accent: "#00e5ff", icon: FiTerminal },
    { value: String(cli.themes.length), label: "Named themes", accent: "#a78bfa", icon: FiDroplet },
    { value: String(cli.completion_shells.length), label: "Completion shells", accent: "#ffd700", icon: FiActivity },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">
        {/* Hero */}
        <section className="relative pt-32 pb-10 overflow-hidden">
          <div className="absolute inset-0 grid-pattern" />
          <Container className="relative z-10">
            <motion.div
              ref={heroRef}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="max-w-4xl"
            >
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8">
                <FiCommand size={14} />
                effgen · {version}
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                {cli.command_count} commands that{" "}
                <span className="gradient-text">answer to a pipe</span> as readily as to a
                terminal
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                The <code className="font-mono text-green-700 dark:text-green-400">effgen</code>{" "}
                command line runs an agent, races models, serves an API, watches the traffic
                going through it and reports what it all cost. Every command has a live view
                on a terminal and a single structured document on a pipe, and the completion
                scripts are generated from the parser rather than maintained by hand, so they
                cannot go stale.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <a
                  href="#commands"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
                >
                  All {cli.command_count} commands
                  <FiArrowRight size={15} />
                </a>
                <a
                  href={DOCS_CLI_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-green-500/50 font-semibold text-sm transition-colors"
                >
                  Reference documentation
                  <FiExternalLink size={14} />
                </a>
              </div>
            </motion.div>
          </Container>
        </section>

        <section className="pb-10 relative">
          <Container className="relative z-10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {headline.map((stat, idx) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={heroInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: idx * 0.08 }}
                  className="rounded-2xl p-5 bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 text-center"
                >
                  <stat.icon className="mx-auto mb-2" style={accentTextStyle(stat.accent)} size={20} />
                  <div className="text-2xl font-black mb-0.5" style={accentTextStyle(stat.accent)}>
                    {stat.value}
                  </div>
                  <div className="text-[10px] text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </Container>
        </section>

        {/* effgen top */}
        <Band
          id="top"
          eyebrow="effgen top"
          title={
            <>
              What is running, <span className="gradient-text">what it costs</span>, and
              which panel measured it
            </>
          }
          lede={
            <>
              Five panels, refreshing in place. Each one names the window and the process it
              measures, and figures from different sources are never combined: Activity
              reads the local run history across every process, Traffic and Per-model read
              one server&rsquo;s counters since it started, Spend reads the local cost
              ledger, and GPU reads the physical devices.
            </>
          }
        >
          <TerminalReplay
            command={top.command}
            output={top.text}
            title="effgen top"
            rows={26}
            interval={70}
            caption={
              <>
                Recorded against a real <code className="font-mono text-xs">effgen serve</code>{" "}
                on this machine with five requests through it, one of which named a model
                that does not exist — which is the error row in Activity and the 100% row in
                Per-model. The GPU panel is this host&rsquo;s eight cards, most of them busy
                with other work.
              </>
            }
          />

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-6">
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-2">
                It is useful without a server
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Activity, Spend and GPU need no server, so the view still says something on
                a host running only local agents. The server-backed panels then read as
                unavailable and name the URL they tried, rather than showing zeros that look
                like quiet.{" "}
                <code className="font-mono text-xs">effgen monitor</code> is an alias for the
                same command.
              </p>
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Piped output, <code className="font-mono text-xs">--json</code>,{" "}
                <code className="font-mono text-xs">--once</code>,{" "}
                <code className="font-mono text-xs">--no-animation</code> and{" "}
                <code className="font-mono text-xs">NO_COLOR</code> all print a single
                snapshot and exit instead of taking over the screen.
              </p>
              <a
                href={DOCS_TOP_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-4 inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
              >
                docs/cli/top.md
                <FiExternalLink size={13} />
              </a>
            </div>
            <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-6">
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                The same snapshot, structured
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                <code className="font-mono text-xs">{topJson.command}</code> emits one
                document carrying the same five panels, so the view is scriptable without
                parsing a table.
              </p>
              <ul className="flex flex-wrap gap-2">
                {topPanels.map((panel) => (
                  <li
                    key={panel}
                    className="px-2.5 py-1 rounded-lg text-xs font-mono bg-gray-100 dark:bg-gray-900 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-800"
                  >
                    .{panel}
                  </li>
                ))}
              </ul>
              <div className="mt-4">
                <CodeSample
                  language="bash"
                  accent="#00e5ff"
                  code={`effgen top --json | jq '.spend | {total_cost_usd, requests, daily_budget_usd}'`}
                  output={`{
  "total_cost_usd": 0.00211725,
  "requests": 13,
  "daily_budget_usd": 1
}`}
                />
              </div>
            </div>
          </div>
        </Band>

        {/* The commands */}
        <Band
          id="commands"
          eyebrow="The command surface"
          tinted
          title={
            <>
              All {cli.command_count} commands,{" "}
              <span className="gradient-text">by what they are for</span>
            </>
          }
          lede={
            <>
              Every summary below is the command&rsquo;s own, from{" "}
              <code className="font-mono text-sm">effgen --help</code> in effGen {version}.{" "}
              {cli.subcommand_count} sub-commands sit under nine of them.
            </>
          }
        >
          <div className="space-y-10">
            {commandGroups.map((group) => (
              <section key={group.id} aria-labelledby={`group-${group.id}`}>
                <div className="flex items-baseline gap-3 mb-1">
                  <h3
                    id={`group-${group.id}`}
                    className="text-xl font-black text-gray-900 dark:text-white"
                  >
                    {group.title}
                  </h3>
                  <span className="text-xs font-mono" style={accentTextStyle(group.accent)}>
                    {group.commands.length}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 max-w-3xl">
                  {group.lede}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {group.commands.map((name) => {
                    const subs = subcommandsOf(name);
                    return (
                      <article
                        key={name}
                        className="relative rounded-xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-4"
                      >
                        <div
                          className="absolute left-0 top-4 bottom-4 w-0.5 rounded-full"
                          style={{ background: group.accent }}
                        />
                        <div className="pl-3">
                          <code
                            className="text-sm font-mono font-bold"
                            style={accentTextStyle(group.accent)}
                          >
                            effgen {name}
                          </code>
                          <p className="mt-1.5 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                            {summaryOf(name)}
                          </p>
                          {subs.length > 0 && (
                            <p className="mt-2 text-[11px] font-mono text-gray-600 dark:text-gray-400 break-words">
                              {subs.join(" · ")}
                            </p>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>

          <div className="mt-12">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
              The flags that come before the command
            </h3>
            <ParamTable
              nameLabel="Flag"
              params={cli.global_options.map((option) => ({
                name: option.name,
                description: option.description,
              }))}
              caption={`effgen --help · effGen ${version}`}
            />
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 max-w-3xl leading-relaxed">
              Long option names mean the same thing everywhere. Two short flags do not:{" "}
              <code className="font-mono text-xs">-c</code> is{" "}
              <code className="font-mono text-xs">--concurrency</code> on{" "}
              <code className="font-mono text-xs">batch</code> and{" "}
              <code className="font-mono text-xs">loadtest</code> but{" "}
              <code className="font-mono text-xs">--config</code> on{" "}
              <code className="font-mono text-xs">run</code>, and{" "}
              <code className="font-mono text-xs">-p</code> is a port on{" "}
              <code className="font-mono text-xs">serve</code>,{" "}
              <code className="font-mono text-xs">monitor</code> and{" "}
              <code className="font-mono text-xs">top</code> but{" "}
              <code className="font-mono text-xs">--print</code> on{" "}
              <code className="font-mono text-xs">code</code>. Both collisions are frozen and
              pinned by a test, so a third meaning cannot be added quietly — and in a script
              the long spelling is the portable one.
            </p>
          </div>
        </Band>

        {/* Themes */}
        <Band
          id="themes"
          eyebrow="Appearance"
          title={
            <>
              {cli.themes.length} named themes, and{" "}
              <span className="gradient-text">NO_COLOR still wins</span>
            </>
          }
          lede={
            <>
              A theme is a table of semantic roles — heading, success, error, warning, cost,
              tool, model, metric, muted — mapped onto colours, so one palette drives every
              table, spinner, panel and status line. Select one with{" "}
              <code className="font-mono text-sm">--theme</code> or{" "}
              <code className="font-mono text-sm">EFFGEN_THEME</code>. When{" "}
              <code className="font-mono text-sm">NO_COLOR</code> is set the console renders
              the structure with no colour at all, whichever theme is chosen.
            </>
          }
        >
          <p className="mb-8 text-sm text-gray-600 dark:text-gray-400 max-w-3xl leading-relaxed">
            These four frames are the same command run under each theme, drawn from the
            colour codes it emitted rather than from a description of them. They are text,
            not screenshots — selectable, searchable, and legible at any zoom.
          </p>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            {themeNotes.map((theme) => {
              const capture = cliCapture(theme.slug);
              return (
                <figure key={theme.name}>
                  <figcaption className="mb-3">
                    <code
                      className="text-sm font-mono font-bold"
                      style={accentTextStyle(theme.accent)}
                    >
                      --theme {theme.name}
                    </code>
                    <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                      {theme.purpose}
                    </p>
                  </figcaption>
                  <Terminal
                    command={capture.command}
                    output={capture.text}
                    spans={capture.spans}
                    ground={theme.ground}
                    title={`--theme ${theme.name}`}
                    maxLines={26}
                  />
                </figure>
              );
            })}
          </div>
        </Band>

        {/* --json */}
        <Band
          id="json"
          eyebrow="The --json contract"
          tinted
          title={
            <>
              One document on stdout,{" "}
              <span className="gradient-text">on a pipe and on a terminal</span>
            </>
          }
          lede={
            <>
              <code className="font-mono text-sm">--json</code> is not a rendering mode that
              a terminal can override. The command emits the same single document either
              way, the live view is skipped, and everything a human reads goes to stderr —
              so <code className="font-mono text-sm">| jq</code> works and a redirect
              captures the document alone.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <ParamTable
                nameLabel="Command"
                params={jsonExamples.map((example) => ({
                  name: example.command,
                  description: example.note,
                }))}
                caption="Each of these was run for this page and its stdout parsed as one document."
              />
            </div>
            <div>
              <CodeSample
                language="bash"
                accent="#a78bfa"
                code={`# stdout is the document; the progress display goes to stderr
effgen run "Say the word ok and nothing else." \\
  -m gemini:gemini-3.1-flash-lite --json > run.json

# so a redirect captures the document, and nothing else
head -c 60 run.json
jq -r ".success, .output" run.json`}
                output={`{
  "task": "Say the word ok and nothing else.",
  "model": 
true
Ok`}
                outputLabel="what those two commands printed"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The same split is why a command that fails still answers in JSON: the
                document carries{" "}
                <code className="font-mono text-xs">&quot;success&quot;: false</code> and a
                typed error rather than leaving a caller to parse a message meant for a
                person.
              </p>
            </div>
          </div>
        </Band>

        {/* battle */}
        <Band
          id="battle"
          eyebrow="effgen battle"
          title={
            <>
              Several models, one prompt,{" "}
              <span className="gradient-text">side by side</span>
            </>
          }
          lede={
            <>
              Where <code className="font-mono text-sm">compare</code> scores a registered
              suite against expected answers,{" "}
              <code className="font-mono text-sm">battle</code> answers &ldquo;which of
              these should I use for this?&rdquo; in one shot. On a terminal each model gets
              a column that fills in as its answer streams; the verdict panel reports what
              was <em>measured</em> — fastest, cheapest, longest — and needs no judge.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <Terminal
              command={battle.command}
              output={battle.text}
              title="effgen battle"
              maxLines={30}
            />
            <div className="space-y-4">
              <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5">
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  A judge is optional, and is reported apart
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  <code className="font-mono text-xs">--judge MODEL</code> asks a separate
                  model to pick the best answer. That pick is reported apart from the
                  measurements and names the judge, because a quality opinion and a measured
                  latency are not the same kind of claim. On{" "}
                  <code className="font-mono text-xs">compare</code>, naming no judge means
                  every model grades its own answers — which the tables say, and the JSON
                  records as <code className="font-mono text-xs">self_judged</code>.
                </p>
              </div>
              <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5">
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Loading is timed apart from generating
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  A local model&rsquo;s start-up shows as loading rather than being counted
                  as slow generation. A model that fails is reported as failed, does not end
                  the race and cannot win the verdict; the command exits 1 only when no
                  model answered at all.
                </p>
              </div>
              <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5">
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Which model can do the work
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  <code className="font-mono text-xs">effgen models info</code> carries the
                  catalog&rsquo;s answer for one id — its context window, its published
                  price, whether it calls tools, and a Coding verdict for{" "}
                  <code className="font-mono text-xs">effgen code</code>.
                </p>
                <div className="mt-4">
                  <Terminal
                    command={modelsInfo.command}
                    output={modelsInfo.text}
                    title="effgen models info"
                    maxLines={24}
                  />
                </div>
              </div>
            </div>
          </div>
        </Band>

        {/* Reports and history */}
        <Band
          id="reports"
          eyebrow="Reports, run cards and history"
          tinted
          title={
            <>
              Something to <span className="gradient-text">send someone</span>
            </>
          }
          lede={
            <>
              <code className="font-mono text-sm">compare</code>,{" "}
              <code className="font-mono text-sm">battle</code>,{" "}
              <code className="font-mono text-sm">eval</code>,{" "}
              <code className="font-mono text-sm">cost</code> and{" "}
              <code className="font-mono text-sm">loadtest</code> each take{" "}
              <code className="font-mono text-sm">--report out.html</code>, and{" "}
              <code className="font-mono text-sm">run --card</code> writes one run as a
              single file. Every style, script and chart is inline, so the file opens from
              disk or an email attachment with no network access — a claim that was checked
              for this page: the report written here reaches no external host at all.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <CodeSample
                language="bash"
                accent="#ffd700"
                code={`# one run as a shareable card, alongside the usual output
effgen run "What is 18723 * 4409? Use the calculator tool." \\
  -m gemini:gemini-3.1-flash-lite -t calculator --card run-card.html

# a result captured now, rendered later, with no model call
effgen run "Name one property of a B-tree." \\
  -m gemini:gemini-3.1-flash-lite -o saved.json
effgen report saved.json -o for-team.html`}
                output={`✓ Done in 0.8s · 1 tool · 188 tokens · $0.000077
1 tool step — run with --trace to see the timeline
✓ Run card written to run-card.html

HTML report written to for-team.html (run)`}
                outputLabel="what those commands printed"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                A card carries the task, the model that answered it, a succeeded or failed
                badge, the full answer, every tool step with its input and its result or
                typed failure, the sources and citations, and the run&rsquo;s tokens, cost
                and latency. A model with no published price reads{" "}
                <code className="font-mono text-xs">unpriced</code> and an absent metric
                reads <code className="font-mono text-xs">—</code>; neither is rendered as
                $0. A run already in history exports as a summary card, and says on its face
                that it is one.
              </p>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  A report is never written blank
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                  Handed a document that carries none of the fields a report needs,{" "}
                  <code className="font-mono text-xs">effgen report</code> lists the kinds it
                  recognises, exits 2 and writes no file.
                </p>
                <Terminal
                  command={reportRefusal.command}
                  output={reportRefusal.text}
                  title="effgen report"
                />
              </div>

              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  What it cost, today
                </h3>
                <Terminal
                  command={costToday.command}
                  output={costToday.text}
                  title="effgen cost today"
                  maxLines={16}
                />
              </div>
            </div>
          </div>

          <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                Runs are durable and addressable
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Every run is recorded with an id, whether it came from the command line, a
                script or the server. <code className="font-mono text-xs">runs show</code>{" "}
                opens one; the run below is the deliberate failure from the{" "}
                <code className="font-mono text-xs">effgen top</code> recording above, and
                its error names the fix rather than only the status code.
              </p>
              <Terminal
                command={runsList.command}
                output={runsList.text}
                title="effgen runs list"
                maxLines={24}
              />
              <div className="mt-4">
                <Terminal
                  command={runsShow.command}
                  output={runsShow.text}
                  title="effgen runs show"
                  maxLines={22}
                />
              </div>
            </div>

            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                So are conversations
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                <code className="font-mono text-xs">effgen sessions</code> lists, shows,
                browses, exports and cleans up saved conversations. It is one store:{" "}
                <code className="font-mono text-xs">effgen chat --session-id</code>,{" "}
                <code className="font-mono text-xs">effgen code --session-id</code> and the
                server all read and write the same sessions, so a conversation started in a
                coding session can be continued in chat.
              </p>
              <Terminal
                command={sessionsList.command}
                output={sessionsList.text}
                title="effgen sessions list"
                maxLines={14}
              />
              <div className="mt-6">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3 leading-relaxed">
                  <code className="font-mono text-xs">effgen chat</code> carries its own{" "}
                  {siteData.code.chat_slash_command_count} slash commands, a smaller set than
                  a coding session&rsquo;s {siteData.code.slash_command_count} because a chat
                  writes no files.
                </p>
                <Terminal
                  command={chatSession.command}
                  output={chatSession.text}
                  title="effgen chat"
                  maxLines={24}
                />
              </div>
            </div>
          </div>
        </Band>

        {/* First run */}
        <Band
          id="first-run"
          eyebrow="The first five minutes"
          title={
            <>
              Scaffold a project, then{" "}
              <span className="gradient-text">check the machine</span>
            </>
          }
          lede={
            <>
              <code className="font-mono text-sm">effgen quickstart --init DIR</code> writes
              a config, an environment template, a runnable example and a{" "}
              <code className="font-mono text-sm">.gitignore</code>, sets a daily spend cap
              when none is configured, and prints the next three commands. It prompts for
              nothing and calls no model.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <TerminalReplay
                command={quickstart.command}
                output={quickstart.text}
                title="effgen quickstart --init"
                rows={18}
                interval={110}
                caption={
                  <>
                    Four files, the cap it found already configured, and the model it picked
                    because a key for that provider was present on the machine.
                  </>
                }
              />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                <code className="font-mono text-xs">effgen doctor</code> is the second of
                those three commands. It reports which providers have a key and which
                environment variable each one reads, the system it found, and the three
                things <code className="font-mono text-xs">effgen code</code> needs — the
                workspace it would use, the sandbox backend and what that backend actually
                enforces, and whether git is present. Anything not ready comes with the fix.
              </p>
              <Terminal
                command={doctor.command}
                output={doctor.text}
                title="effgen doctor"
                maxLines={30}
              />
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Recorded with no provider keys in the environment, which is what a first run
                looks like. It names the variables, never a value.
              </p>
            </div>
          </div>
        </Band>

        {/* Completion */}
        <Band
          id="completion"
          eyebrow="Shell completion"
          tinted
          title={
            <>
              Completion that <span className="gradient-text">cannot go stale</span>
            </>
          }
          lede={
            <>
              The scripts for {cli.completion_shells.join(", ")} are generated by
              introspecting the live parser and the registries, not maintained by hand — so
              the command list, the{" "}
              <code className="font-mono text-sm">--preset</code> choices and the{" "}
              <code className="font-mono text-sm">--tools</code> names complete to what this
              installation actually has.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <CodeSample
                language="bash"
                accent="#5eead4"
                code={`# Bash — add to ~/.bashrc
eval "$(effgen --completion bash)"
complete -p effgen

# Zsh — add to ~/.zshrc
eval "$(effgen --completion zsh)"

# Fish
effgen --completion fish | source`}
                output={`complete -F _effgen_completion effgen`}
                outputLabel="what complete -p printed after sourcing it"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The generated script carries all {cli.command_count} commands, all{" "}
                {siteData.presets.count} presets and all {siteData.tools.count} tool names.
                Install a plugin that registers a tool and the next shell completes it,
                because the list is read rather than written down.
              </p>
            </div>
            <Terminal
              command={completion.command}
              output={completion.text}
              title="effgen --completion bash"
              maxLines={24}
            />
          </div>
        </Band>

        {/* Documentation */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-green-500/25 bg-green-500/[0.04] p-8 md:p-10 flex flex-col md:flex-row md:items-center gap-6 justify-between">
              <div className="max-w-2xl">
                <h2 className="text-2xl font-black text-gray-900 dark:text-white">
                  The reference for <span className="gradient-text">the command line</span>
                </h2>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  Per-command flags and exit codes, the short-flag table,{" "}
                  <code className="font-mono text-xs">compare</code> and its scoring modes,
                  the report formats, and what each <code className="font-mono text-xs">top</code>{" "}
                  panel measures.
                </p>
              </div>
              <a
                href={DOCS_CLI_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex shrink-0 items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
              >
                <FiFileText size={15} />
                docs/dx/cli.md
                <FiExternalLink size={14} />
              </a>
            </div>
          </Container>
        </section>
      </main>
      <Footer />
    </div>
  );
}
