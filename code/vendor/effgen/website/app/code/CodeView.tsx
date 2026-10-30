"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiArrowRight,
  FiCheck,
  FiExternalLink,
  FiGitBranch,
  FiRotateCcw,
  FiShield,
  FiSliders,
  FiTerminal,
  FiX,
} from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Terminal from "@/components/ui/Terminal";
import TerminalReplay from "@/components/ui/TerminalReplay";
import ParamTable from "@/components/ui/ParamTable";
import CodeSample from "@/components/ui/CodeSample";
import { codeCapture, codeDocument } from "@/components/captures";
import { siteData, version } from "@/components/siteData";
import {
  DOCS_CODE_URL,
  environment,
  exitCodes,
  failureModes,
  loopSteps,
  permissionModes,
  reviewTargets,
  sessionRestores,
} from "./codeData";
import { accentTextStyle } from "@/components/accentText";

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

const code = siteData.code;

/* ── A band, in the rhythm the landing sections already use ── */

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
      className={`py-16 relative scroll-mt-24 ${
        tinted ? "bg-gray-50 dark:bg-[#030f07]" : ""
      }`}
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
          {lede && (
            <p className="mt-4 text-gray-600 dark:text-gray-400 leading-relaxed">{lede}</p>
          )}
        </motion.div>
        {children}
      </Container>
    </section>
  );
}

export default function CodeView() {
  const { ref: heroRef, inView: heroInView } = useInView({ triggerOnce: true, threshold: 0.05 });

  const loop = codeCapture("code-loop");
  const plan = codeCapture("code-plan");
  const session = codeCapture("code-session");
  const review = codeCapture("code-review");
  const reviewNothing = codeCapture("code-review-nothing");
  const undo = codeCapture("code-undo");
  const failedActions = codeCapture("code-failed-actions");
  const twoModes = codeCapture("code-two-modes");
  const reviewNoRepo = codeCapture("code-review-norepo");
  const gitRefusals = codeCapture("code-git-refusals");
  const jsonDoc = codeDocument("code-json");

  const jsonKeys = Object.keys(jsonDoc.document).sort();

  const headline = [
    { value: String(permissionModes.length), label: "Permission modes", accent: "#00ff88", icon: FiShield },
    { value: String(code.slash_command_count), label: "Slash commands", accent: "#00e5ff", icon: FiTerminal },
    { value: String(code.undo_journal_entries), label: "Edits you can undo", accent: "#a78bfa", icon: FiRotateCcw },
    { value: String(code.git_refused.length), label: "Git commands refused", accent: "#ff9500", icon: FiGitBranch },
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
                <FiTerminal size={14} />
                effgen code · {version}
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                A coding agent that shows you{" "}
                <span className="gradient-text">every change as a diff</span> before it
                touches disk
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                <code className="font-mono text-green-700 dark:text-green-400">effgen code</code>{" "}
                runs a plan, act and observe loop over one directory: it proposes an
                approach, writes files, executes code in a sandbox, reads the real output
                and iterates. Every write and every command passes a permission gate you
                choose, every applied edit can be reversed, and the files it edits stay
                inside the workspace root.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <a
                  href="#loop"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
                >
                  Watch a real session
                  <FiArrowRight size={15} />
                </a>
                <a
                  href={DOCS_CODE_URL}
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

        {/* Headline figures */}
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

        {/* The centrepiece: a real session, replayed */}
        <Band
          id="loop"
          eyebrow="The loop"
          title={
            <>
              One task, <span className="gradient-text">start to finish</span>
            </>
          }
          lede={
            <>
              This is a recording of a real run against a four-file Python package with an{" "}
              <code className="font-mono text-sm">AGENTS.md</code>, on a clean git tree. It
              is played back a line at a time; the whole transcript is in the page from the
              first frame, so it can be selected, copied and read at any point.
            </>
          }
        >
          <TerminalReplay
            command={loop.command}
            output={loop.text}
            title="effgen code"
            rows={24}
            caption={
              <>
                Three files written, each shown as a unified diff first, and the tests run
                in the sandbox. The agent reports that the test process was killed by the
                sandbox&rsquo;s memory limit on this machine rather than claiming a pass —
                run outside the sandbox, the three tests pass. Recorded against effGen{" "}
                {version}.
              </>
            }
          />

          <div className="mt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {loopSteps.map((step, index) => (
              <article
                key={step.id}
                className="relative rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-5"
              >
                <div
                  className="absolute left-0 top-5 bottom-5 w-0.5 rounded-full"
                  style={{ background: step.accent }}
                />
                <div className="pl-3">
                  <span
                    className="text-xs font-mono font-bold tabular-nums"
                    style={accentTextStyle(step.accent)}
                  >
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <h3 className="mt-1 text-base font-bold text-gray-900 dark:text-white">
                    {step.title}
                  </h3>
                  <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {step.body}
                  </p>
                </div>
              </article>
            ))}
          </div>
        </Band>

        {/* Permission modes */}
        <Band
          id="permissions"
          eyebrow="Permission modes"
          tinted
          title={
            <>
              You decide <span className="gradient-text">what it may do</span>
            </>
          }
          lede={
            <>
              Pick at most one. Naming two is an error rather than a silent precedence
              rule. Without a terminal there is nobody to confirm, so the default becomes
              plan mode and the run writes nothing unless you opt in.
            </>
          }
        >
          <div className="overflow-x-auto rounded-2xl border border-gray-200 dark:border-green-500/15">
            <table className="w-full text-left text-sm border-collapse">
              <caption className="sr-only">
                What each permission mode allows, for file writes, sandboxed runs, shell
                commands and commits
              </caption>
              <thead>
                <tr className="bg-gray-50 dark:bg-[#071a10]">
                  <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300">Mode</th>
                  <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300">Writes</th>
                  <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300">Sandboxed runs</th>
                  <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300">Shell commands</th>
                  <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300">Commit</th>
                </tr>
              </thead>
              <tbody>
                {permissionModes.map((mode) => (
                  <tr
                    key={mode.flag}
                    className="border-t border-gray-200 dark:border-green-500/10 align-top"
                  >
                    <th scope="row" className="px-4 py-3 text-left font-normal">
                      <code
                        className="font-mono text-[13px] font-semibold"
                        style={accentTextStyle(mode.accent)}
                      >
                        {mode.flag}
                      </code>
                      <span className="block mt-1 text-xs text-gray-600 dark:text-gray-400 max-w-xs">
                        {mode.summary}
                      </span>
                    </th>
                    <td className="px-4 py-3 text-gray-600 dark:text-gray-400 whitespace-nowrap">{mode.writes}</td>
                    <td className="px-4 py-3 text-gray-600 dark:text-gray-400 whitespace-nowrap">{mode.sandboxed}</td>
                    <td className="px-4 py-3 text-gray-600 dark:text-gray-400 whitespace-nowrap">{mode.shell}</td>
                    <td className="px-4 py-3 text-gray-600 dark:text-gray-400 whitespace-nowrap">{mode.commit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                Plan mode, on the same package
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                The diff is rendered, the write is withheld with the reason, and the footer
                counts what was proposed. Nothing on disk changed: the directory held the
                same four files before and after.
              </p>
              <Terminal command={plan.command} output={plan.text} title="effgen code --plan" maxLines={26} />
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                Naming two modes stops the run
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Rather than picking one for you, the command says which flags conflict and
                exits. The same holds for <code className="font-mono text-xs">--review</code>,
                which cannot be combined with a permission flag,{" "}
                <code className="font-mono text-xs">--commit</code> or{" "}
                <code className="font-mono text-xs">--undo</code> — each asks for something
                a read-only run will not do.
              </p>
              <Terminal command={twoModes.command} output={twoModes.text} title="conflicting flags" />

              <h3 className="mt-8 text-base font-bold text-gray-900 dark:text-white mb-3">
                The environment the run reads
              </h3>
              <ParamTable
                nameLabel="Variable"
                params={environment.map((item) => ({
                  name: item.name,
                  description: item.description,
                }))}
              />
            </div>
          </div>
        </Band>

        {/* Diffs and undo */}
        <Band
          id="undo"
          eyebrow="Diffs and undo"
          title={
            <>
              Every applied edit is <span className="gradient-text">reversible</span>
            </>
          }
          lede={
            <>
              Applied edits are journaled per workspace — the last{" "}
              {code.undo_journal_entries} of them — so a change can be reversed after the
              session has ended and after the shell has closed. A restored file returns to
              its previous content; a file the run created is removed.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <Terminal command={undo.command} output={undo.text} title="effgen code --undo" />
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The journal is per workspace and lives on disk, so it survives a restart —
                there is nothing about it for a session to restore.{" "}
                <code className="font-mono text-xs">--undo-count N</code> reverses several,
                and <code className="font-mono text-xs">/undo [n]</code> does the same
                inside a session.
              </p>
            </div>
            <div className="space-y-4">
              <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5">
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Staging edits without writing them
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  In a session, <code className="font-mono text-xs">/plan</code> stages
                  edits without writing them,{" "}
                  <code className="font-mono text-xs">/diff</code> shows what is staged,{" "}
                  <code className="font-mono text-xs">/apply [n]</code> writes them — all of
                  them, or one by number — and{" "}
                  <code className="font-mono text-xs">/reject [n]</code> discards them.
                </p>
              </div>
              <div className="rounded-2xl border border-cyan-500/25 bg-cyan-500/[0.04] p-5">
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  When a hunk no longer applies
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  A staged edit is prepared against the file as it was. If the file changed
                  underneath, the hunks that still match are applied and the rest are
                  reported by name. The file is never overwritten with a stale version, so
                  an edit you made outside the session is not silently lost.
                </p>
              </div>
            </div>
          </div>
        </Band>

        {/* Review */}
        <Band
          id="review"
          eyebrow="Review"
          tinted
          title={
            <>
              A review holds <span className="gradient-text">no tool that writes</span>
            </>
          }
          lede={
            <>
              <code className="font-mono text-sm">--review</code> is not a permission mode
              with a different prompt. The run holds no tool that writes a file, runs code
              or runs a shell command: the file tool is narrowed to its reading operations
              and its schema carries no write, and git is the read-only surface, pinned to
              the workspace. The permission gate still stands behind them, so a review is
              read-only by the tools it holds <em>and</em> by the mode it runs in.
            </>
          }
        >
          <Terminal
            command={review.command}
            output={review.text}
            title="effgen code --review"
            maxLines={30}
          />
          <p className="mt-3 text-sm text-gray-600 dark:text-gray-400">
            The change under review is handed to the model as context, because the only
            route to a diff would be a shell and a read-only run has none. A diff over the
            budget is truncated with the cut marked and the remainder counted, never
            silently. The record reports{" "}
            <code className="font-mono text-xs">&quot;read_only&quot;: true</code> and a{" "}
            <code className="font-mono text-xs">review</code> block naming the target, and{" "}
            <code className="font-mono text-xs">files_written</code> is always empty.
          </p>

          <div className="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                What you can point it at
              </h3>
              <ParamTable
                nameLabel="Target"
                params={reviewTargets.map((target) => ({
                  name: target.target,
                  description: target.means,
                }))}
                caption="effgen code --review [TARGET] · -f/--file is repeatable"
              />
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                And when there is nothing to review
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Outside a repository with no <code className="font-mono text-xs">-f</code>,
                the run exits 1 naming the three ways to give it a subject rather than
                reviewing something else. On a clean tree it says so instead of returning
                an empty review.
              </p>
              <Terminal command={reviewNoRepo.command} output={reviewNoRepo.text} title="not a repository" />
              <div className="mt-4">
                <Terminal command={reviewNothing.command} output={reviewNothing.text} title="clean tree" />
              </div>
            </div>
          </div>
        </Band>

        {/* The interactive session */}
        <Band
          id="session"
          eyebrow="The interactive session"
          title={
            <>
              {code.slash_command_count} slash commands,{" "}
              <span className="gradient-text">and a session you can leave</span>
            </>
          }
          lede={
            <>
              On a terminal with no task,{" "}
              <code className="font-mono text-sm">effgen code</code> opens a session. Type{" "}
              <code className="font-mono text-sm">/</code> on its own for the menu; tab
              completes command names. <code className="font-mono text-sm">effgen chat</code>{" "}
              has its own {code.chat_slash_command_count}.
            </>
          }
        >
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 items-start">
            <Terminal
              command={session.command}
              output={session.text}
              title="effgen code"
              maxLines={34}
            />
            <div>
              <div
                className="overflow-x-auto rounded-2xl border border-gray-200 dark:border-green-500/15 max-h-[34em] overflow-y-auto focus-visible:outline focus-visible:outline-2 focus-visible:outline-green-500 focus-visible:-outline-offset-2"
                tabIndex={0}
                role="group"
                aria-label="Slash commands table"
              >
                <table className="w-full text-left text-sm border-collapse">
                  <caption className="sr-only">
                    Every slash command a coding session accepts, and what each does
                  </caption>
                  <thead className="sticky top-0">
                    <tr className="bg-gray-50 dark:bg-[#071a10]">
                      <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300">Command</th>
                      <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300">Does</th>
                    </tr>
                  </thead>
                  <tbody>
                    {code.slash_commands.map((command) => (
                      <tr
                        key={command.name}
                        className="border-t border-gray-200 dark:border-green-500/10 align-top"
                      >
                        <th scope="row" className="px-4 py-2.5 text-left">
                          <code className="font-mono text-[13px] text-green-700 dark:text-green-400">
                            {command.name}
                          </code>
                        </th>
                        <td className="px-4 py-2.5 text-gray-600 dark:text-gray-400">
                          {command.summary}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-3 text-xs text-gray-600 dark:text-gray-400">
                Read from the session&rsquo;s own command table in effGen {version}.
              </p>
            </div>
          </div>

          <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-6">
              <h3 className="text-base font-bold text-gray-900 dark:text-white">
                What a turn shows while it runs
              </h3>
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                An interactive turn shows a status line naming the tool in flight, each
                proposed edit&rsquo;s diff before it is written, and a tick per decided
                action. On a provider that streams its tool calls the answer is written to
                the screen as the model produces it, rather than arriving in one block when
                the turn ends. The status line and the answer take turns owning the
                terminal: the status line runs while the model is thinking and dispatching
                tools, and hands over from the first word of the answer.
              </p>
            </div>
            <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-6">
              <h3 className="text-base font-bold text-gray-900 dark:text-white">
                And where it does not stream
              </h3>
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                A model whose tool calls are not streamed — the local engines among them —
                prints its answer once the turn finishes, and so does every non-interactive
                surface: piped output, <code className="font-mono text-xs">--json</code>,{" "}
                <code className="font-mono text-xs">-q</code>,{" "}
                <code className="font-mono text-xs">--no-animation</code> and{" "}
                <code className="font-mono text-xs">NO_COLOR</code>. Those three also render
                plain text with no escape codes, which is what makes a captured session
                readable as text rather than as a screenful of control characters.
              </p>
            </div>
          </div>

          <div className="mt-12">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
              Continuing a session
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-6 max-w-3xl leading-relaxed">
              <code className="font-mono text-xs">--session-id ID</code> (or{" "}
              <code className="font-mono text-xs">--resume ID</code>) continues a stored
              session — the same store as{" "}
              <code className="font-mono text-xs">effgen chat --session-id</code> and{" "}
              <code className="font-mono text-xs">effgen sessions list</code>. What it
              restores, and what it deliberately does not:
            </p>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {sessionRestores.map((item) => (
                <li
                  key={item.item}
                  className="flex gap-3 rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-4"
                >
                  <span className="mt-0.5 shrink-0" aria-hidden="true">
                    {item.restored ? (
                      <FiCheck className="text-green-700 dark:text-green-400" size={16} />
                    ) : (
                      <FiX className="text-orange-700 dark:text-orange-400" size={16} />
                    )}
                  </span>
                  <span>
                    <span className="block text-sm font-semibold text-gray-900 dark:text-white">
                      {item.item}
                      <span className="sr-only">
                        {item.restored ? " — restored" : " — not restored"}
                      </span>
                    </span>
                    <span className="block mt-1 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                      {item.state}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Band>

        {/* Repository awareness */}
        <Band
          id="git"
          eyebrow="Repository awareness"
          tinted
          title={
            <>
              It reads your repository. It{" "}
              <span className="gradient-text">will not rewrite it</span>
            </>
          }
          lede={
            <>
              In a git repository the branch, a short status and a bounded file layout
              (ignored files excluded) are read before the first model call and become part
              of the agent&rsquo;s context, along with an{" "}
              <code className="font-mono text-sm">AGENTS.md</code> in the workspace if there
              is one. The single repository change a session can make is a commit of the
              files it wrote, after an explicit confirmation.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <Terminal
                command={gitRefusals.command}
                output={gitRefusals.text}
                title="the git allow-list"
                maxLines={22}
              />
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The refusal is not a prompt-level instruction. Every git command a session
                can reach passes the same check, and it reads the command line the model
                asked for — so <code className="font-mono text-xs">git push</code> is
                refused whether it is asked for directly, wrapped in{" "}
                <code className="font-mono text-xs">bash -c</code>, or spawned from a Python
                one-liner.
              </p>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border border-green-500/25 bg-green-500/[0.04] p-5">
                <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-3">
                  It may run
                </h3>
                <ul className="flex flex-wrap gap-2">
                  {code.git_allowed.map((name) => (
                    <li
                      key={name}
                      className="px-2.5 py-1 rounded-lg text-xs font-mono bg-green-500/10 text-green-800 dark:text-green-400 border border-green-500/25"
                    >
                      git {name}
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                  The commit is limited to the paths the run wrote, so work you had staged
                  for other files stays staged and out of it. The plan — repository, exact
                  paths, message — is printed before the confirmation.
                </p>
              </div>

              <div className="rounded-2xl border border-orange-500/25 bg-orange-500/[0.04] p-5">
                <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-3">
                  It refuses, in every mode
                </h3>
                <ul className="flex flex-wrap gap-2">
                  {code.git_refused.map((name) => (
                    <li
                      key={name}
                      className="px-2.5 py-1 rounded-lg text-xs font-mono bg-orange-500/10 text-orange-800 dark:text-orange-400 border border-orange-500/25"
                    >
                      {name}
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                  Along with {code.git_refused_flags.length} flags wherever they appear:{" "}
                  <span className="font-mono">{code.git_refused_flags.join(", ")}</span>.
                </p>
              </div>
            </div>
          </div>
        </Band>

        {/* Scripting */}
        <Band
          id="scripting"
          eyebrow="Scripting"
          title={
            <>
              One JSON document on stdout,{" "}
              <span className="gradient-text">everything else on stderr</span>
            </>
          }
          lede={
            <>
              Piped or with <code className="font-mono text-sm">--json</code>, stdout carries
              only the result — the answer text, or one JSON document — and everything a
              human reads goes to stderr. That is what makes the command safe to put in a
              pipeline.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <CodeSample
                language="bash"
                accent="#00e5ff"
                code={`# the task as an argument, the result as one JSON document
effgen code -p "what does this package export?" --plan --json \\
  -m gemini:gemini-3.1-flash-lite < /dev/null | jq -r .permission_mode

# a failing log becomes context in front of the task
cat pytest.log | effgen code -p "why did this fail?" --plan --json \\
  -m gemini:gemini-3.1-flash-lite | jq -r .tool_calling

# with no task at all, the piped text is the task
echo "what does textkit/wrap.py do?" | effgen code --plan --json \\
  -m gemini:gemini-3.1-flash-lite | jq -r .task

# nothing to pipe in? say so, or the read to EOF holds the run
effgen code -p "write fib.py with fib(n) and print fib(10)" -w ../ws \\
  --auto-edit --json -m openai:gpt-5-mini < /dev/null | jq -r ".files_written[]"`}
                output={`plan
hybrid
what does textkit/wrap.py do?
fib.py`}
                outputLabel="what the four commands printed, in order"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                When stdin is not a terminal it is read to EOF before the run starts, which
                is what lets a producer that writes slowly — a build log,{" "}
                <code className="font-mono text-xs">tail -f</code> — be folded in whole. The
                consequence is that a pipe which never closes holds the run, so after about
                two seconds the command prints a line to stderr naming what it is waiting
                for and how to skip it.
              </p>
            </div>

            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                What the document carries
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                The {jsonKeys.length} keys below are the top level of the document{" "}
                <code className="font-mono text-xs">{jsonDoc.command}</code> printed.
                Every proposed edit appears in{" "}
                <code className="font-mono text-xs">diffs</code>; the ones that reached disk
                carry <code className="font-mono text-xs">&quot;applied&quot;: true</code>,
                so <code className="font-mono text-xs">--plan --json</code> reports the
                changes it would make without writing any of them.
              </p>
              <ul className="flex flex-wrap gap-2">
                {jsonKeys.map((key) => (
                  <li
                    key={key}
                    className="px-2.5 py-1 rounded-lg text-xs font-mono bg-gray-100 dark:bg-gray-900 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-800"
                  >
                    {key}
                  </li>
                ))}
              </ul>
              <dl className="mt-6 space-y-3 text-sm">
                <div>
                  <dt className="font-semibold text-gray-900 dark:text-white font-mono text-xs">tool_calling</dt>
                  <dd className="text-gray-600 dark:text-gray-400">
                    Names the path the model&rsquo;s tool calls travelled on. hybrid and
                    native send the definitions to the provider&rsquo;s tool-calling API;
                    react reads the calls out of the model&rsquo;s text.
                  </dd>
                </div>
                <div>
                  <dt className="font-semibold text-gray-900 dark:text-white font-mono text-xs">answer_source</dt>
                  <dd className="text-gray-600 dark:text-gray-400">
                    Names where the answer came from when the loop had to recover one, and
                    is empty when the model wrote it.
                  </dd>
                </div>
                <div>
                  <dt className="font-semibold text-gray-900 dark:text-white font-mono text-xs">coding_suitability</dt>
                  <dd className="text-gray-600 dark:text-gray-400">
                    Carries the verdict on the chosen model — suitable, limited, unsuitable,
                    or unknown when the catalog does not know the id.
                  </dd>
                </div>
                <div>
                  <dt className="font-semibold text-gray-900 dark:text-white font-mono text-xs">actions</dt>
                  <dd className="text-gray-600 dark:text-gray-400">
                    The full log of what was allowed, withheld, declined or refused, and
                    why.
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        </Band>

        {/* Failure */}
        <Band
          id="failures"
          eyebrow="When it does not work"
          tinted
          title={
            <>
              A run that did not do the work{" "}
              <span className="gradient-text">says so</span>
            </>
          }
          lede={
            <>
              A coding agent that reports success for a turn in which nothing was written is
              worse than one that fails, because the failure is invisible until someone
              reads the diff. These are the states that get their own report rather than
              being rounded up to an answer.
            </>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {failureModes.map((mode) => (
              <article
                key={mode.id}
                className="relative rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-5"
              >
                <div
                  className="absolute left-0 top-5 bottom-5 w-0.5 rounded-full"
                  style={{ background: mode.accent }}
                />
                <div className="pl-3">
                  <h3 className="text-base font-bold text-gray-900 dark:text-white">
                    {mode.title}
                  </h3>
                  <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {mode.body}
                  </p>
                  {mode.reason && (
                    <code
                      className="mt-3 block text-[11px] font-mono break-all"
                      style={accentTextStyle(mode.accent)}
                    >
                      {mode.reason}
                    </code>
                  )}
                </div>
              </article>
            ))}
          </div>

          <div className="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                A failed action stays a failed action
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                In this recording the agent tried to run the test suite twice and both
                attempts failed — once because the sandbox killed the process, once because
                the code it wrote was not valid on its own. Both are marked as failures in
                the action list, and the footer names where the answer came from because the
                model never wrote one.
              </p>
              <Terminal
                command={failedActions.command}
                output={failedActions.text}
                title="a run with failed actions"
                maxLines={26}
              />
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">Exit codes</h3>
              <ParamTable
                nameLabel="Code"
                params={exitCodes.map((exit) => ({
                  name: exit.code,
                  description: exit.meaning,
                }))}
              />
              <div className="mt-6 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5">
                <h4 className="text-sm font-bold text-gray-900 dark:text-white">
                  Before the first call, not after the last
                </h4>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  Not every model can complete a coding turn: some receive no tool
                  definitions at all because their chat template renders none, some answer
                  the question directly, and some write the call out as prose. Each of those
                  finishes with an answer, no files written and exit 0 — which reads like
                  success. So the command says one line up front when the chosen model is a
                  poor fit, carrying the date it was measured. It never blocks the run,{" "}
                  <code className="font-mono text-xs">-q</code> suppresses it, and{" "}
                  <code className="font-mono text-xs">effgen models info</code> shows the
                  same verdict as a Coding row.
                </p>
              </div>
            </div>
          </div>
        </Band>

        {/* Options */}
        <Band
          id="options"
          eyebrow="Options"
          title={
            <>
              Every flag <span className="gradient-text">effgen code</span> takes
            </>
          }
          lede={
            <>
              Read out of <code className="font-mono text-sm">effgen code --help</code> in
              effGen {version}, in the order it prints them, with its wording rather than a
              paraphrase.
            </>
          }
        >
          <ParamTable
            nameLabel="Flag"
            params={[
              {
                name: "task",
                description:
                  "What to build, change or debug (omit on a terminal to open an interactive session)",
              },
              ...code.options.map((option) => ({
                name: option.name,
                description: option.description,
              })),
            ]}
            caption={`effgen code --help · effGen ${version}`}
          />

          <div className="mt-8 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-6">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-2">
              One short flag to watch
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
              <code className="font-mono text-xs">-p</code> means{" "}
              <code className="font-mono text-xs">--print</code> here and{" "}
              <code className="font-mono text-xs">--port</code> on{" "}
              <code className="font-mono text-xs">serve</code>,{" "}
              <code className="font-mono text-xs">monitor</code> and{" "}
              <code className="font-mono text-xs">top</code>, so{" "}
              <code className="font-mono text-xs">effgen code -p 8000</code> and{" "}
              <code className="font-mono text-xs">effgen serve -p 8000</code> mean unrelated
              things. In a script, prefer the long spelling. The binding is frozen and a
              test pins it, so a third meaning cannot be added quietly.
            </p>
          </div>
        </Band>

        {/* Documentation */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-green-500/25 bg-green-500/[0.04] p-8 md:p-10 flex flex-col md:flex-row md:items-center gap-6 justify-between">
              <div className="max-w-2xl">
                <h2 className="text-2xl font-black text-gray-900 dark:text-white">
                  The reference for <span className="gradient-text">effgen code</span>
                </h2>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  The workspace resolution order, the sandbox states and what each enforces,
                  the full session table, the review contract, and the JSON document field
                  by field.
                </p>
              </div>
              <a
                href={DOCS_CODE_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex shrink-0 items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
              >
                <FiSliders size={15} />
                docs/cli/code.md
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
