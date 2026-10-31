// The prose and the tables `/code` renders.
//
// Two rules hold everything here together:
//
//  1. **No frame is written here.** Every terminal frame on the page resolves
//     to a recording in `data/captures/` through `components/captures.ts`, so a
//     plausible-looking session that was never run cannot reach the page.
//  2. **No count is written here.** The slash-command count, the size of the
//     undo journal and the git lists come from `data/effgen.json`, which
//     `scripts/gen_site_data.py` reads off the installed package.
//
// What is written here is the explanation: what each mode gates, what a session
// restores, and what the failures mean. Each of those was checked against
// `effgen code --help` and the framework's `docs/cli/code.md` and, where it is
// observable, against a run.

export const DOCS_CODE_URL =
  "https://github.com/ctrl-gaurav/effGen/blob/main/docs/cli/code.md";

/* ── The loop ── */

export interface LoopStep {
  id: string;
  title: string;
  body: string;
  accent: string;
}

export const loopSteps: LoopStep[] = [
  {
    id: "read",
    title: "Read",
    body:
      "Before the first model call the agent builds an inventory of the workspace: " +
      "the branch, a short git status and a bounded file layout with ignored files " +
      "excluded, plus an AGENTS.md if the workspace has one. Outside a repository the " +
      "same inventory comes from the directory itself.",
    accent: "#00e5ff",
  },
  {
    id: "plan",
    title: "Plan",
    body:
      "The model proposes an approach and the files it wants to touch. In plan mode " +
      "that is where the run stops: the diffs are rendered and nothing is written.",
    accent: "#a78bfa",
  },
  {
    id: "diff",
    title: "Diff",
    body:
      "Every proposed edit is rendered as a unified diff before the write is decided — " +
      "in every mode, including plan mode and including piped output, where the diff " +
      "goes to stderr so stdout keeps carrying only the result.",
    accent: "#00ff88",
  },
  {
    id: "apply",
    title: "Apply",
    body:
      "The permission mode decides what happens next: propose, confirm each, apply " +
      "writes but confirm shell commands, or apply everything. Each applied edit is " +
      "journaled per workspace so it can be reversed.",
    accent: "#ffd700",
  },
  {
    id: "run",
    title: "Run",
    body:
      "Code runs in the sandbox effGen already ships — Docker when its daemon is " +
      "reachable, otherwise a subprocess sandbox that isolates the network and confines " +
      "writes to the workspace. The run reports which of those it actually enforced " +
      "rather than assuming.",
    accent: "#ff9500",
  },
  {
    id: "fix",
    title: "Fix",
    body:
      "The agent reads the real output — not a prediction of it — and iterates until the " +
      "task is done or the iteration cap is reached. A run that spends every step " +
      "without an answer says so, and reports what it reached as partial progress.",
    accent: "#ff6b6b",
  },
];

/* ── Permission modes ── */

export interface PermissionMode {
  flag: string;
  summary: string;
  writes: string;
  sandboxed: string;
  shell: string;
  commit: string;
  accent: string;
}

// The four rows of the table in `docs/cli/code.md`, and the four the `--help`
// epilogue prints. The line each mode prints at the top of a run is in the
// captures, so the wording here can be checked against a recording.
export const permissionModes: PermissionMode[] = [
  {
    flag: "--plan",
    summary: "Propose only. Show the diffs, write nothing, run nothing.",
    writes: "proposed only",
    sandboxed: "no",
    shell: "no",
    commit: "no",
    accent: "#a78bfa",
  },
  {
    flag: "(default)",
    summary: "With a terminal, show each diff and confirm every write and command.",
    writes: "confirm each",
    sandboxed: "confirm each",
    shell: "confirm each",
    commit: "confirm",
    accent: "#00e5ff",
  },
  {
    flag: "--auto-edit",
    summary: "Apply writes and sandboxed runs. Shell commands still need confirming.",
    writes: "applied",
    sandboxed: "applied",
    shell: "confirm each",
    commit: "confirm",
    accent: "#00ff88",
  },
  {
    flag: "--yes",
    summary: "Apply writes, sandboxed runs and shell commands without asking.",
    writes: "applied",
    sandboxed: "applied",
    shell: "applied",
    commit: "applied",
    accent: "#ffd700",
  },
];

/* ── What a session restores ── */

export const sessionRestores: { item: string; state: string; restored: boolean }[] = [
  { item: "The conversation", state: "restored — the next turn can answer from what earlier turns said", restored: true },
  { item: "Files in context", state: "restored, minus any path that no longer exists, which is named", restored: true },
  { item: "Files the session wrote", state: "restored, so a commit still knows its own paths", restored: true },
  { item: "Model and provider", state: "restored when -m was not given, and announced", restored: true },
  { item: "Permission mode", state: "restored only on a terminal and only when no permission flag was given — a stored yes is never restored into a piped run", restored: true },
  { item: "The workspace", state: "not adopted: -w/--workspace or the current directory always wins, and a stored workspace that differs is reported in one line", restored: false },
  { item: "Edits staged by /plan", state: "not stored; the files may have changed underneath, so re-run /plan", restored: false },
  { item: "The undo journal", state: "nothing to restore — it is per workspace and already on disk, so --undo works across restarts", restored: true },
];

/* ── Failure modes ── */

export interface FailureMode {
  id: string;
  title: string;
  body: string;
  reason?: string;
  accent: string;
}

export const failureModes: FailureMode[] = [
  {
    id: "written-tool-call",
    title: "The model describes a call instead of making one",
    body:
      "Some small models answer with the tool call written out as text. Nothing was " +
      "written and nothing ran, so the turn is reported as a failure naming the tool " +
      "whose call was written out — not as an answer describing work that did not " +
      "happen. An answer that recaps a call the run really made keeps its result.",
    reason: "written_tool_call",
    accent: "#ff9500",
  },
  {
    id: "iteration-cap",
    title: "The run stops at its iteration cap",
    body:
      "A run that spends every step without writing a final answer has no answer to " +
      "report. The run states what stopped it and what to do about it, and whatever it " +
      "had reached is reported separately as partial progress — tool output and " +
      "reasoning, never presented as a result.",
    reason: "max_iterations_partial · max_iterations_exhausted",
    accent: "#ff6b6b",
  },
  {
    id: "answer-source",
    title: "The answer is not what the model wrote",
    body:
      "A model that keeps repeating the same call, or returns no final answer, leaves " +
      "the loop with nothing to hand back but the last tool result or its own recovered " +
      "text. The run completed, so the footer says so — and the line under it names " +
      "where the answer came from instead of passing a tool result off as an answer.",
    reason: "loop_detected · repeated_tool_result · null_final_from_model",
    accent: "#a78bfa",
  },
  {
    id: "hunk",
    title: "A hunk no longer applies",
    body:
      "When a file changed underneath a staged edit, the hunks that still match are " +
      "applied and the rest are reported by name. The file is not overwritten with a " +
      "stale version, so work done outside the session is not lost.",
    accent: "#00e5ff",
  },
  {
    id: "withheld",
    title: "There was nobody to confirm",
    body:
      "Without a terminal there is nobody to answer a prompt, so the default becomes " +
      "plan mode: the run reports what it would do and writes nothing. A run that " +
      "completed but withheld its changes for that reason exits 2, which includes a " +
      "commit that could not be confirmed.",
    accent: "#ffd700",
  },
];

export const exitCodes: { code: string; meaning: string }[] = [
  { code: "0", meaning: "Completed." },
  { code: "1", meaning: "Failed." },
  {
    code: "2",
    meaning:
      "Completed, but the changes were withheld because there was no terminal to " +
      "confirm on and neither --auto-edit nor --yes was given. A --commit that could " +
      "not be confirmed exits here too.",
  },
];

/* ── Review targets ── */

export const reviewTargets: { target: string; means: string }[] = [
  { target: "uncommitted", means: "the default — git diff HEAD, staged and unstaged together" },
  { target: "staged", means: "git diff --cached" },
  { target: "HEAD~3", means: "any revision git accepts" },
  { target: "main...HEAD", means: "any range git accepts" },
  { target: "-f PATH", means: "one file in full, repeatable — works with a target or on its own, which is how a directory that is not a repository is reviewed" },
];

/* ── The environment the run reads ── */

export const environment: { name: string; description: string }[] = [
  {
    name: "EFFGEN_WORKSPACE",
    description:
      "The directory the agent reads and writes, and the only one sandboxed code may " +
      "write to (created if missing). Unset: the current directory. -w/--workspace " +
      "overrides it.",
  },
  {
    name: "EFFGEN_SANDBOX_BACKEND",
    description:
      "docker or subprocess. Docker confines the filesystem and network for executed " +
      "code; the subprocess fallback isolates the network and confines writes to the " +
      "workspace, leaving the rest of the filesystem readable but read-only.",
  },
];
