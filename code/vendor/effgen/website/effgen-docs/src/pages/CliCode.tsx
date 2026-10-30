import { Code2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { siteData } from '../siteData';

const codeOptions = siteData.code.options;
const slashCommands = siteData.code.slash_commands;
const { git_allowed, git_refused, git_refused_flags, undo_journal_entries } = siteData.code;

export default function CliCode() {
  return (
    <DocPage
      subtitle="A coding agent in the terminal: it proposes, writes, runs the code and reads the real output — showing every change as a unified diff before it touches disk."
      icon={<Code2 size={48} />}
    >
      <p>
        <code>effgen code</code> runs a plan → act → observe loop over one directory. The model
        proposes an approach, writes files, executes code in the sandbox, reads what actually came
        back, and iterates until the task is done or the iteration cap is reached. Every file change
        is rendered as a unified diff <em>before</em> it is written, every write and command passes
        a permission gate, and nothing outside the workspace is touched.
      </p>

      <h2>The shortest useful run</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen code "write fib.py with fib(n) and print fib(10)" -w ws -m openai:gpt-5-nano --auto-edit`}
      />

      <Terminal
        command={`effgen code -p "write fib.py with fib(n) and print fib(10)" -w ws -m openai:gpt-5-nano --auto-edit`}
        output={`Workspace: /data/wang/gks/effgen-d6/ws
No git repository · 0 file(s)
Permissions: auto-edit — apply writes and sandboxed runs; confirm shell commands

new file fib.py (+10/-0)
--- a/fib.py
+++ b/fib.py
@@ -0,0 +1,10 @@
+def fib(n):
+    if n < 0:
+        raise ValueError("n must be non-negative")
+    a, b = 0, 1
+    for _ in range(n):
+        a, b = b, a + b
+    return a
+
+if __name__ == "__main__":
+    print(fib(10))
✓ Done in 24.7s · 2 tools · 8,604 tokens · $0.0019
Tool calling: hybrid — provider tool API first, falling back to the text
Files written in /data/wang/gks/effgen-d6/ws: fib.py
Result: fib(10) = 55

Observed output when running the provided script/invocation:
55

Conclusion: The fib(n) implementation works for n = 10, returning 55 as expected.`}
        maxLines={26}
        caption={
          <>
            <code>55</code> is not the model's claim about what the code would print — it is what
            the code printed when the run executed it in the sandbox.
          </>
        }
      />

      <p>
        With a task, <code>-p</code>, piped stdin or <code>--json</code>, the command runs once and
        exits. On a terminal with no task it opens an interactive session instead.
      </p>

      <h2>Permission modes</h2>

      <p>
        Pick at most one. Naming two is an error rather than a silent precedence rule.
      </p>

      <ApiTable
        headers={['Mode', 'Writes', 'Sandboxed runs', 'Shell commands', 'Commit']}
        rows={[
          [<code>--plan</code>, 'proposed only', 'no', 'no', 'no'],
          ['default, with a terminal', 'confirm each', 'confirm each', 'confirm each', 'confirm'],
          [<code>--auto-edit</code>, 'applied', 'applied', 'confirm each', 'confirm'],
          [<code>-y, --yes</code>, 'applied', 'applied', 'applied', 'applied'],
        ]}
      />

      <p>
        Without a terminal there is nobody to confirm, so <strong>the default becomes{' '}
        <code>--plan</code></strong>: the run reports what it would do and writes nothing. A scripted
        run that is meant to change files has to opt in with <code>--auto-edit</code> or{' '}
        <code>--yes</code>. A run that completed but withheld its changes for that reason exits{' '}
        <code>2</code>.
      </p>

      <Terminal
        command={`effgen code -p "add a docstring to hello.py" -w ws -m gemini:gemini-3.1-flash-lite < /dev/null; echo "rc=$?"`}
        output={`Workspace: /data/wang/gks/effgen-d6/ws
No git repository · 1 file(s)
Permissions: plan — propose only; no file is written and no command runs
edit hello.py (+3/-0)
--- a/hello.py
+++ b/hello.py
@@ -1 +1,4 @@
+"""
+This module prints hello.
+"""
 print("hello")
✓ Done in 2.2s · 2 tools · 4,237 tokens · $0.0012
Tool calling: hybrid — provider tool API first, falling back to the text
No changes were made: 1 action(s) (write) needed confirmation and this session has no terminal to
confirm on. Re-run with --auto-edit (or --yes) to allow them without asking.
rc=2`}
        maxLines={18}
        caption={
          <>
            No terminal, no permission flag: the diff is still shown, nothing is written, and the
            exit code says why. That <code>2</code> is what a CI job should treat as "the change was
            never applied", distinct from <code>1</code>, which means the run failed.
          </>
        }
      />

      <ApiTable
        headers={['Exit code', 'Means']}
        rows={[
          [<code>0</code>, 'Completed.'],
          [<code>1</code>, 'Failed.'],
          [
            <code>2</code>,
            <>
              Completed, but changes were withheld because there was no terminal to confirm on —
              including a <code>--commit</code> that could not be confirmed.
            </>,
          ],
        ]}
      />

      <h2>The workspace</h2>

      <p>The agent reads, edits and runs in exactly one directory, chosen in this order:</p>

      <ol>
        <li>
          <code>-w/--workspace DIR</code>, created if it does not exist;
        </li>
        <li>
          otherwise <code>EFFGEN_WORKSPACE</code>;
        </li>
        <li>otherwise the directory the command was started in.</li>
      </ol>

      <p>
        A path outside that root is refused with the path named, not silently redirected. The
        deny-list for credential files and sensitive locations — <code>~/.ssh</code>,{' '}
        <code>~/.aws</code>, <code>.env</code> — applies inside the workspace too.
      </p>

      <h3>What the sandbox actually enforces</h3>

      <p>
        Executed code runs in the sandbox the framework already ships: Docker where its daemon is
        reachable, otherwise a subprocess sandbox that isolates the network and confines writes to
        the workspace, leaving the rest of the filesystem readable but read-only.{' '}
        <code>EFFGEN_SANDBOX_BACKEND=docker|subprocess</code> pins one. Only Docker also confines{' '}
        <em>reads</em>.
      </p>

      <p>
        Confinement is not assumed. The sandbox proves it can hold before claiming it, and reports
        what each run actually enforced — a host that allows the namespaces but refuses the locked
        read-only mounts keeps network isolation and says writes are not confined; a host with no
        unprivileged user namespaces at all isolates neither, and says so. On the subprocess
        backend the run prints what it can and cannot do before starting:
      </p>

      <Terminal
        title="sandbox warning"
        output={`┌─────────────────────────────────────────────────────────────┐
│  effGen SANDBOX WARNING                                      │
│                                                              │
│  Code execution is using SubprocessSandbox, which provides   │
│  PARTIAL isolation only.                                     │
│                                                              │
│  Limitations:                                                │
│  • Executed code can READ any host file the calling          │
│    process's user can read — reads are NOT confined          │
│  • Writes are confined to the run's working directory only   │
│    where the kernel allows it; each result reports what was  │
│    enforced in 'filesystem_confined' / 'writable_root'       │
│  • Network isolation via unshare (may require privileges)    │
│  • Memory limit is advisory, not hard-enforced               │
│                                                              │
│  To confine reads as well, install Docker and ensure the     │
│  daemon is running and accessible by the current user.       │
└─────────────────────────────────────────────────────────────┘`}
        maxLines={20}
        caption={
          <>
            Printed on the run above, on a host with no Docker daemon.{' '}
            <Link to="/execution">Code execution and the sandbox</Link> covers the backends;{' '}
            <code>effgen doctor</code> names which of the three states applies here.
          </>
        }
      />

      <h2>Reviewing instead of changing</h2>

      <p>
        <code>--review</code> is not a permission mode with a stricter prompt. The run{' '}
        <strong>holds no tool that writes a file, runs code or runs a shell command</strong>: the
        file tool is narrowed to its reading operations — its schema carries no <code>write</code> —
        and <code>git</code> is the read-only surface, pinned to the workspace. The permission gate
        still stands behind them, so a review is read-only by the tools it holds <em>and</em> by the
        mode it runs in.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen code --review                                   # everything not committed
effgen code --review staged -p "is this safe to merge?"
effgen code --review HEAD~3
effgen code --review main...HEAD --json | jq -r .answer
effgen code -f src/app.py -f src/db.py -p "where can this raise?"`}
      />

      <Terminal
        command={`effgen code --review -m gemini:gemini-3.1-flash-lite --max-tokens 8192`}
        output={`Workspace: /data/wang/gks/effgen-d6/repo
Repo: repo · branch master · 1 change(s) · 1 file(s)
Reviewing uncommitted changes, 15 line(s)
Read-only review: no file is written and no command runs.
✓ Done in 3.1s · 1 tool · 2,479 tokens · $0.0010
Tool calling: hybrid — provider tool API first, falling back to the text
### Correctness Risks

* **\`wrap.py\`, line 13:** The \`wrap\` function does not handle the case where a single word is
longer than the specified \`width\`. If \`len(w) > width\`, the \`if\` condition
\`len(cur) + len(w) + 1 > width\` will be true, but \`cur\` will be set to \`w\` and appended to
\`lines\` in the next iteration or at the end, resulting in a line that exceeds the intended
\`width\`.

### Other Observations

* **\`wrap.py\`, line 12:** The logic \`lines.append(cur)\` inside the loop and after the loop will
append an empty string \`""\` to the \`lines\` list if \`wrap\` is called with an empty string.`}
        maxLines={22}
        caption={
          <>
            A real finding on a real 15-line diff. With no task the question defaults to a review
            brief: correctness risks first, one finding per bullet, each naming{' '}
            <code>file:line</code>.
          </>
        }
      />

      <ApiTable
        headers={['TARGET', 'Reviews']}
        rows={[
          [
            <code>uncommitted</code>,
            <>
              The default — <code>git diff HEAD</code>, staged and unstaged together.
            </>,
          ],
          [
            <code>staged</code>,
            <>
              <code>git diff --cached</code>.
            </>,
          ],
          [
            'Any revision or range git accepts',
            <>
              <code>HEAD~3</code>, <code>main...HEAD</code>, a tag, a sha.
            </>,
          ],
          [
            <>
              <code>-f/--file PATH</code>, repeatable
            </>,
            'One file in full. Works with a target, or on its own — which is how a directory that is not a repository is reviewed.',
          ],
        ]}
      />

      <p>
        The change under review is handed to the model <strong>as context</strong>, because the only
        route to a diff would be a shell and a read-only run has none. A diff over the budget is
        truncated with the cut marked and the remainder counted, never silently. The record reports{' '}
        <code>"read_only": true</code> and a <code>review</code> block naming the target;{' '}
        <code>files_written</code> is always <code>[]</code>, and <code>permission_mode</code> keeps
        reporting one of the four documented modes — <code>plan</code> for a review — so nothing
        that reads it has to learn a fifth value.
      </p>

      <Terminal
        command={`effgen code --review -m openai:gpt-5-nano --json | jq "{read_only, review, files_written, permission_mode}"`}
        output={`{
  "read_only": true,
  "review": {
    "kind": "diff",
    "ref": "uncommitted",
    "files": [],
    "line_count": 15,
    "truncated_at": 0
  },
  "files_written": [],
  "permission_mode": "plan"
}`}
      />

      <p>
        A model that asks for a write anyway gets a refusal it can read, the action log records it
        as <code>refused</code>, and the run still exits <code>0</code>.{' '}
        <code>--review</code> cannot be combined with <code>--plan</code>,{' '}
        <code>--auto-edit</code>, <code>--yes</code>, <code>--commit</code> or <code>--undo</code>:
      </p>

      <Terminal
        command="effgen code --review --plan"
        output={`✗ --review cannot be combined with --plan — a review writes nothing, runs nothing and commits
nothing. Pass one.`}
        caption="Exit 1."
      />

      <p>
        Inside an interactive session, <code>/review [TARGET]</code> runs one read-only turn: the
        tools and the mode are swapped for that turn and put back afterwards — including when it
        fails — so the next turn can write again.
      </p>

      <h2>Diffs, apply, reject and undo</h2>

      <p>
        Each proposed edit is rendered as a colourised unified diff <em>before</em> the write is
        decided, so a change is visible in every mode — including <code>--plan</code>, and including
        piped output, where the diff goes to stderr and stdout keeps the result.
      </p>

      <p>
        Every applied edit is journaled per workspace, so it can be reversed. The journal keeps the
        last <code>{undo_journal_entries}</code> edits and lives on disk, so <code>--undo</code>{' '}
        works after a restart and in a different shell.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen code --undo -w ws            # reverse the last applied edit
effgen code --undo --undo-count 3 -w ws`}
      />

      <Terminal
        command="effgen code --undo -w ws"
        output={`Removed fib.py (created by a coding run).
0 earlier change(s) can still be undone.`}
        caption="A restored file returns to its previous content; a file the run created is removed."
      />

      <h2>Repository awareness</h2>

      <p>
        In a git repository the branch, a short <code>git status</code> and a bounded file layout
        (ignored files excluded) are read before the first model call and become part of the
        agent's context, along with an <code>AGENTS.md</code> in the workspace if there is one.
        Outside a repository the same inventory is built from the workspace itself. You can see the
        line it reports at the top of every run above: <code>Repo: repo · branch master · 1
        change(s) · 1 file(s)</code>.
      </p>

      <h3>What a session may do to a repository</h3>

      <p>
        Exactly one thing: commit the files it wrote, after an explicit confirmation. The plan —
        repository, exact paths, message — is printed before the y/N, only those paths are staged,
        and work you had staged yourself stays staged and out of the commit.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen code "fix the failing test" --auto-edit --commit`}
      />

      <ApiTable
        headers={['git sub-command', 'Status']}
        rows={[
          [
            <span className="param-mono">{git_allowed.join(' · ')}</span>,
            'Allowed. `commit` only through the confirmed path above.',
          ],
          [
            <span className="param-mono">{git_refused.join(' · ')}</span>,
            'Refused in every mode, including when the model tries to reach them through the shell.',
          ],
          [
            <span className="param-mono">{git_refused_flags.join(' · ')}</span>,
            'Refused wherever they appear, on any sub-command.',
          ],
        ]}
        caption="Read off the allow-list the coding agent's git tool enforces. History is never rewritten and nothing is ever pushed."
      />

      <h2>The interactive session</h2>

      <p>
        Run <code>effgen code</code> on a terminal with no task. Type <code>/</code> on its own for
        the menu; tab completes command names.
      </p>

      <ApiTable
        headers={['Command', 'Does']}
        rows={slashCommands.map((command) => [<code>{command.name}</code>, command.summary])}
        caption={
          <>
            All <code>{siteData.code.slash_command_count}</code>, read off the session's own command
            table. <code>effgen chat</code> has a shorter set of{' '}
            <code>{siteData.code.chat_slash_command_count}</code> —{' '}
            <Link to="/cli/run">run and chat</Link> lists them.
          </>
        }
      />

      <h3>What a turn shows while it runs</h3>

      <p>
        A status line naming the tool in flight, each proposed edit's diff before it is written, and
        a tick per decided action. On a model whose provider streams its tool calls — openai,
        gemini, groq, together, fireworks and cerebras — the answer is written to the screen as the
        model produces it rather than arriving in one block at the end. The status line and the
        answer take turns owning the terminal: the status line runs while the model is thinking and
        dispatching tools, and hands over from the first word of the answer.
      </p>

      <p>
        Everything else prints the answer once the turn finishes: a model whose tool calls are not
        streamed (the local engines among them), and every non-interactive surface —{' '}
        <code>--json</code>, <code>--quiet</code>, <code>--no-animation</code>, <code>NO_COLOR</code>{' '}
        and piped output.
      </p>

      <h2>Continuing a session</h2>

      <p>
        <code>--session-id ID</code> (or <code>--resume ID</code>) continues a stored session — the
        same store as <code>effgen chat --session-id</code> and{' '}
        <Link to="/cli/history"><code>effgen sessions list</code></Link>. On a terminal with no task
        it reopens the session; with a task it appends that turn.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen code --session-id refactor-42 -p "start by listing what needs to change" --plan
effgen code --session-id refactor-42            # later, in a new shell`}
      />

      <ApiTable
        headers={['On resume', 'What happens']}
        rows={[
          ['The conversation', 'Restored — the next turn can answer from what earlier turns said.'],
          ['Files in context', 'Restored, minus any path that no longer exists, which is named.'],
          [
            'Files the session wrote',
            <>
              Restored, so <code>/git commit</code> still knows its own paths.
            </>,
          ],
          [
            'Model and provider',
            <>
              Restored when <code>-m</code> was not given, and announced.
            </>,
          ],
          [
            'Permission mode',
            'Restored only on a terminal and only when no permission flag was given — a stored `yes` is never restored into a piped run.',
          ],
          [
            'The workspace',
            <>
              <strong>Not</strong> adopted. <code>-w/--workspace</code> or the current directory
              always wins, and a stored workspace that differs is reported in one line.
            </>,
          ],
          [
            <>
              Edits staged by <code>/plan</code>
            </>,
            <>
              <strong>Not</strong> stored — the files may have changed underneath, so re-run{' '}
              <code>/plan</code>.
            </>,
          ],
          [
            'The undo journal',
            <>
              Nothing to restore: it is per workspace and already on disk, so <code>--undo</code>{' '}
              and <code>/undo</code> work across restarts.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="Two stores that look alike">
        <p>
          <code>/save</code> and <code>/load</code> are a separate, file-based store under the chat
          history directory; <code>effgen sessions</code> does not list those. Use{' '}
          <code>/session</code> and <code>--session-id</code> for a session you want to continue by
          id.
        </p>
      </Callout>

      <h2>Stdin</h2>

      <p>
        When stdin is not a terminal it is read <strong>to EOF before the run starts</strong>. With
        a task already given the piped text becomes context in front of it; with no task the piped
        text <em>is</em> the task.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`cat pytest.log | effgen code -p "why did this fail?"   # log becomes context
echo "add a --retries flag to cli.py" | effgen code    # stdin is the task`}
      />

      <p>
        Reading to EOF is what lets a slow producer — a build log, a <code>tail -f</code> — be
        folded in whole. The consequence is that a pipe which never closes holds the run: an open
        stdin inherited from a supervisor or a job-control shell keeps the command waiting even
        though a task was supplied. That wait is announced rather than silent — after about two
        seconds a line goes to stderr naming what it is waiting for:
      </p>

      <Terminal
        title="stderr"
        output={`Reading piped stdin as context before starting; the stream has not closed.
Close it (Ctrl-D) or re-run with < /dev/null to skip it.`}
        caption={
          <>
            With no task the note names the other case instead —{' '}
            <em>Reading the task from piped stdin</em>. Pass <code>&lt; /dev/null</code> when a
            caller has no context to pipe in.
          </>
        }
      />

      <h2>Scripting</h2>

      <p>
        Piped or with <code>--json</code>, stdout carries only the result — the answer text, or one
        JSON document — and everything a human reads goes to stderr.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen code -p "write hello.py that prints hello" -w ws -m openai:gpt-5-nano \\
  --auto-edit --json < /dev/null \\
  | jq "{files_written, iterations, tool_calling, cost_usd, diffs: [.diffs[].applied]}"`}
      />

      <Terminal
        command={`effgen code -p "write hello.py that prints hello" -w ws -m openai:gpt-5-nano --auto-edit --json < /dev/null | jq "{files_written, iterations, tool_calling, cost_usd, diffs: [.diffs[].applied]}"`}
        output={`{
  "files_written": [
    "hello.py"
  ],
  "iterations": 5,
  "tool_calling": "hybrid",
  "cost_usd": 0.00175698,
  "diffs": [
    true
  ]
}`}
      />

      <ApiTable
        headers={['JSON field', 'Carries']}
        rows={[
          [<code>answer</code>, 'The final text.'],
          [<code>files_written</code>, 'Paths that reached disk. Always `[]` under `--plan` and `--review`.'],
          [
            <code>diffs</code>,
            <>
              Every <em>proposed</em> edit. The ones that reached disk carry{' '}
              <code>"applied": true</code>, so <code>--plan --json</code> reports the changes it
              would make without writing any of them.
            </>,
          ],
          [
            <code>actions</code>,
            'The full action log — what was allowed, withheld, declined or refused, and why.',
          ],
          [
            <code>tool_calling</code>,
            <>
              The path the model's tool calls travelled: <code>hybrid</code> and <code>native</code>{' '}
              send the definitions to the provider's tool-calling API, <code>react</code> reads the
              calls out of the model's text.
            </>,
          ],
          [
            <code>answer_source</code>,
            <>
              Where the answer came from when the loop had to recover one —{' '}
              <code>loop_detected</code>, <code>repeated_tool_result</code>,{' '}
              <code>null_final_from_model</code>. Empty when the model wrote it.
            </>,
          ],
          [
            <>
              <code>read_only</code>, <code>review</code>
            </>,
            'Describe a `--review` run.',
          ],
          [
            <code>coding_suitability</code>,
            'The verdict on the chosen model — see below.',
          ],
          [
            <code>partial_output</code>,
            'What a run that hit its iteration cap had reached. Tool output and reasoning, never a result.',
          ],
          [
            <>
              <code>iterations</code>, <code>tool_calls</code>, <code>tokens</code>,{' '}
              <code>cost_usd</code>, <code>duration_s</code>
            </>,
            'The run\'s own measurements.',
          ],
        ]}
      />

      <h2>Whether a model can do the work</h2>

      <p>
        Not every model can complete a coding turn. Some receive no tool definitions at all —
        their chat template renders none — some receive them and answer the question directly, and
        some write the call out as prose. Each of those finishes with an answer,{' '}
        <code>files_written: []</code> and exit <code>0</code>, which reads like success.
      </p>

      <p>Before the first call, <code>effgen code</code> says one line when the model is a poor fit:</p>

      <Terminal
        title="stderr"
        output={`transformers:google/gemma-2-2b-it for coding: its chat template renders no tool
definitions, so it never receives the coding tools and can only describe a
change. Use Qwen/Qwen2.5-7B-Instruct locally, or a keyed cloud model.
(measured 2026-08-06)`}
        caption={
          <>
            It never blocks the run, <code>-q</code> suppresses it, and <code>--json</code> carries
            the same fact under <code>coding_suitability</code>. The verdict is{' '}
            <code>suitable</code> (nothing is printed), <code>limited</code>,{' '}
            <code>unsuitable</code>, or <code>unknown</code> when the catalog does not know the id.
          </>
        }
      />

      <p>
        <code>effgen models info &lt;id&gt;</code> shows the same verdict as a <code>Coding</code>{' '}
        row, and <code>effgen models list --json</code> carries it per model.
      </p>

      <h2>Options</h2>

      <ParamTable
        nameLabel="Flag"
        params={codeOptions.map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={
          <>
            Every flag <code>effgen code --help</code> declares, in its order. Note that{' '}
            <code>-p</code> here is <code>--print</code>, not a port.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              Exit <code>2</code>, <code>No changes were made: N action(s) needed confirmation and
              this session has no terminal</code>
            </>,
            'A scripted run in the default mode. Nothing was applied, by design.',
            <>
              Add <code>--auto-edit</code> or <code>--yes</code>. Treat exit <code>2</code> as
              "withheld", not as a failure.
            </>,
          ],
          [
            <>
              <code>"reason": "written_tool_call"</code>, exit <code>1</code>
            </>,
            'The model wrote the tool call out as text instead of making it, so nothing ran and nothing was written. The turn is reported as a failure rather than as an answer describing work that did not happen.',
            <>
              On the <code>native</code> or <code>hybrid</code> path, the model does not call tools
              — replace it. On the <code>react</code> text path, ask for the native path.{' '}
              <code>effgen models list</code> marks the models that advertise tool calling.
            </>,
          ],
          [
            <>
              <code>"reason": "max_iterations_partial"</code> or{' '}
              <code>max_iterations_exhausted</code>
            </>,
            'The run spent every step without writing a final answer.',
            <>
              Raise <code>--max-iterations</code>, or use a model that needs fewer steps. Whatever
              it reached is reported separately as <code>partial_output</code> and under a{' '}
              <em>Partial progress</em> heading.
            </>,
          ],
          [
            <>
              <code>The model did not write this answer; it is the last tool result</code>
            </>,
            <>
              The model repeated the same call or returned no final answer, so the loop reported
              what it had. <code>answer_source</code> carries the same fact in JSON.
            </>,
            <>
              A larger model, or a higher <code>--max-tokens</code> if it is a reasoning model
              spending its budget before writing anything.
            </>,
          ],
          [
            <>
              <code>… is not inside a git repository, so there is no diff to review</code>
            </>,
            <>
              <code>--review</code> outside a repository, with no <code>-f/--file</code>.
            </>,
            <>
              Name files with <code>-f</code> (repeatable), or run the review from inside a
              repository.
            </>,
          ],
          [
            <>
              <code>There is nothing to review …: nothing is staged</code>
            </>,
            <>
              <code>--review staged</code> on a clean index.
            </>,
            <>
              Stage something, or pass a target: <code>--review HEAD~1</code>,{' '}
              <code>--review uncommitted</code>.
            </>,
          ],
          [
            'The command hangs before doing anything',
            'Stdin is a pipe that has not closed, and the run reads it to EOF first.',
            <>
              Close it (Ctrl-D), or pass <code>&lt; /dev/null</code>. After about two seconds the
              wait is announced on stderr.
            </>,
          ],
          [
            'A write outside the workspace fails with a read-only filesystem error',
            'The sandbox is doing its job: only the workspace is writable.',
            <>
              Correct the path. The message names where writes are allowed, which is what lets the
              model fix it itself on the next iteration.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          The whole command is new in this release. <code>--review</code>, <code>--undo</code>,{' '}
          <code>--session-id</code>, the four permission modes, the git allow-list and the{' '}
          <code>coding_suitability</code> verdict all arrived with it.{' '}
          <code>effgen quickstart</code> offers a coding step that writes and runs a small program
          end to end inside <code>~/.effgen/quickstart-code</code>, if you would rather see the loop
          before pointing it at real code.
        </p>
      </Callout>

      <SeeAlso paths={['/cli/run', '/execution', '/sessions']} />
    </DocPage>
  );
}
