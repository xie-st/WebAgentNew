import { NotebookPen } from 'lucide-react';
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

export default function Jupyter() {
  return (
    <DocPage
      subtitle="Three IPython magics that run a model, an agent or a metrics snapshot from a notebook cell."
      icon={<NotebookPen size={48} />}
    >
      <p>
        <code>effgen.jupyter</code> is an IPython extension. Loading it adds{' '}
        <code>%effgen_chat</code> for a single message, <code>%%effgen_agent</code> for a cell that
        is a task, and <code>%effgen_metrics</code> for a counter snapshot. Each renders effGen's
        own result card in the notebook — the markdown answer, a metric strip, and a collapsible
        step trace when tools ran.
      </p>

      <h2>Getting started</h2>

      <CodeBlock language="bash" filename="terminal" code={`pip install "effgen[jupyter]"`} />

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`%load_ext effgen.jupyter
%effgen_chat --model openai:gpt-5-nano Tell me a one-sentence joke about gradient descent`}
      />

      <Terminal
        title="cell output"
        output={`**effGen** (openai:gpt-5-nano, 3.05s)

42

AgentResponse(success=True, output='42', mode='single', iterations=1,
              tool_calls=ToolCallList(total=0), tokens_used=168,
              execution_time=1.07s, trace_steps=1, reason='final_answer')`}
        caption={
          <>
            What the cell displays, captured from a real IPython shell: a markdown header naming
            the model and the wall-clock time, then the response itself. In a notebook the response
            renders as effGen's result card; in a terminal IPython it falls back to the text
            representation above.
          </>
        }
      />

      <h2>The three magics</h2>

      <h3>
        <code>%effgen_chat</code> — one message
      </h3>

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`%effgen_chat [--model MODEL] [--server URL] <message...>`}
      />

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`# A specific model
%effgen_chat --model openai:gpt-5-nano Explain attention in transformers in two sentences

# Route through a running effGen server instead of loading in-process
%effgen_chat --server http://localhost:8080 What is 42 * 7?`}
      />

      <p>
        The output is an <code>effGen (model, Xs)</code> header followed by the result card: the
        answer rendered as markdown, a compact strip of status, time, tokens and cost, and — when
        tools ran — a collapsible step trace. That card is{' '}
        <code>AgentResponse._repr_html_</code>, the same one <Link to="/agents">a response</Link>{' '}
        renders anywhere IPython is displaying it.
      </p>

      <h3>
        <code>%%effgen_agent</code> — the cell is the task
      </h3>

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`%%effgen_agent [preset] [--model MODEL] [--tools TOOL [TOOL ...]]
<task description in the cell body>`}
      />

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`%%effgen_agent math --model openai:gpt-5-nano
Compute 17*23. Answer with the number only.`}
      />

      <Terminal
        title="cell output"
        output={`**effGen** (agent:math · openai:gpt-5-nano, 7.65s)

391

AgentResponse(success=True, output='391', mode='single', iterations=2,
              tool_calls=ToolCallList(['calculator', 'python_repl'], total=2),
              tokens_used=1958, execution_time=7.07s, trace_steps=7,
              reason='final_answer')`}
        caption={
          <>
            <code>391</code> because the calculator computed it. The header names the preset as
            well as the model, and <code>tool_calls</code> shows which tools ran — in a notebook
            they are the collapsible step trace on the card.
          </>
        }
      />

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`%%effgen_agent coding --model openai:gpt-5-nano
Write a Python function to compute the nth Fibonacci number iteratively.
Return type-annotated code with a docstring.`}
      />

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`%%effgen_agent --model openai:gpt-5-nano --tools calculator
What is the population of France in millions, divided by 3.14159?`}
      />

      <p>
        When <code>[preset]</code> names a built-in preset the agent is created through{' '}
        <code>effgen.presets.create_agent</code>, so it arrives wired with that preset's tools and
        system prompt. That is the difference between a computed answer and a guessed one:{' '}
        <code>math</code> and <code>general</code> include a calculator, so{' '}
        <code>Compute 17*23.</code> returns 391 because something multiplied it.{' '}
        <Link to="/presets">Presets</Link> lists all nine.
      </p>

      <Callout type="warning" title="A tool-heavy preset can exceed a small model's context">
        <p>
          Every tool's description is sent on every request. <code>general</code> attaches 31 tools
          and roughly 7,900 tokens of schema, which will not fit a local model with a 4K–8K window.
          When that happens the magic prints a hint naming a smaller preset or a larger-context
          model rather than passing a raw provider error through.
        </p>
      </Callout>

      <h3>
        <code>%effgen_metrics</code> — a counter snapshot
      </h3>

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`%effgen_metrics`}
      />

      <p>
        Renders an HTML table of every registered Prometheus metric name and its current value —
        the <code>effgen_*</code> families among them. It reads the in-process registry, so it
        reports what this kernel has done, not what a server has done.{' '}
        <Link to="/metrics">Metrics</Link> covers the families themselves.
      </p>

      <h2>Choosing the model</h2>

      <ParamTable
        nameLabel="Variable"
        params={[
          {
            name: 'EFFGEN_JUPYTER_MODEL',
            default: 'unset',
            description: 'Jupyter-specific default model for all three magics.',
          },
          {
            name: 'EFFGEN_DEFAULT_MODEL',
            default: 'unset',
            description: (
              <>
                Framework-wide default model, used when <code>EFFGEN_JUPYTER_MODEL</code> is unset.
              </>
            ),
          },
          {
            name: 'EFFGEN_JUPYTER_SERVER_URL',
            default: 'unset',
            description: (
              <>
                When set, <code>%effgen_chat</code> routes through this server instead of running
                in-process.
              </>
            ),
          },
        ]}
        caption={
          <>
            Both model variables are overridden per cell by <code>--model</code>, and the server
            variable by <code>--server</code>.
          </>
        }
      />

      <Callout type="note" title="There is no hidden default model">
        <p>
          The magics never quietly pick a model that might be a paid one. With no{' '}
          <code>--model</code> and neither variable set, nothing is sent anywhere and the cell says
          what to do instead:
        </p>
      </Callout>

      <Terminal
        title="cell output"
        output={`**No model configured.** Pass one on the magic line, e.g.

\`\`\`
%effgen_chat -m gpt-5-nano Hello!
\`\`\`

or set a default once with \`EFFGEN_DEFAULT_MODEL\` (effGen never picks a paid cloud model for
you). Run \`effgen models list\` to see options or \`effgen doctor\` to check which providers are
usable.`}
        caption={
          <>
            The magic returns <code>None</code> and makes no call. <code>-m</code> is accepted as
            the short form of <code>--model</code>.
          </>
        }
      />

      <h2>In-process or through a server</h2>

      <ApiTable
        headers={['What is compared', 'In-process (default)', 'Through --server']}
        rows={[
          [
            'What runs the model',
            'The kernel. The provider SDK has to be installed in the notebook environment.',
            'The server. The kernel only makes an HTTP call.',
          ],
          [
            'Where the keys live',
            'The kernel environment.',
            'The server. Nothing sensitive has to reach the notebook.',
          ],
          [
            'Cost and metrics',
            <>
              Recorded by this kernel — which is what <code>%effgen_metrics</code> reads.
            </>,
            <>
              Recorded by the server, and visible on its{' '}
              <Link to="/dashboard">dashboard</Link>.
            </>,
          ],
          [
            'Local weights',
            <>
              Loaded into the kernel's own process, so they stay resident between cells.{' '}
              <Link to="/local-models">Local models</Link>.
            </>,
            'Served by whatever the server has loaded.',
          ],
          [
            'Which magics',
            'All three.',
            <>
              <code>%effgen_chat</code>. <code>%%effgen_agent</code> runs in-process.
            </>,
          ],
        ]}
      />

      <h2>Using the library directly</h2>

      <p>
        The magics are a convenience over the same API a script uses, and a notebook can drop to it
        whenever the magic's flags run out. A response renders as the same card either way, because
        the card belongs to the response.
      </p>

      <CodeBlock
        language="python"
        filename="notebook cell"
        code={`from effgen import Agent, AgentConfig
from effgen.tools import get_registry

calculator = get_registry().get_tool_sync("calculator")   # instances, not names
agent = Agent(AgentConfig(model="openai:gpt-5-nano", tools=[calculator]))
response = agent.run("What is 18723 * 4409?")
response                       # renders the result card`}
        caption={
          <>
            Close it with <code>agent.close()</code>, or use{' '}
            <code>with Agent(config) as agent:</code> — a long-lived kernel that never closes an
            agent warns when it is garbage-collected. <Link to="/agents">Agents</Link> covers the
            configuration surface.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>UsageError: Line magic function `%effgen_chat` not found</code>
            </>,
            'The extension is not loaded in this kernel.',
            <>
              <code>%load_ext effgen.jupyter</code>, once per kernel. Put it in an IPython startup
              file to have it always loaded.
            </>,
          ],
          [
            <>
              <code>ModuleNotFoundError: No module named 'IPython'</code> on the load
            </>,
            <>
              effGen is installed without the Jupyter extra.
            </>,
            <>
              <code>pip install "effgen[jupyter]"</code> into the environment the{' '}
              <em>kernel</em> runs in, which is not always the one the terminal is in.
            </>,
          ],
          [
            'A message asking you to name a model',
            'No `--model`, and neither `EFFGEN_JUPYTER_MODEL` nor `EFFGEN_DEFAULT_MODEL` is set. The magics will not pick one for you.',
            <>
              Pass <code>--model</code>, or set one of the two variables before starting the kernel.
            </>,
          ],
          [
            'A hint suggesting a smaller preset',
            'The preset\'s tool schemas do not fit the model\'s context window.',
            <>
              Use <code>math</code> or <code>minimal</code>, name the tools you need with{' '}
              <code>--tools</code>, or move to a larger-context model.
            </>,
          ],
          [
            <>
              A warning that an agent was garbage-collected without <code>close()</code>
            </>,
            'A kernel keeps objects alive across cells, so an agent created in one cell and forgotten is collected later.',
            <>
              <code>agent.close()</code>, or the context-manager form. The warning is harmless but
              it means a client stayed open longer than it needed to.
            </>,
          ],
          [
            'The card renders as plain text',
            'Not a notebook — a terminal IPython console has no HTML display.',
            'Expected. The text representation carries the same answer and the same figures.',
          ],
          [
            <>
              <code>%effgen_metrics</code> is empty
            </>,
            'Nothing has run in this kernel yet, so no counter has been incremented.',
            'Run a cell first. It reads the in-process registry, not a server’s.',
          ],
        ]}
      />

      <Callout type="note" title="Set EFFGEN_DEV_MODE=1 only for a local server">
        <p>
          Routing <code>--server</code> at a local <code>effgen serve</code> that has no OIDC
          configured needs <code>EFFGEN_DEV_MODE=1</code> on the <em>server</em>, which disables its
          authentication and says so loudly at start-up. Never set it on anything reachable from
          another machine. <Link to="/api-server">The API server</Link> covers the alternatives.
        </p>
      </Callout>

      <SeeAlso paths={['/agents', '/vscode', '/metrics']} />
    </DocPage>
  );
}
