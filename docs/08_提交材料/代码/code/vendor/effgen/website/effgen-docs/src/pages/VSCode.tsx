import { FileCode2 } from 'lucide-react';
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

export default function VSCode() {
  return (
    <DocPage
      subtitle="An editor extension for prompt-template completion, hover docs and a Run button that sends the prompt to your server."
      icon={<FileCode2 size={48} />}
    >
      <Callout type="warning" title="Experimental, and built from source">
        <p>
          The extension is a developer-experience preview shipped inside the framework repository at{' '}
          <code>tools/vscode-effgen</code>. It is <strong>not</strong> published to the VS Code
          Marketplace and is <strong>not</strong> covered by effGen's stability guarantees. Build
          the <code>.vsix</code> yourself, as below.
        </p>
      </Callout>

      <p>
        In a Python file the extension offers effGen's prompt templates as completions, documents
        one on hover, and puts a <strong>▶ Run with effGen</strong> lens above any line that starts
        a prompt. Clicking the lens posts the prompt to a running effGen server and writes the reply
        into an <strong>effGen</strong> output channel — the editor never runs a model itself.
      </p>

      <h2>Building it</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`cd tools/vscode-effgen
npm install
npm run compile

# to package it as a .vsix
npm install -g @vscode/vsce
vsce package`}
        caption={
          <>
            Then <strong>Extensions → Install from VSIX…</strong> in VS Code.{' '}
            <code>npm run watch</code> recompiles incrementally while you work on it.
          </>
        }
      />

      <ApiTable
        headers={['Requirement', 'Version']}
        rows={[
          [
            'VS Code',
            <>
              <code>^1.85.0</code>, from the extension's <code>engines</code> field.
            </>,
          ],
          ['Node.js', '18 or newer, to build the .vsix from source.'],
          [
            'A running effGen server',
            <>
              Only for the Run lens and for server-supplied templates.{' '}
              <Link to="/api-server">The API server</Link>.
            </>,
          ],
        ]}
      />

      <h2>What it adds</h2>

      <ApiTable
        headers={['Feature', 'How it triggers', 'What happens']}
        rows={[
          [
            'Prompt-template completion',
            <>
              A line containing <code>LibraryPrompt(</code>, <code>effgen.prompts.</code> or{' '}
              <code>%effgen_</code>
            </>,
            'Offers the templates it knows, each inserted as a snippet with its placeholders ready to tab through.',
          ],
          [
            'Hover docs',
            'Hovering a quoted template name',
            "The template's description, its category and a usage snippet.",
          ],
          [
            'Run code lens',
            <>
              A line starting with <code>LibraryPrompt(</code>, <code>effgen_chat(</code>,{' '}
              <code>%%effgen_agent</code> or <code>%effgen_chat</code>
            </>,
            <>
              A <strong>▶ Run with effGen</strong> button. Clicking it extracts the prompt, posts it
              to the server and prints the reply in the output channel.
            </>,
          ],
          [
            'Prompt registry viewer',
            <>
              <strong>effGen: Show Prompt Registry</strong> from the Command Palette (
              <code>Ctrl+Shift+P</code>)
            </>,
            'Opens a webview listing every template it knows.',
          ],
        ]}
        caption={
          <>
            The extension activates on a Python file, or on a workspace that contains one —{' '}
            <code>onLanguage:python</code> and <code>workspaceContains:**/*.py</code>.
          </>
        }
      />

      <h2>Settings</h2>

      <ParamTable
        nameLabel="Setting"
        params={[
          {
            name: 'effgen.serverUrl',
            type: 'string',
            default: 'http://localhost:8080',
            description: 'URL of the effGen API server',
          },
          {
            name: 'effgen.defaultModel',
            type: 'string',
            default: 'gpt-5-nano',
            description:
              'Default model id sent to the effGen server when running prompts from the editor (e.g. gpt-5-nano, or a local HF repo id)',
          },
          {
            name: 'effgen.enableCompletion',
            type: 'boolean',
            default: 'true',
            description: 'Enable prompt-template auto-completion',
          },
          {
            name: 'effgen.enableHover',
            type: 'boolean',
            default: 'true',
            description: 'Enable hover documentation for prompt templates',
          },
        ]}
        caption={
          <>
            Read off the extension's <code>contributes.configuration</code>. Open{' '}
            <strong>Settings</strong> (<code>Ctrl+,</code>) and search for <code>effgen</code>.
          </>
        }
      />

      <h2>Using it</h2>

      <CodeBlock
        language="python"
        filename="prompts.py"
        code={`LibraryPrompt("cod")     # Tab here completes to "coding" with its placeholders

# A ▶ Run with effGen lens appears above this line ↓
result = effgen_chat("Explain gradient descent in one paragraph")

LibraryPrompt("reasoning", problem="...")
#              ↑ hover here for the template's description and category`}
      />

      <p>
        The eight templates the extension ships with, used when no server answers:
      </p>

      <ApiTable
        headers={['Template', 'For', 'Inserts']}
        rows={[
          [
            <code>general</code>,
            'General-purpose assistant prompt with configurable persona',
            <code>LibraryPrompt("general", topic="…")</code>,
          ],
          [
            <code>coding</code>,
            'Code generation and review, language-aware',
            <code>LibraryPrompt("coding", language="…", task="…")</code>,
          ],
          [
            <code>reasoning</code>,
            'Step-by-step reasoning, chain-of-thought',
            <code>LibraryPrompt("reasoning", problem="…")</code>,
          ],
          [
            <code>analysis</code>,
            'Data or text analysis with structured output',
            <code>LibraryPrompt("analysis", data="…", goal="…")</code>,
          ],
          [
            <code>summarize</code>,
            'Summarisation with a configurable length',
            <code>LibraryPrompt("summarize", text="…", max_words=…)</code>,
          ],
          [
            <code>translate</code>,
            'Translation with source and target languages',
            <code>LibraryPrompt("translate", text="…", target_language="…")</code>,
          ],
          [
            <code>eval_rubric</code>,
            'A judging rubric',
            <code>LibraryPrompt("eval_rubric", response="…", rubric="…")</code>,
          ],
          [
            <code>agent_system</code>,
            'A system prompt for a tool-using agent',
            <code>LibraryPrompt("agent_system", tools="…", persona="…")</code>,
          ],
        ]}
        caption={
          <>
            These are the extension's own built-in list. When a server is reachable it replaces them
            with whatever that server reports, so the completions follow the deployment rather than
            the extension's release. The framework's own library is larger —{' '}
            <Link to="/prompts">the prompt library</Link> covers it.
          </>
        }
      />

      <h2>What the Run lens actually does</h2>

      <p>
        It reads the first string argument on the line, posts one chat completion to{' '}
        <code>&lt;serverUrl&gt;/v1/chat/completions</code> with{' '}
        <code>effgen.defaultModel</code>, waits up to 30 seconds, and writes the reply into the
        output channel. That is the whole mechanism: it is an OpenAI-protocol client, so anything
        that speaks that protocol will answer it.
      </p>

      <Terminal
        title="effGen output channel"
        output={`[effGen] Running: Explain gradient descent in one paragraph…
[effGen] Server: http://localhost:8080  Model: gpt-5-nano

[effGen] Response:

…`}
        caption={
          <>
            The two header lines name the server and the model that were used, which is what makes
            an unexpected answer traceable to the wrong setting.
          </>
        }
      />

      <p>
        A server that needs a key will refuse: the extension sends no{' '}
        <code>Authorization</code> header, so point it at a local server started in development
        mode, or at one behind a proxy that adds the credential.{' '}
        <Link to="/openai-api">The OpenAI-compatible API</Link> covers the endpoint itself.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>[effGen] Request failed: …</code> and a tip about starting the server
            </>,
            'Nothing is listening at `effgen.serverUrl`, or the request timed out after 30 seconds.',
            <>
              <code>effgen serve --port 8080</code>, or correct <code>effgen.serverUrl</code>. The
              extension's default port is <code>8080</code>, while <code>effgen serve</code> binds{' '}
              <code>8000</code> unless told otherwise.
            </>,
          ],
          [
            <>
              <code>[effGen] Error 401: …</code>
            </>,
            'The server requires an API key and the extension does not send one.',
            <>
              Use a local development-mode server, or put a proxy in front that adds the header.
              Never expose a keyless server beyond loopback.
            </>,
          ],
          [
            <>
              <code>[effGen] Error 404: …</code>
            </>,
            <>
              The configured <code>effgen.defaultModel</code> is not one the server can serve.
            </>,
            <>
              <code>curl &lt;serverUrl&gt;/v1/models</code> lists what it will accept.
            </>,
          ],
          [
            'No completions appear',
            <>
              The line does not carry a trigger, the file is not Python, or{' '}
              <code>effgen.enableCompletion</code> is off.
            </>,
            <>
              The triggers are <code>LibraryPrompt(</code>, <code>effgen.prompts.</code> and{' '}
              <code>%effgen_</code>. Completion is offered inside a quote — type{' '}
              <code>LibraryPrompt("</code> first.
            </>,
          ],
          [
            'The templates offered are not the ones my deployment has',
            'No server answered, so the built-in list above was used.',
            <>
              Point <code>effgen.serverUrl</code> at a reachable server. The refresh is
              best-effort with a two-second timeout and fails quietly by design.
            </>,
          ],
          [
            'The extension does not activate',
            'The workspace contains no Python file.',
            <>
              Open a <code>.py</code> file. The activation events are{' '}
              <code>onLanguage:python</code> and <code>workspaceContains:**/*.py</code>.
            </>,
          ],
          [
            'The Run lens sends the wrong text',
            'The prompt is extracted from the first quoted string on the line; a multi-line or computed prompt does not match.',
            <>
              Put the literal on one line, or run it from a terminal —{' '}
              <Link to="/cli/run">effgen run</Link> takes the same prompt.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="If the editor is not where you want to be">
        <p>
          Two other surfaces cover the same ground with fewer moving parts:{' '}
          <Link to="/jupyter">the Jupyter magics</Link>, which run in-process from a notebook cell
          and need no server, and <Link to="/cli/code">effgen code</Link>, which is a coding agent
          in the terminal with diffs, permission modes and undo.
        </p>
      </Callout>

      <SeeAlso paths={['/jupyter', '/cli/code', '/prompts']} />
    </DocPage>
  );
}
