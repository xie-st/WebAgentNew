import { Layers } from 'lucide-react';
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

export default function CliBatch() {
  return (
    <DocPage
      subtitle="Running many prompts from one file, and wiring effgen into a script that nobody is watching."
      icon={<Layers size={48} />}
    >
      <p>
        <code>effgen batch</code> reads a file of queries, runs them concurrently, and writes one
        row per query. It takes JSONL, CSV, JSON or plain text, writes JSONL, CSV or JSON, validates
        every row against a schema if you give it one, and resumes an interrupted job without
        repeating the rows that already succeeded.
      </p>

      <h2>The shortest useful batch</h2>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen batch queries.jsonl -o answers.jsonl -m openai:gpt-5-nano -c 3`}
      />

      <CodeBlock
        language="json"
        filename="queries.jsonl"
        code={`{"query": "In one word, what colour is the sky on a clear day?"}
{"query": "In one word, what is the capital of France?"}
{"query": "In one word, how many legs does a spider have?"}`}
      />

      <Terminal
        command="effgen batch queries.jsonl -o answers.jsonl -m openai:gpt-5-nano -c 3 --no-animation"
        output={`Loading queries from queries.jsonl...

Batch complete: 3/3 succeeded in 2.27s · 758 tokens · $0.0003
Results written to answers.jsonl`}
      />

      <Terminal
        title="answers.jsonl"
        output={`{"index": 1, "query": "In one word, what is the capital of France?", "output": "Paris", "success": true, "execution_time": 1.685, "cost_usd": 5.66e-05, "prompt_tokens": 28, "completion_tokens": 138, "total_tokens": 166}
{"index": 0, "query": "In one word, what colour is the sky on a clear day?", "output": "blue", "success": true, "execution_time": 2.008, "cost_usd": 0.00010795, "prompt_tokens": 31, "completion_tokens": 266, "total_tokens": 297}
{"index": 2, "query": "In one word, how many legs does a spider have?", "output": "eight", "success": true, "execution_time": 2.265, "cost_usd": 0.00010785, "prompt_tokens": 29, "completion_tokens": 266, "total_tokens": 295}`}
        caption={
          <>
            Look at the order. Row <code>1</code> is written first because it finished first — see
            the section below.
          </>
        }
      />

      <Callout type="warning" title="A .jsonl file is in completion order, not input order">
        <p>
          At any <code>--concurrency</code> above 1, <code>.jsonl</code> rows are written as each
          query finishes, so line <em>N</em> of the output is not input row <em>N</em>. Every row
          carries an <code>index</code> field pointing back at its input position — sort on it if
          your consumer assumes the two line up. <code>.csv</code> and <code>.json</code> outputs
          are different: they are written once at the end, in input order.
        </p>
      </Callout>

      <h2>Input formats</h2>

      <ApiTable
        headers={['Extension', 'How the query is found']}
        rows={[
          [
            <code>.jsonl</code>,
            <>
              One JSON object per line. The query is the <code>query</code> field, or whatever{' '}
              <code>--query-field</code> names.
            </>,
          ],
          [
            <code>.csv</code>,
            <>
              A header row, then one query per row, in the <code>query</code> column or the one{' '}
              <code>--query-field</code> names. Other columns travel with the row.
            </>,
          ],
          [<code>.json</code>, 'An array of objects, read the same way as JSONL.'],
          [
            'Anything else',
            'Plain text: one query per line. This is the form to reach for when the queries have no metadata.',
          ],
        ]}
      />

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`printf 'What is 2+2?\\nWhat is 3+3?\\n' > prompts.txt
effgen batch prompts.txt -m openai:gpt-5-nano -q --no-animation --json \\
  | jq -r '.rows[] | "\\(.index): \\(.output)"'`}
      />

      <Terminal
        command={`effgen batch prompts.txt -m openai:gpt-5-nano -q --no-animation --json | jq -r '.rows[] | "\\(.index): \\(.output)"'`}
        output={`Loading queries from prompts.txt...

Batch complete: 2/2 succeeded in 1.46s · 324 tokens · $0.0001
0: 4
1: 6`}
        caption={
          <>
            The two summary lines are on stderr; <code>jq</code> is reading stdout, which carries
            only the document. <code>2&gt;/dev/null</code> removes them from a capture.
          </>
        }
      />

      <h2>Options</h2>

      <ParamTable
        nameLabel="Flag"
        params={[
          {
            name: 'INPUT',
            description: (
              <>
                Input file (JSONL, CSV, JSON, or plain text). Same as <code>-i/--input</code>.
              </>
            ),
          },
          { name: '-i INPUT, --input INPUT', description: 'Input file (JSONL, CSV, JSON, or plain text)' },
          {
            name: '-o OUTPUT, --output OUTPUT',
            description: (
              <>
                Output file (JSONL, CSV, or JSON). <code>.jsonl</code> rows are written as each
                query finishes, so their file order is completion order, not input order, at any{' '}
                <code>--concurrency</code> above 1; <code>.csv</code>/<code>.json</code> rows are
                written once at the end in input order. Every row carries an <code>index</code>{' '}
                field back to its input position.
              </>
            ),
          },
          {
            name: '-c CONCURRENCY, --concurrency CONCURRENCY',
            type: 'int',
            default: '5',
            description: 'Max concurrent queries',
          },
          { name: '--batch-size BATCH_SIZE', type: 'int', description: 'Batch size (0 = all at once)' },
          { name: '--timeout TIMEOUT', type: 'float', description: 'Timeout per query in seconds' },
          { name: '--retries RETRIES', type: 'int', description: 'Retries for failed queries' },
          { name: '-m MODEL, --model MODEL', description: 'Model to use' },
          {
            name: '--preset {coding,general,math,media,minimal,multimodal,notify,rag,research}',
            description: 'Use a preset agent configuration',
          },
          {
            name: '--guardrails NAME',
            description: (
              <>
                Apply a guardrail preset to redact/block PII and screen for prompt injection on
                every row: <code>strict</code>, <code>standard</code> (alias{' '}
                <code>default</code>/<code>balanced</code>), <code>phi</code> (alias{' '}
                <code>hipaa</code>/<code>deidentify</code>), <code>minimal</code>, or{' '}
                <code>none</code>.
              </>
            ),
          },
          {
            name: '--system-prompt TEXT, --persona TEXT',
            description:
              'System prompt applied to every row — a target language, a glossary, a tone. Overrides the preset’s default prompt.',
          },
          {
            name: '--query-field QUERY_FIELD',
            default: 'query',
            description: 'Field name for queries in JSONL/CSV',
          },
          {
            name: '--max-tokens MAX_TOKENS',
            type: 'int',
            description: 'Max output tokens per query (raise for token-heavy or reasoning models)',
          },
          {
            name: '--temperature TEMPERATURE',
            type: 'float',
            description:
              'Sampling temperature per query (0 for deterministic reruns where the provider supports it)',
          },
          {
            name: '--schema SCHEMA_PATH',
            description:
              'JSON Schema file; each row is validated against it and its parsed object is written',
          },
          {
            name: '--output-model OUTPUT_MODEL',
            description: (
              <>
                Pydantic model as <code>module:ClassName</code> to validate each row against
              </>
            ),
          },
          {
            name: '--strict',
            type: 'flag',
            description: 'Abort on the first malformed input line instead of skipping it',
          },
          {
            name: '--resume',
            type: 'flag',
            description: (
              <>
                Skip input rows already present in the JSONL <code>--output</code> file and append
                the rest
              </>
            ),
          },
          {
            name: '--excel, --bom',
            type: 'flag',
            description: (
              <>
                Prepend a UTF-8 BOM to CSV output so Excel on Windows reads non-Latin scripts
                (Arabic, CJK, Devanagari, …) as UTF-8 on double-click rather than as mojibake. Only
                affects{' '}
                <code>--output</code> ending in <code>.csv</code>.
              </>
            ),
          },
          { name: '-q, --quiet', type: 'flag', description: 'Quiet output (suppress the progress bar)' },
          {
            name: '--no-animation',
            type: 'flag',
            description: 'Disable the live progress bar (plain output)',
          },
          {
            name: '--json',
            type: 'flag',
            description: (
              <>
                Emit the job summary and every row as a JSON document to stdout, in addition to any{' '}
                <code>-o</code> file. Human output goes to stderr; combine with <code>-q</code> for
                clean stdout.
              </>
            ),
          },
        ]}
        caption={
          <>
            Every flag <code>effgen batch --help</code> declares. Note that <code>-c</code> here is{' '}
            <code>--concurrency</code>, not <code>--config</code> as it is on <code>effgen run</code>.
          </>
        }
      />

      <h2>Structured output</h2>

      <p>
        Two flags turn a batch into a typed extraction job. <code>--schema</code> validates each
        row against a JSON Schema file and writes the parsed object;{' '}
        <code>--output-model module:ClassName</code> does the same against a Pydantic model. A row
        that does not validate is reported as a failed row rather than written as free text.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen batch tickets.csv --query-field body -o triage.jsonl \\
  -m openai:gpt-5-nano --schema triage.schema.json --temperature 0`}
        caption={
          <>
            <code>--temperature 0</code> where the provider honours it makes a re-run comparable
            with the last one. <Link to="/generation">Generation controls</Link> covers the rest of
            the structured-output surface.
          </>
        }
      />

      <h2>Resuming a job that stopped</h2>

      <p>
        <code>--resume</code> reads the JSONL <code>--output</code> file, skips the input rows
        already in it, and appends the rest. It matches on the input row, so it is safe to re-run
        with the same command line after a rate limit, an interruption or a provider outage.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen batch big.jsonl -o out.jsonl -m openai:gpt-5-nano -c 8
# interrupted
effgen batch big.jsonl -o out.jsonl -m openai:gpt-5-nano -c 8 --resume`}
        caption={
          <>
            It needs a <code>.jsonl</code> output: the CSV and JSON writers only produce a file at
            the end, so an interrupted job leaves nothing to resume from.
          </>
        }
      />

      <h2>In a script</h2>

      <p>
        The scripting contract is the same as the rest of the command line:{' '}
        <code>--json</code> puts one document on stdout and everything a person reads on stderr, and{' '}
        <code>-q</code> silences the human half. That is what makes the exit status and the document
        the only two things a caller has to handle.
      </p>

      <CodeBlock
        language="bash"
        filename="nightly.sh"
        code={`#!/usr/bin/env bash
set -euo pipefail

effgen batch queries.jsonl -o answers.jsonl -m openai:gpt-5-nano \\
  -c 8 --retries 2 --timeout 60 --guardrails standard \\
  -q --no-animation --json 2>run.log > summary.json

failed=$(jq '.total - .succeeded' summary.json)
if [ "$failed" -gt 0 ]; then
  jq -r '.rows[] | select(.success == false) | "\\(.index)\\t\\(.error)"' summary.json
  exit 1
fi`}
        caption={
          <>
            <code>--no-animation</code> keeps carriage returns out of <code>run.log</code>.{' '}
            <Link to="/cli/appearance">Appearance and themes</Link> lists the other switches.
          </>
        }
      />

      <Terminal
        command={`effgen batch queries.jsonl -m openai:gpt-5-nano -q --no-animation --json | jq "{total: .total, succeeded: .succeeded, cost: .total_cost_usd}"`}
        output={`{
  "total": 3,
  "succeeded": 3,
  "cost": 0.0003236
}`}
      />

      <h2>The other commands a script reaches for</h2>

      <ApiTable
        headers={['Command', 'For']}
        rows={[
          [
            <>
              <code>effgen run … --json</code>
            </>,
            <>
              One task, one document. <Link to="/cli/run">run and chat</Link>.
            </>,
          ],
          [
            <>
              <code>effgen workflow run FILE --json</code>
            </>,
            <>
              A DAG of steps with dependencies between them, defined in YAML, gating on the exit
              status. <Link to="/workflows">Workflows</Link>.
            </>,
          ],
          [
            <>
              <code>effgen eval --suite … --json</code>
            </>,
            <>
              A pass/fail gate in CI. <Link to="/evaluation">Evaluation and CI gates</Link>.
            </>,
          ],
          [
            <>
              <code>effgen code -p … --auto-edit --json</code>
            </>,
            <>
              A change to a repository, non-interactively. Read the exit code:{' '}
              <code>2</code> means the change was withheld. <Link to="/cli/code">effgen code</Link>.
            </>,
          ],
          [
            <>
              <code>effgen top --json</code>
            </>,
            <>
              A health or spend check with no terminal. <Link to="/cli/top">effgen top</Link>.
            </>,
          ],
          [
            <>
              <code>effgen report result.json</code>
            </>,
            <>
              Turning any of the above into an HTML file to attach to the job's output.{' '}
              <Link to="/cli/reports">Reports and run cards</Link>.
            </>,
          ],
        ]}
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>Warning: no -o/--output and no --json — each row's answer, cost, tokens, and any
              error detail will be discarded</code>
            </>,
            'The batch will run and only the final summary line will survive it.',
            <>
              Add <code>-o out.jsonl</code> or <code>--json</code>. The warning exists because a
              long batch that keeps nothing is almost always a mistake.
            </>,
          ],
          [
            <>
              <code>✗ Could not read missing.jsonl: [Errno 2] No such file or directory</code>,
              exit <code>1</code>
            </>,
            'The input path is wrong.',
            'Check the path. The file is read before any model call, so nothing was billed.',
          ],
          [
            'Output rows are in the wrong order',
            <>
              A <code>.jsonl</code> output above concurrency 1 is in completion order.
            </>,
            <>
              Sort on <code>index</code>, or write <code>.csv</code>/<code>.json</code>, which are
              emitted at the end in input order.
            </>,
          ],
          [
            'Some rows failed with rate-limit errors',
            'Concurrency is above what the provider will accept on this key.',
            <>
              Lower <code>-c</code>, add <code>--retries</code>, and re-run with{' '}
              <code>--resume</code> so the successful rows are not paid for twice.
            </>,
          ],
          [
            'Rows are silently missing from the output',
            'Malformed input lines are skipped by default.',
            <>
              <code>--strict</code> aborts on the first one instead, which is what you want in a
              pipeline.
            </>,
          ],
          [
            'Non-Latin text is mojibake when the CSV is opened in Excel',
            'Excel on Windows guesses the encoding without a byte-order mark.',
            <>
              <code>--excel</code> (alias <code>--bom</code>) writes one. It affects{' '}
              <code>.csv</code> output only.
            </>,
          ],
          [
            'The job cost far more than expected',
            'Every row is a full agent run, and a preset with many tools sends its whole tool schema on every call.',
            <>
              <code>--preset minimal</code> where no tools are needed, and check the per-preset
              token cost with <code>effgen presets</code>. <Link to="/cost">Cost and budgets</Link>{' '}
              covers capping it.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>--resume</code>, <code>--json</code>, <code>--guardrails</code>,{' '}
          <code>--schema</code>, <code>--output-model</code>, <code>--strict</code> and{' '}
          <code>--excel</code> are new on this command, and every output row now carries{' '}
          <code>index</code>, its own cost and its own token counts.
        </p>
      </Callout>

      <SeeAlso paths={['/cli/run', '/workflows', '/cost']} />
    </DocPage>
  );
}
