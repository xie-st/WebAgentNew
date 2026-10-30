import { Scale } from 'lucide-react';
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

const compareOptions = siteData.cli.command_options['compare'] ?? [];
const battleOptions = siteData.cli.command_options['battle'] ?? [];

export default function Compare() {
  return (
    <DocPage
      subtitle="Running the same work through several models and reading the difference."
      icon={<Scale size={48} />}
    >
      <p>
        Two commands answer two different questions. <code>effgen compare</code> runs a whole test
        suite through several models and recommends one on accuracy, cost or latency —{' '}
        it is how you pick a model. <code>effgen battle</code> puts several models on{' '}
        <em>one</em> prompt side by side and reports what was measured — it is how you look at the
        answers themselves.
      </p>

      <h2>A bake-off</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen compare --models openai:gpt-5-nano,gemini:gemini-3.1-flash-lite \\
  --suite math --max-cases 4 --no-animation`} />

      <Terminal
        command="effgen compare --models openai:gpt-5-nano,gemini:gemini-3.1-flash-lite --suite math --max-cases 4"
        output={`Loading model openai:gpt-5-nano...
Loading model gemini:gemini-3.1-flash-lite...

Comparing 2 models on math (4 cases)...
# Model Comparison

## Accuracy

| Model | math |
|-------|-------|
| gemini:gemini-3.1-flash-lite | 100.0% |
| openai:gpt-5-nano | 100.0% |

## Avg Latency (s)

| Model | math |
|-------|-------|
| gemini:gemini-3.1-flash-lite | 2.382 |
| openai:gpt-5-nano | 3.341 |

## Avg Cost (USD/run)

| Model | math |
|-------|-------|
| gemini:gemini-3.1-flash-lite | $0.000021 |
| openai:gpt-5-nano | $0.000078 |

## Recommendations (optimized for accuracy)

- **math**: gemini:gemini-3.1-flash-lite — highest accuracy at 100%, tied with 1
other model and faster at 2.382s/run ($0.000021/run)`}
        caption={
          <>
            Three tables and a recommendation. On a tie in accuracy the recommendation breaks on
            latency, then on tokens — and it says which rule it applied.
          </>
        }
      />

      <Callout type="note" title="compare reports, eval gates">
        <p>
          <code>compare</code> always exits 0. It is a bake-off, not a build gate — a model losing a
          comparison is a result, not a failure. Use{' '}
          <Link to="/evaluation"><code>eval --fail-under</code></Link> when you want an exit code.
        </p>
      </Callout>

      <h2>Optimising for something other than accuracy</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen compare --models openai:gpt-5-nano,gemini:gemini-3.1-flash-lite \\
  --suite math --max-cases 4 --optimize cost --no-animation`} />

      <Terminal
        command="effgen compare … --optimize cost"
        output={`Loading model openai:gpt-5-nano...
Loading model gemini:gemini-3.1-flash-lite...

Comparing 2 models on math (4 cases)...
# Model Comparison

## Accuracy

| Model | math |
|-------|-------|
| gemini:gemini-3.1-flash-lite | 100.0% |
| openai:gpt-5-nano | 100.0% |

## Avg Latency (s)

| Model | math |
|-------|-------|
| gemini:gemini-3.1-flash-lite | 2.327 |
| openai:gpt-5-nano | 3.122 |

## Avg Cost (USD/run)

| Model | math |
|-------|-------|
| gemini:gemini-3.1-flash-lite | $0.000021 |
| openai:gpt-5-nano | $0.000065 |

## Recommendations (optimized for cost)

- **math**: gemini:gemini-3.1-flash-lite — cheapest at 100% accuracy — 
$0.000021/run vs $0.000065/run for openai:gpt-5-nano`}
        maxLines={26}
      />

      <ApiTable
        headers={['--optimize', 'What it recommends']}
        rows={[
          [
            <code>accuracy</code>,
            'The highest accuracy. The default. Ties break on lower latency, then on fewer tokens.',
          ],
          [
            <code>cost</code>,
            <>
              The cheapest model that still meets <code>--threshold</code> accuracy — falling back to
              the whole field if none qualify. Ties break on higher accuracy.
            </>,
          ],
          [<code>latency</code>, 'The same rule, for the fastest.'],
        ]}
        caption={
          <>
            The fallback matters: <code>--optimize cost</code> never recommends a model that failed
            the accuracy bar unless every candidate did, and the wording of the recommendation says
            which case you are in.
          </>
        }
      />

      <h2>Machine-readable</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen compare --models openai:gpt-5-nano,gemini:gemini-3.1-flash-lite \\
  --suite math --max-cases 3 --json --no-animation | python -m json.tool | head -40`} />

      <Terminal command="effgen compare … --json" output={`{
    "scores": [
        {
            "model": "openai:gpt-5-nano",
            "suite": "math",
            "accuracy": 1.0,
            "avg_latency": 3.209303858379523,
            "total_tokens": 781,
            "avg_cost_usd": 8.371666666666666e-05,
            "avg_tool_accuracy": 0.0,
            "error": null,
            "error_count": 0,
            "responses": [
                {
                    "query": "What is 2 + 3?",
                    "output": "5",
                    "score": 1.0,
                    "passed": true,
                    "error": null
                },
                {
                    "query": "What is 15 * 7?",
                    "output": "105",
                    "score": 1.0,
                    "passed": true,
                    "error": null
                },
                {
                    "query": "What is 144 / 12?",
                    "output": "12",
                    "score": 1.0,
                    "passed": true,
                    "error": null
                }
            ]
        },
        {
            "model": "gemini:gemini-3.1-flash-lite",
            "suite": "math",
            "accuracy": 1.0,`} maxLines={24} />

      <p>
        Each entry in <code>scores</code> carries the model, the suite, <code>accuracy</code>,{' '}
        <code>avg_latency</code>, <code>total_tokens</code>, <code>avg_cost_usd</code>,{' '}
        <code>avg_tool_accuracy</code>, an <code>error</code> and <code>error_count</code>, and every
        individual response with its score. <code>recommendations</code> carries the pick.
      </p>

      <h2>compare options</h2>

      <ParamTable
        nameLabel="Flag"
        params={compareOptions.map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={
          <>
            Every flag <code>effgen compare --help</code> declares, read from the binary.{' '}
            <code>--suite</code> takes one suite name or one path — run the command again for a
            second suite.
          </>
        }
      />

      <h2>A battle</h2>

      <p>
        One prompt, every model at once. On a terminal the answers stream side by side, each column
        showing its own time to first token, elapsed time, tokens and cost, and a verdict panel
        closes the race.
      </p>

      <CodeBlock language="bash" filename="terminal" code={`effgen battle "Explain a B-tree in two sentences." \\
  -m openai:gpt-5-nano,gemini:gemini-3.1-flash-lite --no-animation`} />

      <Terminal
        command="effgen battle 'Explain a B-tree in two sentences.' -m openai:gpt-5-nano,gemini:gemini-3.1-flash-lite"
        output={`# Model Battle

**Prompt:** Explain a B-tree in two sentences.

| Model | Result | TTFT | Latency | Tokens (in/out) | Cost |
|-------|--------|------|---------|-----------------|------|
| openai:gpt-5-nano | answered | 5.86s | 7.06s | 25/903 | $0.000362 |
| gemini:gemini-3.1-flash-lite | answered | 1.92s | 3.09s | 21/60 | $0.000095 |

## Verdict

- **Fastest**: gemini:gemini-3.1-flash-lite — answered in 3.09s
- **Cheapest**: gemini:gemini-3.1-flash-lite — $0.000095 for this run
- **Longest**: openai:gpt-5-nano — 315 characters

## openai:gpt-5-nano

A B-tree is a self-balancing, multi-way search tree designed for efficient disk storage, where each node stores multiple keys and child pointers in sorted order. All leaves are at the same depth, and searches, inserts, and deletes take logarithmic time because nodes split or merge to preserve capacity and balance.

## gemini:gemini-3.1-flash-lite

A B-tree is a self-balancing search tree data structure that maintains sorted data and allows for efficient insertion, deletion, and search operations in logarithmic time. By allowing nodes to contain more than two children, it minimizes disk I/O and remains optimized for systems that store large blocks of data.

_2/2 answered in 10.65s; total cost $0.000458._`}
        maxLines={28}
        caption={
          <>
            This is the piped form — the live side-by-side view is skipped when stdout is not a
            terminal, and one structured result is printed instead. Both answers are here in full;
            nothing is elided to make a table fit.
          </>
        }
      />

      <h3>The verdict reports what was measured</h3>

      <p>
        Fastest, cheapest and longest need no judge — they are measurements.{' '}
        <code>--judge MODEL</code> adds a separate model that picks a winner on quality, and that
        pick is reported apart from the measurements and names the judge that made it. A model that
        fails is reported as failed and cannot win.
      </p>

      <CodeBlock language="bash" filename="terminal" code={`effgen battle "Name one prime number between 20 and 30." \\
  -m openai:gpt-5-nano,gemini:gemini-3.1-flash-lite --json \\
  | python -c "
import json, sys
battle = json.load(sys.stdin)
for c in battle['contenders']:
    print(f\\"{c['model']:30} {c['state']:8} latency={c['latency_s']}s cost={c['cost_usd']}\\")
print('fastest :', battle['verdict']['fastest'])
print('cheapest:', battle['verdict']['cheapest'])
print('total   :', battle['total_cost_usd'], 'in', battle['wall_s'], 's')
"`} />

      <Terminal command="effgen battle … --json | python …" output={`openai:gpt-5-nano              done     latency=4.2316s cost=5.66e-05
gemini:gemini-3.1-flash-lite   done     latency=3.5051s cost=9.25e-06
fastest : {'model': 'gemini:gemini-3.1-flash-lite', 'detail': 'answered in 3.51s'}
cheapest: {'model': 'gemini:gemini-3.1-flash-lite', 'detail': '$0.000009 for this run'}
total   : 6.585e-05 in 7.7987 s`} />

      <ApiTable
        headers={['Field', 'On', 'What it holds']}
        rows={[
          [<code>prompt</code>, 'the battle', 'The prompt every model answered.'],
          [<code>contenders</code>, 'the battle', 'One entry per model.'],
          [<code>verdict</code>, 'the battle', <><code>fastest</code>, <code>cheapest</code>, <code>longest</code>, each with a model and a detail line.</>],
          [<code>wall_s</code>, 'the battle', 'Wall-clock seconds for the whole race.'],
          [<code>total_cost_usd</code>, 'the battle', 'Summed across contenders that reported a cost.'],
          [<code>model</code>, 'a contender', 'The model id.'],
          [<code>answer</code>, 'a contender', 'The full answer.'],
          [<code>state</code>, 'a contender', <><code>done</code> or <code>failed</code>.</>],
          [<code>error</code>, 'a contender', <>The failure, or <code>null</code>.</>],
          [<code>load_s</code>, 'a contender', 'Time spent loading the model.'],
          [<code>ttft_s</code>, 'a contender', 'Time to first token.'],
          [<code>latency_s</code>, 'a contender', 'Time to the last token.'],
          [
            <>
              <code>prompt_tokens</code>, <code>completion_tokens</code>, <code>total_tokens</code>
            </>,
            'a contender',
            'Provider counts where the provider reports them.',
          ],
          [
            <code>cost_usd</code>,
            'a contender',
            <>
              <code>null</code> for an <Link to="/cost">unpriced model</Link>, never an invented
              zero.
            </>,
          ],
          [<code>estimated_tokens</code>, 'a contender', 'True when the counts came from a tokenizer rather than the provider.'],
        ]}
        caption={
          <>
            <code>--json</code>, a pipe, <code>--no-animation</code> and <code>NO_COLOR</code> each
            skip the live view and print one structured result carrying every model's full answer.
          </>
        }
      />

      <h2>battle options</h2>

      <ParamTable
        nameLabel="Flag"
        params={battleOptions.map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={
          <>
            Every flag <code>effgen battle --help</code> declares. The prompt is a positional
            argument, and at least two models are required.
          </>
        }
      />

      <h2>Reports</h2>

      <p>
        Both commands take <code>--report out.html</code> for a self-contained report — for{' '}
        <code>compare</code>, the recommended model and why, a per-model table, and accuracy, cost
        and latency charts — and <code>-o PATH</code>, whose extension picks the format:{' '}
        <code>.html</code> renders the report, <code>.md</code> writes Markdown, anything else writes
        JSON.
      </p>

      <CodeBlock language="bash" filename="terminal" code={`effgen compare --models openai:gpt-5-nano,gemini:gemini-3.1-flash-lite \\
  --suite math --max-cases 3 --no-animation --report bakeoff.html > /dev/null
ls -l bakeoff.html | awk '{print $5, "bytes", $NF}'
echo "off-host references: $(grep -oE 'https?://[^"'"'"' )]+' bakeoff.html | wc -l)"`} />

      <Terminal
        command="effgen compare … --report bakeoff.html"
        output={`14830 bytes bakeoff.html
off-host references: 0`}
        caption="Under 15 kB, and zero references to anything off the machine — every style, script and chart is inline, so the file opens with the network off."
      />

      <p>
        A result captured earlier renders without re-running the models:{' '}
        <code>effgen report bakeoff.json</code>, or <code>--kind comparison</code> when the shape
        cannot be inferred.
      </p>

      <h2>Which command do I want?</h2>

      <ApiTable
        headers={['You want to', 'Use']}
        rows={[
          ['Pick a model for a job you can describe as test cases', <code>effgen compare</code>],
          ['See how several models phrase one answer', <code>effgen battle</code>],
          [
            'Fail a build when quality drops',
            <>
              <Link to="/evaluation"><code>effgen eval --fail-under</code></Link>
            </>,
          ],
          [
            'Know what the field costs before committing',
            <>
              Either — both report per-run cost. <Link to="/catalog">The catalog</Link> has published
              prices without spending anything.
            </>,
          ],
        ]}
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'A model in the field reports an error and scores zero',
            'It failed rather than answered — usually a missing key or an id that provider does not have.',
            <>
              Its <code>error</code> is in the JSON. A failed model is reported as failed and cannot
              win a battle.
            </>,
          ],
          [
            'The recommendation changes between two identical runs',
            'The models are sampling, and a tie broke the other way.',
            <>
              <code>--temperature 0</code> where the provider supports it, and more than four cases.
            </>,
          ],
          [
            <>
              <code>--optimize cost</code> recommends the least accurate model
            </>,
            <>
              No model met <code>--threshold</code>, so it fell back to the whole field.
            </>,
            'The recommendation text says so. Lower the threshold deliberately, or add a stronger candidate.',
          ],
          [
            'A bare model id is not found',
            'No provider prefix, and no --provider to supply one.',
            <>
              Use <code>provider:model</code> per id, or pass <code>--provider</code> for all the
              bare ones.
            </>,
          ],
          [
            <>
              <code>cost_usd</code> is <code>null</code> for one contender
            </>,
            'That model publishes no per-token rate.',
            <>
              Expected, and better than a fabricated zero — <Link to="/cost">Cost and budgets</Link>.
              It also means it cannot win "cheapest".
            </>,
          ],
          [
            'The battle is much slower than the slowest model',
            'A local model id had to be loaded first, which is counted separately as load_s.',
            'Read load_s against latency_s. A hosted id has no load time.',
          ],
          [
            'A judge picks a model the measurements did not favour',
            'It is grading quality, which the measurements do not.',
            'That is the design. The two are reported separately, and the judge is named.',
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>effgen battle</code> is new, and so is the shareable HTML report{' '}
          <code>--report</code> writes for <code>compare</code>, <code>eval</code>,{' '}
          <code>cost</code> and <code>loadtest</code> — along with{' '}
          <code>effgen report &lt;result.json&gt;</code> to render a document saved earlier.{' '}
          <code>--provider</code> and <code>--temperature</code> reached both{' '}
          <code>eval</code> and <code>compare</code> in the same release.
        </p>
      </Callout>

      <SeeAlso paths={['/evaluation', '/cost', '/catalog']} />
    </DocPage>
  );
}
