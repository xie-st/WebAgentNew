import { Waves } from 'lucide-react';
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

const loadtestOptions = siteData.cli.command_options['loadtest'] ?? [];

export default function LoadTest() {
  return (
    <DocPage
      subtitle="Putting a server or a provider under load, and injecting failure to see what holds."
      icon={<Waves size={48} />}
    >
      <p>
        Three harnesses answer three different questions. <code>effgen loadtest</code> asks how much
        traffic a deployment takes and what its latency looks like under it. The chaos harness
        injects deterministic faults so you can watch retries, breakers and fallback actually work.
        The fuzz harness throws hostile input at tools, messages and the router and checks that
        nothing crashes and no secret escapes.
      </p>

      <h2>A load test in one command</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen loadtest --duration 5 --concurrency 4`} />

      <Terminal command="effgen loadtest --duration 5 --concurrency 4" output={`Starting load test  scenario=fixed  concurrency=4  duration=5.0s
────────────────────────────────────────────────────
  Load Test Report — fixed
────────────────────────────────────────────────────
  Concurrency   : 4
  Duration      : 5.0s
  Total requests: 9216
  Successful    : 9216
  Failed        : 0
  Error rate    : 0.00%
  Throughput    : 1841.75 req/s
  Latency p50   : 2.1ms
  Latency p95   : 2.2ms
  Latency p99   : 2.3ms
  Latency mean  : 2.2ms
  Latency stdev : 0.6ms
────────────────────────────────────────────────────`} />

      <Callout type="note" title="With no --provider and no --url, this is a mock">
        <p>
          The default target returns immediately, so the numbers above measure the harness and the
          machine, not a model. That is the point of the default: it is how you check the harness
          works and how you size a run before spending anything. Add <code>--provider</code> to drive
          a provider adapter, or <code>--url</code> to drive a running server over HTTP.
        </p>
      </Callout>

      <h2>Against a running server</h2>

      <p>
        <code>--url</code> points the load at a{' '}
        <Link to="/api-server">running <code>effgen serve</code></Link> and drives{' '}
        <code>POST {'{URL}'}/v1/chat/completions</code> over HTTP — so auth, rate limiting, roles,
        budgets, the audit log and the rest of the middleware stack are all in the path, not just the
        provider adapter. It needs <code>--model</code>, and it is mutually exclusive with{' '}
        <code>--provider</code>.
      </p>

      <CodeBlock language="bash" filename="terminal" code={`effgen loadtest --url http://127.0.0.1:8000 --model openai:gpt-5-nano \\
  --concurrency 2 --duration 10`} />

      <Terminal
        command="effgen loadtest --url http://127.0.0.1:8000 --model openai:gpt-5-nano --concurrency 2 --duration 10"
        output={`Server mode: url=http://127.0.0.1:8000  model=openai:gpt-5-nano
Starting load test  scenario=fixed  concurrency=2  duration=10.0s
────────────────────────────────────────────────────
  Load Test Report — fixed
────────────────────────────────────────────────────
  Concurrency   : 2
  Duration      : requested 10.0s, wall 11.0s (incl. 1.0s draining in-flight requests after the window closed)
  Total requests: 14
  Successful    : 14
  Failed        : 0
  Error rate    : 0.00%
  Throughput    : 1.27 req/s
  Latency p50   : 1402.4ms
  Latency p95   : 2398.5ms
  Latency p99   : 2492.1ms
  Latency mean  : 1533.7ms
  Latency stdev : 454.9ms
────────────────────────────────────────────────────`}
        caption={
          <>
            Fourteen real completions against a real model, so the latency figures are the provider's
            and the throughput is what two concurrent callers actually got. Note the wall time: the
            harness drains in-flight requests after the window closes rather than cutting them off
            and counting them as failures.
          </>
        }
      />

      <Callout type="warning" title="A live run spends money">
        <p>
          Every request is a billed call. Start with a short <code>--duration</code> and a low{' '}
          <code>--concurrency</code>, and set a budget first —{' '}
          <Link to="/cost">Cost and budgets</Link>. The mock target exists so you can get the shape
          of the run right before any of it is real.
        </p>
      </Callout>

      <h2>Options</h2>

      <ParamTable
        nameLabel="Flag"
        params={loadtestOptions.map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={
          <>
            Every flag <code>effgen loadtest --help</code> declares, read from the binary.
          </>
        }
      />

      <h3>Scenarios</h3>

      <ApiTable
        headers={['Scenario', 'What each virtual user sends']}
        rows={[
          [<code>fixed</code>, 'The same short prompt every time. Isolates the system from prompt variation.'],
          [<code>synthetic</code>, 'Prompts cycled from a library of ten varied questions.'],
          [
            <code>multi_tool</code>,
            'Calculator expressions that vary by virtual user and request index, so the tool path is exercised.',
          ],
        ]}
      />

      <CodeBlock language="bash" filename="terminal" code={`effgen loadtest --scenario multi_tool --duration 5 --concurrency 4 --output report.json`} />

      <Terminal
        command="effgen loadtest --scenario multi_tool --duration 5 --concurrency 4 --output report.json"
        output={`Starting load test  scenario=multi_tool  concurrency=4  duration=5.0s
────────────────────────────────────────────────────
  Load Test Report — multi_tool
────────────────────────────────────────────────────
  Concurrency   : 4
  Duration      : 5.0s
  Total requests: 5887
  Successful    : 5887
  Failed        : 0
  Error rate    : 0.00%
  Throughput    : 1175.93 req/s
  Latency p50   : 4.2ms
  Latency p95   : 5.5ms
  Latency p99   : 5.6ms
  Latency mean  : 3.4ms
  Latency stdev : 1.8ms
────────────────────────────────────────────────────`}
      />

      <h2>The report</h2>

      <p>
        The report is printed to stdout and, with <code>--output</code>, written to a file whose
        extension chooses the format: <code>.html</code> renders the self-contained report,{' '}
        anything else writes JSON. <code>--report out.html</code> writes the HTML alongside whatever{' '}
        <code>--output</code> did. Parent directories are created.
      </p>

      <CodeBlock
        language="json"
        filename="report.json"
        code={`{
  "scenario": "multi_tool",
  "concurrency": 4,
  "duration": 5.0,
  "total_requests": 5887,
  "successful_requests": 5887,
  "failed_requests": 0,
  "error_rate": 0.0,
  "throughput": 1175.93,
  "p50_latency": 0.0042,
  "p95_latency": 0.0055,
  "p99_latency": 0.0056,
  "min_latency": 0.0009,
  "max_latency": 0.0121,
  "mean_latency": 0.0034,
  "stdev_latency": 0.0018,
  "provider": null,
  "model": null,
  "requested_duration": 5.0,
  "error_breakdown": {}
}`}
        caption={
          <>
            The field names of <code>LoadReport</code>. <strong>Every latency is in seconds</strong>{' '}
            — the terminal view converts to milliseconds, the JSON does not.{' '}
            <code>error_breakdown</code> counts failures by category, and{' '}
            <code>raw_results</code> on the object (not in the JSON) holds every individual{' '}
            <code>RequestResult</code>.
          </>
        }
      />

      <h2>From Python</h2>

      <CodeBlock filename="library.py" code={`from effgen.tools.loadgen import LoadConfig, LoadGenerator, LoadScenario

cfg = LoadConfig(
    concurrency=4,
    duration=5.0,
    scenario=LoadScenario.SYNTHETIC,
    request_timeout=60.0,
)
report = LoadGenerator(cfg).run()

print(f"requests   : {report.total_requests}")
print(f"throughput : {report.throughput:.1f} req/s")
print(f"p95 latency: {report.p95_latency * 1000:.2f} ms")
print(f"error rate : {report.error_rate * 100:.2f}%")`} />

      <Terminal command="python library.py" output={`requests   : 5308
throughput : 1060.4 req/s
p95 latency: 5.41 ms
error rate : 0.00%`} />

      <ParamTable
        nameLabel="LoadConfig"
        params={[
          { name: 'concurrency', type: 'int', default: '10', description: 'Virtual users.' },
          { name: 'duration', type: 'float', default: '30.0', description: 'Seconds of load, before draining.' },
          { name: 'scenario', type: 'LoadScenario', default: 'LoadScenario.FIXED', description: 'One of the three above.' },
          { name: 'ramp_up', type: 'float', default: '0.0', description: 'Stagger virtual-user starts over this many seconds.' },
          {
            name: 'request_timeout',
            type: 'float',
            default: '60.0',
            description: 'Per-request limit. Always explicit — never None.',
          },
          { name: 'think_time', type: 'float', default: '0.0', description: 'Pause between requests, per virtual user.' },
          { name: 'provider', type: 'str | None', default: 'None', description: 'Drive a provider adapter directly.' },
          { name: 'model', type: 'str | None', default: 'None', description: 'The model id for a live run.' },
          { name: 'output_path', type: 'Path | None', default: 'None', description: 'Write the report here as well as to stdout.' },
        ]}
        caption={
          <>
            From <code>effgen.tools.loadgen.LoadConfig</code>.
          </>
        }
      />

      <h3>Your own target</h3>

      <CodeBlock filename="target.py" code={`import asyncio

from effgen.tools.loadgen import LoadConfig, LoadGenerator


async def my_target(prompt: str) -> str:
    await asyncio.sleep(0.1)          # stand in for your own backend
    return f"Response to: {prompt}"


report = LoadGenerator(LoadConfig(concurrency=4, duration=3.0), target=my_target).run()
print(f"{report.total_requests} requests, p50 {report.p50_latency * 1000:.0f} ms")`} />

      <Terminal
        command="python target.py"
        output={`120 requests, p50 100 ms`}
        caption="An async callable taking a prompt and returning a string is all the harness needs, so it will drive a backend that has nothing to do with effGen."
      />

      <h2>Chaos</h2>

      <p>
        A load test tells you what happens when everything works. The chaos harness tells you what
        happens when it does not — deterministically, so a failure you find is a failure you can
        reproduce.
      </p>

      <CodeBlock filename="chaos.py" code={`from effgen.reliability.chaos import Chaos, ChaosHttp429Error, ChaosHttp5xxError, Http429, Http5xx

chaos = Chaos(seed=42)
chaos.add_rule("primary", Http5xx, every_nth=3)
chaos.add_rule("primary", Http429, every_nth=6, retry_after=2.0)

for call in range(1, 7):
    try:
        chaos.maybe_inject("primary")
        print(f"call {call}: ok")
    except ChaosHttp429Error as exc:
        print(f"call {call}: {type(exc).__name__} retry_after={exc.retry_after}")
    except ChaosHttp5xxError as exc:
        print(f"call {call}: {type(exc).__name__} status={exc.status_code}")`} />

      <Terminal
        command="python chaos.py"
        output={`call 1: ok
call 2: ok
call 3: ChaosHttp5xxError status=500
call 4: ok
call 5: ok
call 6: ChaosHttp5xxError status=500`}
        caption={
          <>
            The 5xx rule fires on calls 3 and 6. The 429 rule is also due on call 6 — the first
            matching rule wins, so only one fault is injected per call. Give a rule{' '}
            <code>max_fires</code> when you want it to fire a fixed number of times.
          </>
        }
      />

      <ApiTable
        headers={['Fault', 'Raises', 'Carries']}
        rows={[
          [<code>NetworkTimeout</code>, <code>ChaosNetworkTimeout</code>, <><code>provider</code>, <code>limit</code></>],
          [<code>Http5xx</code>, <code>ChaosHttp5xxError</code>, <><code>provider</code>, <code>status_code</code></>],
          [
            <code>Http429</code>,
            <code>ChaosHttp429Error</code>,
            <>
              <code>provider</code>, <code>retry_after</code>, <code>status_code=429</code>
            </>,
          ],
          [<code>SlowResponse</code>, 'nothing — it delays', <code>delay_s</code>],
          [<code>PartialResponse</code>, <code>ChaosPartialResponseError</code>, <code>partial_content</code>],
          [<code>MalformedJSON</code>, <code>ChaosMalformedJSONError</code>, <code>raw</code>],
        ]}
        caption={
          <>
            All importable from <code>effgen.reliability</code>.
          </>
        }
      />

      <p>
        The injected exceptions are not a parallel world: they classify the way the real ones do, so
        the retry policy and the circuit breaker treat them identically.
      </p>

      <CodeBlock filename="transient.py" code={`from effgen.reliability import is_transient_error
from effgen.reliability.chaos import Chaos, NetworkTimeout

chaos = Chaos(seed=1)
chaos.add_rule("primary", NetworkTimeout, every_nth=1, limit=60.0)
try:
    chaos.maybe_inject("primary")
except TimeoutError as exc:
    print(type(exc).__name__, "is a TimeoutError:", isinstance(exc, TimeoutError))
    print("retryable:", is_transient_error(exc))`} />

      <Terminal command="python transient.py" output={`ChaosNetworkTimeout is a TimeoutError: True
retryable: True`} />

      <ParamTable
        nameLabel="add_rule"
        params={[
          {
            name: 'provider',
            type: 'str',
            required: true,
            description: (
              <>
                The provider to target. <code>"*"</code> matches all of them.
              </>
            ),
          },
          { name: 'fault_type', type: 'type[FaultBase]', required: true, description: 'Which fault to inject.' },
          {
            name: 'every_nth',
            type: 'int | None',
            default: 'None',
            description: 'Fire on every nth call. Counter-based, so it is seed-independent.',
          },
          {
            name: 'fault_rate',
            type: 'float',
            default: '0.0',
            description: 'Probability per call. Consumes the seeded PRNG, so the same seed replays the same sequence.',
          },
          { name: 'max_fires', type: 'int | None', default: 'None', description: 'Cap on total fires. None is unlimited.' },
          {
            name: '**params',
            type: 'Any',
            description: (
              <>
                Passed to the fault — <code>retry_after=</code>, <code>delay_s=</code>,{' '}
                <code>limit=</code>.
              </>
            ),
          },
        ]}
        caption={
          <>
            <code>Chaos(seed=…)</code> seeds the PRNG once at construction;{' '}
            <code>chaos.reset()</code> re-seeds it and clears the rule counters, replaying the same
            sequence from the start.
          </>
        }
      />

      <p>
        <code>ProviderRegistry.with_chaos(chaos)</code> attaches the harness to real calls —{' '}
        <code>registry.call(...)</code> and <code>registry.async_call(...)</code> then run fault
        injection before forwarding. When every provider is faulted, the agent-level harness raises{' '}
        <code>AllProvidersFailed</code>, which always carries a non-empty message and a{' '}
        <code>failures</code> dict of provider to last exception — never a silent empty answer.
      </p>

      <h2>Fuzz</h2>

      <p>
        The fuzz harness lives in the repository under <code>tests/fuzz/</code> and runs at least 500
        Hypothesis-generated examples per test. It exists to check three invariants that are easy to
        break and hard to notice.
      </p>

      <ApiTable
        headers={['Module', 'What it fuzzes', 'What it asserts']}
        rows={[
          [
            <code>test_tool_fuzz.py</code>,
            <>
              Every <code>BaseTool</code> subclass discovered by the registry.
            </>,
            <>
              <code>execute()</code> always returns a <code>ToolResult</code>; a{' '}
              <code>success=False</code> always carries a non-empty <code>error</code>; no secret
              appears in any error, including synthetic ones injected into the inputs.
            </>,
          ],
          [
            <code>test_message_fuzz.py</code>,
            <>
              Random <code>ContentPart</code> and <code>Message</code> objects.
            </>,
            <>
              Valid construction never raises; invalid MIME types, negative durations and empty ids
              raise only <code>InvalidMultimodalContent</code>; serialisation never crashes.
            </>,
          ],
          [
            <code>test_router_fuzz.py</code>,
            'Random candidates, constraints and model metadata.',
            <>
              <code>route()</code> returns a decision or raises <code>NoCandidateError</code> —
              never <code>None</code>; a chosen pair is always well formed; complexity scores stay
              in 0–1.
            </>,
          ],
        ]}
      />

      <Terminal
        command="pytest -q -m fuzz tests/fuzz/test_message_fuzz.py"
        output={`............                                                             [100%]
12 passed in 10.80s`}
        caption="From a checkout of the framework repository — the tests are not part of the installed package. No network, no live calls."
      />

      <p>
        A new tool is fuzzed automatically as soon as the registry discovers it; the harness patches{' '}
        <code>_execute</code>, so a tool that hits the network or a system binary still gets its
        validation, coercion and error-handling envelope exercised. The secret-injection test forces{' '}
        <code>_execute</code> to raise an exception echoing its own arguments — the worst case for a
        leak — and checks that <code>BaseTool.execute()</code> scrubbed it. That is the same{' '}
        <Link to="/metrics">redactor</Link> the log encoder uses.
      </p>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Impossibly good numbers — thousands of requests a second',
            'It is the mock target. No model was called.',
            <>
              Add <code>--provider</code> or <code>--url</code>. The mock is the default precisely so
              a first run costs nothing.
            </>,
          ],
          [
            <>
              <code>--url</code> is refused
            </>,
            <>
              It needs <code>--model</code>, and it cannot be combined with{' '}
              <code>--provider</code>.
            </>,
            <>
              With <code>--url</code>, the model id is the full one the server expects —{' '}
              <code>openai:gpt-5-nano</code>, not a bare id.
            </>,
          ],
          [
            'The error rate climbs the moment concurrency rises',
            "You are measuring the server's rate limit, or a provider's.",
            <>
              Raise <code>--rate-limit</code> on the server for the test, or lower concurrency. A
              429 is a real result about the deployment, not a harness artefact.
            </>,
          ],
          [
            'Wall time is longer than the duration asked for',
            'In-flight requests are drained after the window closes so they are not counted as failures.',
            'Expected, and it is reported as a separate figure rather than folded into the duration.',
          ],
          [
            'Latency numbers look 1000× too small',
            'The JSON report is in seconds; the terminal view is in milliseconds.',
            'Multiply, or read the terminal view.',
          ],
          [
            'A chaos rule never fires',
            <>
              An earlier rule for the same provider matched first, or <code>max_fires</code> is
              spent.
            </>,
            <>
              One fault per call. Order the rules, or split them across providers.{' '}
              <code>chaos.reset()</code> puts the counters back.
            </>,
          ],
          [
            'A fault_rate run behaves differently every time',
            'Rate rules draw from the PRNG. Counter rules do not.',
            <>
              Fix the seed. Same seed, same sequence — that is the guarantee the harness is built
              on.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>--url</code> is new: before it, a load test could only drive a provider adapter
          directly. Driving a running server over HTTP is what puts auth, rate limiting, roles and
          budgets in the measurement. <code>--report out.html</code> is new in the same release.
        </p>
      </Callout>

      <SeeAlso paths={['/reliability', '/observability', '/metrics']} />
    </DocPage>
  );
}
