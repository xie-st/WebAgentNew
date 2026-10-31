import { GitBranch } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function Tracing() {
  return (
    <DocPage
      subtitle="The OpenTelemetry spans effGen emits, and exporting them to a collector."
      icon={<GitBranch size={48} />}
    >
      <p>
        Every hot path is instrumented with OpenTelemetry spans, so one agent run is one trace you
        can open in Jaeger, Zipkin, Grafana Tempo or anything else that speaks OTLP. You configure a
        sampler and an exporter once; the agent loop, the model adapters, the tools and the router
        already emit the spans.
      </p>

      <h2>Turn it on</h2>

      <CodeBlock filename="setup.py" code={`from effgen.observability import (
    ParentBasedSampler,
    TraceIdRatioSampler,
    reset_tracing,
    setup_tracing,
)

reset_tracing()
setup_tracing(
    service_name="my-service",
    sampler=ParentBasedSampler(TraceIdRatioSampler(0.10)),
)
print("tracing configured")`} />

      <p>
        That is the whole integration. Running an agent now produces a tree — here read back through
        an in-memory exporter, which is also how you would assert on spans in a test:
      </p>

      <CodeBlock filename="spans.py" code={`from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from effgen import Agent, AgentConfig
from effgen.observability import AlwaysOnSampler, reset_tracing, setup_tracing
from effgen.tools import get_registry

reset_tracing()
exporter = InMemorySpanExporter()
setup_tracing(service_name="docs", sampler=AlwaysOnSampler(), exporter=exporter)

calculator = get_registry().get_tool_sync("calculator")
with Agent(AgentConfig(model="gpt-5-nano", provider="openai", tools=[calculator])) as agent:
    agent.run("What is 127 * 43? Use the calculator.")

for span in exporter.get_finished_spans():
    detail = span.attributes.get("effgen.model.name") or span.attributes.get("effgen.tool.name") or ""
    print(f"{span.name:26} {detail}")`} />

      <Terminal
        command="python spans.py"
        output={`effgen.model.call          gpt-5-nano
effgen.agent.iteration     
effgen.tool.call           calculator
effgen.agent.run           `}
        caption="One run, one trace: the run span, an iteration, the model calls inside it, and the tool call the model made."
      />

      <h2>The span tree</h2>

      <CodeBlock
        language="text"
        filename="hierarchy"
        code={`effgen.agent.run                          # the whole agent.run() call
  effgen.agent.iteration                  # one loop iteration
    effgen.model.call                     # one generation
    effgen.tool.call  [calculator]        # each tool the model asked for
    effgen.tool.call  [web_search]
  effgen.router.decision                  # policy-based routing, when enabled`}
        caption={
          <>
            All of them share one <code>trace_id</code>. The router span is what makes a{' '}
            <Link to="/routing">fallback</Link> legible in a trace viewer — you can see the provider
            change between two model calls.
          </>
        }
      />

      <h2>Samplers</h2>

      <p>
        The sampler is always declared. There is no implicit "sample everything" shipped to a
        production deployment.
      </p>

      <ApiTable
        headers={['Sampler', 'What it does', 'When']}
        rows={[
          [<code>AlwaysOnSampler()</code>, 'Every span.', 'Development, and tests.'],
          [
            <code>AlwaysOffSampler()</code>,
            'No spans.',
            'Turning tracing off without unpicking the configuration.',
          ],
          [
            <code>TraceIdRatioSampler(p)</code>,
            <>
              A fraction <code>p</code> of traces. Deterministic — the same trace id always decides
              the same way.
            </>,
            'A fixed proportion of a steady stream.',
          ],
          [
            <code>RateLimitedSampler(per_second)</code>,
            'At most N traces a second, through a token bucket.',
            'High and bursty traffic, where a ratio would still flood the collector at peak.',
          ],
          [
            <code>ParentBasedSampler(root)</code>,
            "Honours the incoming decision, and delegates a root span to `root`.",
            'What you want in production, so one distributed trace is sampled as a whole.',
          ],
        ]}
      />

      <CodeBlock filename="samplers.py" code={`from effgen.observability import (
    AlwaysOffSampler,
    AlwaysOnSampler,
    ParentBasedSampler,
    RateLimitedSampler,
    TraceIdRatioSampler,
)

for sampler in (
    AlwaysOnSampler(),
    AlwaysOffSampler(),
    TraceIdRatioSampler(0.05),
    RateLimitedSampler(20.0),
    ParentBasedSampler(TraceIdRatioSampler(0.05)),
):
    print(f"{type(sampler).__name__:22} {sampler.get_description()}")`} />

      <Terminal command="python samplers.py" output={`AlwaysOnSampler        AlwaysOnSampler
AlwaysOffSampler       AlwaysOffSampler
TraceIdRatioSampler    TraceIdRatioSampler(0.0500)
RateLimitedSampler     RateLimitedSampler(20.0/s)
ParentBasedSampler     ParentBasedSampler(root=TraceIdRatioSampler(0.0500))`} />

      <h2>Exporters</h2>

      <ParamTable
        nameLabel="setup_tracing"
        params={[
          { name: 'service_name', type: 'str', description: 'The service name attached to every span.' },
          {
            name: 'sampler',
            type: 'Sampler',
            description: 'One of the five above. Declare it rather than inheriting a default.',
          },
          {
            name: 'exporter',
            type: 'SpanExporter',
            description: 'Any OpenTelemetry exporter — OTLP over gRPC or HTTP, or an in-memory one in a test.',
          },
          {
            name: 'export_to_console',
            type: 'bool',
            default: 'False',
            description: 'Print spans to the console. Useful while developing, noisy anywhere else.',
          },
        ]}
      />

      <CodeBlock
        filename="otlp.py"
        code={`from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from effgen.observability import ParentBasedSampler, TraceIdRatioSampler, setup_tracing

setup_tracing(
    service_name="my-effgen-service",
    sampler=ParentBasedSampler(TraceIdRatioSampler(0.1)),
    exporter=OTLPSpanExporter(endpoint="http://localhost:4317"),
)`}
        caption={
          <>
            The OTLP exporter comes from the OpenTelemetry SDK, not from effGen — install it
            alongside. With no SDK installed at all, every tracing helper is a no-op rather than an
            import error.
          </>
        }
      />

      <h2>Span attributes</h2>

      <p>
        Every attribute name lives in <code>effgen/observability/spans.py</code> and is reachable as
        a constant — <code>AgentAttrs</code>, <code>ModelAttrs</code>, <code>ToolAttrs</code> — so a
        query you write against a trace backend is spelled the same way the emitter spells it.
      </p>

      <ApiTable
        headers={['Attribute', 'Type', 'On', 'What it holds']}
        rows={[
          [<code>effgen.agent.preset</code>, 'string', 'run, iteration', 'The agent name or preset.'],
          [<code>effgen.agent.iteration</code>, 'int', 'iteration', 'Loop counter.'],
          [<code>effgen.agent.run_id</code>, 'string', 'run', 'The run id, which the log lines also carry.'],
          [<code>effgen.agent.task</code>, 'string', 'run', 'The task text, truncated to 500 characters.'],
          [<code>effgen.model.provider</code>, 'string', 'model.call', 'Provider name.'],
          [<code>effgen.model.name</code>, 'string', 'model.call', 'Model id.'],
          [<code>effgen.model.input_tokens</code>, 'int', 'model.call', 'Prompt tokens.'],
          [<code>effgen.model.output_tokens</code>, 'int', 'model.call', 'Completion tokens.'],
          [<code>effgen.model.cached_tokens</code>, 'int', 'model.call', 'Cache-hit prompt tokens.'],
          [
            <code>effgen.model.cost_usd</code>,
            'float',
            'model.call',
            'Per-call cost — absent, not zero, when the model publishes no rate.',
          ],
          [
            <code>effgen.model.outcome</code>,
            'string',
            'model.call',
            <>
              <code>ok</code>, <code>error</code> or <code>timeout</code>.
            </>,
          ],
          [<code>effgen.model.latency_ms</code>, 'float', 'model.call', 'End-to-end latency.'],
          [<code>effgen.model.reasoning_effort</code>, 'string', 'model.call', 'On a reasoning model.'],
          [<code>effgen.model.thinking_budget</code>, 'int', 'model.call', 'On a thinking model.'],
          [<code>effgen.model.parts_count</code>, 'int', 'model.call', 'Multimodal content parts.'],
          [<code>effgen.tool.name</code>, 'string', 'tool.call', 'Tool name.'],
          [<code>effgen.tool.input</code>, 'string', 'tool.call', 'Serialised input, capped at 500 characters.'],
          [
            <code>effgen.tool.status</code>,
            'string',
            'tool.call',
            <>
              <code>ok</code>, <code>error</code> or <code>timeout</code>.
            </>,
          ],
          [<code>effgen.tool.latency_ms</code>, 'float', 'tool.call', 'Tool execution latency.'],
          [<code>effgen.router.policy</code>, 'string', 'router.decision', 'The policy that chose.'],
          [<code>effgen.router.selected_provider</code>, 'string', 'router.decision', 'What it chose.'],
          [<code>effgen.router.considered</code>, 'string', 'router.decision', 'The candidate pairs it weighed.'],
          [<code>effgen.router.eliminated_count</code>, 'int', 'router.decision', 'How many it ruled out.'],
        ]}
        caption={
          <>
            <code>effgen.model.cost_usd</code> being absent rather than <code>0.0</code> is the same
            rule <Link to="/cost">cost and budgets</Link> follows everywhere: a real zero means a
            free tier.
          </>
        }
      />

      <h2>Instrumenting your own code</h2>

      <p>
        The same context managers the framework uses are importable, so work you do around an agent
        joins the same trace instead of sitting outside it.
      </p>

      <CodeBlock filename="custom.py" code={`from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from effgen.observability import (
    AlwaysOnSampler,
    ModelAttrs,
    reset_tracing,
    setup_tracing,
    start_agent_run,
    start_model_call,
    start_tool_call,
)

reset_tracing()
exporter = InMemorySpanExporter()
setup_tracing(service_name="docs", sampler=AlwaysOnSampler(), exporter=exporter)

with start_agent_run(preset="my_agent", task="hello", run_id="abc"):
    with start_model_call(provider="openai", model="gpt-5-nano") as span:
        span.set_attribute(ModelAttrs.INPUT_TOKENS, 400)
        span.set_attribute(ModelAttrs.OUTPUT_TOKENS, 60)
    with start_tool_call("calculator", "sqrt(16)"):
        pass

for span in exporter.get_finished_spans():
    print(span.name, dict(span.attributes))`} />

      <Terminal command="python custom.py" output={`effgen.model.call {'effgen.model.provider': 'openai', 'effgen.model.name': 'gpt-5-nano', 'effgen.model.input_tokens': 400, 'effgen.model.output_tokens': 60, 'effgen.model.latency_ms': 0.0, 'effgen.model.outcome': 'ok'}
effgen.tool.call {'effgen.tool.name': 'calculator', 'effgen.tool.input': 'sqrt(16)', 'effgen.tool.latency_ms': 0.0, 'effgen.tool.status': 'ok'}
effgen.agent.run {'effgen.agent.preset': 'my_agent', 'effgen.agent.task': 'hello', 'effgen.agent.run_id': 'abc'}`} maxLines={14} />

      <p>Every one of them behaves the same way on the way out:</p>

      <ul>
        <li>
          <code>latency_ms</code> and a default outcome or status attribute are set automatically.
        </li>
        <li>
          An exception propagating out marks the span <code>ERROR</code> and records the exception.
        </li>
        <li>
          They never raise. A tracing failure is logged at <code>DEBUG</code> and swallowed, because
          losing a span is not a reason to lose a request.
        </li>
      </ul>

      <h3>Retry events</h3>

      <CodeBlock filename="retry_event.py" code={`from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from effgen.observability import (
    AlwaysOnSampler,
    record_retry_attempt,
    reset_tracing,
    setup_tracing,
    start_model_call,
)

reset_tracing()
exporter = InMemorySpanExporter()
setup_tracing(service_name="docs", sampler=AlwaysOnSampler(), exporter=exporter)

with start_model_call(provider="openai", model="gpt-5-nano") as span:
    record_retry_attempt(span, attempt=1, reason="http_429", delay_s=2.0)

for span in exporter.get_finished_spans():
    for event in span.events:
        print(event.name, dict(event.attributes))`} />

      <Terminal
        command="python retry_event.py"
        output={`effgen.retry.attempt {'effgen.retry.attempt': 1, 'effgen.retry.reason': 'http_429', 'effgen.retry.delay_s': 2.0}`}
        caption={
          <>
            The event shows up in a span's detail view as <code>effgen.retry.attempt</code>, with{' '}
            <code>attempt</code>, <code>reason</code> and <code>delay_s</code>. A honoured{' '}
            <code>Retry-After</code> is visible as the delay —{' '}
            <Link to="/reliability">Reliability</Link> covers the policy that sets it.
          </>
        }
      />

      <h2>Traces and logs together</h2>

      <p>
        When a span is active, the log formatter reads the current <code>trace_id</code> and{' '}
        <code>span_id</code> and puts them on every line. Nothing extra is needed beyond calling{' '}
        <code>setup_tracing()</code> before <code>configure_logging()</code>, and it is what lets you
        jump from a log line to the trace it came from.{' '}
        <Link to="/metrics">Metrics and logging</Link> has the line schema.
      </p>

      <h2>Testing</h2>

      <CodeBlock
        filename="test_spans.py"
        code={`from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from effgen.observability import AlwaysOnSampler, reset_tracing, setup_tracing


def test_agent_emits_spans():
    reset_tracing()
    exporter = InMemorySpanExporter()
    setup_tracing(sampler=AlwaysOnSampler(), exporter=exporter)

    # ... run your agent ...

    names = [span.name for span in exporter.get_finished_spans()]
    assert "effgen.agent.run" in names
    assert "effgen.model.call" in names`}
        caption={
          <>
            Call <code>reset_tracing()</code> in the fixture teardown too, so the next test starts
            with a clean provider rather than the previous test's.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'No spans anywhere',
            <>
              <code>setup_tracing()</code> was never called, or the OpenTelemetry SDK is not
              installed.
            </>,
            <>
              Install the SDK and call it once at startup. With no SDK the helpers are no-ops by
              design — nothing raises, so nothing tells you.
            </>,
          ],
          [
            'Spans in development, none in production',
            <>
              A <code>TraceIdRatioSampler</code> is doing its job.
            </>,
            <>
              Raise the ratio, or use <code>RateLimitedSampler</code> so you always get some traces
              regardless of volume.
            </>,
          ],
          [
            'Half a distributed trace',
            <>
              A root sampler was used where <code>ParentBasedSampler</code> was needed, so each
              service decided independently.
            </>,
            <>
              Wrap the root sampler: <code>ParentBasedSampler(TraceIdRatioSampler(0.1))</code>.
            </>,
          ],
          [
            'Spans from the previous test',
            <>
              <code>reset_tracing()</code> was not called between tests.
            </>,
            'Put it in the fixture, on both sides.',
          ],
          [
            <>
              A tool span with <code>status=error</code> but a successful run
            </>,
            'A tool failed and the agent recovered — which is normal, and is exactly what the span is for.',
            <>
              Cross-check against <code>response.tool_calls.failed</code> —{' '}
              <Link to="/observability">Observability</Link> shows the pairing.
            </>,
          ],
          [
            <>
              No <code>effgen.model.cost_usd</code> on some spans
            </>,
            'That model publishes no per-token rate.',
            <>
              Expected. <Link to="/cost">Cost and budgets</Link> explains why an absent cost is
              better than an invented zero.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/observability', '/metrics', '/slos']} />
    </DocPage>
  );
}
