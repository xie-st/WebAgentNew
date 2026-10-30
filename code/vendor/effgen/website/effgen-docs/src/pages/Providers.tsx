import { Plug } from 'lucide-react';
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
import { modelCount, providerCount, providersWithCatalog, siteData } from '../siteData';

const PROVIDERS = siteData.models.providers;

/** The catalogued adapters, in the order the registry lists them. */
const WITH_CATALOG = PROVIDERS.filter((provider) => provider.models > 0);

export default function Providers() {
  return (
    <DocPage
      subtitle="Every provider adapter, the keys it reads, and what it supports."
      icon={<Plug size={48} />}
    >
      <p>
        {providerCount} provider adapters are registered. {providersWithCatalog} of them ship a
        bundled catalog — {modelCount} models between them, each with its context window, its
        prices and what it can do. The tenth, <code>openai_compatible</code>, ships no catalog
        because it serves whatever the endpoint you point it at serves.
      </p>

      <h2>Using one</h2>

      <CodeBlock
        filename="provider.py"
        code={`from effgen import create_agent

agent = create_agent("minimal", "gemini:gemini-3.1-flash-lite")
print(agent.run("Name the largest ocean in one word.").text.strip())`}
      />

      <Terminal command="python provider.py" output={`Pacific`} />

      <p>
        Changing provider is changing the prefix. Nothing else in the program moves — the agent,
        the tools, the response and the errors are the same across all of them.
      </p>

      <h2>What effGen sees in this environment</h2>

      <CodeBlock
        filename="keys.py"
        code={`from effgen import check_keys

for provider, info in sorted(check_keys().items()):
    state = "ready" if info["available"] else "no key"
    print(f"{provider:18s} {state:7s} {info['env_key'] or ', '.join(info['env_keys_checked'])}")`}
      />

      <Terminal command="python keys.py" output={`anthropic          no key  ANTHROPIC_API_KEY
cerebras           ready   CEREBRAS_API_KEY
fireworks          ready   FIREWORKS_API_KEY
gemini             ready   GOOGLE_API_KEY
groq               ready   GROQ_API_KEY
hf                 ready   HF_TOKEN
openai             ready   OPENAI_API_KEY
openai_compatible  no key  EFFGEN_BASE_URL, OPENAI_BASE_URL, OPENAI_API_BASE
replicate          ready   REPLICATE_API_TOKEN
together           ready   TOGETHER_API_KEY`} />

      <p>
        <code>effgen doctor</code> prints the same thing as a table, with a system report beside
        it. Neither reads a key value.
      </p>

      <h2>The adapters</h2>

      <ApiTable
        headers={['Provider', 'Environment variable', 'Catalogued models', 'Default model']}
        rows={PROVIDERS.map((provider) => [
          <code>{provider.name}</code>,
          provider.env_keys.map((key, i) => (
            <span key={key}>
              {i > 0 ? ' or ' : ''}
              <code>{key}</code>
            </span>
          )),
          provider.models === 0 ? <em>the endpoint's own</em> : String(provider.models),
          provider.default ? <code>{provider.default}</code> : '—',
        ])}
        caption="Read from the installed provider registry. A default is the model an adapter reaches for when a call names none."
      />

      <h2>What each catalog carries</h2>

      <ApiTable
        headers={['Provider', 'Tool-capable', 'Vision', 'Audio', 'Free tier', 'Largest window', 'Catalog checked']}
        rows={WITH_CATALOG.map((provider) => [
          <code>{provider.name}</code>,
          String(provider.supports_tools),
          String(provider.supports_vision),
          String(provider.supports_audio),
          String(provider.free_tier),
          provider.max_context ? provider.max_context.toLocaleString() : '—',
          provider.verified_on ?? '—',
        ])}
        caption={
          <>
            Counts of models in each catalog carrying that capability, not a yes/no for the
            provider. Across all of them: {siteData.models.capability_totals.supports_tools}{' '}
            tool-capable, {siteData.models.capability_totals.supports_vision} vision-capable,{' '}
            {siteData.models.capability_totals.supports_audio} audio-capable,{' '}
            {siteData.models.capability_totals.free_tier} free-tier and{' '}
            {siteData.models.capability_totals.priced} priced.
          </>
        }
      />

      <Callout type="tip" title="Filter the catalog rather than reading the table">
        <p>
          <code>effgen models browse --vision --free --min-context 100000</code> answers "which
          models can do this" directly, across every provider at once. See{' '}
          <Link to="/catalog">The model catalog and pricing</Link>.
        </p>
      </Callout>

      <h2>What is different about each one</h2>

      <h3>OpenAI</h3>
      <p>
        Chat models and the reasoning o-series. A reasoning model takes{' '}
        <code>reasoning_effort</code>, and spends part of its output budget on hidden reasoning
        before it emits a visible token — so it needs a larger <code>max_tokens</code> than a chat
        model doing the same job. Prompt prefixes of 1,024 tokens or more are cached
        automatically, and the saving shows up as{' '}
        <code>metadata["cached_input_tokens"]</code>. Structured output is enforced at the token
        level; see <Link to="/generation">Generation controls and structured output</Link>.
      </p>

      <h3>Gemini</h3>
      <p>
        Carries three things the others do not: an explicit thinking budget for extended
        reasoning, Google Search grounding (which populates <code>response.sources</code> from
        what was actually retrieved), and a Files API for uploading a document once and
        referencing it across calls.
      </p>

      <h3>Anthropic</h3>
      <p>
        Extended thinking, including the redacted-thinking blocks that have to be preserved across
        turns for a multi-turn conversation to stay valid — <code>build_assistant_message()</code>{' '}
        keeps them for you. It is also the one provider with <em>explicit</em> prompt caching, set
        per content block rather than inferred from a prefix.
      </p>

      <h4>Prompt caching</h4>
      <p>
        You mark blocks with <code>cache_control</code>; the first request that hits a marked block
        writes the prefix to cache, and later requests sharing that prefix read it back. Anthropic
        allows at most <strong>four</strong> markers per request across tools, system and messages
        combined, and effGen raises <code>ValueError</code> before making the call when there are
        more.
      </p>

      <ApiTable
        headers={['TTL', 'Write cost', 'Read cost', 'For']}
        rows={[
          [<code>"5m"</code>, '1.25× input', '0.1× input', 'Most uses. The default.'],
          [<code>"1h"</code>, '2× input', '0.1× input', 'Long-lived contexts and heavy tool lists.'],
        ]}
      />

      <p>
        Cache markers are evaluated <code>tools → system → messages</code>, and a change at any
        level invalidates that level and everything after it — so the most stable content should
        come first. If you have to drop a marker, the order of value is the system prompt's last
        block, then the last tool spec, then message blocks. A block below the model's minimum
        size is billed at the normal rate with no error.
      </p>

      <CodeBlock
        continues
        code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="anthropic:claude-sonnet-4-6",
    system_prompt=LONG_STABLE_INSTRUCTIONS,
    cache_system_prompt=True,   # default: marks the system prompt's last block
    cache_tools=True,           # default: marks the last tool spec
))

r = agent.run("Summarise the attached policy.")
print(r.metadata["cached_input_tokens"], r.metadata["cache_creation_tokens"])`}
        caption="Both metadata fields are always present, and are 0 when caching did not apply. For direct adapter calls, apply_cache_to_system() and apply_cache_to_last_tool() do the marking."
      />

      <Callout type="note" title="This snippet was not run">
        <p>
          There is no Anthropic credential in the environment this page was written in, so the
          caching example above is the documented contract rather than a captured run. Everything
          else on this page was run.
        </p>
      </Callout>

      <h3>Groq, Cerebras, Together and Fireworks</h3>
      <p>
        Fast serverless inference over open-weights models, all four with native tool calling and
        streaming. Together's catalog is the largest here; some of its models need a dedicated
        endpoint started before they can be called, and those are marked in the catalog and skipped
        by cost-based routing. Cerebras and Groq publish tight free-tier rate limits, which is what{' '}
        <code>RateLimitCoordinator</code> tracks — it waits as a limit approaches rather than
        letting the call fail.
      </p>

      <h3>Replicate</h3>
      <p>
        Reaches models that are not served anywhere else. Two things differ: many public models are
        billed by compute seconds rather than by token, so a token count does not give you a cost;
        and a prediction is polled to completion rather than streamed from the first byte, which
        makes <code>ModelTimeoutError</code> a normal outcome to handle.
      </p>

      <h3>HuggingFace Inference</h3>
      <p>
        <code>hf:&lt;repo&gt;</code> calls the hosted Inference API, routing to whichever
        underlying provider serves that repo. Its catalog is refreshed from the live API rather
        than only bundled, and effGen warns when the bundled snapshot has drifted from what is
        actually served. To run the same repo on your own machine, drop the prefix — see{' '}
        <Link to="/local-models">Local models and engines</Link>.
      </p>

      <h3>openai_compatible</h3>
      <p>
        Not a hosted provider: it is how effGen talks to a server of your own. It reads an
        endpoint, not a credential, from{' '}
        {PROVIDERS.find((provider) => provider.name === 'openai_compatible')?.env_keys.map(
          (key, i) => (
            <span key={key}>
              {i > 0 ? ', then ' : ''}
              <code>{key}</code>
            </span>
          ),
        )}
        . <Link to="/openai-compatible">Any OpenAI-compatible server</Link> is its page.
      </p>

      <h2>Switching provider</h2>

      <CodeBlock
        filename="switch.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

for model_id in ("openai:gpt-5-nano", "gemini:gemini-3.1-flash-lite"):
    agent = Agent(AgentConfig(model=model_id, tools=[Calculator()], temperature=0.0))
    print(model_id, "->", agent.run("What is (17 * 23) + 12?").output.strip())`}
      />

      <Terminal command="python switch.py" output={`openai:gpt-5-nano -> 403

Briefly: 17 × 23 = 391, and 391 + 1
gemini:gemini-3.1-flash-lite -> 403`} />

      <p>
        To have effGen make that choice for you — cheapest first, fastest first, or failing over
        when a provider rate-limits you — see <Link to="/routing">Model routing and fallback</Link>.
      </p>

      <h2>Errors every adapter raises</h2>

      <ParamTable
        nameLabel="Error"
        params={[
          {
            name: 'ModelAuthError',
            type: 'not retryable',
            description: 'The credential was missing or rejected. Fix the key.',
          },
          {
            name: 'ModelNotFoundError',
            type: 'not retryable',
            description: 'The provider does not serve that id. The message suggests near matches.',
          },
          {
            name: 'ModelUnavailableError',
            type: 'not retryable',
            description: 'The model exists but is not available on this tier or endpoint.',
          },
          {
            name: 'InvalidRequestError',
            type: 'not retryable',
            description: 'The request itself was malformed — a bad schema, an unsupported parameter.',
          },
          {
            name: 'ModelRefusalError',
            type: 'not retryable',
            description: 'The model returned a refusal instead of content. The text is on e.refusal_message.',
          },
          {
            name: 'CapabilityNotSupportedError',
            type: 'not retryable',
            description: 'The model cannot do what was asked — vision on a text-only model, for instance.',
          },
          {
            name: 'RateLimitExceeded',
            type: 'retryable',
            description: "The provider's limit was hit. A router fails over on this.",
          },
          {
            name: 'ProviderTransientError',
            type: 'retryable',
            description: 'The provider answered badly — a 5xx. Retrying can succeed.',
          },
          {
            name: 'ModelTimeoutError',
            type: 'retryable',
            description: 'The call did not finish in time.',
          },
          {
            name: 'BackendUnreachableError',
            type: 'never retried',
            description:
              'Nothing answered: a refused connection, an unresolvable host, a missing route. Raised whatever raise_on_error says, because there is no result to inspect.',
          },
        ]}
        caption={
          <>
            All from <code>effgen.models.errors</code>. Whether an error is retryable is what{' '}
            <Link to="/routing">the router</Link> decides failover on, and{' '}
            <Link to="/errors">Errors and exceptions</Link> lists the full hierarchy.
          </>
        }
      />

      <SeeAlso paths={['/models', '/catalog', '/routing']} />
    </DocPage>
  );
}
