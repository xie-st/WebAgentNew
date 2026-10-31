"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiAlertTriangle,
  FiArrowRight,
  FiCpu,
  FiDatabase,
  FiExternalLink,
  FiFileText,
  FiGlobe,
  FiHardDrive,
  FiServer,
} from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Terminal from "@/components/ui/Terminal";
import Figure from "@/components/ui/Figure";
import ParamTable from "@/components/ui/ParamTable";
import CodeSample from "@/components/ui/CodeSample";
import RouteLink from "@/components/ui/RouteLink";
import { modelsCapture } from "@/components/captures";
import { figureOf, webCapture } from "@/components/webCaptures";
import { siteData, version } from "@/components/siteData";
import {
  DOCS_COMPATIBLE_URL,
  DOCS_REGISTRY_URL,
  DOCS_ROUTER_URL,
  adapterNotes,
  baseUrlSources,
  compatibleServers,
  engines,
  providerAliases,
  providerRow,
} from "./modelsData";
import { accentTextStyle } from "@/components/accentText";

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

const models = siteData.models;

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
      className={`py-16 relative scroll-mt-24 ${tinted ? "bg-gray-50 dark:bg-[#030f07]" : ""}`}
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
          {lede && <p className="mt-4 text-gray-600 dark:text-gray-400 leading-relaxed">{lede}</p>}
        </motion.div>
        {children}
      </Container>
    </section>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-6">
      <h3 className="text-base font-bold text-gray-900 dark:text-white mb-2">{title}</h3>
      <div className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed space-y-3">
        {children}
      </div>
    </div>
  );
}

const numeric = (value: number) => value.toLocaleString("en-US");

export default function ModelsView() {
  const { ref: heroRef, inView: heroInView } = useInView({ triggerOnce: true, threshold: 0.05 });

  const browseDefault = modelsCapture("models-browse-default");
  const browseFreeTools = modelsCapture("models-browse-free-tools");
  const browseVision = modelsCapture("models-browse-vision");
  const browseCheap = modelsCapture("models-browse-cheap");
  const browseProvider = modelsCapture("models-browse-provider");
  const browseSearch = modelsCapture("models-browse-search");
  const modelsList = modelsCapture("models-list");
  const modelsInfo = modelsCapture("cli-models-info");

  const headline = [
    {
      value: String(models.adapter_count),
      label: "Provider adapters",
      accent: "#00ff88",
      icon: FiGlobe,
    },
    {
      value: numeric(models.models),
      label: `Catalogued models, ${models.with_catalog_count} providers`,
      accent: "#00e5ff",
      icon: FiDatabase,
    },
    {
      value: String(models.local_engines.length),
      label: "Local engines",
      accent: "#a78bfa",
      icon: FiHardDrive,
    },
    {
      value: "1",
      label: "base_url to reach any other server",
      accent: "#ffd700",
      icon: FiServer,
    },
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
                <FiGlobe size={14} />
                effgen models · {version}
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                Any model,{" "}
                <span className="gradient-text">anywhere you already run it</span>
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                {models.adapter_count} provider adapters, {models.with_catalog_count} of
                them carrying a bundled catalog of {numeric(models.models)} models with
                their context windows, prices and capabilities. The tenth carries none on
                purpose: point it at a server you already run — vLLM, SGLang, TGI,
                llama.cpp, Ollama, LM Studio, a proxy, a gateway — and it drives the model
                that server serves. And{" "}
                {models.local_engines.length} local engines run weights in the process,
                including on Apple Silicon.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <a
                  href="#compatible"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
                >
                  Point it at your own server
                  <FiArrowRight size={15} />
                </a>
                <a
                  href={DOCS_REGISTRY_URL}
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

        <section className="pb-12 relative">
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

        {/* The catalog, above the fold */}
        <section className="pb-4 relative">
          <Container className="relative z-10">
            <Terminal
              command={browseDefault.command}
              output={browseDefault.text}
              title="effgen models browse"
              maxLines={26}
            />
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 max-w-3xl leading-relaxed">
              One table across every provider, so &ldquo;the cheapest vision model over
              128k context&rdquo; is one command rather than nine browser tabs. Every
              figure in it comes from the bundled catalog, and the footer names the
              snapshot it came from and the command that refreshes it.
            </p>
          </Container>
        </section>

        {/* OpenAI-compatible */}
        <Band
          id="compatible"
          eyebrow="Any OpenAI-protocol server"
          tinted
          title={
            <>
              One argument, and effGen drives{" "}
              <span className="gradient-text">the server you already have</span>
            </>
          }
          lede={
            <>
              vLLM, SGLang, TGI, llama.cpp&rsquo;s server, Ollama, LM Studio, LiteLLM and
              most corporate gateways all expose the OpenAI chat-completions API. Give
              effGen a <code className="font-mono text-sm">base_url</code> and it talks to
              the model you are already serving, instead of loading a second copy of the
              weights inside the agent&rsquo;s process.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <CodeSample
                language="python"
                accent="#00ff88"
                code={`from effgen.models import load_model

model = load_model(
    "gemini:gemini-3.1-flash-lite",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)
print(model.generate("What is 6 times 7?").text)`}
                output={`42`}
                outputLabel="what that printed"
              />
              <CodeSample
                language="python"
                accent="#00e5ff"
                code={`from effgen import Agent, AgentConfig

with Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    base_url="http://127.0.0.1:8000/v1",
)) as agent:
    print(agent.run("What is 6 times 7?").output)`}
                output={`42`}
                outputLabel="what that printed"
              />
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Both were run against an{" "}
                <code className="font-mono text-xs">effgen serve</code> on this machine —
                which speaks the same protocol as everything in the list, and is the
                easiest one to reproduce. A <code className="font-mono text-xs">base_url</code>{" "}
                with no provider named is the whole instruction: it selects this adapter
                rather than falling through to a local download of the model id.
              </p>
            </div>

            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {compatibleServers.map((server) => (
                  <div
                    key={server.name}
                    className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-4"
                  >
                    <div className="text-sm font-bold text-gray-900 dark:text-white">
                      {server.name}
                    </div>
                    <p className="mt-1 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                      {server.what}
                    </p>
                  </div>
                ))}
              </div>
              <Card title="Why serve the model separately">
                <p>
                  Loading in-process means one copy of the weights per agent process, no
                  sharing between agents, no continuous batching across callers, and a GPU
                  tied to the agent&rsquo;s lifetime. A shared server fixes all four: the
                  weights load once, every caller&rsquo;s requests batch together, and the
                  GPU outlives any individual run.
                </p>
                <p>
                  It is also the only way to have several frameworks — or several versions
                  of your own service — generate under identical settings, which is what a
                  fair comparison needs.
                </p>
              </Card>
            </div>
          </div>

          <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                Ask the endpoint what it serves
              </h3>
              <CodeSample
                language="python"
                accent="#a78bfa"
                code={`from effgen.models import load_model

model = load_model(
    "gemini:gemini-3.1-flash-lite",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)
print(model.list_served_models())`}
                output={`['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-3.5-turbo-instruct', 'default', 'effgen-default', 'gemini:gemini-3.1-flash-lite']`}
                outputLabel="what that server reported"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                An endpoint that does not implement{" "}
                <code className="font-mono text-xs">/models</code> returns an empty list
                rather than failing — some minimal servers have nothing to say about
                themselves.
              </p>
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                Where the endpoint comes from
              </h3>
              <ParamTable
                nameLabel="Order"
                params={baseUrlSources.map((source) => ({
                  name: source.order,
                  description: source.what,
                }))}
                caption="effGen's own variable is consulted first, so pointing effGen at a server does not redirect every other OpenAI client on the machine."
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                <code className="font-mono text-xs">provider=&quot;openai&quot;</code>{" "}
                <em>with</em> a <code className="font-mono text-xs">base_url</code> routes
                here too, because a URL of your own means the model ids, the context
                window and the pricing are the server&rsquo;s rather than
                OpenAI&rsquo;s. Without one it stays on OpenAI, so a machine-wide{" "}
                <code className="font-mono text-xs">OPENAI_BASE_URL</code> set for
                something unrelated cannot silently reroute a plain OpenAI call.
              </p>
              <ul className="mt-4 flex flex-wrap gap-2">
                {providerAliases.map((alias) => (
                  <li
                    key={alias}
                    className="px-2.5 py-1 rounded-lg text-xs font-mono bg-gray-100 dark:bg-gray-900 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-800"
                  >
                    {alias}
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
                All accepted spellings of the same provider.
              </p>
            </div>
          </div>

          <div className="mt-12">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
              What effGen refuses to assume about your server
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <div>
                <div className="rounded-2xl border border-yellow-500/30 bg-yellow-500/[0.05] p-5 mb-6">
                  <div className="flex items-start gap-3">
                    <FiAlertTriangle
                      className="mt-0.5 shrink-0 text-yellow-600 dark:text-yellow-400"
                      size={18}
                    />
                    <div>
                      <h4 className="text-sm font-bold text-gray-900 dark:text-white">
                        The context window is a guess, and it says so
                      </h4>
                      <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                        The protocol carries no way for a server to publish its window, so
                        effGen assumes 32,768 and warns once, naming the value, the model
                        and the endpoint. effGen plans compaction against that number, so a
                        server started with a smaller window fails later, at the call, far
                        from where the number was chosen.
                      </p>
                    </div>
                  </div>
                </div>
                <CodeSample
                  language="python"
                  accent="#ffd700"
                  code={`from effgen import load_model

assumed = load_model(
    "gemini:gemini-3.1-flash-lite",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)
print("assumed:", assumed.get_context_length())

told = load_model(
    "gemini:gemini-3.1-flash-lite",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
    context_length=1_000_000,
)
print("told:   ", told.get_context_length())`}
                  output={`WARNING effgen.models.openai_compatible_adapter: No context_length given for
'gemini:gemini-3.1-flash-lite' at http://127.0.0.1:8000/v1; assuming 32768
tokens. If the server was started with a smaller window (vLLM's
--max-model-len, TGI's --max-total-tokens), pass context_length=<the real
number> — planning against a window the server does not have fails later,
at the call.
assumed: 32768
told:    1000000`}
                  outputLabel="the warning on stderr, then what the program printed"
                />
              </div>
              <div className="space-y-6">
                <Card title="Cost: nothing rather than $0">
                  <p>
                    A call through this adapter reports no price. What your own server
                    costs is not something effGen can derive from a token count, so it
                    states nothing — and counts the call as unpriced rather than adding
                    zero to a total.
                  </p>
                  <CodeSample
                    className="!bg-transparent"
                    language="python"
                    accent="#ff6b6b"
                    code={`from effgen import Agent, AgentConfig

with Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    base_url="http://127.0.0.1:8000/v1",
)) as agent:
    r = agent.run("Reply with the single word ok.")
    print("tokens:", r.metadata["total_tokens"])
    print("cost:  ", r.metadata.get("cost_usd"))
    print("unpriced calls:", r.metadata.get("unpriced_calls"))`}
                    output={`tokens: 33
cost:   None
unpriced calls: 1`}
                    outputLabel="what that printed"
                  />
                </Card>
                <Card title="Ids, sampling and reasoning">
                  <p>
                    The server serves its own model ids, so no OpenAI catalog is consulted
                    for any of them. The full sampling surface —{" "}
                    <code className="font-mono text-xs">top_p</code>,{" "}
                    <code className="font-mono text-xs">top_k</code>,{" "}
                    <code className="font-mono text-xs">seed</code>, the penalties — is
                    offered, which every implementation of the protocol accepts. Pass{" "}
                    <code className="font-mono text-xs">supports_reasoning=True</code> if
                    what you serve emits a reasoning stream.
                  </p>
                  <p>
                    A local server that authenticates nothing needs no credential — effGen
                    sends a placeholder, which vLLM, SGLang, TGI, llama.cpp and Ollama all
                    accept. Pass a real one for a gateway that checks it.
                  </p>
                </Card>
              </div>
            </div>
          </div>
        </Band>

        {/* The providers */}
        <Band
          id="providers"
          eyebrow="The providers"
          title={
            <>
              {models.adapter_count} adapters,{" "}
              <span className="gradient-text">
                {models.with_catalog_count} with a catalog
              </span>
            </>
          }
          lede={
            <>
              Both numbers are true and they mean different things, so this page says
              which one it means. Every row below — the model count, the capability
              counts, the largest context window, the default model, the date the catalog
              was last checked against the provider&rsquo;s live API and the environment
              variable the adapter reads — is read out of the installed package.
            </>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {adapterNotes.map((adapter) => {
              const row = providerRow(adapter.name);
              const hasCatalog = row.models > 0;
              return (
                <article
                  key={adapter.name}
                  className="relative rounded-xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-5"
                >
                  <div
                    className="absolute left-0 top-5 bottom-5 w-0.5 rounded-full"
                    style={{ background: adapter.accent }}
                  />
                  <div className="pl-4">
                    <div className="flex flex-wrap items-baseline gap-2">
                      <h3 className="text-base font-bold text-gray-900 dark:text-white">
                        {adapter.label}
                      </h3>
                      <code
                        className="text-xs font-mono"
                        style={accentTextStyle(adapter.accent)}
                      >
                        {adapter.name}
                      </code>
                    </div>
                    <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                      {adapter.note}
                    </p>
                    <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      <div className="flex justify-between gap-2">
                        <dt className="text-gray-600 dark:text-gray-400">Models</dt>
                        <dd className="font-mono text-gray-700 dark:text-gray-300">
                          {hasCatalog ? numeric(row.models) : "the server's"}
                        </dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt className="text-gray-600 dark:text-gray-400">Tool-calling</dt>
                        <dd className="font-mono text-gray-700 dark:text-gray-300">
                          {hasCatalog ? numeric(row.supports_tools) : "—"}
                        </dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt className="text-gray-600 dark:text-gray-400">Vision</dt>
                        <dd className="font-mono text-gray-700 dark:text-gray-300">
                          {hasCatalog ? numeric(row.supports_vision) : "—"}
                        </dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt className="text-gray-600 dark:text-gray-400">Free tier</dt>
                        <dd className="font-mono text-gray-700 dark:text-gray-300">
                          {hasCatalog ? numeric(row.free_tier) : "—"}
                        </dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt className="text-gray-600 dark:text-gray-400">Max context</dt>
                        <dd className="font-mono text-gray-700 dark:text-gray-300">
                          {row.max_context ? numeric(row.max_context) : "you say"}
                        </dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt className="text-gray-600 dark:text-gray-400">Checked</dt>
                        <dd className="font-mono text-gray-700 dark:text-gray-300">
                          {row.verified_on ?? "—"}
                        </dd>
                      </div>
                    </dl>
                    <p className="mt-3 text-[11px] font-mono text-gray-600 dark:text-gray-400 break-words">
                      {row.env_keys.join(" · ")}
                    </p>
                    {row.default && (
                      <p className="mt-1 text-[11px] text-gray-600 dark:text-gray-400">
                        Default when a call names no model:{" "}
                        <code className="font-mono">{row.default}</code>
                      </p>
                    )}
                  </div>
                </article>
              );
            })}
          </div>

          <div className="mt-10">
            <Terminal
              command={modelsList.command}
              output={modelsList.text}
              title="effgen models list"
              maxLines={28}
            />
          </div>
        </Band>

        {/* Local engines */}
        <Band
          id="local"
          eyebrow="Local engines"
          tinted
          title={
            <>
              Weights in the process,{" "}
              <span className="gradient-text">including on a Mac</span>
            </>
          }
          lede={
            <>
              A provider adapter calls someone else&rsquo;s server. A local engine loads
              the weights where your code is running — no key, no network, and the model
              id is the repository id. {models.local_engines.length} engines cover a GPU
              box, a throughput server, a laptop with no GPU and Apple Silicon.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-3">
              {engines().map((engine) => (
                <div
                  key={engine.name}
                  className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5"
                >
                  <div className="flex items-center gap-2">
                    <FiCpu className="text-green-700 dark:text-green-400" size={15} />
                    <code className="text-sm font-mono font-bold text-gray-900 dark:text-white">
                      engine=&quot;{engine.name}&quot;
                    </code>
                  </div>
                  <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {engine.what}
                  </p>
                </div>
              ))}
            </div>
            <div>
              <CodeSample
                language="python"
                accent="#a78bfa"
                code={`from effgen.models import load_model

model = load_model("Qwen/Qwen2.5-0.5B-Instruct", engine="transformers")
print(model.generate("Name one primary colour.", max_tokens=16).text.strip())`}
                output={`One primary color is blue.`}
                outputLabel="what that printed"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Run for this page on a machine with the weights already in the local
                cache. The same call with{" "}
                <code className="font-mono text-xs">engine=&quot;vllm&quot;</code> serves
                the same weights at higher throughput,{" "}
                <code className="font-mono text-xs">engine=&quot;gguf&quot;</code> takes a
                quantised file through llama.cpp, and{" "}
                <code className="font-mono text-xs">engine=&quot;mlx&quot;</code> runs on
                Apple Silicon.
              </p>
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                <code className="font-mono text-xs">
                  effgen models browse --include-local
                </code>{" "}
                adds what is already downloaded in the local Hugging Face cache to the
                table, so the zero-key option is visible beside the hosted ones.
              </p>
            </div>
          </div>
        </Band>

        {/* Catalog and pricing */}
        <Band
          id="catalog"
          eyebrow="The catalog and its prices"
          title={
            <>
              A price effGen does not have{" "}
              <span className="gradient-text">is never printed as $0</span>
            </>
          }
          lede={
            <>
              A published rate is shown as it is. A genuine free tier reads{" "}
              <code className="font-mono text-sm">free</code>. A model billed by something
              other than tokens reads <code className="font-mono text-sm">metered</code>.
              Anything else — including a catalog entry that carries an explicit zero with
              no free tier behind it — reads{" "}
              <code className="font-mono text-sm">unpriced</code>, because a fabricated $0
              is the one answer that makes a spend total wrong without looking wrong.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <Terminal
                command={browseCheap.command}
                output={browseCheap.text}
                title="effgen models browse"
                maxLines={26}
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                A price ceiling drops every model whose rate the catalog does not carry at
                all. What survives here are entries carrying an explicit zero — and the
                table refuses to call those $0.
              </p>
            </div>
            <div>
              <Terminal
                command={browseFreeTools.command}
                output={browseFreeTools.text}
                title="effgen models browse"
                maxLines={20}
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                A genuine free tier is a different thing and is labelled as one. Filters
                compose: a model has to satisfy every one supplied.
              </p>
            </div>
          </div>

          <div className="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                Sort, filter, page
              </h3>
              <Terminal
                command={browseVision.command}
                output={browseVision.text}
                title="effgen models browse"
                maxLines={24}
              />
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                A missing number sorts last on an ascending sort, so an unknown price never
                masquerades as the cheapest.
              </p>
            </div>
            <div className="space-y-6">
              <Terminal
                command={browseProvider.command}
                output={browseProvider.text}
                title="effgen models browse"
                maxLines={16}
              />
              <Terminal
                command={browseSearch.command}
                output={browseSearch.text}
                title="effgen models browse"
                maxLines={24}
              />
            </div>
          </div>

          <div className="mt-12">
            <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
              Every filter <code className="font-mono text-sm">browse</code> takes
            </h3>
            <ParamTable
              nameLabel="Flag"
              params={siteData.cli.command_options["models browse"].map((option) => ({
                name: option.name,
                description: option.description,
              }))}
              caption={`effgen models browse --help · effGen ${version}`}
            />
          </div>

          <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                One model, in detail
              </h3>
              <Terminal
                command={modelsInfo.command}
                output={modelsInfo.text}
                title="effgen models info"
                maxLines={24}
              />
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                The same catalog in the browser
              </h3>
              <Figure
                {...figureOf(webCapture("dashboard-catalog"))}
                caption={
                  <>
                    The dashboard&rsquo;s catalog panel reads the same source, with the
                    same pricing labels and the same filters.
                  </>
                }
                command="GET /dashboard/catalog.json"
              />
              <RouteLink
                to="/dashboard"
                className="mt-4 inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
              >
                The web surfaces
                <FiArrowRight size={14} />
              </RouteLink>
            </div>
          </div>
        </Band>

        {/* Routing */}
        <Band
          id="routing"
          eyebrow="Routing"
          tinted
          title={
            <>
              Name more than one model, and{" "}
              <span className="gradient-text">a router is built across them</span>
            </>
          }
          lede={
            <>
              <code className="font-mono text-sm">AgentConfig(models=[...])</code> loads
              every model named and routes between them. For a policy rather than a list,{" "}
              <code className="font-mono text-sm">ModelRouter</code> selects on capability,
              cost or measured latency, and records why every candidate it rejected was
              rejected.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <CodeSample
                language="python"
                accent="#00ff88"
                code={`from effgen import Agent, AgentConfig

with Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    models=["openai:gpt-5-nano"],
)) as agent:
    r = agent.run("Reply with the single word ok.")
    print(r.output)
    print(r.metadata["total_tokens"], "tokens ·  $%.6f" % r.metadata["cost_usd"])`}
                output={`ok
20 tokens ·  $0.000006`}
                outputLabel="what that printed"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                A model in the list that cannot be loaded is reported and skipped rather
                than taking the agent down with it, so a list can name a model that is not
                configured on every machine.
              </p>
            </div>
            <div className="space-y-4">
              <Card title="Three policies">
                <p>
                  <code className="font-mono text-xs">FirstAvailablePolicy</code> takes
                  the first provider with a configured key that supports the capabilities
                  asked for.{" "}
                  <code className="font-mono text-xs">CostBasedPolicy</code> takes the
                  cheapest that fits the budget, ranking a free tier ahead of a paid one
                  and breaking remaining ties deterministically.{" "}
                  <code className="font-mono text-xs">LatencyBasedPolicy</code> takes the
                  fastest that meets a latency budget, from measured p50s rather than from
                  seeds once any real measurement exists.
                </p>
              </Card>
              <Card title="Every rejection is on the record">
                <p>
                  A <code className="font-mono text-xs">RouterDecision</code> carries the
                  pair it chose, the policy that chose it, that policy&rsquo;s score, and{" "}
                  <code className="font-mono text-xs">eliminated</code> — one entry per
                  rejected candidate with the reason: no key and the variable it would
                  come from, a capability the model does not have, a model that needs a
                  dedicated endpoint, or simply not the cheapest.
                </p>
                <p>
                  When nothing fits the budget,{" "}
                  <code className="font-mono text-xs">NoCandidateWithinBudgetError</code>{" "}
                  is raised carrying the cheapest option that exists, so the caller can say
                  what the budget would have to be.
                </p>
                <a
                  href={DOCS_ROUTER_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  docs/models/router.md
                  <FiExternalLink size={13} />
                </a>
              </Card>
            </div>
          </div>
        </Band>

        {/* Failure */}
        <Band
          id="failure"
          eyebrow="When it does not work"
          title={
            <>
              A backend that never answered{" "}
              <span className="gradient-text">is not a result</span>
            </>
          }
          lede={
            <>
              A task that ran and failed is something you can inspect. A connection that
              was refused, a host that does not resolve and a route that does not exist
              are not, and returning one quietly is how a whole batch completes against
              nothing and looks healthy in the summary. So they raise{" "}
              <code className="font-mono text-sm">BackendUnreachableError</code> —
              whatever <code className="font-mono text-sm">raise_on_error</code> says.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <CodeSample
              language="python"
              accent="#ff6b6b"
              code={`from effgen import Agent, AgentConfig
from effgen.models.errors import BackendUnreachableError

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:9/v1",
    raise_on_error=False,
))
try:
    agent.run("What is 6 times 7?")
except BackendUnreachableError as e:
    print(type(e).__name__)
    print(str(e).split(". ")[-1])
finally:
    agent.close()`}
              output={`BackendUnreachableError
Nothing answered at that endpoint — check the server is running and the base_url, host and port are right.`}
              outputLabel="what that printed"
            />
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  A model id that is not there names the ones that are
                </h3>
                <CodeSample
                  language="bash"
                  accent="#ff9500"
                  code={`curl -s http://127.0.0.1:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"cerebras:llama3.1-8b","messages":[{"role":"user","content":"hello"}]}'`}
                  output={`{"error":{"message":"cerebras error (model='llama3.1-8b'): Unknown Cerebras model 'llama3.1-8b'. Did you mean: zai-glm-4.7, gpt-oss-120b? Available cerebras models: gpt-oss-120b, zai-glm-4.7. Model id not found — run \`effgen models list\` to see ids, \`effgen models refresh\` to update the catalog, and verify the id/provider prefix.","type":"invalid_request_error","param":null,"code":"not_found"}}`}
                  outputLabel="the 404 body, verbatim"
                />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  And no endpoint at all names the variable
                </h3>
                <CodeSample
                  language="python"
                  accent="#a78bfa"
                  code={`from effgen.models import load_model

try:
    load_model("my-model", provider="openai_compatible")
except ValueError as e:
    print(e)`}
                  output={`An OpenAI-compatible endpoint needs a base_url. Pass base_url='http://host:port/v1', or set EFFGEN_BASE_URL (or OPENAI_BASE_URL) in the environment. To call OpenAI itself, use provider='openai' instead.`}
                  outputLabel="what that printed"
                />
              </div>
            </div>
          </div>
        </Band>

        {/* Documentation */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-green-500/25 bg-green-500/[0.04] p-8 md:p-10 flex flex-col md:flex-row md:items-center gap-6 justify-between">
              <div className="max-w-2xl">
                <h2 className="text-2xl font-black text-gray-900 dark:text-white">
                  The reference for <span className="gradient-text">models</span>
                </h2>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  The registry and how a model id is resolved, one page per provider with
                  its rate limits and its quirks, the tool-call dialects, and the router
                  with its policies and its capability matrix.
                </p>
                <a
                  href={DOCS_COMPATIBLE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  <FiServer size={13} />
                  docs/models/openai-compatible.md — pointing effGen at your own server
                  <FiExternalLink size={13} />
                </a>
              </div>
              <a
                href={DOCS_REGISTRY_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex shrink-0 items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
              >
                <FiFileText size={15} />
                docs/models/registry.md
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
