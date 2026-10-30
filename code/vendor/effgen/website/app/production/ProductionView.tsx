"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiActivity,
  FiAlertTriangle,
  FiArrowRight,
  FiCheckCircle,
  FiCloud,
  FiCpu,
  FiDollarSign,
  FiExternalLink,
  FiFileText,
  FiLock,
  FiServer,
  FiShield,
  FiTrendingUp,
} from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Terminal from "@/components/ui/Terminal";
import Figure from "@/components/ui/Figure";
import ParamTable from "@/components/ui/ParamTable";
import CodeSample from "@/components/ui/CodeSample";
import RouteLink from "@/components/ui/RouteLink";
import { productionCapture } from "@/components/captures";
import { figureOf, webCapture } from "@/components/webCaptures";
import { siteData, version } from "@/components/siteData";
import {
  DOCS_ALERTING_URL,
  DOCS_AUDIT_URL,
  DOCS_AUTH_URL,
  DOCS_INSTALL_URL,
  DOCS_LOADTEST_URL,
  DOCS_METRICS_URL,
  DOCS_OPENAI_COMPAT_URL,
  DOCS_RBAC_URL,
  DOCS_RELIABILITY_URL,
  DOCS_SANDBOX_URL,
  DOCS_SLO_URL,
  DOCS_TRACING_URL,
  auditFields,
  auditOmissions,
  primitives,
  targets,
} from "./productionData";
import { accentTextStyle } from "@/components/accentText";

// The samples on this page, held here rather than inline so each one is the
// file that was run, character for character. Every `output=` beside them in
// the JSX is that run's stdout, pasted from the transcript.

const SAMPLE_SLO = `from effgen.observability.slo import SLO, get_tracker

tracker = get_tracker()
tracker.register(SLO(name="model_call_success", target_pct=99.0, window_seconds=3600))

for i in range(100):
    tracker.record("model_call_success", ok=(i % 50 != 0))

slo = tracker.get_slo("model_call_success")
print("target:      ", slo.target_pct, "%")
print("error budget:", slo.error_budget_fraction)
print("events:      ", tracker.total_count("model_call_success"),
      "· bad:", tracker.bad_count("model_call_success"))
print("burn rate:    %.1fx" % tracker.burn_rate("model_call_success"))
print("fast burn:   ", tracker.burn_rate("model_call_success") > 14.4)
print("status:      ", tracker.status("model_call_success"))`;
const SAMPLE_ALERTS = `from pathlib import Path

import yaml

from effgen import validate_alert_rules_yaml

rules = Path("docs/observability/alert_rules.yaml")
ok, errors = validate_alert_rules_yaml(rules)
print("valid:", ok, "· errors:", errors)

for group in yaml.safe_load(rules.read_text())["groups"]:
    for rule in group["rules"]:
        print(f"{rule['alert']:22} {rule['labels']['severity']:9} "
              f"for {rule.get('for', 'instant')}")`;
const SAMPLE_ERRORS = `from effgen import RateLimitExceeded
from effgen.models.errors import (BackendUnreachableError, InvalidRequestError,
                                  ModelAuthError, ModelNotFoundError,
                                  ModelRefusalError, ModelTimeoutError,
                                  ProviderTransientError,
                                  classify_provider_error)

for exc in (ModelAuthError("x"), ModelNotFoundError("x"), RateLimitExceeded("x"),
            ModelTimeoutError("x"), ProviderTransientError("x"),
            ModelRefusalError("x"), InvalidRequestError("x"),
            BackendUnreachableError("x"), ValueError("something unrecognised")):
    classified = classify_provider_error(exc)
    print(f"{type(exc).__name__:26} {classified.category:16} "
          f"retry={classified.should_retry}")`;
const SAMPLE_CIRCUIT = `import time

from effgen import CircuitBreaker

breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=1)

for attempt in range(1, 5):
    if not breaker.is_available("web_search"):
        print(f"attempt {attempt}: not called — {breaker.get_state('web_search')}")
        continue
    breaker.record_failure("web_search")
    print(f"attempt {attempt}: called, failed — {breaker.get_state('web_search')}")

time.sleep(1.1)
print("after the cooldown:      ", breaker.get_state("web_search"))
print("would it be called now?  ", breaker.is_available("web_search"))
breaker.record_success("web_search")
print("after that call succeeds:", breaker.get_state("web_search"))`;
const SAMPLE_PII = `from effgen import PIIGuardrail

guard = PIIGuardrail(action="redact")
record = ("Patient: Marta Reyes\\nMRN: 55-2213\\nDOB: January 5, 1980\\n"
          "Contact marta@example.com or 555-0142. Card 4111 1111 1111 1111.")

result = guard.check(record)
print("passed:", result.passed)
print(result.modified_content)`;
const SAMPLE_GUARDRAILS = `from effgen import get_guardrail_preset

for name in ("minimal", "standard", "strict", "phi"):
    chain = get_guardrail_preset(name)
    print(f"{name:9} {len(chain.guardrails)} guardrails: "
          f"{[g.name for g in chain.guardrails]}")`;
const SAMPLE_SANDBOX = `import asyncio
import os

from effgen.security.sandbox import SandboxConfig, get_sandbox

CODE = ("import os\\n"
        "print('processes visible: ', len([p for p in os.listdir('/proc') if p.isdigit()]))\\n"
        "print('~/.ssh contains:   ', os.listdir(os.path.expanduser('~/.ssh')))\\n"
        "open('scratch.txt', 'w').write('allowed')\\n"
        "print('wrote in the scratch space')\\n"
        "try:\\n"
        "    open(os.path.expanduser('~/escaped.txt'), 'w')\\n"
        "except OSError as e:\\n"
        "    print('outside it:        ', type(e).__name__, e.strerror)\\n")


async def main() -> None:
    sandbox = await get_sandbox()
    result = await sandbox.run(CODE, "python", SandboxConfig())

    print(result.stdout.rstrip())
    print()
    print("backend:                ", result.backend_used)
    print("network_isolated:       ", result.network_isolated)
    print("filesystem_confined:    ", result.filesystem_confined)
    print("process_table_isolated: ", result.process_table_isolated)
    print("credential_reads_masked:", result.credential_reads_masked)
    print()
    print("on the host, for comparison:")
    print("  processes:", len([p for p in os.listdir("/proc") if p.isdigit()]))
    print("  ~/.ssh:   ", os.listdir(os.path.expanduser("~/.ssh")))


asyncio.run(main())`;

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

const production = siteData.production;

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

function Mono({ children }: { children: React.ReactNode }) {
  return <code className="font-mono text-xs">{children}</code>;
}

export default function ProductionView() {
  const { ref: heroRef, inView: heroInView } = useInView({ triggerOnce: true, threshold: 0.05 });

  const metrics = productionCapture("serve-metrics");
  const unauthenticated = productionCapture("serve-unauthenticated");
  const completion = productionCapture("serve-completion");
  const rateLimit = productionCapture("serve-rate-limit");
  const audit = productionCapture("serve-audit");
  const loadtest = productionCapture("serve-loadtest");
  const gatePass = productionCapture("eval-gate-pass");
  const gateFail = productionCapture("eval-gate-fail");
  const baseline = productionCapture("eval-baseline");
  const budget = productionCapture("cost-budget");

  const headline = [
    {
      value: String(production.metrics.length),
      label: "Instruments on /metrics",
      accent: "#00ff88",
      icon: FiActivity,
    },
    {
      value: String(production.rbac_roles.length),
      label: "Roles the server ships",
      accent: "#00e5ff",
      icon: FiLock,
    },
    {
      value: String(production.remediation_categories.length),
      label: "Error categories, each with a next step",
      accent: "#ffd700",
      icon: FiAlertTriangle,
    },
    {
      value: String(targets.length),
      label: "Deployment targets, one application",
      accent: "#a78bfa",
      icon: FiCloud,
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
                <FiServer size={14} />
                effgen serve · {version}
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                The same agents,{" "}
                <span className="gradient-text">behind an API someone else calls</span>
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                <Mono>effgen serve</Mono> puts an OpenAI-compatible server in front of
                everything on this site, and it is never unauthenticated by default. What
                it adds beyond the protocol is the part that decides whether you can run
                it: who called, what they were allowed to do, what it cost, how close the
                error budget is to being spent, and what happens when a provider stops
                answering.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <a
                  href="#access"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
                >
                  Who is allowed to call it
                  <FiArrowRight size={15} />
                </a>
                <a
                  href={DOCS_OPENAI_COMPAT_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-green-500/50 font-semibold text-sm transition-colors"
                >
                  Server reference
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

        {/* The artefact, above the fold */}
        <section className="pb-4 relative">
          <Container className="relative z-10">
            <Terminal
              command={metrics.command}
              output={metrics.text}
              title="GET /metrics"
              maxLines={26}
            />
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 max-w-3xl leading-relaxed">
              A real scrape of a server this page drove traffic through: four completions
              across two providers, one more naming a model that does not exist, two
              requests rejected for carrying no credential, and eighteen refused by the
              rate limit. Latency is a histogram per provider and model, tokens are counted
              by provider, model and direction, and the request counter carries the route
              and the status — so a rise in errors can be attributed to a route and a
              backend before anyone opens a log.
            </p>
          </Container>
        </section>

        {/* The server */}
        <Band
          id="server"
          eyebrow="The server"
          tinted
          title={
            <>
              An OpenAI-compatible API,{" "}
              <span className="gradient-text">so your existing client already works</span>
            </>
          }
          lede={
            <>
              <Mono>/v1/chat/completions</Mono>, <Mono>/v1/completions</Mono>,{" "}
              <Mono>/v1/models</Mono> and <Mono>/v1/embeddings</Mono>, spoken the way
              every OpenAI client expects. Point the official SDK at it with a{" "}
              <Mono>base_url</Mono> and nothing else in your code changes — the model id
              carries the provider, so one endpoint serves every backend effGen reaches.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <Terminal
                command={completion.command}
                output={completion.text}
                title="POST /v1/chat/completions"
                maxLines={14}
              />
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The response is the shape a client expects, with one addition: an{" "}
                <Mono>effgen</Mono> block carrying the model that was requested, the model
                that actually served it, whether an alias was applied, what the call cost
                and the run id — which is the same id the trace spans and the run history
                carry, so a bill, a trace and a stored run all join up.
              </p>
              <Card title="One worker by default">
                <p>
                  <Mono>effgen serve</Mono> binds to loopback and runs a single worker.
                  For more, run the application factory under uvicorn or gunicorn —{" "}
                  <Mono>
                    uvicorn effgen.server.app:create_app --factory --workers 4
                  </Mono>{" "}
                  — which is the same object the container image and the Lambda handler
                  use. <Mono>--host 0.0.0.0</Mono> exposes it, and the help text says to
                  set a key first.
                </p>
              </Card>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  What <Mono>serve</Mono> takes
                </h3>
                <ParamTable
                  nameLabel="Flag"
                  params={siteData.cli.command_options["serve"].map((option) => ({
                    name: option.name,
                    description: option.description,
                  }))}
                  caption={`effgen serve --help · effGen ${version}`}
                />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  And what it reads from the environment
                </h3>
                <ParamTable
                  nameLabel="Variable"
                  params={siteData.cli.serve_env.map((option) => ({
                    name: option.name,
                    description: option.description,
                  }))}
                  caption="The operational settings the command documents itself"
                />
              </div>
            </div>
          </div>
        </Band>

        {/* Access */}
        <Band
          id="access"
          eyebrow="Auth, roles, limits and the audit log"
          title={
            <>
              Never unauthenticated,{" "}
              <span className="gradient-text">even when you forget to configure it</span>
            </>
          }
          lede={
            <>
              With no OIDC issuer, no static key and no development flag set,{" "}
              <Mono>effgen serve</Mono> mints an ephemeral key at startup and prints it
              once. There is no state in which the server is running and open, and the
              only way to turn auth off is a flag that prints a warning while it does so.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <Terminal
                command={unauthenticated.command}
                output={unauthenticated.text}
                title="no credential"
                maxLines={10}
              />
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                A rejected request comes back in the same error envelope as everything
                else, so a client has one shape to parse. Bearer tokens from any OIDC
                provider and a static <Mono>X-API-Key</Mono> are both accepted; the static
                key is compared in constant time.
              </p>
              <Card title="What answers without a credential">
                <p>
                  {production.public_endpoints.map((endpoint, index) => (
                    <span key={endpoint}>
                      {index > 0 && ", "}
                      <Mono>{endpoint}</Mono>
                    </span>
                  ))}{" "}
                  — liveness and readiness probes, the aggregate SLO status, and the API
                  schema. None of them carries a request body, a cost or anyone&rsquo;s
                  data.
                </p>
                <p>
                  Everything else needs credentials, including every <Mono>/v1</Mono>{" "}
                  route and the data endpoints behind the dashboard and the playground.
                  The two pages themselves always load, so they can prompt for a key.{" "}
                  <Mono>/metrics</Mono> is protected by default.
                </p>
              </Card>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  The roles it ships with
                </h3>
                <ParamTable
                  nameLabel="Role"
                  params={production.rbac_roles.map((role) => ({
                    name: role.name,
                    type: role.tools === "none" ? "no tools" : "all tools",
                    description:
                      role.max_cost_per_day === 0
                        ? "no daily spend cap"
                        : `$${role.max_cost_per_day.toFixed(2)} a day`,
                  }))}
                  caption={`The policy registry in effGen ${version}. Read at run time, so a custom policy file replaces them.`}
                />
                <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  A principal&rsquo;s effective policy is the union of its roles, and the
                  most permissive one wins — so adding a role can only widen access, never
                  narrow it. A principal whose roles are all read-only is permitted no
                  tools at all, and one with no recognised role falls back to read-only
                  rather than failing open. Spend is counted per principal per UTC day;
                  past the cap the call is refused rather than charged.
                </p>
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  And the rate limit, per client address
                </h3>
                <Terminal
                  command={rateLimit.command}
                  output={rateLimit.text}
                  title="the rate limit"
                  maxLines={10}
                />
                <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  Health probes are always exempt, so a limit cannot make an orchestrator
                  think the service is down. The client address is the raw socket peer
                  unless <Mono>--trust-proxy</Mono> is set, because any caller can put
                  whatever they like in an <Mono>X-Forwarded-For</Mono> header.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                Every request, on one line
              </h3>
              <Terminal
                command={audit.command}
                output={audit.text}
                title="the audit log"
                maxLines={12}
              />
              <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                One JSON line per request/response pair, in a file per UTC day. The three
                lines above are the completion that succeeded, the request with no
                credential, and the one naming a model that does not exist — recorded as{" "}
                <Mono>ok</Mono>, <Mono>denied</Mono> and <Mono>error</Mono>.
              </p>
            </div>
            <div className="space-y-6">
              <ParamTable
                nameLabel="Field"
                params={auditFields.map((field) => ({
                  name: field.name,
                  description: field.what,
                }))}
                caption="What a record carries"
              />
              <div className="rounded-2xl border border-green-500/25 bg-green-500/[0.04] p-5">
                <div className="flex items-start gap-3">
                  <FiShield
                    className="mt-0.5 shrink-0 text-green-700 dark:text-green-400"
                    size={18}
                  />
                  <div>
                    <h4 className="text-sm font-bold text-gray-900 dark:text-white">
                      And what it deliberately leaves out
                    </h4>
                    <ul className="mt-2 space-y-1 text-sm text-gray-600 dark:text-gray-400">
                      {auditOmissions.map((omission) => (
                        <li key={omission}>— {omission}</li>
                      ))}
                    </ul>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                      An audit log that carries prompts is a second copy of your
                      users&rsquo; data with a different retention policy. This one
                      records who did what, when, and how it ended.
                    </p>
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap gap-4">
                {[
                  { href: DOCS_AUTH_URL, label: "docs/server/auth.md" },
                  { href: DOCS_RBAC_URL, label: "docs/server/rbac.md" },
                  { href: DOCS_AUDIT_URL, label: "docs/server/audit.md" },
                ].map((link) => (
                  <a
                    key={link.href}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                  >
                    {link.label}
                    <FiExternalLink size={13} />
                  </a>
                ))}
              </div>
            </div>
          </div>
        </Band>

        {/* Observability */}
        <Band
          id="observability"
          eyebrow="Metrics, traces, SLOs and alerts"
          tinted
          title={
            <>
              Instrumented where it matters,{" "}
              <span className="gradient-text">not where it was easy</span>
            </>
          }
          lede={
            <>
              The three latencies that decide whether a service is usable — the model
              call, the tool call and one turn of the agent loop — are histograms with the
              labels you need to attribute a regression. Everything below is on{" "}
              <Mono>/metrics</Mono> in Prometheus format, with no exporter to configure.
            </>
          }
        >
          <ParamTable
            nameLabel="Metric"
            params={production.metrics.map((instrument) => ({
              name: instrument.name,
              type: instrument.kind,
              description: instrument.labels.length
                ? `${instrument.help}. Labels: ${instrument.labels.join(", ")}.`
                : instrument.help,
            }))}
            caption={`Every instrument the observability module registers, effGen ${version}`}
            className="mb-10"
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  An objective, and how fast its budget is burning
                </h3>
                <CodeSample
                  language="python"
                  accent="#00ff88"
                  code={SAMPLE_SLO}
                  output={`target:       99.0 %
error budget: 0.01
events:       100 · bad: 2
burn rate:    2.0x
fast burn:    False
status:       {'name': 'model_call_success', 'target_pct': 99.0, 'window_seconds': 3600, 'query': '', 'total_events': 100, 'good_events': 98, 'bad_events': 2, 'good_ratio': 0.98, 'bad_ratio': 0.02, 'burn_rate': 2.0, 'within_budget': False}`}
                  outputLabel="what that printed"
                />
                <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  A burn rate of <Mono>1.0</Mono> is exactly on budget; two failures in a
                  hundred against a 99% objective is twice the budget, which is the number
                  above. The rolling window evicts old events as it is read, so the figure
                  is about now rather than about the whole life of the process.
                </p>
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  The alert pack that goes with it
                </h3>
                <CodeSample
                  language="python"
                  accent="#ffd700"
                  code={SAMPLE_ALERTS}
                  output={`valid: True · errors: []
HighErrorRate          critical  for 10m
HighP95Latency         warning   for 5m
CostBurnHigh           warning   for 0m
SLOFastBurn            critical  for 0m
SLOSlowBurn            warning   for 60m
CircuitBreakerOpen     warning   for 1m`}
                  outputLabel="what that printed"
                />
                <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  An Alertmanager rule file is in the repository at{" "}
                  <Mono>docs/observability/alert_rules.yaml</Mono> — the path above is
                  relative to a clone, not to an installed package — and{" "}
                  <Mono>validate_alert_rules_yaml</Mono> will tell you it is well-formed
                  before Prometheus does.{" "}
                  <Mono>AlertWebhook</Mono> posts an alert to Slack or Discord and works
                  out which from the URL.
                </p>
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  Every hot path emits a span
                </h3>
                <Terminal
                  output={`effgen.agent.run                          # entire agent.run() call
  effgen.agent.iteration                  # one ReAct iteration
    effgen.model.call                     # model inference
    effgen.tool.call  [calculator]        # tool 1
    effgen.tool.call  [web_search]        # tool 2
    effgen.tool.call  [wikipedia]         # tool 3
  effgen.router.decision                  # policy-based routing (if enabled)`}
                  title="the span tree"
                />
                <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  OpenTelemetry, so it goes to Jaeger, Zipkin, Tempo or anything else that
                  speaks OTLP. The agent loop, the model adapters, the tools and the
                  router already emit spans —{" "}
                  <Mono>setup_tracing(service_name=..., sampler=...)</Mono> at startup is
                  the whole integration. Five samplers ship, and a parent-based one over a
                  ratio sampler is the one to reach for: a trace is either sampled whole
                  or not at all.
                </p>
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  What it does under load
                </h3>
                <Terminal
                  command={loadtest.command}
                  output={loadtest.text}
                  title="effgen loadtest"
                  maxLines={22}
                />
                <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  Pointed at a URL rather than at a provider, the load test goes through
                  the whole middleware stack — auth, the rate limit, the audit log — and
                  not just the adapter. It reports the tail, not only the mean, and says
                  how long it spent draining requests that were still in flight when the
                  window closed.
                </p>
              </div>
              <div className="flex flex-wrap gap-4">
                {[
                  { href: DOCS_METRICS_URL, label: "docs/observability/metrics.md" },
                  { href: DOCS_TRACING_URL, label: "docs/observability/tracing.md" },
                  { href: DOCS_SLO_URL, label: "docs/observability/slos.md" },
                  { href: DOCS_ALERTING_URL, label: "docs/observability/alerting.md" },
                  { href: DOCS_LOADTEST_URL, label: "docs/observability/loadtest.md" },
                ].map((link) => (
                  <a
                    key={link.href}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                  >
                    {link.label}
                    <FiExternalLink size={13} />
                  </a>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-12">
            <RouteLink
              to="/dashboard"
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
            >
              The same numbers, drawn — the dashboard
              <FiArrowRight size={14} />
            </RouteLink>
          </div>
        </Band>

        {/* Reliability */}
        <Band
          id="reliability"
          eyebrow="Reliability and errors"
          title={
            <>
              Every failure is classified{" "}
              <span className="gradient-text">before anything decides to retry it</span>
            </>
          }
          lede={
            <>
              Retrying an authentication failure wastes a minute and fixes nothing;
              refusing to retry a transient blip turns a hiccup into an outage. So an
              exception is put in a category first, and the category decides. An
              unrecognised error is treated as retryable, so a genuine blip is not turned
              into a hard failure by a class name nobody had seen before.
            </>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
            {primitives.map((primitive) => (
              <div
                key={primitive.name}
                className="relative rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5"
              >
                <div
                  className="absolute left-0 top-5 bottom-5 w-0.5 rounded-full"
                  style={{ background: primitive.accent }}
                />
                <div className="pl-4">
                  <h3 className="text-sm font-bold text-gray-900 dark:text-white">
                    {primitive.name}
                  </h3>
                  <code className="text-[11px] font-mono text-gray-600 dark:text-gray-400">
                    effgen.reliability.{primitive.module}
                  </code>
                  <p className="mt-1.5 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                    {primitive.what}
                  </p>
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <CodeSample
                language="python"
                accent="#00e5ff"
                code={SAMPLE_ERRORS}
                output={`ModelAuthError             auth             retry=False
ModelNotFoundError         not_found        retry=False
RateLimitExceeded          rate_limited     retry=True
ModelTimeoutError          timeout          retry=True
ProviderTransientError     transient        retry=True
ModelRefusalError          refusal          retry=False
InvalidRequestError        invalid_request  retry=False
BackendUnreachableError    unreachable      retry=True
ValueError                 unknown          retry=True`}
                outputLabel="what that printed"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The classifier recognises effGen&rsquo;s own typed errors first, then
                falls back to the SDK exception name, the HTTP status and the message —
                so a raw provider exception still lands in one of these categories.{" "}
                {production.remediation_categories.length} categories each carry a next
                step, which is what turns an error message into one that names the fix.
              </p>
            </div>
            <div className="space-y-6">
              <CodeSample
                language="python"
                accent="#ffd700"
                code={SAMPLE_CIRCUIT}
                output={`attempt 1: called, failed — CircuitState.CLOSED
attempt 2: called, failed — CircuitState.CLOSED
attempt 3: called, failed — CircuitState.OPEN
attempt 4: not called — CircuitState.OPEN
after the cooldown:       CircuitState.OPEN
would it be called now?   True
after that call succeeds: CircuitState.CLOSED`}
                outputLabel="what that printed"
              />
              <Card title="Why an unreachable backend is its own case">
                <p>
                  Every category above describes something that answered. A connection
                  refused, a host that does not resolve and a route that does not exist
                  describe a backend that never did — and a batch that quietly completes
                  against nothing looks healthy in the summary. So{" "}
                  <Mono>BackendUnreachableError</Mono> is raised whatever{" "}
                  <Mono>raise_on_error</Mono> says. It is the one error with no opt-out,
                  and one of 1.0.0&rsquo;s three breaking changes.
                </p>
                <RouteLink
                  to="/models"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  What that looks like against a dead endpoint
                  <FiArrowRight size={14} />
                </RouteLink>
              </Card>
              <a
                href={DOCS_RELIABILITY_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
              >
                docs/observability/reliability.md
                <FiExternalLink size={13} />
              </a>
            </div>
          </div>
        </Band>

        {/* Safety */}
        <Band
          id="safety"
          eyebrow="Guardrails, redaction and the sandbox"
          tinted
          title={
            <>
              Checks over the text,{" "}
              <span className="gradient-text">and a wall around the code</span>
            </>
          }
          lede={
            <>
              A guardrail sits at one of {production.guardrail_positions.length} positions
              — {production.guardrail_positions.join(", ")} — and can pass, block or
              rewrite what goes through it. Four chains ship configured; the{" "}
              <Mono>phi</Mono> one is the health-record shape.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <CodeSample
                language="python"
                accent="#00ff88"
                code={SAMPLE_GUARDRAILS}
                output={`minimal   2 guardrails: ['LengthGuardrail', 'PromptInjectionGuardrail']
standard  5 guardrails: ['LengthGuardrail', 'PromptInjectionGuardrail', 'PIIGuardrail', 'ToolInputGuardrail', 'ToolOutputGuardrail']
strict    7 guardrails: ['LengthGuardrail', 'PromptInjectionGuardrail', 'SystemPromptLeakGuardrail', 'ToxicityGuardrail', 'PIIGuardrail', 'ToolInputGuardrail', 'ToolOutputGuardrail']
phi       6 guardrails: ['LengthGuardrail', 'PromptInjectionGuardrail', 'SystemPromptLeakGuardrail', 'PIIGuardrail', 'ToolInputGuardrail', 'ToolOutputGuardrail']`}
                outputLabel="what that printed"
              />
              <CodeSample
                language="python"
                accent="#ff6b6b"
                code={SAMPLE_PII}
                output={`passed: True
Patient: [NAME REDACTED]
MRN: [MRN REDACTED]
DOB: [DOB REDACTED]
Contact [EMAIL REDACTED] or 555-0142. Card [CC REDACTED].`}
                outputLabel="what that printed"
              />
              <Card title="What redaction keeps, and what it admits">
                <p>
                  The label survives and the value goes, so a redacted record is still
                  readable as a record. Social security numbers, emails, phone numbers,
                  Luhn-checked card numbers and IP addresses are matched by shape;
                  API keys and private-key headers are matched too, so a leaked credential
                  is treated as sensitive rather than as ordinary text; and the
                  label-anchored fields a health or insurance record carries — name, date
                  of birth, medical record number, member and policy ids — are matched
                  from their labels.
                </p>
                <p>
                  That last group is where the limit is: a name with no label in front of
                  it is not matched. The framework documents that boundary rather than
                  implying there is none.
                </p>
              </Card>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  Code the model wrote, run where it cannot reach anything
                </h3>
                <CodeSample
                  language="python"
                  accent="#a78bfa"
                  code={SAMPLE_SANDBOX}
                  output={`processes visible:  1
~/.ssh contains:    []
wrote in the scratch space
outside it:         OSError Read-only file system

backend:                 subprocess
network_isolated:        True
filesystem_confined:     True
process_table_isolated:  True
credential_reads_masked: True

on the host, for comparison:
  processes: 1603
  ~/.ssh:    ['known_hosts.old', 'known_hosts']`}
                  outputLabel="what that printed on this machine"
                />
              </div>
              <Card title="What that run actually enforced">
                <p>
                  This is the subprocess backend, which is what runs when there is no
                  Docker daemon. The network namespace has no interfaces, so nothing gets
                  out. Every mount is remounted read-only except one scratch directory —
                  the same directory the file and shell tools use — so generated code can
                  write the files the agent just made and nothing else.
                </p>
                <p>
                  Reads are not confined, and the framework says so rather than implying
                  they are. What it does instead is mask the credential stores:{" "}
                  <Mono>~/.ssh</Mono>, <Mono>~/.aws</Mono>, <Mono>~/.gnupg</Mono>,{" "}
                  <Mono>~/.kube</Mono>, <Mono>~/.docker</Mono>, <Mono>~/.azure</Mono>, the
                  gcloud configuration, the credential files beside them,{" "}
                  <Mono>/etc/shadow</Mono> and mounted secrets are each covered by an
                  empty directory or by <Mono>/dev/null</Mono>, so a read succeeds and
                  returns nothing. Use the Docker backend when reads must be confined
                  rather than masked.
                </p>
                <p>
                  The run also gets its own PID namespace, which is why one process is
                  visible above against the host&rsquo;s. Both are reported on the result
                  as <Mono>credential_reads_masked</Mono> and{" "}
                  <Mono>process_table_isolated</Mono> — a host that cannot create the
                  namespace says so on the result instead of pretending.
                </p>
                <a
                  href={DOCS_SANDBOX_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  docs/security/codeexecutor.md
                  <FiExternalLink size={13} />
                </a>
              </Card>
            </div>
          </div>
        </Band>

        {/* Evaluation */}
        <Band
          id="evaluation"
          eyebrow="Evaluation and CI"
          title={
            <>
              A prompt change is a code change,{" "}
              <span className="gradient-text">so give it an exit code</span>
            </>
          }
          lede={
            <>
              <Mono>effgen eval</Mono> runs a suite and exits non-zero when accuracy falls
              under a threshold you set, which is all a CI job needs.{" "}
              <Mono>--compare-baseline</Mono> goes further: a regression against a stored
              baseline fails the job whatever the threshold says, because the question is
              not whether the number is good but whether it got worse.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <Terminal
                command={gatePass.command}
                output={gatePass.text}
                title="the gate passing"
                maxLines={22}
              />
              <Terminal
                command={gateFail.command}
                output={gateFail.text}
                title="the gate failing"
                maxLines={22}
              />
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Same suite, same result, different threshold — and the exit code is the
                only thing a CI job has to read. Five scoring modes ship, from exact match
                to a model as judge, and <Mono>--temperature 0</Mono> makes the run
                reproducible wherever the provider supports it.
              </p>
              <ParamTable
                nameLabel="Flag"
                params={siteData.cli.command_options["eval"].map((option) => ({
                  name: option.name,
                  description: option.description,
                }))}
                caption={`effgen eval --help · effGen ${version}`}
              />
            </div>

            <div className="space-y-6">
              <Terminal
                command={baseline.command}
                output={baseline.text}
                title="against the stored baseline"
                maxLines={26}
              />
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The regression report is Markdown, so it goes straight into a pull-request
                comment. It carries the version each side was measured at, which is what
                stops a comparison between two different releases being read as a
                behaviour change.
              </p>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  And the same run as a page you can send someone
                </h3>
                <Figure
                  {...figureOf(webCapture("eval-report"))}
                  caption={
                    <>
                      <Mono>--report out.html</Mono> writes one self-contained file: the
                      gate&rsquo;s verdict, the headline figures, accuracy by difficulty
                      and every case with what it expected and what it got. No external
                      reference of any kind, so it opens from disk with the network off
                      and survives being attached to an email.{" "}
                      <Mono>compare</Mono>, <Mono>cost</Mono> and <Mono>loadtest</Mono>{" "}
                      take the same flag.
                    </>
                  }
                  command="effgen eval --suite math -m openai:gpt-5-nano --max-cases 5 --fail-under 0.8 --report math-eval.html"
                  frameClassName="max-h-[640px] overflow-y-auto"
                />
              </div>
            </div>
          </div>
        </Band>

        {/* Cost */}
        <Band
          id="cost"
          eyebrow="Cost and budgets"
          tinted
          title={
            <>
              What it spent,{" "}
              <span className="gradient-text">per model, against a cap</span>
            </>
          }
          lede={
            <>
              Spend is recorded per provider and per model as it happens, and a daily
              budget is a number rather than an alert someone reads later. A scaffolded
              project gets a $1.00 daily cap when none is set, so the first thing a new
              project cannot do is spend without a limit.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <Terminal
              command={budget.command}
              output={budget.text}
              title="effgen cost"
              maxLines={20}
            />
            <div className="space-y-4">
              <Card title="Two caps, at two layers">
                <p>
                  The one above is the local one, kept in a database beside the session
                  and run history, and it covers everything run from this machine. The
                  server has its own: each role carries a daily cap in dollars, spend is
                  counted per principal for the UTC day, and a call past the cap is
                  refused with a 429 rather than charged and reported afterwards.
                </p>
              </Card>
              <Card title="A price effGen does not have is never printed as $0">
                <p>
                  A model with no published rate — a fine-tune, an uncatalogued id, your
                  own server — reports no cost and is counted as unpriced. A fabricated
                  zero is the one answer that makes a spend total wrong without looking
                  wrong, which is why the totals here can be smaller than the traffic and
                  say so.
                </p>
                <RouteLink
                  to="/models"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  The catalog and its prices
                  <FiArrowRight size={14} />
                </RouteLink>
              </Card>
              <Card title="Where it is written down">
                <p>
                  Cost, sessions, run history and the rate-limit counters all live under{" "}
                  <Mono>$EFFGEN_HOME</Mono> (<Mono>~/.effgen</Mono> by default), which is
                  the one directory a container has to mount to keep any of it. Every
                  store has its own variable when one directory is not what you want.
                </p>
              </Card>
            </div>
          </div>
        </Band>

        {/* Deployment */}
        <Band
          id="deployment"
          eyebrow="Deployment and hardware"
          title={
            <>
              One application,{" "}
              <span className="gradient-text">{targets.length} places to put it</span>
            </>
          }
          lede={
            <>
              The object every one of these serves is the same application factory,{" "}
              <Mono>effgen.server.app:create_app</Mono>. Nothing is re-implemented per
              target, so auth, RBAC, the audit log and the <Mono>/v1</Mono> routes behave
              the same wherever it runs.
            </>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
            {targets.map((target) => (
              <article
                key={target.name}
                className="relative rounded-xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-5"
              >
                <div
                  className="absolute left-0 top-5 bottom-5 w-0.5 rounded-full"
                  style={{ background: target.accent }}
                />
                <div className="pl-4">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <h3 className="text-base font-bold text-gray-900 dark:text-white">
                      {target.name}
                    </h3>
                    <span className="text-xs" style={accentTextStyle(target.accent)}>
                      {target.shape}
                    </span>
                  </div>
                  <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {target.what}
                  </p>
                </div>
              </article>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                <FiCheckCircle
                  className="inline mb-0.5 mr-1.5 text-green-700 dark:text-green-400"
                  size={15}
                />
                A container that can prove it is up
              </h3>
              <CodeSample
                language="bash"
                accent="#00e5ff"
                code={`docker run --rm -p 8080:8080 \\
  -e EFFGEN_OIDC_ISSUER=https://your-issuer.example.com \\
  -e EFFGEN_OIDC_CLIENT_ID=your-client-id \\
  -v ~/.effgen:/home/effgen/.effgen:ro \\
  --read-only --tmpfs /tmp \\
  effgen:${version}

curl http://localhost:8080/health`}
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Read-only root, a non-root user, and the state directory mounted rather
                than baked in. <Mono>/health</Mono> answers without a credential, which is
                what a probe needs and why the readiness endpoints are on the public list
                above.
              </p>
            </div>
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  <FiCpu
                    className="inline mb-0.5 mr-1.5 text-green-700 dark:text-green-400"
                    size={15}
                  />
                  If you are serving the weights yourself
                </h3>
                <Card title="The CUDA trap, and what effGen does about it">
                  <p>
                    A PyTorch wheel is built against one CUDA runtime and a driver is
                    forward-compatible only, so a CUDA-13 wheel on a 12.4 driver leaves
                    you with <Mono>torch.cuda.is_available() == False</Mono> and
                    everything running slowly on the CPU with no error. The framework
                    detects that at runtime: when it sees physical NVIDIA cards that torch
                    cannot use, it prints one warning naming the torch build and the
                    driver&rsquo;s version instead of running silently.
                  </p>
                  <p>
                    The second half of the trap is a later{" "}
                    <Mono>pip install</Mono> quietly upgrading torch back. A constraints
                    file per CUDA line pins torch and its companions, so an extras install
                    cannot move them.
                  </p>
                  <a
                    href={DOCS_INSTALL_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                  >
                    docs/installation.md — the wheel and constraint tables
                    <FiExternalLink size={13} />
                  </a>
                </Card>
              </div>
              <Card title="Or do not serve them from the agent at all">
                <p>
                  Loading weights in the agent&rsquo;s process means one copy per process,
                  no batching across callers and a GPU tied to the run&rsquo;s lifetime.
                  Running vLLM, SGLang or TGI separately and pointing effGen at it with a{" "}
                  <Mono>base_url</Mono> fixes all three, and is the same code path this
                  server exposes.
                </p>
                <RouteLink
                  to="/models"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  Any model, anywhere
                  <FiArrowRight size={14} />
                </RouteLink>
              </Card>
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
                  The reference for{" "}
                  <span className="gradient-text">running it</span>
                </h2>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  The server&rsquo;s endpoints and its error envelope, OIDC and the static
                  key, the role model and its union semantics, the audit record, every
                  metric and span attribute, the SLO maths, the reliability primitives,
                  the sandbox threat model, and a page per deployment target.
                </p>
                <a
                  href={DOCS_SLO_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  <FiTrendingUp size={13} />
                  docs/observability/slos.md — error budgets and burn rates
                  <FiExternalLink size={13} />
                </a>
              </div>
              <div className="flex shrink-0 flex-col gap-3">
                <a
                  href={DOCS_OPENAI_COMPAT_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
                >
                  <FiFileText size={15} />
                  docs/server/openai-compat.md
                  <FiExternalLink size={14} />
                </a>
                <RouteLink
                  to="/agents"
                  className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-green-500/50 font-semibold text-sm transition-colors"
                >
                  <FiDollarSign size={14} />
                  The library behind it
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>
      </main>
      <Footer />
    </div>
  );
}
