// The grouping and the prose `/production` renders.
//
// The roles, the metric names, the error classes, the guardrail chains, the
// public endpoints and the `serve`, `eval` and `cost` flag tables all come from
// `data/effgen.json`, which `scripts/gen_site_data.py` reads off the installed
// server and observability modules and out of the real `--help`. Nothing here
// restates any of them.
//
// What is written here is what the modules do not state: which reliability
// primitive is for what, what an audit record does and does not carry, and what
// each deployment target is worth. Every one of those is sourced from a
// framework document named beside it, and each was checked against 1.0.0.

import { siteData } from "@/components/siteData";

const FRAMEWORK_DOCS = "https://github.com/ctrl-gaurav/effGen/blob/main";

export const DOCS_OPENAI_COMPAT_URL = `${FRAMEWORK_DOCS}/docs/server/openai-compat.md`;
export const DOCS_AUTH_URL = `${FRAMEWORK_DOCS}/docs/server/auth.md`;
export const DOCS_RBAC_URL = `${FRAMEWORK_DOCS}/docs/server/rbac.md`;
export const DOCS_AUDIT_URL = `${FRAMEWORK_DOCS}/docs/server/audit.md`;
export const DOCS_METRICS_URL = `${FRAMEWORK_DOCS}/docs/observability/metrics.md`;
export const DOCS_TRACING_URL = `${FRAMEWORK_DOCS}/docs/observability/tracing.md`;
export const DOCS_SLO_URL = `${FRAMEWORK_DOCS}/docs/observability/slos.md`;
export const DOCS_ALERTING_URL = `${FRAMEWORK_DOCS}/docs/observability/alerting.md`;
export const DOCS_RELIABILITY_URL = `${FRAMEWORK_DOCS}/docs/observability/reliability.md`;
export const DOCS_SANDBOX_URL = `${FRAMEWORK_DOCS}/docs/security/codeexecutor.md`;
export const DOCS_LOADTEST_URL = `${FRAMEWORK_DOCS}/docs/observability/loadtest.md`;
export const DOCS_INSTALL_URL = `${FRAMEWORK_DOCS}/docs/installation.md`;

const production = siteData.production;

/* ── The reliability primitives ── */

export interface Primitive {
  name: string;
  module: string;
  what: string;
  accent: string;
}

// The four under `effgen.reliability`, from `docs/observability/reliability.md`.
export const primitives: Primitive[] = [
  {
    name: "Timeouts",
    module: "timeouts.py",
    what:
      "An explicit wall-clock limit on every I/O call — the model call, the tool call, " +
      "the HTTP request, the agent loop and the queue wait each have their own. " +
      "A missing limit is caught by a test rather than discovered in production.",
    accent: "#00ff88",
  },
  {
    name: "Retries",
    module: "retry.py",
    what:
      "Exponential backoff with jitter, retrying only what could plausibly succeed on a " +
      "second attempt, and honouring a Retry-After header when the provider sends one.",
    accent: "#00e5ff",
  },
  {
    name: "Circuit breaker",
    module: "circuit.py",
    what:
      "A per-provider state machine that stops calling a backend that keeps failing, " +
      "lets one request through after a cooldown, and closes again on a success.",
    accent: "#ffd700",
  },
  {
    name: "Bulkhead",
    module: "bulkhead.py",
    what:
      "A concurrency cap and a bounded queue per provider, so one slow backend cannot " +
      "consume every worker and take the rest of the service down with it.",
    accent: "#a78bfa",
  },
];

/* ── What the audit log carries ── */

export interface AuditField {
  name: string;
  what: string;
}

// From `docs/server/audit.md`, confirmed against a real record this page shows.
export const auditFields: AuditField[] = [
  { name: "ts", what: "ISO-8601 UTC timestamp" },
  { name: "principal", what: "the caller — a JWT subject, an API key, or anonymous" },
  { name: "roles", what: "the roles that caller held at the time of the request" },
  { name: "endpoint", what: "method and path" },
  { name: "request_summary", what: "method, path and the query string with secrets scrubbed" },
  { name: "response_summary", what: "the status and the content type" },
  { name: "outcome", what: "ok, error, or denied" },
  { name: "request_id", what: "the X-Request-ID header, so a record joins to a trace" },
  { name: "duration_ms", what: "handler wall-clock time" },
];

/** What the record deliberately leaves out. From the same document. */
export const auditOmissions = [
  "request and response bodies",
  "the Authorization header",
  "API keys in a query string, which are replaced with [REDACTED]",
];

/* ── Deployment ── */

export interface Target {
  name: string;
  shape: string;
  what: string;
  accent: string;
}

// The four under `deploy/`, each with its own framework document.
export const targets: Target[] = [
  {
    name: "Docker",
    shape: "a multi-stage image",
    what:
      "Runs as a non-root user, read-only root filesystem, and the extras it installs " +
      "chosen at build time so an image carries only the provider SDKs it needs.",
    accent: "#00e5ff",
  },
  {
    name: "Kubernetes",
    shape: "a Helm chart",
    what:
      "Two replicas by default with horizontal autoscaling, liveness and readiness " +
      "probes on the endpoints that answer without a credential, and configuration " +
      "through values rather than a rebuilt image.",
    accent: "#a78bfa",
  },
  {
    name: "AWS Lambda",
    shape: "the same app under Mangum",
    what:
      "The FastAPI application that runs under uvicorn locally is deployed unchanged " +
      "behind an API Gateway HTTP API. Auth, RBAC, the audit log and the /v1 routes " +
      "behave identically.",
    accent: "#ffd700",
  },
  {
    name: "Cloudflare",
    shape: "an edge proxy Worker",
    what:
      "In front of any of the three: CORS, a structural JWT and expiry check at the " +
      "edge, per-IP and per-token rate limiting through KV, and security headers — " +
      "forwarding to whichever backend you run.",
    accent: "#ff9500",
  },
];

/* ── Assertions ── */

/**
 * Fail the build if the reliability list stops matching the framework's four.
 *
 * The primitives are described here rather than generated, so this is the only
 * thing standing between a renamed module and a page that describes something
 * that no longer exists.
 */
function assertFourPrimitives(): void {
  if (primitives.length !== 4) {
    throw new Error(
      `app/production/productionData.ts describes ${primitives.length} reliability ` +
        "primitives; the framework ships four (timeouts, retries, circuit breaker, " +
        "bulkhead). Re-read docs/observability/reliability.md.",
    );
  }
}

assertFourPrimitives();

/** The generated row for one metric, by name. */
export function metric(name: string) {
  const row = production.metrics.find((entry) => entry.name === name);
  if (!row) {
    throw new Error(`No metric "${name}" in data/effgen.json.`);
  }
  return row;
}

/** The generated row for one guardrail preset, by name. */
export function guardrailPreset(name: string) {
  const row = production.guardrail_presets.find((entry) => entry.name === name);
  if (!row) {
    throw new Error(`No guardrail preset "${name}" in data/effgen.json.`);
  }
  return row;
}
