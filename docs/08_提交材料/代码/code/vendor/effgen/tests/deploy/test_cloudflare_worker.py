"""Tests for the effGen Cloudflare Worker edge proxy.

Two test modes:

1.  **Structural tests** (always run) — parse wrangler.toml, validate required
    keys, KV binding, CORS/rate-limit vars, and the worker.js file itself for
    required patterns (fetch export, CORS, rate-limit logic, upstream forward).

2.  **Fetch-handler tests** (run when Node ≥18 is available) — exercise the
    fetch handler with a tiny Node harness that feeds synthetic ``Request``
    objects and asserts on the returned ``Response`` objects.  No Cloudflare
    account or wrangler login required; skips with a message when Node is
    absent.

Exit criteria:
- wrangler.toml parses and contains all required keys.
- worker.js contains all required implementation patterns.
- Fetch-handler unit tests pass (OPTIONS/CORS, 401 missing auth, 401 expired
  JWT, 429 rate-limit stub, 502 bad gateway stub).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
CF_DIR = REPO_ROOT / "deploy" / "cloudflare"
WORKER_JS = CF_DIR / "worker.js"
WRANGLER_TOML = CF_DIR / "wrangler.toml"

# ── Helpers ────────────────────────────────────────────────────────────────

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def _load_toml(path: Path) -> dict:
    """Load a TOML file, falling back to a regex-based partial parse."""
    if tomllib is not None:
        return tomllib.loads(path.read_text())
    # Minimal fallback: extract top-level string/int keys and [[kv_namespaces]]
    text = path.read_text()
    result: dict = {}
    for m in re.finditer(r'^(\w+)\s*=\s*"([^"]*)"', text, re.MULTILINE):
        result[m.group(1)] = m.group(2)
    for m in re.finditer(r"^(\w+)\s*=\s*(\d+)", text, re.MULTILINE):
        result[m.group(1)] = int(m.group(2))
    result["_raw"] = text
    return result


def _has_node(min_major: int = 18) -> bool:
    node = shutil.which("node")
    if not node:
        return False
    try:
        out = subprocess.check_output([node, "--version"], text=True).strip()
        maj = int(out.lstrip("v").split(".")[0])
        return maj >= min_major
    except Exception:
        return False


NODE_AVAILABLE = _has_node(18)

# ── wrangler.toml structural tests ────────────────────────────────────────


class TestWranglerToml:
    """Validate the wrangler.toml configuration file."""

    def test_file_exists(self):
        assert WRANGLER_TOML.exists(), f"{WRANGLER_TOML} does not exist"

    def test_worker_js_exists(self):
        assert WORKER_JS.exists(), f"{WORKER_JS} does not exist"

    def test_name_present(self):
        cfg = _load_toml(WRANGLER_TOML)
        assert "name" in cfg, "wrangler.toml must have a 'name' key"
        assert cfg["name"], "Worker name must not be empty"

    def test_main_present(self):
        cfg = _load_toml(WRANGLER_TOML)
        assert "main" in cfg, "wrangler.toml must have a 'main' key"
        assert cfg["main"] == "worker.js", "'main' must point to worker.js"

    def test_compatibility_date(self):
        cfg = _load_toml(WRANGLER_TOML)
        assert "compatibility_date" in cfg, "wrangler.toml must have 'compatibility_date'"
        date_str = cfg["compatibility_date"]
        assert re.match(r"\d{4}-\d{2}-\d{2}", date_str), (
            f"compatibility_date '{date_str}' is not YYYY-MM-DD"
        )

    def test_kv_namespace_binding(self):
        raw = WRANGLER_TOML.read_text()
        assert "[[kv_namespaces]]" in raw, "wrangler.toml must declare [[kv_namespaces]]"
        assert 'binding' in raw, "KV namespace must have a 'binding' key"
        assert "RATE_LIMIT" in raw, "KV binding must be named RATE_LIMIT"

    def test_backend_url_var(self):
        raw = WRANGLER_TOML.read_text()
        assert "EFFGEN_BACKEND_URL" in raw, "wrangler.toml must define EFFGEN_BACKEND_URL var"

    def test_cors_origins_var(self):
        raw = WRANGLER_TOML.read_text()
        assert "EFFGEN_CORS_ORIGINS" in raw, "wrangler.toml must define EFFGEN_CORS_ORIGINS var"

    def test_rate_limit_vars(self):
        raw = WRANGLER_TOML.read_text()
        assert "EFFGEN_RATE_LIMIT_WINDOW_SECONDS" in raw
        assert "EFFGEN_RATE_LIMIT_IP_MAX" in raw
        assert "EFFGEN_RATE_LIMIT_TOKEN_MAX" in raw

    def test_env_staging_defined(self):
        raw = WRANGLER_TOML.read_text()
        assert "[env.staging]" in raw, "Staging environment must be defined"

    def test_env_production_defined(self):
        raw = WRANGLER_TOML.read_text()
        assert "[env.production]" in raw, "Production environment must be defined"


# ── worker.js structural tests ────────────────────────────────────────────


class TestWorkerJs:
    """Validate worker.js source patterns."""

    @pytest.fixture(autouse=True)
    def _source(self):
        self.src = WORKER_JS.read_text()

    def test_exports_default_fetch(self):
        # Either named export or default export with fetch
        assert "export" in self.src and "fetch" in self.src, (
            "worker.js must export a fetch handler"
        )

    def test_cors_handling(self):
        assert "OPTIONS" in self.src, "worker.js must handle CORS OPTIONS preflight"
        assert "Access-Control" in self.src, (
            "worker.js must set Access-Control-* headers"
        )

    def test_bearer_auth_check(self):
        assert "Bearer" in self.src or "bearer" in self.src, (
            "worker.js must check Bearer token"
        )
        assert "401" in self.src, "worker.js must return 401 for missing/invalid auth"

    def test_rate_limit_logic(self):
        assert "RATE_LIMIT" in self.src, "worker.js must reference RATE_LIMIT KV binding"
        assert "429" in self.src, "worker.js must return 429 on rate-limit exceeded"

    def test_upstream_forward(self):
        assert "EFFGEN_BACKEND_URL" in self.src, (
            "worker.js must use EFFGEN_BACKEND_URL for forwarding"
        )
        assert "502" in self.src, "worker.js must return 502 on upstream failure"

    def test_security_headers(self):
        assert "X-Content-Type-Options" in self.src, (
            "worker.js must set X-Content-Type-Options"
        )
        assert "X-Frame-Options" in self.src, "worker.js must set X-Frame-Options"

    def test_public_paths_match_the_origin_server(self):
        for path in ("/health", "/healthz", "/livez", "/ready", "/readyz", "/slo"):
            assert f'"{path}"' in self.src, f"worker.js must skip auth for {path}"

    def test_metrics_is_not_public_at_the_edge(self):
        public = self.src.split("const PUBLIC_PATHS")[1].split("]")[0]
        assert "/metrics" not in public, (
            "worker.js must not exempt /metrics from auth — the origin protects it"
        )

    def test_jwt_expiry_check(self):
        assert "exp" in self.src, "worker.js must check JWT exp claim"

    def test_x_forwarded_headers(self):
        assert "X-Forwarded" in self.src, (
            "worker.js must propagate X-Forwarded-* to upstream"
        )


# ── Fetch-handler Node harness tests ─────────────────────────────────────


# Inline Node test harness: a tiny ESM module that stubs the global environment
# (Request, Response, Headers, URL, fetch, atob) and runs the worker fetch
# handler with synthetic inputs.
_NODE_HARNESS = textwrap.dedent(r"""
import { URL as NodeURL, pathToFileURL } from "url";
import { createServer } from "http";

// ── Minimal Web API stubs for Node < 21 ─────────────────────────────────

globalThis.URL = globalThis.URL ?? NodeURL;

if (typeof globalThis.atob === "undefined") {
  globalThis.atob = (b64) => Buffer.from(b64, "base64").toString("binary");
}
if (typeof globalThis.btoa === "undefined") {
  globalThis.btoa = (s) => Buffer.from(s, "binary").toString("base64");
}

// Minimal KV stub (in-memory)
class KVStub {
  constructor() { this._store = new Map(); this._ttls = new Map(); }
  async get(k) {
    const exp = this._ttls.get(k);
    if (exp && Date.now() > exp) { this._store.delete(k); this._ttls.delete(k); return null; }
    return this._store.get(k) ?? null;
  }
  async put(k, v, opts) {
    this._store.set(k, v);
    if (opts?.expirationTtl) this._ttls.set(k, Date.now() + opts.expirationTtl * 1000);
  }
}

// ── Load worker ───────────────────────────────────────────────────────────
//
// Import the worker from its real file:// URL rather than a data: URL. A data:
// URL gives the module a synthetic base and its own global scope, which masks
// module-scope binding bugs — notably a top-level `export function fetch` that
// would shadow the global `fetch` the worker uses for upstream calls. Loading
// the real file mirrors how the Cloudflare runtime evaluates the module.

const workerPath = process.argv[2];
const mod = await import(pathToFileURL(workerPath).href);
const handler = mod.default?.fetch ?? mod.fetch;

if (!handler) { console.error("No fetch export found"); process.exit(1); }

// ── Test helpers ──────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;
const results = [];

function makeJwt(payload) {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }))
    .replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
  const body = btoa(JSON.stringify(payload))
    .replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
  return `${header}.${body}.fakesig`;
}

async function test(name, fn) {
  try {
    await fn();
    results.push({ name, ok: true });
    passed++;
  } catch (e) {
    results.push({ name, ok: false, error: e.message });
    failed++;
  }
}

function assert(cond, msg) { if (!cond) throw new Error(msg || "Assertion failed"); }

function makeRequest(method, path, headers = {}, body = null) {
  const url = `https://edge.example.com${path}`;
  return new Request(url, { method, headers: new Headers(headers), body });
}

// The worker uses env._fetch if provided (test injection hook).
// For a 502 scenario, simply omit env._fetch so the worker's try/catch fires.
function mockFetch(responseFn) {
  return async (url, opts) => responseFn(url, opts);
}
function noFetch() {
  return async () => { throw new Error("Upstream not available"); };
}

const baseEnv = {
  EFFGEN_BACKEND_URL: "https://backend.example.com",
  EFFGEN_CORS_ORIGINS: "https://app.example.com",
  EFFGEN_RATE_LIMIT_WINDOW_SECONDS: "60",
  EFFGEN_RATE_LIMIT_IP_MAX: "100",
  EFFGEN_RATE_LIMIT_TOKEN_MAX: "200",
  RATE_LIMIT: new KVStub(),
  _fetch: noFetch(),
};

const ctx = { waitUntil: () => {}, passThroughOnException: () => {} };

// ── Tests ──────────────────────────────────────────────────────────────────

// 1. OPTIONS preflight returns 204 with CORS headers
await test("OPTIONS preflight → 204 + CORS headers", async () => {
  const req = makeRequest("OPTIONS", "/v1/chat/completions", {
    Origin: "https://app.example.com",
    "Access-Control-Request-Method": "POST",
  });
  const res = await handler(req, baseEnv, ctx);
  assert(res.status === 204, `Expected 204, got ${res.status}`);
  assert(res.headers.get("Access-Control-Allow-Methods"), "Missing CORS allow-methods");
});

// 2. Missing Authorization → 401
await test("Missing Authorization → 401", async () => {
  const req = makeRequest("GET", "/v1/models", {});
  const res = await handler(req, baseEnv, ctx);
  assert(res.status === 401, `Expected 401, got ${res.status}`);
  const body = await res.json();
  assert(body.error?.code === "unauthorized", `Expected 'unauthorized' code, got ${body.error?.code}`);
  // Same envelope the origin server returns: {message, type, param, code}.
  const keys = Object.keys(body.error).sort().join(",");
  assert(keys === "code,message,param,type", `Expected the shared error envelope, got ${keys}`);
  assert(body.error.type === "invalid_request_error", `Expected 'invalid_request_error', got ${body.error.type}`);
  assert(body.error.param === null, "Expected param to be null");
});

// 3. Malformed JWT (not 3 parts) → 401 invalid_token
await test("Malformed JWT → 401 invalid_token", async () => {
  const req = makeRequest("GET", "/v1/models", { Authorization: "Bearer notajwt" });
  const res = await handler(req, baseEnv, ctx);
  assert(res.status === 401, `Expected 401, got ${res.status}`);
  const body = await res.json();
  assert(body.error?.code === "invalid_token", `Expected 'invalid_token', got ${body.error?.code}`);
});

// 4. Expired JWT → 401 token_expired
await test("Expired JWT → 401 token_expired", async () => {
  const expiredToken = makeJwt({ sub: "user123", exp: Math.floor(Date.now() / 1000) - 3600 });
  const req = makeRequest("GET", "/v1/models", { Authorization: `Bearer ${expiredToken}` });
  const res = await handler(req, baseEnv, ctx);
  assert(res.status === 401, `Expected 401, got ${res.status}`);
  const body = await res.json();
  assert(body.error?.code === "token_expired", `Expected 'token_expired', got ${body.error?.code}`);
});

// 5. Valid JWT (not expired) + upstream reachable → response forwarded
await test("Valid JWT + upstream 200 → forwarded response", async () => {
  const validToken = makeJwt({ sub: "user456", exp: Math.floor(Date.now() / 1000) + 3600 });
  const env5 = {
    ...baseEnv,
    _fetch: mockFetch(async () => new Response(JSON.stringify({ object: "list" }), {
      status: 200, headers: { "Content-Type": "application/json" },
    })),
  };
  const req = makeRequest("GET", "/v1/models", { Authorization: `Bearer ${validToken}` });
  const res = await handler(req, env5, ctx);
  assert(res.status === 200, `Expected 200, got ${res.status}`);
});

// 6. Valid JWT + upstream throws → 502 bad_gateway
await test("Valid JWT + upstream unreachable → 502", async () => {
  const validToken = makeJwt({ sub: "user789", exp: Math.floor(Date.now() / 1000) + 3600 });
  // baseEnv._fetch already throws
  const req = makeRequest("GET", "/v1/chat/completions", { Authorization: `Bearer ${validToken}` });
  const res = await handler(req, baseEnv, ctx);
  assert(res.status === 502, `Expected 502, got ${res.status}`);
  const body = await res.json();
  assert(body.error?.code === "bad_gateway", `Expected 'bad_gateway', got ${body.error?.code}`);
  assert(body.error.type === "upstream_error", `Expected 'upstream_error', got ${body.error.type}`);
});

// 7. The origin's public endpoints are public at the edge too, forwarded upstream.
await test("public paths skip auth → forwarded", async () => {
  const env7 = {
    ...baseEnv,
    _fetch: mockFetch(async () => new Response(JSON.stringify({ status: "ok" }), {
      status: 200, headers: { "Content-Type": "application/json" },
    })),
  };
  // A trailing slash names the same endpoint, as it does at the origin.
  for (const path of ["/health", "/health/", "/healthz", "/livez", "/ready", "/readyz", "/slo"]) {
    const res = await handler(makeRequest("GET", path, {}), env7, ctx);
    assert(res.status === 200, `Expected 200 for ${path}, got ${res.status}`);
  }
});

// 7b. /metrics is protected at the origin, so the edge must not exempt it.
await test("/metrics without a token → 401", async () => {
  const res = await handler(makeRequest("GET", "/metrics", {}), baseEnv, ctx);
  assert(res.status === 401, `Expected 401, got ${res.status}`);
  const body = await res.json();
  assert(body.error?.code === "unauthorized", `Expected 'unauthorized', got ${body.error?.code}`);
});

// 8. Rate limit exceeded → 429
await test("IP rate-limit exceeded → 429", async () => {
  const limitEnv = {
    ...baseEnv,
    EFFGEN_RATE_LIMIT_IP_MAX: "1",
    RATE_LIMIT: new KVStub(),
    _fetch: mockFetch(async () => new Response("{}", { status: 200 })),
  };
  const validToken = makeJwt({ sub: "userRL", exp: Math.floor(Date.now() / 1000) + 3600 });

  // First request: should pass
  const req1 = makeRequest("GET", "/v1/models", {
    Authorization: `Bearer ${validToken}`,
    "CF-Connecting-IP": "1.2.3.4",
  });
  const res1 = await handler(req1, limitEnv, ctx);
  assert(res1.status === 200, `First request should be 200, got ${res1.status}`);

  // Second request from same IP: should be 429
  const req2 = makeRequest("GET", "/v1/models", {
    Authorization: `Bearer ${validToken}`,
    "CF-Connecting-IP": "1.2.3.4",
  });
  const res2 = await handler(req2, limitEnv, ctx);
  assert(res2.status === 429, `Second request should be 429, got ${res2.status}`);

});

// 9. Security headers present on response
await test("Security headers injected on response", async () => {
  const validToken = makeJwt({ sub: "userSH", exp: Math.floor(Date.now() / 1000) + 3600 });
  const env9 = {
    ...baseEnv,
    _fetch: mockFetch(async () => new Response("{}", { status: 200 })),
  };
  const req = makeRequest("GET", "/v1/models", { Authorization: `Bearer ${validToken}` });
  const res = await handler(req, env9, ctx);
  assert(res.headers.get("X-Content-Type-Options") === "nosniff", "Missing X-Content-Type-Options");
  assert(res.headers.get("X-Frame-Options") === "DENY", "Missing X-Frame-Options");
});

// ── Real-fetch round-trip (no _fetch injection) ─────────────────────────────
//
// Scenarios 1-9 inject env._fetch, so they never exercise the worker's real
// upstream `fetch` call. These two start a tiny in-process HTTP origin and let
// the worker forward to it via the genuine global fetch. This guards against:
//   * a module-scope `fetch` export shadowing the global (recursion → bad URL)
//   * a streamed request body missing the required `duplex: "half"` init
async function withOrigin(handlerFn) {
  const server = createServer(handlerFn);
  await new Promise((res) => server.listen(0, "127.0.0.1", res));
  const { port } = server.address();
  const baseUrl = `http://127.0.0.1:${port}`;
  return { baseUrl, close: () => new Promise((res) => server.close(res)) };
}

// 10. Real GET forward to a live origin (exercises the global fetch binding)
await test("Real upstream GET forward (global fetch)", async () => {
  const origin = await withOrigin((req, res) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok: true, path: req.url }));
  });
  try {
    const env10 = { ...baseEnv, EFFGEN_BACKEND_URL: origin.baseUrl, RATE_LIMIT: new KVStub() };
    delete env10._fetch;  // force the worker to use the real global fetch
    const req = makeRequest("GET", "/health", {});
    const res = await handler(req, env10, ctx);
    assert(res.status === 200, `Expected 200, got ${res.status}`);
    const body = await res.json();
    assert(body.ok === true, "Upstream body not forwarded");
    assert(body.path === "/health", `Path not forwarded: ${body.path}`);
  } finally {
    await origin.close();
  }
});

// 11. Real POST forward with a body (exercises duplex + body passthrough)
await test("Real upstream POST forward with body (global fetch)", async () => {
  let received = "";
  const origin = await withOrigin((req, res) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      received = Buffer.concat(chunks).toString();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ echoed: received }));
    });
  });
  try {
    const env11 = { ...baseEnv, EFFGEN_BACKEND_URL: origin.baseUrl, RATE_LIMIT: new KVStub() };
    delete env11._fetch;
    const validToken = makeJwt({ sub: "poster", exp: Math.floor(Date.now() / 1000) + 3600 });
    const payload = JSON.stringify({ hello: "world" });
    const req = makeRequest("POST", "/v1/chat/completions",
      { Authorization: `Bearer ${validToken}`, "Content-Type": "application/json" }, payload);
    const res = await handler(req, env11, ctx);
    assert(res.status === 200, `Expected 200, got ${res.status}`);
    const body = await res.json();
    assert(body.echoed === payload, `Request body not forwarded intact: got ${body.echoed}`);
  } finally {
    await origin.close();
  }
});

// ── Report ─────────────────────────────────────────────────────────────────

console.log(JSON.stringify({ passed, failed, results }, null, 2));
process.exit(failed > 0 ? 1 : 0);
""")


def _write_harness(tmp: Path) -> Path:
    harness = tmp / "harness.mjs"
    harness.write_text(_NODE_HARNESS)
    return harness


def _run_node_harness(worker_path: Path) -> tuple[dict, int]:
    """Run the Node harness against the worker file.

    Returns a ``(parsed_json_result, returncode)`` tuple.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        harness = _write_harness(Path(tmpdir))
        result = subprocess.run(
            ["node", "--experimental-vm-modules", str(harness), str(worker_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw_stdout = result.stdout.strip()
        # Find the JSON block (last {...} block in output)
        # Node may print warnings before our JSON
        for line in reversed(raw_stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line), result.returncode
        # Try multiline
        try:
            return json.loads(raw_stdout), result.returncode
        except json.JSONDecodeError:
            # Return raw output for debugging
            return {"error": raw_stdout or result.stderr, "passed": 0, "failed": -1}, result.returncode


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node ≥18 not found; skipping fetch-handler tests")
class TestFetchHandler:
    """Exercise the worker fetch handler via a Node ESM harness."""

    @pytest.fixture(scope="class")
    def harness_result(self):
        result, rc = _run_node_harness(WORKER_JS)
        return result, rc

    def test_harness_ran(self, harness_result):
        result, _ = harness_result
        assert "results" in result or "error" not in result, (
            f"Harness failed to run: {result.get('error', result)}"
        )

    def test_all_scenarios_pass(self, harness_result):
        result, rc = harness_result
        failed = result.get("failed", -1)
        passed = result.get("passed", 0)
        failures = [r for r in result.get("results", []) if not r.get("ok")]
        summary = "\n".join(f"  FAIL: {r['name']}: {r.get('error','')}" for r in failures)
        assert rc == 0 and failed == 0, (
            f"{failed} fetch-handler test(s) failed, {passed} passed\n{summary}"
        )

    def test_options_preflight(self, harness_result):
        result, _ = harness_result
        for r in result.get("results", []):
            if "OPTIONS" in r["name"] or "preflight" in r["name"]:
                assert r["ok"], f"CORS preflight test failed: {r.get('error')}"
                return
        pytest.skip("CORS preflight test not found in results")

    def test_missing_auth_401(self, harness_result):
        result, _ = harness_result
        for r in result.get("results", []):
            if "Missing Authorization" in r["name"]:
                assert r["ok"], f"Auth test failed: {r.get('error')}"
                return
        pytest.skip("Auth test not found in results")

    def test_expired_jwt_401(self, harness_result):
        result, _ = harness_result
        for r in result.get("results", []):
            if "Expired" in r["name"] or "expired" in r["name"]:
                assert r["ok"], f"JWT expiry test failed: {r.get('error')}"
                return
        pytest.skip("JWT expiry test not found in results")

    def test_rate_limit_429(self, harness_result):
        result, _ = harness_result
        for r in result.get("results", []):
            if "rate" in r["name"].lower() or "429" in r["name"]:
                assert r["ok"], f"Rate-limit test failed: {r.get('error')}"
                return
        pytest.skip("Rate-limit test not found in results")

    def test_upstream_502(self, harness_result):
        result, _ = harness_result
        for r in result.get("results", []):
            if "502" in r["name"] or "unreachable" in r["name"]:
                assert r["ok"], f"502 test failed: {r.get('error')}"
                return
        pytest.skip("502 upstream test not found in results")

    def test_security_headers(self, harness_result):
        result, _ = harness_result
        for r in result.get("results", []):
            if "Security" in r["name"] or "security" in r["name"]:
                assert r["ok"], f"Security header test failed: {r.get('error')}"
                return
        pytest.skip("Security header test not found in results")

    def test_real_upstream_get(self, harness_result):
        """Forward to a live origin via the worker's real global fetch.

        Regression guard: a top-level ``export function fetch`` would shadow the
        global ``fetch`` and make the worker recurse into itself ("Invalid URL").
        """
        result, _ = harness_result
        for r in result.get("results", []):
            if "Real upstream GET" in r["name"]:
                assert r["ok"], f"Real GET forward failed: {r.get('error')}"
                return
        pytest.skip("Real upstream GET test not found in results")

    def test_real_upstream_post_body(self, harness_result):
        """Forward a POST body to a live origin via the real global fetch.

        Regression guard: a streamed request body needs ``duplex: "half"`` in the
        fetch init, otherwise undici/the Workers runtime throws.
        """
        result, _ = harness_result
        for r in result.get("results", []):
            if "Real upstream POST" in r["name"]:
                assert r["ok"], f"Real POST forward failed: {r.get('error')}"
                return
        pytest.skip("Real upstream POST test not found in results")
