/**
 * effGen Edge Proxy — Cloudflare Worker
 *
 * A thin edge proxy that handles:
 *   1. CORS preflight + response headers
 *   2. JWT Bearer-token validation (structure + expiry check only; signature
 *      verification is delegated to the upstream effGen server's OIDC/JWKS).
 *   3. Fixed-window rate limiting via Cloudflare KV (per IP + per token).
 *      Note: KV is eventually consistent, so counters are best-effort across
 *      data centres. For strict limits, back this with a Durable Object.
 *   4. Request forwarding to the upstream effGen backend.
 *   5. Response passthrough with security headers injected.
 *
 * Configuration (set via wrangler.toml [vars] or `wrangler secret put`):
 *
 *   EFFGEN_BACKEND_URL              — upstream server (no trailing slash)
 *   EFFGEN_CORS_ORIGINS             — comma-separated allowed origins ("*" = any)
 *   EFFGEN_RATE_LIMIT_WINDOW_SECONDS — rate-limit window size in seconds (default 60)
 *   EFFGEN_RATE_LIMIT_IP_MAX        — max requests per IP per window (0 = disabled)
 *   EFFGEN_RATE_LIMIT_TOKEN_MAX     — max requests per token per window (0 = disabled)
 *   EFFGEN_BACKEND_TOKEN            — secret: Bearer token for backend auth
 *
 * KV binding:
 *   RATE_LIMIT  — namespace bound in wrangler.toml [[kv_namespaces]]
 *
 * @module effgen-edge-proxy
 */

// ── Constants ──────────────────────────────────────────────────────────────

/** Headers added to every response for defence-in-depth. */
const SECURITY_HEADERS = {
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
  "X-Powered-By": "effgen",
};

/**
 * Public paths that skip JWT Bearer validation.
 *
 * The set mirrors the origin server's public endpoints — the liveness and
 * readiness probes and the aggregate SLO status — so one auth rule decides
 * access at the edge and at the origin. `/metrics` is deliberately absent: the
 * origin protects it by default, and the worker forwards a backend token, so
 * exempting it here would serve metrics to an unauthenticated caller.
 */
const PUBLIC_PATHS = new Set([
  "/health",
  "/healthz",
  "/livez",
  "/ready",
  "/readyz",
  "/slo",
  "/favicon.ico",
]);

/**
 * Return true when *path* is public, ignoring a trailing slash so `/health/`
 * is the probe endpoint the origin also treats as public.
 *
 * @param {string} path - Request path
 * @returns {boolean}
 */
function isPublicPath(path) {
  if (PUBLIC_PATHS.has(path)) return true;
  return path.length > 1 && path.endsWith("/") && PUBLIC_PATHS.has(path.replace(/\/+$/, ""));
}

// ── Helpers ────────────────────────────────────────────────────────────────

/**
 * Parse and return environment variables with defaults.
 *
 * @param {Record<string,string>} env - Cloudflare env bindings
 * @returns {object} Parsed config
 */
function parseConfig(env) {
  return {
    backendUrl: (env.EFFGEN_BACKEND_URL || "https://api.example.com").replace(/\/$/, ""),
    corsOrigins: (env.EFFGEN_CORS_ORIGINS || "*")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
    windowSeconds: Math.max(1, parseInt(env.EFFGEN_RATE_LIMIT_WINDOW_SECONDS || "60", 10)),
    ipMax: parseInt(env.EFFGEN_RATE_LIMIT_IP_MAX || "100", 10),
    tokenMax: parseInt(env.EFFGEN_RATE_LIMIT_TOKEN_MAX || "200", 10),
    backendToken: env.EFFGEN_BACKEND_TOKEN || "",
  };
}

/**
 * Build CORS headers for a given request origin.
 *
 * @param {string|null} origin - The Origin header from the request (or null)
 * @param {string[]} allowedOrigins - Configured allowed origins
 * @returns {Record<string,string>}
 */
function corsHeaders(origin, allowedOrigins) {
  const headers = {
    "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Request-Id",
    "Access-Control-Max-Age": "86400",
    "Access-Control-Allow-Credentials": "true",
  };

  if (!origin) return headers;

  const any = allowedOrigins.includes("*");
  if (any || allowedOrigins.includes(origin)) {
    headers["Access-Control-Allow-Origin"] = any ? "*" : origin;
    if (!any) {
      headers["Vary"] = "Origin";
    }
  }
  return headers;
}

/**
 * Decode a base64url string to a UTF-8 string.
 * Cloudflare Workers support atob() but not TextDecoder for base64url directly.
 *
 * @param {string} b64url
 * @returns {string}
 */
function base64urlDecode(b64url) {
  // Convert base64url → base64
  const b64 = b64url.replace(/-/g, "+").replace(/_/g, "/").padEnd(
    b64url.length + ((4 - (b64url.length % 4)) % 4),
    "="
  );
  try {
    return atob(b64);
  } catch {
    throw new Error("Invalid base64url encoding");
  }
}

/**
 * Minimal structural JWT parse: no signature verification here.
 * Full verification is performed by the upstream effGen server (OIDC JWKS).
 * At the edge we reject obviously malformed tokens and extract the `sub` claim
 * for rate-limiting purposes.
 *
 * @param {string} token - Raw JWT (without "Bearer " prefix)
 * @returns {{ sub: string, exp: number, iat: number } | null} Parsed claims or null
 */
function parseJwtClaims(token) {
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  try {
    const payload = JSON.parse(base64urlDecode(parts[1]));
    if (typeof payload !== "object" || payload === null) return null;
    return {
      sub: typeof payload.sub === "string" ? payload.sub : "",
      exp: typeof payload.exp === "number" ? payload.exp : 0,
      iat: typeof payload.iat === "number" ? payload.iat : 0,
    };
  } catch {
    return null;
  }
}

/**
 * Check and enforce a fixed-window rate limit stored in Cloudflare KV.
 *
 * The key stores the current request count as a plain string. The KV TTL
 * (set on the first request of a window) expires the counter, which resets
 * the window — i.e. a fixed window of `windowSeconds`, not a true sliding one.
 *
 * @param {KVNamespace} kv - Bound KV namespace
 * @param {string} key - Rate-limit key (e.g. "ip:<ip>" or "tok:<hash>")
 * @param {number} max - Max requests per window
 * @param {number} windowSeconds - Window size
 * @returns {Promise<{ ok: boolean; count: number; remaining: number }>}
 */
async function checkRateLimit(kv, key, max, windowSeconds) {
  if (max <= 0) return { ok: true, count: 0, remaining: Infinity };

  const raw = await kv.get(key);
  const count = raw ? parseInt(raw, 10) : 0;
  const next = count + 1;

  if (count === 0) {
    // First request in this window — set with TTL
    await kv.put(key, "1", { expirationTtl: windowSeconds });
  } else {
    // Increment. Cloudflare KV `put` without a TTL would reset the key to
    // no-expiry, so we re-assert the window TTL on every write. This slightly
    // extends the window on each request (acceptable for a best-effort proxy).
    await kv.put(key, String(next), { expirationTtl: windowSeconds });
  }

  const remaining = Math.max(0, max - next);
  return { ok: next <= max, count: next, remaining };
}

/**
 * Map an HTTP status to the OpenAI error `type` the origin server uses, so an
 * error raised at the edge carries the same fields as one from the origin.
 *
 * @param {number} status - HTTP status code
 * @returns {string}
 */
function errorType(status) {
  const byStatus = {
    400: "invalid_request_error",
    401: "invalid_request_error",
    403: "permission_error",
    404: "invalid_request_error",
    413: "invalid_request_error",
    429: "rate_limit_exceeded",
    502: "upstream_error",
    503: "upstream_unavailable",
    504: "timeout",
  };
  return byStatus[status] || (status >= 500 ? "server_error" : "invalid_request_error");
}

/**
 * Build a JSON error response with standard headers.
 *
 * Uses the origin server's `{"error": {message, type, param, code}}` envelope
 * so a client reads edge and origin failures the same way.
 *
 * @param {number} status - HTTP status code
 * @param {string} code - Machine-readable error code
 * @param {string} message - Human-readable message
 * @param {Record<string,string>} [extraHeaders={}]
 * @returns {Response}
 */
function errorResponse(status, code, message, extraHeaders = {}) {
  return new Response(
    JSON.stringify({ error: { message, type: errorType(status), param: null, code } }),
    {
      status,
      headers: {
        "Content-Type": "application/json",
        ...SECURITY_HEADERS,
        ...extraHeaders,
      },
    }
  );
}

// ── Main fetch handler ─────────────────────────────────────────────────────

/**
 * Cloudflare Worker fetch handler — entry point.
 *
 * @param {Request} request
 * @param {Record<string,any>} env  - Worker environment bindings (vars + KV)
 * @param {ExecutionContext} ctx    - Execution context (waitUntil, passThroughOnException)
 * @returns {Promise<Response>}
 */
async function handleRequest(request, env, ctx) {
  const url = new URL(request.url);
  const path = url.pathname;
  const origin = request.headers.get("Origin");
  const cfg = parseConfig(env);
  const cors = corsHeaders(origin, cfg.corsOrigins);

  // ── 1. CORS preflight ────────────────────────────────────────────────────
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: { ...cors, ...SECURITY_HEADERS } });
  }

  // ── 2. JWT Bearer validation (skip public paths) ─────────────────────────
  let tokenSub = "";
  const isPublic = isPublicPath(path);

  if (!isPublic) {
    const authHeader = request.headers.get("Authorization") || "";
    if (!authHeader.toLowerCase().startsWith("bearer ")) {
      return errorResponse(401, "unauthorized", "Bearer token required", cors);
    }

    const rawToken = authHeader.slice(7).trim();
    const claims = parseJwtClaims(rawToken);

    if (!claims) {
      return errorResponse(401, "invalid_token", "Malformed JWT", cors);
    }

    // Check token expiry (structural check only; signature verified upstream)
    const nowSec = Math.floor(Date.now() / 1000);
    if (claims.exp > 0 && claims.exp < nowSec) {
      return errorResponse(401, "token_expired", "JWT has expired", cors);
    }

    tokenSub = claims.sub || rawToken.slice(-16); // fallback: last 16 chars as opaque id
  }

  // ── 3. Rate limiting ─────────────────────────────────────────────────────
  const kv = env.RATE_LIMIT;
  if (kv) {
    const clientIp =
      request.headers.get("CF-Connecting-IP") ||
      request.headers.get("X-Forwarded-For")?.split(",")[0].trim() ||
      "unknown";

    // IP-based rate limit
    const ipResult = await checkRateLimit(
      kv,
      `ip:${clientIp}`,
      cfg.ipMax,
      cfg.windowSeconds
    );
    if (!ipResult.ok) {
      return errorResponse(
        429,
        "rate_limit_exceeded",
        `IP rate limit: ${cfg.ipMax} req/${cfg.windowSeconds}s`,
        {
          ...cors,
          "Retry-After": String(cfg.windowSeconds),
          "X-RateLimit-Limit": String(cfg.ipMax),
          "X-RateLimit-Remaining": "0",
          "X-RateLimit-Reset": String(Math.floor(Date.now() / 1000) + cfg.windowSeconds),
        }
      );
    }

    // Per-token rate limit (only for authenticated requests)
    if (tokenSub && cfg.tokenMax > 0) {
      // Truncate the sub claim to keep the KV key bounded (Cloudflare key limit
      // is 512 bytes; 64 chars is ample for a `sub` identifier).
      const tokenKey = `tok:${tokenSub.slice(0, 64)}`;
      const tokResult = await checkRateLimit(
        kv,
        tokenKey,
        cfg.tokenMax,
        cfg.windowSeconds
      );
      if (!tokResult.ok) {
        return errorResponse(
          429,
          "rate_limit_exceeded",
          `Token rate limit: ${cfg.tokenMax} req/${cfg.windowSeconds}s`,
          {
            ...cors,
            "Retry-After": String(cfg.windowSeconds),
            "X-RateLimit-Limit": String(cfg.tokenMax),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": String(Math.floor(Date.now() / 1000) + cfg.windowSeconds),
          }
        );
      }
    }
  }

  // ── 4. Forward to upstream effGen backend ───────────────────────────────
  const upstreamUrl = `${cfg.backendUrl}${path}${url.search}`;

  // Clone headers; inject backend auth token if configured
  const upstreamHeaders = new Headers(request.headers);
  upstreamHeaders.delete("Host");           // let Cloudflare set the correct Host
  upstreamHeaders.set("X-Forwarded-Host", url.host);
  upstreamHeaders.set("X-Forwarded-Proto", url.protocol.replace(":", ""));

  if (cfg.backendToken) {
    // Override Authorization for the backend call with a machine-to-machine token
    upstreamHeaders.set("X-Effgen-Proxy-Token", cfg.backendToken);
  }

  // env._fetch is an optional test-injection hook (not used in production Workers).
  // In the Cloudflare runtime `fetch` is the global built-in; in unit tests
  // callers inject a stub via env._fetch to avoid real network calls.
  const _fetch = env._fetch ?? fetch;

  const hasBody = !["GET", "HEAD"].includes(request.method);
  const upstreamInit = {
    method: request.method,
    headers: upstreamHeaders,
    body: hasBody ? request.body : null,
    redirect: "follow",
  };
  // When streaming a request body (a ReadableStream), the Fetch standard
  // requires `duplex: "half"`. The Cloudflare runtime and undici both enforce
  // this; omitting it throws "duplex option is required when sending a body".
  if (hasBody) {
    upstreamInit.duplex = "half";
  }

  let upstreamResponse;
  try {
    upstreamResponse = await _fetch(upstreamUrl, upstreamInit);
  } catch (err) {
    return errorResponse(
      502,
      "bad_gateway",
      `Upstream unreachable: ${err.message}`,
      cors
    );
  }

  // ── 5. Pass response through with security + CORS headers ────────────────
  const responseHeaders = new Headers(upstreamResponse.headers);
  for (const [k, v] of Object.entries(SECURITY_HEADERS)) {
    responseHeaders.set(k, v);
  }
  for (const [k, v] of Object.entries(cors)) {
    responseHeaders.set(k, v);
  }

  return new Response(upstreamResponse.body, {
    status: upstreamResponse.status,
    statusText: upstreamResponse.statusText,
    headers: responseHeaders,
  });
}

// ── Default export (Workers module syntax) ─────────────────────────────────
// NOTE: the handler is named `handleRequest`, not `fetch`. A top-level
// `export function fetch(...)` would shadow the global `fetch` within this
// module, so the worker's own upstream `fetch()` call would recurse into the
// handler instead of making a network request. The Workers runtime only needs
// `default.fetch`, which we satisfy here.
export { handleRequest };
export default { fetch: handleRequest };
