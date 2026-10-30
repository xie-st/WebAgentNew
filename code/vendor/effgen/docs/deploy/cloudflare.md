# Deploying effGen behind a Cloudflare Worker

effGen's core is Python.  Cloudflare Workers run JS/WASM at the edge.
The `deploy/cloudflare/` directory ships a **thin proxy Worker** that sits in
front of your backend effGen deployment (Docker/K8s/Lambda) and handles:

- ✅ **CORS** — preflight handling and per-origin response headers
- ✅ **JWT Bearer validation** — structural + expiry check at the edge;
  full signature verification delegated to the upstream (OIDC JWKS)
- ✅ **Fixed-window rate limiting** — per-IP and per-token via Cloudflare KV
- ✅ **Security headers** — `X-Content-Type-Options`, `X-Frame-Options`, etc.
- ✅ **Request forwarding** — proxies to any `EFFGEN_BACKEND_URL` with
  `X-Forwarded-{Host,Proto}` headers injected

---

## Architecture

```
Client
  │
  ▼
Cloudflare Edge PoP
  ┌──────────────────────────────────────────────────────┐
  │  Worker: effgen-edge-proxy                            │
  │                                                       │
  │  1. CORS preflight?  → 204 immediately               │
  │  2. Public path?     → skip JWT check                │
  │  3. JWT present?     → parse, check expiry           │
  │  4. Rate limit?      → check KV counters             │
  │  5. Forward          → upstream effGen server        │
  │  6. Inject headers   → security + CORS               │
  └──────────────────────────────────────────────────────┘
  │
  ▼
Backend effGen server  (Docker / K8s / AWS Lambda)
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| Node.js ≥18 | <https://nodejs.org> |
| Wrangler CLI v4 | `npm install -g wrangler` |
| Cloudflare account | <https://dash.cloudflare.com/sign-up> |

---

## Quick start

### 1. Clone & configure

```bash
git clone https://github.com/your-org/effgen.git
cd effgen/deploy/cloudflare
```

Edit `wrangler.toml`:

```toml
[vars]
# Point to your running effGen backend
EFFGEN_BACKEND_URL = "https://api.your-domain.com"

# CORS: comma-separated allowed origins, or "*" for any
EFFGEN_CORS_ORIGINS = "https://app.your-domain.com"
```

### 2. Create the KV namespace

```bash
# Create the namespace and note the returned id
wrangler kv namespace create RATE_LIMIT
# → Created namespace "RATE_LIMIT": id = "abc123..."

wrangler kv namespace create RATE_LIMIT --preview
# → Created namespace "RATE_LIMIT_preview": id = "def456..."
```

Update `wrangler.toml`:

```toml
[[kv_namespaces]]
binding    = "RATE_LIMIT"
id         = "abc123..."        # ← real id from above
preview_id = "def456..."        # ← preview id
```

### 3. Set secrets

```bash
# Backend-to-worker M2M auth token (optional but recommended)
wrangler secret put EFFGEN_BACKEND_TOKEN
# Paste your secret token when prompted
```

### 4. Deploy

```bash
# Deploy to the Workers.dev subdomain (no custom domain needed)
wrangler deploy

# Or deploy to a specific environment
wrangler deploy --env staging
wrangler deploy --env production
```

### 5. Verify

```bash
# Replace with your worker URL
curl -I https://effgen-edge-proxy.<your-account>.workers.dev/health
# → HTTP/2 200
# → X-Content-Type-Options: nosniff

curl -s https://effgen-edge-proxy.<your-account>.workers.dev/v1/models
# → {"error":{"message":"Bearer token required","type":"invalid_request_error",
#             "param":null,"code":"unauthorized"}}

# With a valid token
curl -H "Authorization: Bearer $TOKEN" \
     https://effgen-edge-proxy.<your-account>.workers.dev/v1/models
# → proxied response from your backend
```

---

## Configuration reference

All configuration is in `wrangler.toml` `[vars]` or via `wrangler secret put`.

| Variable | Default | Description |
|----------|---------|-------------|
| `EFFGEN_BACKEND_URL` | `https://api.example.com` | Upstream effGen server URL (no trailing slash) |
| `EFFGEN_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated; `*` = any) |
| `EFFGEN_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window size in seconds |
| `EFFGEN_RATE_LIMIT_IP_MAX` | `100` | Max requests per IP per window (`0` = disabled) |
| `EFFGEN_RATE_LIMIT_TOKEN_MAX` | `200` | Max requests per token per window (`0` = disabled) |
| `EFFGEN_BACKEND_TOKEN` | _(secret)_ | Bearer token injected as `X-Effgen-Proxy-Token` for M2M backend auth |

### KV binding

The `RATE_LIMIT` KV namespace stores fixed-window counters.  Keys are:

- `ip:<client-ip>` — per-IP counter; TTL = `EFFGEN_RATE_LIMIT_WINDOW_SECONDS`
- `tok:<sub-prefix>` — per-token counter; same TTL

---

## Public paths (auth bypass)

These paths skip JWT validation and are forwarded directly:

```
/health   /healthz   /livez   /ready   /readyz   /slo   /favicon.ico
```

A trailing slash names the same endpoint (`/health/` is public too). The set
matches the effGen server's own public endpoints, so one auth rule decides
access at the edge and at the origin.

`/metrics` is **not** in the list: the server protects it by default, and the
Worker forwards a backend token, so exempting it at the edge would serve metrics
to an unauthenticated caller. Scrape it with a token, or set
`EFFGEN_PUBLIC_METRICS=1` on the backend and add the path here deliberately.

---

## Security headers

Every response (including errors) includes:

| Header | Value |
|--------|-------|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Permissions-Policy` | `geolocation=(), camera=(), microphone=()` |
| `X-Powered-By` | `effgen` |

---

## Rate limiting details

The Worker enforces two independent fixed-window counters stored in KV:

1. **IP-based** — keyed by `CF-Connecting-IP` (Cloudflare's authoritative
   client IP).  Limits all requests from a single IP regardless of auth status.
2. **Token-based** — keyed by the JWT `sub` claim (truncated to 64 chars).
   Limits authenticated requests per identity.

Each window opens on the first request and is reset by the KV TTL
(`EFFGEN_RATE_LIMIT_WINDOW_SECONDS`). KV is eventually consistent, so counters
are best-effort across data centres; for strict global limits, back the counter
with a Durable Object.

When either limit is exceeded, the Worker returns:

```http
HTTP/2 429 Too Many Requests
Retry-After: 60
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1716768000

{"error":{"message":"IP rate limit: 100 req/60s","type":"rate_limit_exceeded",
          "param":null,"code":"rate_limit_exceeded"}}
```

Errors raised at the edge use the same `{"error": {message, type, param, code}}`
envelope the origin server returns, so a client reads both the same way.

---

## Local development

```bash
# No Cloudflare account needed for local dev (entry comes from wrangler.toml `main`)
wrangler dev

# The dev server binds a local KV namespace automatically
# Access at http://localhost:8787
curl http://localhost:8787/health
```

---

## Environments

`wrangler.toml` ships with `staging` and `production` environments:

```bash
wrangler deploy --env staging
wrangler deploy --env production
```

Override any var per-environment in `wrangler.toml`:

```toml
[env.production]
name = "effgen-edge-proxy"
vars = { EFFGEN_BACKEND_URL = "https://api.example.com", EFFGEN_RATE_LIMIT_IP_MAX = "100" }
```

---

## Testing

Unit tests for the fetch handler run with Node (no Cloudflare account):

```bash
conda activate effgen
pytest tests/deploy/test_cloudflare_worker.py -v
```

Tests cover: CORS preflight, missing auth (401), malformed JWT (401),
expired JWT (401), valid token + upstream 200, upstream unreachable (502),
public path bypass, IP rate limiting (429), security headers.

Wrangler config validation:

```bash
wrangler deploy --dry-run
# → prints binding summary and exits; no deployment occurs
```

---

## Custom domain

To bind to a custom hostname instead of `*.workers.dev`:

```toml
[[routes]]
pattern  = "api.your-domain.com/*"
zone_name = "your-domain.com"
```

Then redeploy:

```bash
wrangler deploy --env production
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `401 unauthorized` | No `Authorization: Bearer ...` header | Add header or check client config |
| `401 invalid_token` | JWT has fewer than 3 dot-separated parts | Client sending wrong token format |
| `401 token_expired` | JWT `exp` claim is in the past | Refresh the token |
| `429 rate_limit_exceeded` | Too many requests in window | Back off and retry after `Retry-After` seconds |
| `502 bad_gateway` | Worker cannot reach `EFFGEN_BACKEND_URL` | Check backend is running and URL is correct |
| CORS errors | Origin not in `EFFGEN_CORS_ORIGINS` | Add origin to the list |
