"""Production middleware for the effGen API server.

Attached to an existing FastAPI app via :func:`install_production_middleware`.
Provides:

- Request ID injection (``X-Request-ID``)
- CORS configuration
- Response compression (gzip)
- Request/response validation is handled by FastAPI/Pydantic by default
- Shutdown hooks that drain in-flight requests (SIGTERM/SIGINT)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_rate_limit(rate_limit_per_minute: int | None) -> int:
    """Resolve the requests-per-minute limit (param overrides EFFGEN_RATE_LIMIT).

    Returns ``0`` (disabled) when neither source provides a positive integer.
    """
    if rate_limit_per_minute is None:
        raw = (os.getenv("EFFGEN_RATE_LIMIT", "") or "").strip()
        if not raw:
            return 0
        try:
            rate_limit_per_minute = int(raw)
        except ValueError:
            logger.warning(
                "EFFGEN_RATE_LIMIT=%r is not an integer — request rate limiting disabled",
                raw,
            )
            return 0
    return max(0, int(rate_limit_per_minute))


def _resolve_trust_proxy(trust_proxy: bool | None) -> bool:
    """Resolve whether to trust ``X-Forwarded-For`` (param overrides
    ``EFFGEN_TRUST_PROXY``). Defaults to ``False``: a directly-exposed server
    must not let a client-supplied header override its own rate-limit key.
    """
    if trust_proxy is not None:
        return bool(trust_proxy)
    raw = (os.getenv("EFFGEN_TRUST_PROXY", "") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def install_production_middleware(
    app: Any,
    *,
    cors_origins: Iterable[str] | None = None,
    dev_mode: bool = False,
    allow_credentials: bool | None = None,
    enable_gzip: bool = True,
    gzip_min_size: int = 500,
    enable_request_id: bool = True,
    shutdown_timeout: float = 10.0,
    rate_limit_per_minute: int | None = None,
    trust_proxy: bool | None = None,
) -> None:
    """Install the standard middleware stack on a FastAPI ``app``.

    Safe to call even if FastAPI/Starlette is not importable (no-op).

    CORS is fail-closed: when ``cors_origins`` is not provided, cross-origin
    access is **disabled** in production (no CORS middleware is installed) and
    only opened to ``*`` in dev mode. A wildcard origin is never combined with
    ``allow_credentials=True`` (that combination is both insecure and rejected
    by browsers), so credentials are enabled only for an explicit origin list.

    Args:
        app: The FastAPI application to install onto.
        cors_origins: Origins allowed to call the API cross-origin.
        dev_mode: Open CORS to any origin, for local development only.
        allow_credentials: Whether cross-origin requests may carry credentials;
            never combined with a wildcard origin.
        enable_gzip: Compress responses above *gzip_min_size*.
        gzip_min_size: Smallest response body, in bytes, that is compressed.
        enable_request_id: Stamp and echo a request id on every response.
        shutdown_timeout: Seconds in-flight requests get to finish on shutdown.
        rate_limit_per_minute: Per-client request cap, or ``None`` for no limit.
        trust_proxy: Read the client IP from the first ``X-Forwarded-For`` hop.
    """
    try:
        from starlette.middleware.cors import CORSMiddleware
        from starlette.middleware.gzip import GZipMiddleware
    except Exception:  # pragma: no cover
        logger.warning("fastapi/starlette not available — middleware skipped")
        return

    # 1. CORS
    origins = list(cors_origins or [])
    if not origins and dev_mode:
        origins = ["*"]
    wildcard = "*" in origins
    if allow_credentials is None:
        # Credentials only for an explicit, non-wildcard origin allow-list.
        allow_credentials = bool(origins) and not wildcard
    if wildcard and allow_credentials:
        logger.warning(
            "CORS: refusing to combine wildcard origin with credentials; "
            "disabling allow_credentials."
        )
        allow_credentials = False

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    else:
        logger.info(
            "CORS: no cross-origin access configured (set EFFGEN_CORS_ORIGINS "
            "or pass cors_origins to enable)."
        )

    # 2. gzip compression
    if enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=gzip_min_size)

    # 3. Request ID injection. Implemented as a *pure-ASGI* middleware (not the
    # @app.middleware("http") / BaseHTTPMiddleware form) because
    # BaseHTTPMiddleware buffers the whole response body before forwarding it,
    # which breaks server-sent-event streaming (the /v1 SSE responses) — it
    # would hang or raise "No response returned." Pure-ASGI middleware forwards
    # each chunk as it is produced, so streaming stays truly incremental.
    if enable_request_id:
        app.add_middleware(RequestIDMiddleware)  # type: ignore[arg-type]

    # 4. Per-client request rate limiting (opt-in via EFFGEN_RATE_LIMIT).
    # Added last so it wraps as the outermost layer: a request flood is rejected
    # cheaply (per-IP, before auth/route work). Disabled unless a positive limit
    # is configured. ``X-Forwarded-For`` is trusted only when the deployment
    # opts in (``trust_proxy=True`` / ``EFFGEN_TRUST_PROXY=1``) — a directly
    # exposed server must key the limiter on the real socket peer, since any
    # caller can set that header to a value of their choosing.
    limit = _resolve_rate_limit(rate_limit_per_minute)
    trust_xff = _resolve_trust_proxy(trust_proxy)
    if limit > 0:
        app.add_middleware(  # type: ignore[arg-type]
            RateLimitMiddleware, requests_per_minute=limit, trust_x_forwarded_for=trust_xff
        )
        logger.info(
            "Request rate limiting enabled: %d req/min per client (trust_proxy=%s)",
            limit, trust_xff,
        )

    # 5. Graceful shutdown
    _install_graceful_shutdown(app, shutdown_timeout)

    # 6. Status-coded request counter — outermost layer (added last) so it
    # observes the status actually sent to the client: rate-limit 429s,
    # auth 401s, RBAC 403s, upstream 502/503s, and the shutdown-drain 503
    # above, not just successful routed calls.
    app.add_middleware(RequestMetricsMiddleware)  # type: ignore[arg-type]


class RequestIDMiddleware:
    """Pure-ASGI middleware that stamps every request/response with an id.

    Reads an inbound ``X-Request-ID`` (or generates one), exposes it on
    ``scope["state"]["request_id"]``, and echoes it on the response. Unlike a
    ``BaseHTTPMiddleware`` it never buffers the response body, so streaming
    (SSE) responses are forwarded chunk-by-chunk.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        inbound = headers.get(b"x-request-id")
        req_id = inbound.decode("latin-1", errors="replace") if inbound else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = req_id
        req_id_bytes = req_id.encode("latin-1", errors="replace")

        async def _send(message: dict) -> None:
            if message.get("type") == "http.response.start":
                hdrs = [
                    (k, v) for k, v in message.get("headers", [])
                    if k.lower() != b"x-request-id"
                ]
                hdrs.append((b"x-request-id", req_id_bytes))
                message = {**message, "headers": hdrs}
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception:
            logger.exception("request %s failed", req_id)
            raise


# Health/liveness/readiness probes are exempt from rate limiting so that a
# frequently-polling load balancer or K8s probe can never throttle real traffic.
_RATE_LIMIT_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/health", "/healthz", "/livez", "/readyz", "/ready"}
)


class RateLimitMiddleware:
    """Pure-ASGI fixed-window per-client request limiter.

    Honours the ``EFFGEN_RATE_LIMIT`` knob (requests/minute). Keyed by client IP
    so a flood is rejected before auth/route work, mirroring the per-IP limit the
    Cloudflare worker enforces at the edge. The window is per-process (each
    uvicorn worker keeps its own counters); behind multiple workers the effective
    limit is ``requests_per_minute × workers``. Probe paths are exempt and
    ``OPTIONS`` preflight requests are never counted. On breach it returns a
    redacted ``429`` with a ``Retry-After`` header — no buffering, so SSE
    responses on the allowed path stay incremental. Every response — allowed
    or throttled — also carries ``RateLimit-Limit``/``RateLimit-Remaining``/
    ``RateLimit-Reset`` headers so a client can pace itself before hitting 429.

    ``trust_x_forwarded_for`` defaults to ``False``: the limiter keys on the raw
    socket peer (``scope["client"]``), which a caller cannot spoof. A directly
    exposed server that honoured ``X-Forwarded-For`` unconditionally would let
    any caller defeat its own rate limit by sending a fresh value on every
    request. Set it to ``True`` only when the server sits behind a proxy that
    is trusted to set that header accurately (and strips any client-supplied
    one first) — the first hop is then used as the client IP.
    """

    def __init__(
        self,
        app: Any,
        *,
        requests_per_minute: int,
        window_seconds: int = 60,
        trust_x_forwarded_for: bool = False,
    ) -> None:
        self.app = app
        self.limit = max(1, int(requests_per_minute))
        self.window = max(1, int(window_seconds))
        self.trust_x_forwarded_for = trust_x_forwarded_for
        # client-ip -> [window_start_epoch, count]
        self._buckets: dict[str, list[float]] = {}

    def _client_ip(self, scope: dict) -> str:
        # Only consider X-Forwarded-For when the deployment explicitly opted
        # in to trusting a proxy; otherwise a client-supplied header must never
        # influence the rate-limit key.
        if self.trust_x_forwarded_for:
            for k, v in scope.get("headers", []):
                if k == b"x-forwarded-for":
                    first = v.decode("latin-1", errors="replace").split(",")[0].strip()
                    if first:
                        return first
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _check(self, ip: str) -> tuple[bool, int, int]:
        """Return (allowed, remaining, reset_seconds).

        ``remaining`` is the requests still allowed in the current window;
        ``reset_seconds`` is how long until the window rolls over (also the
        ``Retry-After`` value on a 429).
        """
        now = time.monotonic()
        bucket = self._buckets.get(ip)
        if bucket is None or now - bucket[0] >= self.window:
            self._buckets[ip] = [now, 1]
            # Opportunistically prune stale buckets to bound memory.
            if len(self._buckets) > 4096:
                cutoff = now - self.window
                self._buckets = {
                    k: b for k, b in self._buckets.items() if b[0] >= cutoff
                }
            return True, self.limit - 1, self.window
        reset = max(1, int(self.window - (now - bucket[0])))
        if bucket[1] < self.limit:
            bucket[1] += 1
            return True, self.limit - bucket[1], reset
        return False, 0, reset

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in _RATE_LIMIT_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        allowed, remaining, reset = self._check(self._client_ip(scope))
        # Standard rate-limit headers (draft IETF RateLimit-* shape) so a
        # well-behaved client can pace itself before ever hitting 429, not
        # just learn about it after the fact via Retry-After.
        rate_headers = [
            (b"ratelimit-limit", str(self.limit).encode()),
            (b"ratelimit-remaining", str(remaining).encode()),
            (b"ratelimit-reset", str(reset).encode()),
        ]
        if allowed:
            async def _send_with_rate_headers(message: Any) -> None:
                if message["type"] == "http.response.start":
                    message = {
                        **message,
                        "headers": list(message.get("headers", [])) + rate_headers,
                    }
                await send(message)

            await self.app(scope, receive, _send_with_rate_headers)
            return

        import json as _json

        try:
            from effgen.api.openai_compat import error_envelope

            payload = error_envelope(
                429, "Rate limit exceeded. Please retry later.", redact=False
            )
        except Exception:  # noqa: BLE001 - keep the limiter working if the helper is unavailable
            payload = {"error": {"message": "Rate limit exceeded. Please retry later.",
                                 "type": "rate_limit_exceeded", "param": None,
                                 "code": "rate_limit_exceeded"}}
        body = _json.dumps(payload).encode()
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(reset).encode()),
                *rate_headers,
            ],
        })
        await send({"type": "http.response.body", "body": body})


#: Routes tracked with their own label; anything else collapses to "other" so
#: a scanner probing random paths cannot grow the metric's label cardinality.
_KNOWN_HTTP_ROUTES: frozenset[str] = frozenset({
    "/health", "/healthz", "/livez", "/ready", "/readyz", "/metrics",
    "/whoami", "/rbac/policy", "/rbac/roles",
    "/v1/models", "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
})


class RequestMetricsMiddleware:
    """Pure-ASGI middleware recording ``effgen_http_requests_total{route,method,status}``.

    Reads the status code off the ``http.response.start`` message (never
    buffers the body, so SSE stays incremental) and records it once the
    response — or an unhandled exception — has completed. Placed as the
    outermost layer so it sees the status the client actually received,
    including responses produced by other middleware (rate limiting, auth,
    shutdown drain), not just successful routed calls.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "GET")
        path = scope.get("path", "")
        route = path if path in _KNOWN_HTTP_ROUTES else "other"
        status_holder = {"status": 500}

        async def _send(message: dict) -> None:
            if message.get("type") == "http.response.start":
                status_holder["status"] = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            try:
                from effgen.observability.metrics import record_http_request

                record_http_request(
                    route=route, method=method, status=status_holder["status"]
                )
            except Exception:  # noqa: BLE001 - metrics must never break a request
                pass


def _install_graceful_shutdown(app: Any, timeout: float) -> None:
    """Drain in-flight requests before exiting on SIGTERM/SIGINT."""
    inflight: set = set()
    shutting_down = {"value": False}

    class _InflightMiddleware:
        """Pure-ASGI in-flight tracker (no response buffering — SSE-safe)."""

        def __init__(self, asgi_app: Any) -> None:
            self.app = asgi_app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return
            if shutting_down["value"]:
                import json as _json

                message = "Server is shutting down; retry against another instance."
                try:
                    from effgen.api.openai_compat import error_envelope

                    payload = error_envelope(
                        503, message, code="server_shutting_down", redact=False
                    )
                except Exception:  # noqa: BLE001 - the drain path must not depend on the import
                    payload = {
                        "error": {
                            "message": message,
                            "type": "upstream_unavailable",
                            "param": None,
                            "code": "server_shutting_down",
                        }
                    }
                body = _json.dumps(payload).encode()
                await send({
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return
            task = asyncio.current_task()
            if task is not None:
                inflight.add(task)
            try:
                await self.app(scope, receive, send)
            finally:
                if task is not None:
                    inflight.discard(task)

    app.add_middleware(_InflightMiddleware)  # type: ignore[arg-type]

    @app.on_event("shutdown")
    async def _drain() -> None:
        shutting_down["value"] = True
        deadline = asyncio.get_event_loop().time() + timeout
        while inflight and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.1)
        if inflight:
            logger.warning(
                "shutdown: %d requests still in-flight after %.1fs",
                len(inflight),
                timeout,
            )

    # Best-effort signal hookup; uvicorn handles SIGTERM itself, but when
    # embedded elsewhere this ensures flags still flip.
    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig, lambda: shutting_down.__setitem__("value", True)
                )
            except (NotImplementedError, RuntimeError):
                pass
    except RuntimeError:
        pass
