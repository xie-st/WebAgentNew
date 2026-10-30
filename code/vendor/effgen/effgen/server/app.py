"""effGen API Server — main FastAPI application with auth/RBAC/audit wired in.

Usage::

    # Production (requires EFFGEN_OIDC_ISSUER + EFFGEN_OIDC_CLIENT_ID)
    uvicorn effgen.server.app:create_app --factory --host 0.0.0.0 --port 8080

    # Dev mode (auth disabled, loud warning)
    EFFGEN_DEV_MODE=1 uvicorn effgen.server.app:create_app --factory ...

Environment variables
---------------------
EFFGEN_DEV_MODE          — "1" disables auth (warn only, never silently)
EFFGEN_OIDC_ISSUER       — OIDC issuer URL
EFFGEN_OIDC_CLIENT_ID    — JWT audience claim
EFFGEN_OIDC_JWKS_URI     — JWKS endpoint (auto-discovered if omitted)
EFFGEN_RBAC_POLICY_FILE  — path to JSON role policy file
EFFGEN_AUDIT_DIR         — override audit log directory
EFFGEN_METRICS_AUTH      — "1" to require auth on /metrics
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Machine-readable ``code`` for the errors the framework itself raises, so a
# client can branch on a stable string instead of parsing the message.
_HTTP_ERROR_CODES: dict[int, str] = {
    404: "not_found",
    405: "method_not_allowed",
}

# The application is assembled from sibling modules — middleware, the default
# runner + model pool, the dashboard/playground surfaces, and the dashboard
# data builders. Every name they define is re-exported here so
# ``from effgen.server.app import X`` and patches against this module resolve
# unchanged.
from effgen.server.app_dashboard_data import (  # noqa: E402,F401  re-exported for import/patch parity
    _PROM_LINE_RE,
    _bucket_bounds,
    _build_dashboard_data,
    _build_dashboard_history,
    _collect_dashboard_metrics,
    _effgen_version,
    _get_prompt_templates,
    _get_recent_runs,
    _get_recent_spans,
    _histogram_quantile,
    _http_route_breakdown,
    _http_status_breakdown,
    _http_status_detail,
    _latency_percentiles,
    _metric_key,
    _model_breakdown,
    _parse_prometheus_labels,
    _parse_prometheus_text,
    _sum_samples,
)
from effgen.server.app_middleware import (  # noqa: E402
    MaxBodySizeMiddleware,
    RBACBudgetMiddleware,
    _enforce_max_body_size,  # noqa: F401  re-exported for import/patch parity
    _reject_json,  # noqa: F401  re-exported for import/patch parity
)
from effgen.server.app_runner import (  # noqa: E402,F401  re-exported for import/patch parity
    _KNOWN_PROVIDERS,
    _LOADING_LOCKS,
    _MODEL_POOL,
    _MODEL_POOL_LOCK,
    _SERVED_MODEL_IDS,
    _SERVED_MODEL_LOCK,
    _SERVED_MODEL_MAX,
    _TRACE_ARG_MAX,
    _TRACE_RESULT_MAX,
    _build_default_runner,
    _extract_cost,
    _extract_tool_trace,
    _extract_usage,
    _get_pooled_model,
    _model_pool_max,
    _mount_existing_routers,
    _normalize_model_id,
    _record_served_model,
    _resolve_tools,
    _served_model_ids,
    _StreamWithUsage,
    _tool_name,
)
from effgen.server.app_web import (  # noqa: E402
    _PLAYGROUND_SAFE_TOOLS,  # noqa: F401  re-exported for import/patch parity
    _asset_not_found,  # noqa: F401  re-exported for import/patch parity
    _build_playground_bootstrap,  # noqa: F401  re-exported for import/patch parity
    _FastAPIRequest,
    _mount_dashboard,
    _mount_playground,
    _resolve_web_asset,  # noqa: F401  re-exported for import/patch parity
)


def _server_version() -> str:
    """Return the effGen package version without paying for a heavy import.

    Reads the installed distribution metadata first (cheap; does not import the
    ``effgen`` package). Falls back to the package attribute, then a sentinel.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("effgen")
        except PackageNotFoundError:  # no install metadata (source/editable run); use the fallback version
            pass
    except Exception:  # noqa: BLE001 - importlib.metadata always present on 3.11
        pass
    try:
        from effgen import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _truthy_env(name: str) -> bool:
    """Return True when env var *name* is set to a truthy value."""
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    dev_mode: bool | None = None,
    oidc_issuer: str | None = None,
    oidc_client_id: str | None = None,
    oidc_jwks_uri: str | None = None,
    api_key: str | None = None,
    metrics_auth: bool = False,
    public_metrics: bool | None = None,
    public_dashboard: bool | None = None,
    public_playground: bool | None = None,
    cors_origins: list[str] | None = None,
    rate_limit_per_minute: int | None = None,
    trust_proxy: bool | None = None,
    runner: Any = None,
    extra_models: Any = None,
) -> Any:
    """Create and return the FastAPI application.

    All parameters fall back to environment variables when ``None``.

    Parameters
    ----------
    runner:
        Optional callable ``runner(prompt, *, model, tools, stream, ...)`` used
        by the OpenAI-compatible ``/v1`` endpoints. When ``None`` a default
        agent-backed runner is constructed lazily on first use.
    extra_models:
        Optional callable returning model ids to list in ``GET /v1/models``
        alongside the legacy aliases. When ``None`` this defaults to the ids
        the default runner has actually served a successful response for.
    trust_proxy:
        Whether the per-IP rate limiter should trust the first
        ``X-Forwarded-For`` hop as the client IP. Defaults to ``False`` (the
        raw socket peer is used); enable only when the deployment sits behind
        a reverse proxy that sets/overwrites this header, since any direct
        caller can otherwise set it to bypass the limit.

    Args:
        dev_mode: Relax CORS for local development.
        oidc_issuer: Issuer whose tokens the server accepts.
        oidc_client_id: Audience the tokens must carry.
        oidc_jwks_uri: Where the issuer's signing keys are fetched from.
        api_key: The API key clients authenticate with; one is minted when absent.
        metrics_auth: Require authentication on ``/metrics``.
        public_metrics: Serve ``/metrics`` without authentication.
        public_dashboard: Serve the dashboard without authentication.
        public_playground: Serve the playground without authentication.
        cors_origins: Origins allowed to call the API cross-origin.
        rate_limit_per_minute: Per-client request cap, or ``None`` for no limit.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError("fastapi is required: pip install 'effgen[server]'") from exc

    _dev = dev_mode if dev_mode is not None else (os.getenv("EFFGEN_DEV_MODE", "0") == "1")

    # Observability endpoints are protected by default. They become public only
    # in dev mode or when an operator explicitly opts in (constructor flag or
    # EFFGEN_PUBLIC_METRICS / EFFGEN_PUBLIC_DASHBOARD). The legacy
    # ``metrics_auth`` flag (and EFFGEN_METRICS_AUTH) still force metrics auth on.
    if public_metrics is None:
        public_metrics = _dev or _truthy_env("EFFGEN_PUBLIC_METRICS")
    if metrics_auth or _truthy_env("EFFGEN_METRICS_AUTH"):
        public_metrics = False
    if public_dashboard is None:
        public_dashboard = _dev or _truthy_env("EFFGEN_PUBLIC_DASHBOARD")
    # The playground drives real, billed model calls, so its local-view opt-in
    # is a flag of its own (not shared with the dashboard's) — an operator must
    # knowingly authorize spend from a page anyone with the URL can reach.
    if public_playground is None:
        public_playground = _dev or _truthy_env("EFFGEN_PUBLIC_PLAYGROUND")

    app = FastAPI(
        title="effGen API",
        description="effGen — Efficient Agent Framework API",
        version=_server_version(),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Every error this app returns — validation (422), a raised HTTPException
    # (including the router's own 404/405), and an unhandled exception (500) —
    # is rendered with the same `{"error": {message, type, param, code}}`
    # envelope the model routes use, so a client branches on `err.type` /
    # `err.code` uniformly instead of switching between that and FastAPI's
    # `{"detail": ...}`.
    try:
        from fastapi.exceptions import RequestValidationError
        from starlette.exceptions import HTTPException as _StarletteHTTPException

        from effgen.api.openai_compat import _classify_http, error_envelope

        @app.exception_handler(RequestValidationError)
        async def _validation_handler(request: Any, exc: RequestValidationError) -> Any:
            errs = exc.errors()
            try:
                first = errs[0]
                loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
                msg = f"{loc}: {first.get('msg')}" if loc else str(first.get("msg"))
            except Exception:  # noqa: BLE001 - fall back to a generic message
                msg = "invalid request body"
            return JSONResponse(
                status_code=422,
                content=error_envelope(422, msg, code="invalid_request_body", redact=False),
            )

        @app.exception_handler(_StarletteHTTPException)
        async def _http_exception_handler(request: Any, exc: Any) -> Any:
            detail = exc.detail if isinstance(exc.detail, str) else "request failed"
            # A 404/405 from the router is about the URL, not about a model, so
            # it is typed as an invalid request. Response headers the exception
            # carries are part of the HTTP contract (``Allow`` on a 405,
            # ``WWW-Authenticate`` on a 401) and survive the re-render.
            return JSONResponse(
                status_code=exc.status_code,
                content=error_envelope(
                    exc.status_code,
                    detail,
                    code=_HTTP_ERROR_CODES.get(exc.status_code),
                    error_type=(
                        "invalid_request_error" if exc.status_code in (404, 405) else None
                    ),
                ),
                headers=getattr(exc, "headers", None),
            )

        @app.exception_handler(Exception)
        async def _unhandled_handler(request: Any, exc: Exception) -> Any:
            # The handler runs outside the ``except`` block, so attach the
            # exception explicitly to keep the traceback in the log.
            logger.error(
                "Unhandled error serving %s",
                getattr(request, "url", ""),
                exc_info=exc,
            )
            status, _err_type, code = _classify_http(exc)
            if status < 500:
                status, code = 500, None
            # The message is redacted before it leaves the process so an
            # upstream error carrying a key does not reach the client.
            return JSONResponse(
                status_code=status,
                content=error_envelope(status, str(exc) or exc.__class__.__name__, code=code),
            )
    except Exception:  # noqa: BLE001 - error handlers are best-effort
        logger.debug("Could not install error handlers", exc_info=True)

    # ------------------------------------------------------------------
    # Middleware stack. ``add_middleware`` is LIFO, so the *last* added wraps
    # as the outermost layer. We want, outer → inner:
    #
    #   production (CORS/gzip/request-id)  →  audit  →  auth  →  RBAC/budget  →  route
    #
    # Auth must run before RBAC/budget so request.state.user is populated, and
    # RBAC/budget sits innermost (just outside the route) so its request-body
    # replay survives the production BaseHTTPMiddleware layers above it.
    # ------------------------------------------------------------------

    # 1. RBAC + budget enforcement (innermost; reads body for /v1 endpoints)
    app.add_middleware(RBACBudgetMiddleware)  # type: ignore[arg-type]

    # 1b. Body-size cap for body-accepting routes RBAC/budget doesn't cover
    # (e.g. /v1/embeddings, which needs no RBAC/budget enforcement but must
    # still bound how much a client can make the server buffer).
    app.add_middleware(MaxBodySizeMiddleware)  # type: ignore[arg-type]

    # 2. Auth middleware (validates JWT, populates request.state.user)
    from effgen.server.auth import AuthMiddleware

    app.add_middleware(
        AuthMiddleware,  # type: ignore[arg-type]
        issuer=oidc_issuer,
        client_id=oidc_client_id,
        jwks_uri=oidc_jwks_uri,
        api_key=api_key,
        public_metrics=public_metrics,
        public_dashboard=public_dashboard,
        public_playground=public_playground,
        dev_mode=_dev,
    )

    # Record the resolved playground settings so the playground bootstrap route
    # can decide whether to expose a local-view session key and report whether
    # spend is authorized without re-reading the environment.
    app.state.public_playground = bool(public_playground)
    app.state.dev_mode = bool(_dev)
    app.state.api_key = api_key if api_key is not None else os.getenv("EFFGEN_API_KEY", "")

    # 3. Audit middleware (logs every request; reads user from state set by auth)
    from effgen.server.audit import AuditMiddleware

    app.add_middleware(AuditMiddleware)  # type: ignore[arg-type]

    # 4. Production middleware (CORS, gzip, request-ID) — outermost.
    # CORS is fail-closed: outside dev mode, cross-origin access is disabled
    # unless ``cors_origins`` (or EFFGEN_CORS_ORIGINS) is explicitly configured.
    if cors_origins is None:
        _env_origins = os.getenv("EFFGEN_CORS_ORIGINS", "").strip()
        if _env_origins:
            cors_origins = [o.strip() for o in _env_origins.split(",") if o.strip()]
    try:
        from effgen.api.middleware import install_production_middleware

        install_production_middleware(
            app,
            cors_origins=cors_origins,
            dev_mode=_dev,
            rate_limit_per_minute=rate_limit_per_minute,
            trust_proxy=trust_proxy,
        )
    except ImportError:
        logger.warning("Production middleware not available")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", tags=["ops"])
    @app.get("/healthz", tags=["ops"])
    @app.get("/livez", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness probe — always public.

        ``/health`` is the canonical name; ``/healthz`` and ``/livez`` are
        aliases for the conventional Kubernetes liveness probe path. A 200 here
        means the process is up and serving requests.
        """
        return {"status": "ok", "version": _server_version()}

    @app.get("/ready", tags=["ops"])
    @app.get("/readyz", tags=["ops"])
    async def ready() -> dict[str, str]:
        """Readiness probe — always public.

        Returns 200 once the application is constructed and able to accept
        traffic. K8s readiness probes conventionally use ``/readyz``; ``/ready``
        is provided as an alias. Kept lightweight (no per-request model load) so
        a load balancer can poll it cheaply.
        """
        return {"status": "ready", "version": _server_version()}

    @app.get("/metrics", tags=["ops"])
    async def metrics(request: _FastAPIRequest) -> Any:  # type: ignore[valid-type]
        """Prometheus metrics (requires auth by default; EFFGEN_PUBLIC_METRICS=1 opens it)."""
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            from starlette.responses import Response

            payload = generate_latest().decode("utf-8", errors="replace")
            try:
                from effgen.observability.metrics import export_metrics

                payload += "\n" + export_metrics()
            except Exception:  # noqa: BLE001 - metrics export is optional; omit it on failure
                pass
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            from effgen.api.openai_compat import error_envelope

            return JSONResponse(
                error_envelope(
                    503,
                    "Metrics are unavailable: prometheus_client is not installed. "
                    "Install it with `pip install 'effgen[server]'`.",
                    code="metrics_unavailable",
                    redact=False,
                ),
                status_code=503,
            )

    @app.get("/slo", tags=["ops"])
    async def slo_status() -> dict[str, Any]:
        """SLO burn-rate status for every registered SLO (public, like /health).

        Lists objectives this process registered via
        ``effgen.observability.slo.get_tracker().register(...)`` and recorded
        events against with ``tracker.record(name, ok=...)``. It does not
        compute latency or availability from request metrics, so it stays empty
        on a server that has served traffic but registered no objective; the
        measured percentiles and availability for that traffic are in the
        ``slo`` block of ``/dashboard/data.json``. An empty list carries a
        ``detail`` note saying so.
        """
        from effgen.observability.slo import EMPTY_SLO_DETAIL, get_tracker

        statuses = get_tracker().all_statuses()
        payload: dict[str, Any] = {"slos": statuses}
        if not statuses:
            payload["detail"] = EMPTY_SLO_DETAIL
        return payload

    @app.get("/whoami", tags=["auth"])
    async def whoami(request: _FastAPIRequest) -> dict[str, Any]:  # type: ignore[valid-type]
        """Return the current principal's identity (requires auth in non-dev mode)."""
        user = getattr(request.state, "user", None)
        if user is None:
            return {"principal": "anonymous"}
        return {
            "sub": user.sub,
            "iss": user.iss,
            "roles": user.roles,
            "email": user.email,
        }

    @app.get("/rbac/policy", tags=["auth"])
    async def rbac_policy(request: _FastAPIRequest) -> dict[str, Any]:  # type: ignore[valid-type]
        """Return the effective RBAC policy for the current principal."""
        from effgen.api.openai_compat import error_envelope
        from effgen.server.rbac import PolicyDenied, resolve_policy

        user = getattr(request.state, "user", None)
        roles: list[str] = getattr(user, "roles", []) if user else []
        try:
            policy = resolve_policy(roles)
        except PolicyDenied as exc:
            # Same envelope the RBAC middleware returns when it denies a model
            # call, so both denials read identically to a client.
            return JSONResponse(
                error_envelope(exc.status_code, str(exc), redact=False),
                status_code=exc.status_code,
            )
        return {
            "roles": [r.name for r in policy.roles],
            "allowed_tools": sorted(policy.allowed_tools) or ["*"],
            "allowed_models": sorted(policy.allowed_models) or ["*"],
            "max_cost_per_day": policy.max_cost_per_day,
        }

    @app.get("/rbac/roles", tags=["auth"])
    async def rbac_roles() -> list[dict[str, Any]]:
        """List all defined RBAC roles."""
        from effgen.server.rbac import list_roles

        return [r.to_dict() for r in list_roles()]

    # Import and mount the OpenAI-compatible + embeddings routers, with RBAC
    # and per-principal cost-cap enforcement layered on top.
    _mount_existing_routers(app, runner=runner, extra_models=extra_models)

    # Mount the local dashboard (static SPA + data.json + SSE spans).
    _mount_dashboard(app)

    # Mount the in-browser playground (static SPA + bootstrap config).
    _mount_playground(app)

    return app
