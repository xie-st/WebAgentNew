"""OIDC/JWT authentication for the effGen API server.

Bearer JWT validation on every non-public endpoint.
Configurable via environment variables:

  EFFGEN_OIDC_ISSUER       — OIDC issuer URL (e.g. https://accounts.google.com)
  EFFGEN_OIDC_CLIENT_ID    — Audience claim that JWTs must carry
  EFFGEN_OIDC_JWKS_URI     — JWKS endpoint (auto-discovered if omitted)
  EFFGEN_DEV_MODE          — set to "1" to disable auth (loud warning)

Public endpoints (no JWT required):
  /health, /healthz, /livez, /ready, /readyz  — liveness and readiness probes
  /slo                                        — aggregate burn-rate status
  /openapi.json, /docs, /redoc                — API schema, no data
  /dashboard, /playground and their static assets — the page shells

Everything else requires credentials, /metrics included; set
EFFGEN_PUBLIC_METRICS=1 to serve metrics without auth (EFFGEN_METRICS_AUTH=1
forces auth back on). A rejected request is answered with the server's shared
``{"error": {message, type, param, code}}`` envelope.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

from effgen.errors import quote_for_message, with_next_step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEV_MODE_WARNED = False  # module-level flag so we warn only once per process
_AUTH_UNCONFIGURED_WARNED = False
_UNVERIFIED_DECODE_WARNED = False


def _is_dev_mode() -> bool:
    return os.getenv("EFFGEN_DEV_MODE", "0").strip() == "1"


def _warn_auth_unconfigured() -> None:
    """Warn once when a bearer token is rejected for missing OIDC config."""
    global _AUTH_UNCONFIGURED_WARNED
    if not _AUTH_UNCONFIGURED_WARNED:
        logger.error(
            "Bearer token rejected: no OIDC issuer/JWKS configured and not in "
            "dev mode. Set EFFGEN_OIDC_ISSUER/EFFGEN_OIDC_JWKS_URI to enable "
            "JWT auth, or EFFGEN_DEV_MODE=1 for local development."
        )
        _AUTH_UNCONFIGURED_WARNED = True


def _warn_unverified_decode() -> None:
    """Warn once when a token is decoded without signature verification."""
    global _UNVERIFIED_DECODE_WARNED
    if not _UNVERIFIED_DECODE_WARNED:
        logger.warning(
            "JWT decoded WITHOUT signature verification (dev mode / explicit "
            "opt-in). Never do this in production."
        )
        _UNVERIFIED_DECODE_WARNED = True


def _warn_dev_mode() -> None:
    global _DEV_MODE_WARNED
    if not _DEV_MODE_WARNED:
        warnings.warn(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  ⚠  EFFGEN_DEV_MODE=1: AUTHENTICATION IS DISABLED  ⚠       ║\n"
            "║  Never run with EFFGEN_DEV_MODE=1 in production!            ║\n"
            "╚══════════════════════════════════════════════════════════════╝",
            stacklevel=3,
        )
        logger.critical(
            "EFFGEN_DEV_MODE=1 — JWT authentication is DISABLED. "
            "Do NOT use this in production."
        )
        _DEV_MODE_WARNED = True


# ---------------------------------------------------------------------------
# JWKS cache
# ---------------------------------------------------------------------------


@dataclass
class _JWKSCache:
    """Simple in-memory JWKS cache with TTL."""

    keys: dict[str, Any] = field(default_factory=dict)
    fetched_at: float = 0.0
    ttl: float = 3600.0  # 1 hour


_jwks_cache = _JWKSCache()


def _fetch_jwks(jwks_uri: str) -> dict[str, Any]:
    """Fetch JWKS from the IdP.  Returns a dict keyed by ``kid``."""
    now = time.time()
    if _jwks_cache.keys and (now - _jwks_cache.fetched_at) < _jwks_cache.ttl:
        return _jwks_cache.keys

    try:
        import httpx

        resp = httpx.get(jwks_uri, timeout=10.0)
        resp.raise_for_status()
        raw: dict[str, Any] = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise AuthError(f"Failed to fetch JWKS from {jwks_uri}: {exc}") from exc

    keys: dict[str, Any] = {}
    for jwk in raw.get("keys", []):
        kid = jwk.get("kid", "__default__")
        keys[kid] = jwk

    _jwks_cache.keys = keys
    _jwks_cache.fetched_at = now
    return keys


def _discover_jwks_uri(issuer: str) -> str:
    """OIDC discovery: GET {issuer}/.well-known/openid-configuration."""
    discovery_url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        import httpx

        resp = httpx.get(discovery_url, timeout=10.0)
        resp.raise_for_status()
        cfg: dict[str, Any] = resp.json()
        return cfg["jwks_uri"]
    except Exception as exc:  # noqa: BLE001
        raise AuthError(
            f"OIDC discovery failed for issuer '{issuer}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Raised when a JWT cannot be validated.

    The message names the check that failed and then what the caller can do
    about it, because a bare rejection leaves an operator guessing which of
    the token, the clock and the server configuration is wrong.

    Attributes:
        status_code: The HTTP status this maps to (401 by default).
    """

    _FOLLOW_ON = (
        "Send a current token in the Authorization header — mint one with the "
        "key `effgen serve` printed at startup (EFFGEN_API_KEY), and check the "
        "issuer, audience and clock skew match the server configuration."
    )

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(with_next_step(quote_for_message(message), self._FOLLOW_ON))
        self.status_code = status_code


@dataclass
class TokenPayload:
    """Decoded and verified JWT claims."""

    sub: str
    iss: str
    aud: str | list[str]
    exp: int
    iat: int
    roles: list[str] = field(default_factory=list)
    email: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_claims(cls, claims: dict[str, Any]) -> "TokenPayload":
        """Build a payload from decoded JWT claims (``scope`` accepted for roles)."""
        aud = claims.get("aud", "")
        roles_raw = claims.get("roles", claims.get("scope", ""))
        if isinstance(roles_raw, str):
            roles = [r for r in roles_raw.split() if r]
        else:
            roles = list(roles_raw)

        known = {"sub", "iss", "aud", "exp", "iat", "roles", "scope", "email"}
        extra = {k: v for k, v in claims.items() if k not in known}

        return cls(
            sub=claims.get("sub", ""),
            iss=claims.get("iss", ""),
            aud=aud,
            exp=int(claims.get("exp", 0)),
            iat=int(claims.get("iat", 0)),
            roles=roles,
            email=claims.get("email", ""),
            extra=extra,
        )


def verify_jwt(
    token: str,
    *,
    issuer: str | None = None,
    client_id: str | None = None,
    jwks_uri: str | None = None,
    allow_unverified: bool = False,
) -> TokenPayload:
    """Validate a Bearer JWT and return its decoded payload.

    Parameters
    ----------
    token:
        Raw JWT string (without ``Bearer `` prefix).
    issuer:
        Expected ``iss`` claim value.  Falls back to ``EFFGEN_OIDC_ISSUER``.
    client_id:
        Expected ``aud`` claim value.  Falls back to ``EFFGEN_OIDC_CLIENT_ID``.
    jwks_uri:
        JWKS endpoint.  Falls back to ``EFFGEN_OIDC_JWKS_URI`` or OIDC
        discovery via ``issuer``.
    allow_unverified:
        When ``True`` *and* no issuer/JWKS is configured, decode the token
        **without** signature verification. This is intended only for explicit
        local development (``EFFGEN_DEV_MODE=1``); it is **never** enabled on a
        normal request path. With the default ``False`` an unconfigured server
        rejects every bearer token instead of trusting it (fail closed).
    """
    issuer = issuer or os.getenv("EFFGEN_OIDC_ISSUER", "")
    client_id = client_id or os.getenv("EFFGEN_OIDC_CLIENT_ID", "")
    jwks_uri = jwks_uri or os.getenv("EFFGEN_OIDC_JWKS_URI", "")

    try:
        import jwt as pyjwt
        from jwt.algorithms import RSAAlgorithm
    except ImportError as exc:
        raise AuthError("PyJWT is required for JWT validation") from exc

    # Decode header to find key id
    try:
        header = pyjwt.get_unverified_header(token)
    except Exception as exc:
        raise AuthError(f"Malformed JWT header: {exc}") from exc

    if jwks_uri:
        # Fetch public key from JWKS
        kid = header.get("kid", "__default__")
        jwks = _fetch_jwks(jwks_uri)
        jwk = jwks.get(kid) or next(iter(jwks.values()), None)
        if jwk is None:
            raise AuthError(f"No JWK found for kid={kid!r}")

        try:
            public_key = RSAAlgorithm.from_jwk(json.dumps(jwk))
        except Exception as exc:
            raise AuthError(f"Cannot load public key from JWK: {exc}") from exc

        options: dict[str, Any] = {"verify_aud": bool(client_id)}
        decode_kwargs: dict[str, Any] = {
            "key": public_key,
            "algorithms": ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
            "options": options,
        }
        if client_id:
            decode_kwargs["audience"] = client_id
        if issuer:
            decode_kwargs["issuer"] = issuer

        try:
            claims: dict[str, Any] = pyjwt.decode(token, **decode_kwargs)
        except pyjwt.ExpiredSignatureError as exc:
            raise AuthError("JWT has expired") from exc
        except pyjwt.InvalidAudienceError as exc:
            raise AuthError(f"JWT audience mismatch (expected {client_id!r})") from exc
        except pyjwt.InvalidIssuerError as exc:
            raise AuthError(f"JWT issuer mismatch (expected {issuer!r})") from exc
        except pyjwt.PyJWTError as exc:
            raise AuthError(f"JWT validation failed: {exc}") from exc

    elif issuer:
        # Discover JWKS from OIDC configuration
        discovered_jwks_uri = _discover_jwks_uri(issuer)
        return verify_jwt(
            token,
            issuer=issuer,
            client_id=client_id,
            jwks_uri=discovered_jwks_uri,
        )
    elif allow_unverified:
        # No JWKS URI and no issuer, and the caller explicitly opted in to
        # signature-free decoding (dev mode only). This trusts the token as-is
        # and must NEVER be reachable on a production request path.
        _warn_unverified_decode()
        try:
            claims = pyjwt.decode(
                token,
                options={"verify_signature": False, "verify_aud": False},
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "HS256"],
            )
        except pyjwt.PyJWTError as exc:
            raise AuthError(f"JWT decode failed: {exc}") from exc
    else:
        # FAIL CLOSED: no issuer and no JWKS configured, and signature-free
        # decoding was not explicitly permitted. Refuse the token rather than
        # trusting an attacker-supplied signature. A production server that
        # forgets its OIDC env vars therefore rejects all bearer tokens instead
        # of silently accepting forged ones.
        _warn_auth_unconfigured()
        raise AuthError(
            "Server authentication is not configured: no OIDC issuer or JWKS "
            "URI is set, so bearer tokens cannot be verified. Configure "
            "EFFGEN_OIDC_ISSUER (and/or EFFGEN_OIDC_JWKS_URI), or set "
            "EFFGEN_DEV_MODE=1 for local development."
        )

    return TokenPayload.from_claims(claims)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

# Health/liveness/readiness probes — always public (no sensitive data, needed
# by load balancers and K8s probes before any token is available).
_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/livez",
    # Aggregate SLO burn-rate status (no request bodies, costs, or user data) —
    # same sensitivity class as the probes above, public like an uptime badge.
    "/slo",
    # API schema / interactive docs carry no data (just the API shape) and need
    # to load before a token is available, like the health probes.
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
})

# Dashboard *data* endpoints expose operational metrics/traces. They are
# protected by default and only become public in dev mode or when an operator
# opts in (``public_dashboard``). The static SPA shell (HTML/JS/CSS) carries no
# data and stays public so the page can load and prompt for a token.
_DASHBOARD_DATA_PATHS: frozenset[str] = frozenset({
    "/dashboard/data.json",
    "/dashboard/spans",
    "/dashboard/catalog.json",
    "/dashboard/history.json",
    "/dashboard/topology.json",
})


def _is_public_path(path: str, public_paths: frozenset[str]) -> bool:
    """Return True when *path* names one of the always-public endpoints.

    A trailing slash is ignored, so ``/health/`` is the probe endpoint and not
    an unknown protected path — the router redirects the two spellings to the
    same route, and a probe configured either way behaves the same.
    """
    if path in public_paths:
        return True
    return len(path) > 1 and path.endswith("/") and path.rstrip("/") in public_paths


# File extensions treated as inert static dashboard assets.
_STATIC_ASSET_SUFFIXES: tuple[str, ...] = (
    ".js", ".css", ".html", ".png", ".svg", ".ico", ".map", ".woff", ".woff2",
)


def _is_dashboard_static(path: str) -> bool:
    """Return True for inert static dashboard assets (never data endpoints)."""
    if path in ("/dashboard", "/dashboard/"):
        return True
    if path in _DASHBOARD_DATA_PATHS:
        return False
    return path.startswith("/dashboard/") and path.endswith(_STATIC_ASSET_SUFFIXES)


# The playground can drive real (billed) model calls, so its data endpoint
# (``/playground/bootstrap`` — presets, tools, and any local-view session key)
# is protected by default and opens only when an operator explicitly opts in via
# ``EFFGEN_PUBLIC_PLAYGROUND`` (a flag separate from the dashboard's, since it
# authorizes spend). The static SPA shell stays public so the page can load and
# prompt for a key.
_PLAYGROUND_DATA_PATHS: frozenset[str] = frozenset({"/playground/bootstrap"})


def _is_playground_static(path: str) -> bool:
    """Return True for inert static playground assets (never data endpoints)."""
    if path in ("/playground", "/playground/"):
        return True
    if path in _PLAYGROUND_DATA_PATHS:
        return False
    return path.startswith("/playground/") and path.endswith(_STATIC_ASSET_SUFFIXES)


def get_auth_dependency() -> Any:
    """Return a FastAPI Depends-compatible callable for JWT auth.

    Usage::

        from fastapi import Depends
        from effgen.server.auth import get_auth_dependency

        AuthDep = Annotated[TokenPayload, Depends(get_auth_dependency())]

        @router.get("/secure")
        async def secure(user: AuthDep):
            return {"sub": user.sub}
    """
    try:
        from fastapi import Request
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fastapi is required for get_auth_dependency()") from exc

    _bearer = HTTPBearer(auto_error=False)

    async def _auth_dep(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = None,
    ) -> TokenPayload | None:
        # Public paths are exempt
        if _is_public_path(request.url.path, _PUBLIC_PATHS):
            return None

        # Dev-mode bypass
        if _is_dev_mode():
            _warn_dev_mode()
            return TokenPayload(
                sub="dev-user",
                iss="dev",
                aud="dev",
                exp=int(time.time()) + 86400,
                iat=int(time.time()),
                roles=["admin"],
                email="dev@localhost",
            )

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        raw_token = auth_header.removeprefix("Bearer ").strip()
        try:
            payload = verify_jwt(raw_token)
        except AuthError as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        return payload

    return _auth_dep


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------


class AuthMiddleware:
    """ASGI middleware that validates Bearer JWTs on every non-public request.

    Adds ``request.state.user`` (a :class:`TokenPayload`) for downstream
    handlers to consume.  Rejected requests receive a JSON 401.
    """

    def __init__(
        self,
        app: Any,
        *,
        issuer: str | None = None,
        client_id: str | None = None,
        jwks_uri: str | None = None,
        api_key: str | None = None,
        api_key_roles: list[str] | None = None,
        public_paths: frozenset[str] | None = None,
        metrics_auth: bool = False,
        public_metrics: bool | None = None,
        public_dashboard: bool | None = None,
        public_playground: bool | None = None,
        dev_mode: bool | None = None,
    ) -> None:
        self.app = app
        self.issuer = issuer or os.getenv("EFFGEN_OIDC_ISSUER", "")
        self.client_id = client_id or os.getenv("EFFGEN_OIDC_CLIENT_ID", "")
        self.jwks_uri = jwks_uri or os.getenv("EFFGEN_OIDC_JWKS_URI", "")
        # Static API-key auth: a shared-secret middle ground between full OIDC
        # and dev mode. When configured, a request authenticates by presenting
        # the key as ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.
        self.api_key = api_key if api_key is not None else os.getenv("EFFGEN_API_KEY", "")
        if api_key_roles is not None:
            self.api_key_roles = list(api_key_roles)
        else:
            _roles_env = os.getenv("EFFGEN_API_KEY_ROLES", "admin").strip()
            self.api_key_roles = [r.strip() for r in _roles_env.split(",") if r.strip()] or ["admin"]
        # dev_mode: explicit bool overrides the env var; None means "read env at call time"
        self._dev_mode_override: bool | None = dev_mode

        # /metrics is protected by default. It is public only when explicitly
        # opted in (constructor flag / EFFGEN_PUBLIC_METRICS) and never when the
        # legacy EFFGEN_METRICS_AUTH=1 forces auth on.
        _force_metrics_auth = metrics_auth or os.getenv("EFFGEN_METRICS_AUTH", "0") == "1"
        if public_metrics is None:
            public_metrics = (not _force_metrics_auth) and (
                self._effective_dev_mode()
                or os.getenv("EFFGEN_PUBLIC_METRICS", "0").strip() == "1"
            )
        self.public_metrics: bool = bool(public_metrics) and not _force_metrics_auth

        if public_dashboard is None:
            public_dashboard = self._effective_dev_mode() or (
                os.getenv("EFFGEN_PUBLIC_DASHBOARD", "0").strip() == "1"
            )
        self.public_dashboard: bool = bool(public_dashboard)

        if public_playground is None:
            public_playground = self._effective_dev_mode() or (
                os.getenv("EFFGEN_PUBLIC_PLAYGROUND", "0").strip() == "1"
            )
        self.public_playground: bool = bool(public_playground)

        _extra = set(public_paths or set())
        _default_public = set(_PUBLIC_PATHS)
        if self.public_metrics:
            _default_public.add("/metrics")
        self.public_paths: frozenset[str] = frozenset(_default_public | _extra)

        if self._effective_dev_mode():
            _warn_dev_mode()

    def _effective_dev_mode(self) -> bool:
        """Return the effective dev-mode flag.

        If ``dev_mode`` was passed explicitly to the constructor that value is
        used; otherwise the env-var is re-read on every call so that tests that
        monkeypatch ``EFFGEN_DEV_MODE`` still work as expected.
        """
        if self._dev_mode_override is not None:
            return self._dev_mode_override
        return _is_dev_mode()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "/")

        # Dashboard exemption: inert static assets (the SPA shell) are public so
        # the page can load, but the data endpoints (/dashboard/data.json,
        # /dashboard/spans) require auth unless the operator opted into a public
        # dashboard. This closes the broad ``startswith("/dashboard/")`` hole
        # that would otherwise make any future dashboard API public.
        _is_dashboard_public = self.public_dashboard and (
            path == "/dashboard" or path.startswith("/dashboard/")
        )
        _is_dashboard = _is_dashboard_public or _is_dashboard_static(path)
        # Playground exemption: the static shell is always public; the bootstrap
        # data endpoint opens only when the operator opted into a public
        # playground (it can carry a local-view session key that authorizes
        # spend).
        _is_playground_public = self.public_playground and (
            path == "/playground" or path.startswith("/playground/")
        )
        _is_playground = _is_playground_public or _is_playground_static(path)
        _dev = self._effective_dev_mode()
        if (
            _is_public_path(path, self.public_paths)
            or _is_dashboard
            or _is_playground
            or _dev
        ):
            if _dev:
                scope.setdefault("state", {})["user"] = TokenPayload(
                    sub="dev-user",
                    iss="dev",
                    aud="dev",
                    exp=int(time.time()) + 86400,
                    iat=int(time.time()),
                    roles=["admin"],
                    email="dev@localhost",
                )
            await self.app(scope, receive, send)
            return

        # Extract Authorization header
        headers = dict(scope.get("headers", []))
        auth_bytes: bytes = headers.get(b"authorization", b"")
        auth_str = auth_bytes.decode("latin-1", errors="replace")

        # Static API-key path (when EFFGEN_API_KEY / api_key is configured).
        # Accept the key as a bearer token or via the X-API-Key header. A
        # configured key is REQUIRED on protected routes — there is no fallthrough
        # to unauthenticated access (fail closed).
        if self.api_key:
            x_api_key = headers.get(b"x-api-key", b"").decode("latin-1", errors="replace")
            presented = (
                auth_str.removeprefix("Bearer ").strip()
                if auth_str.startswith("Bearer ")
                else x_api_key.strip()
            )
            if not presented:
                await self._reject(
                    scope, send, 401,
                    "Missing API key (send 'Authorization: Bearer <key>' or "
                    "'X-API-Key: <key>')",
                )
                return
            if not hmac.compare_digest(presented, self.api_key):
                await self._reject(
                    scope, send, 401,
                    "Invalid API key. Check the key the client sends against "
                    "the one this server was started with (EFFGEN_API_KEY).",
                )
                return
            scope.setdefault("state", {})["user"] = TokenPayload(
                sub="api-key",
                iss="static-api-key",
                aud="effgen",
                exp=int(time.time()) + 86400,
                iat=int(time.time()),
                roles=list(self.api_key_roles),
                email="",
            )
            await self.app(scope, receive, send)
            return

        if not auth_str.startswith("Bearer "):
            await self._reject(
                scope, send, 401,
                "Missing or invalid Authorization header. Send "
                "'Authorization: Bearer <token>' with a token this server "
                "accepts.",
            )
            return

        raw_token = auth_str.removeprefix("Bearer ").strip()
        try:
            payload = verify_jwt(
                raw_token,
                issuer=self.issuer or None,
                client_id=self.client_id or None,
                jwks_uri=self.jwks_uri or None,
            )
        except AuthError as exc:
            await self._reject(scope, send, exc.status_code, str(exc))
            return

        scope.setdefault("state", {})["user"] = payload
        await self.app(scope, receive, send)

    async def _reject(self, scope: Any, send: Any, status: int, detail: str) -> None:
        # WebSocket handshakes cannot be denied with an HTTP response message
        # (uvicorn raises "Expected 'websocket.accept'/'websocket.close'…"); the
        # ASGI way to refuse one before accepting is ``websocket.close``, which
        # the server surfaces to the client as an HTTP 403.
        if scope.get("type") == "websocket":
            await send({"type": "websocket.close", "code": 1008})  # policy violation
            return
        try:
            from effgen.api.openai_compat import error_envelope

            # The auth messages are static, server-authored, secret-free hints
            # (they literally spell out the header syntax) — don't run them
            # through the key scrubber, which would corrupt the guidance.
            envelope = error_envelope(status, detail, redact=False)
        except Exception:  # noqa: BLE001 - keep auth working even if the helper is unavailable
            envelope = {"error": {"message": detail, "type": "invalid_request_error",
                                  "param": None, "code": None}}
        body = json.dumps(envelope).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
