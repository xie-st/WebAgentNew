"""Server security hardening regression tests.

Covers the fail-closed auth contract and the secure-by-default posture of the
effGen API server:

* JWT auth must fail **closed** when no OIDC issuer/JWKS is configured outside
  dev mode — forged/unsigned/wrong-alg/issuer/aud/expired tokens are rejected.
* A correctly configured JWKS validates genuine tokens and rejects tampered
  ones.
* ``/metrics`` and dashboard *data* endpoints require auth by default.
* Production CORS is never wildcard + credentials.
* RBAC ``viewer`` cannot execute tools; unknown roles are strict by default.
* Budget is reserved then reconciled — failed calls are not charged.
* Oversized request bodies are rejected before buffering.
* The server version is sourced from package metadata, not a hardcoded literal.
"""
from __future__ import annotations

import json
import secrets
import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from effgen.server.app import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _dummy_runner(prompt, *, model, tools=None, stream=False, **kw):  # noqa: ANN001
    return "ok"


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    """Ensure a clean, non-dev, no-OIDC environment for each test."""
    for var in (
        "EFFGEN_DEV_MODE",
        "EFFGEN_OIDC_ISSUER",
        "EFFGEN_OIDC_CLIENT_ID",
        "EFFGEN_OIDC_JWKS_URI",
        "EFFGEN_METRICS_AUTH",
        "EFFGEN_PUBLIC_METRICS",
        "EFFGEN_PUBLIC_DASHBOARD",
        "EFFGEN_CORS_ORIGINS",
        "EFFGEN_RBAC_STRICT_ROLES",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EFFGEN_BUDGET_PERSIST", "0")
    yield


class _RSAKey:
    """An RSA signing key + its JWKS public-key dict."""

    ISSUER = "https://issuer.example.com"
    AUDIENCE = "effgen-api"
    KID = "test-key-1"

    def __init__(self) -> None:
        self._priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        from jwt.algorithms import RSAAlgorithm

        pub_jwk = json.loads(RSAAlgorithm.to_jwk(self._priv.public_key()))
        pub_jwk["kid"] = self.KID
        pub_jwk["alg"] = "RS256"
        self.jwk = pub_jwk

    def sign(self, claims: dict, alg: str = "RS256") -> str:
        return pyjwt.encode(claims, self._priv, algorithm=alg, headers={"kid": self.KID})

    def valid_claims(self, **over) -> dict:
        now = int(time.time())
        claims = {
            "sub": "alice",
            "iss": self.ISSUER,
            "aud": self.AUDIENCE,
            "iat": now,
            "exp": now + 3600,
            "roles": ["admin"],
        }
        claims.update(over)
        return claims


@pytest.fixture
def rsa_key():
    return _RSAKey()


@pytest.fixture
def jwks_app(rsa_key, monkeypatch):
    """An app configured with a (mocked) JWKS issuer + a dummy runner."""
    from effgen.server import auth as auth_mod

    monkeypatch.setattr(auth_mod, "_fetch_jwks", lambda uri: {rsa_key.KID: rsa_key.jwk})
    app = create_app(
        dev_mode=False,
        oidc_issuer=_RSAKey.ISSUER,
        oidc_client_id=_RSAKey.AUDIENCE,
        oidc_jwks_uri="https://issuer.example.com/jwks",
        runner=_dummy_runner,
    )
    return app


def _client(app):
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. Fail-closed when unconfigured
# ---------------------------------------------------------------------------


class TestFailClosedUnconfigured:
    def setup_method(self):
        self.app = create_app(dev_mode=False, runner=_dummy_runner)
        self.client = _client(self.app)

    def test_no_credentials_rejected(self):
        assert self.client.get("/whoami").status_code == 401

    def test_forged_hs256_rejected(self):
        forged = pyjwt.encode(
            {"sub": "attacker", "roles": ["admin"], "exp": 9999999999, "iat": 1},
            secrets.token_hex(16), algorithm="HS256",
        )
        r = self.client.get("/whoami", headers={"Authorization": f"Bearer {forged}"})
        assert r.status_code == 401

    def test_alg_none_rejected(self):
        unsigned = pyjwt.encode(
            {"sub": "attacker", "roles": ["admin"], "exp": 9999999999, "iat": 1},
            key=None, algorithm="none",
        )
        r = self.client.get("/whoami", headers={"Authorization": f"Bearer {unsigned}"})
        assert r.status_code == 401

    def test_forged_token_cannot_reach_v1(self):
        forged = pyjwt.encode(
            {"sub": "attacker", "roles": ["admin"], "exp": 9999999999, "iat": 1},
            secrets.token_hex(16), algorithm="HS256",
        )
        r = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {forged}"},
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 401

    def test_verify_jwt_unconfigured_raises(self):
        from effgen.server.auth import AuthError, verify_jwt

        forged = pyjwt.encode({"sub": "x"}, "k", algorithm="HS256")
        with pytest.raises(AuthError):
            verify_jwt(forged)


# ---------------------------------------------------------------------------
# 2. Configured JWKS validates real tokens, rejects tampered
# ---------------------------------------------------------------------------


class TestConfiguredJWKS:
    def test_valid_token_accepted(self, jwks_app, rsa_key):
        token = rsa_key.sign(rsa_key.valid_claims())
        r = _client(jwks_app).get("/whoami", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["sub"] == "alice"

    def test_wrong_issuer_rejected(self, jwks_app, rsa_key):
        token = rsa_key.sign(rsa_key.valid_claims(iss="https://evil.example.com"))
        r = _client(jwks_app).get("/whoami", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401

    def test_wrong_audience_rejected(self, jwks_app, rsa_key):
        token = rsa_key.sign(rsa_key.valid_claims(aud="some-other-api"))
        r = _client(jwks_app).get("/whoami", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401

    def test_expired_rejected(self, jwks_app, rsa_key):
        now = int(time.time())
        token = rsa_key.sign(rsa_key.valid_claims(iat=now - 7200, exp=now - 3600))
        r = _client(jwks_app).get("/whoami", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401

    def test_hs256_forgery_against_rsa_jwks_rejected(self, jwks_app, rsa_key):
        """A forged HS256 token must not validate against an RS256 JWKS."""
        forged = pyjwt.encode(rsa_key.valid_claims(), "attacker", algorithm="HS256")
        r = _client(jwks_app).get("/whoami", headers={"Authorization": f"Bearer {forged}"})
        assert r.status_code == 401

    def test_tampered_signature_rejected(self, jwks_app, rsa_key):
        token = rsa_key.sign(rsa_key.valid_claims())
        tampered = token[:-3] + ("AAA" if token[-3:] != "AAA" else "BBB")
        r = _client(jwks_app).get("/whoami", headers={"Authorization": f"Bearer {tampered}"})
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# 3. Metrics / dashboard auth defaults
# ---------------------------------------------------------------------------


class TestObservabilityAuth:
    def test_metrics_protected_by_default(self):
        c = _client(create_app(dev_mode=False, runner=_dummy_runner))
        assert c.get("/metrics").status_code == 401

    def test_dashboard_data_protected_by_default(self):
        c = _client(create_app(dev_mode=False, runner=_dummy_runner))
        assert c.get("/dashboard/data.json").status_code == 401
        assert c.get("/dashboard/spans").status_code == 401

    def test_dashboard_static_shell_public(self):
        c = _client(create_app(dev_mode=False, runner=_dummy_runner))
        # The SPA shell must load (so it can prompt for a token); never 401.
        assert c.get("/dashboard").status_code != 401

    def test_health_always_public(self):
        c = _client(create_app(dev_mode=False, runner=_dummy_runner))
        assert c.get("/health").status_code == 200

    def test_public_metrics_opt_in(self):
        c = _client(create_app(dev_mode=False, public_metrics=True, runner=_dummy_runner))
        assert c.get("/metrics").status_code == 200

    def test_public_dashboard_opt_in(self):
        c = _client(create_app(dev_mode=False, public_dashboard=True, runner=_dummy_runner))
        assert c.get("/dashboard/data.json").status_code == 200

    def test_dev_mode_makes_observability_public(self):
        c = _client(create_app(dev_mode=True, runner=_dummy_runner))
        assert c.get("/metrics").status_code == 200
        assert c.get("/dashboard/data.json").status_code == 200


# ---------------------------------------------------------------------------
# 4. CORS hardening
# ---------------------------------------------------------------------------


class TestCORS:
    def test_no_cross_origin_in_production_by_default(self):
        c = _client(create_app(dev_mode=False, runner=_dummy_runner))
        r = c.get("/health", headers={"Origin": "https://evil.example.com"})
        # No CORS middleware installed → no allow-origin header echoed back.
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    def test_explicit_origin_allowed_with_credentials(self):
        c = _client(create_app(
            dev_mode=False, cors_origins=["https://app.example.com"], runner=_dummy_runner,
        ))
        r = c.get("/health", headers={"Origin": "https://app.example.com"})
        assert r.headers.get("access-control-allow-origin") == "https://app.example.com"
        assert r.headers.get("access-control-allow-credentials") == "true"

    def test_wildcard_never_combined_with_credentials(self):
        from effgen.api import middleware as mw

        captured = {}

        class _FakeApp:
            def add_middleware(self, cls, **kw):  # noqa: ANN001
                if cls.__name__ == "CORSMiddleware":
                    captured["called"] = True
                    captured.update(kw)

            def middleware(self, *_a, **_k):  # noqa: ANN002, ANN003
                return lambda fn: fn

            def on_event(self, *_a, **_k):  # noqa: ANN002, ANN003
                return lambda fn: fn

        mw.install_production_middleware(
            _FakeApp(), cors_origins=["*"], allow_credentials=True,
        )
        assert captured.get("allow_credentials") is False


# ---------------------------------------------------------------------------
# 5. RBAC viewer + unknown roles
# ---------------------------------------------------------------------------


class TestRBAC:
    def test_viewer_cannot_run_tools(self):
        from effgen.server.rbac import resolve_policy

        policy = resolve_policy(["viewer"])
        assert policy.allows_tool("web_search") is False

    def test_limited_user_can_run_tools(self):
        from effgen.server.rbac import resolve_policy

        policy = resolve_policy(["limited_user"])
        assert policy.allows_tool("web_search") is True
        assert policy.max_cost_per_day == 5.0

    def test_unknown_role_strict_by_default(self):
        from effgen.server.rbac import PolicyDenied, resolve_policy

        with pytest.raises(PolicyDenied):
            resolve_policy(["totally-unknown"])

    def test_unknown_role_lenient_opt_out(self):
        from effgen.server.rbac import resolve_policy

        policy = resolve_policy(["totally-unknown"], strict=False)
        assert policy.allows_tool("web_search") is False

    def test_viewer_tool_request_denied_via_server(self, jwks_app, rsa_key):
        token = rsa_key.sign(rsa_key.valid_claims(roles=["viewer"]))
        r = _client(jwks_app).post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "web_search"}}],
            },
        )
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# 6. Budget reserve / reconcile
# ---------------------------------------------------------------------------


class TestBudget:
    def setup_method(self):
        from effgen.server import budget

        budget.reset()

    def test_reserve_then_reconcile_charges_estimate(self):
        from effgen.server import budget

        token = budget.reserve("p1", 0.01, cap=1.0)
        assert budget.get_spend("p1") == 0.0  # not charged yet
        budget.reconcile("p1", token)
        assert budget.get_spend("p1") == pytest.approx(0.01)

    def test_release_does_not_charge(self):
        from effgen.server import budget

        token = budget.reserve("p2", 0.01, cap=1.0)
        budget.release("p2", token)
        assert budget.get_spend("p2") == 0.0

    def test_reservation_counts_against_cap(self):
        from effgen.server import budget

        budget.reserve("p3", 5.0, cap=5.0)  # reserves the whole cap
        with pytest.raises(budget.BudgetExceeded):
            budget.reserve("p3", 1.0, cap=5.0)

    def test_failed_call_not_charged_via_server(self, jwks_app, rsa_key, monkeypatch):
        """A 500 from the route releases the reservation (no charge)."""
        from effgen.server import budget

        budget.reset()

        def _boom(prompt, *, model, tools=None, stream=False, **kw):  # noqa: ANN001
            raise RuntimeError("provider exploded")

        app = create_app(
            dev_mode=False,
            oidc_issuer=_RSAKey.ISSUER,
            oidc_client_id=_RSAKey.AUDIENCE,
            oidc_jwks_uri="https://issuer.example.com/jwks",
            runner=_boom,
        )
        from effgen.server import auth as auth_mod

        monkeypatch.setattr(auth_mod, "_fetch_jwks", lambda uri: {rsa_key.KID: rsa_key.jwk})

        token = rsa_key.sign(rsa_key.valid_claims(sub="charlie", roles=["researcher"]))
        r = _client(app).post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code >= 500
        assert budget.get_spend("charlie") == 0.0

    def test_successful_call_charged_via_server(self, jwks_app, rsa_key):
        from effgen.server import budget

        budget.reset()
        token = rsa_key.sign(rsa_key.valid_claims(sub="dave", roles=["researcher"]))
        r = _client(jwks_app).post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert budget.get_spend("dave") == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 7. Body-size limit
# ---------------------------------------------------------------------------


class TestBodySize:
    def test_oversized_body_rejected(self, jwks_app, rsa_key, monkeypatch):
        monkeypatch.setenv("EFFGEN_MAX_BODY_BYTES", "1024")
        # Rebuild app so middleware re-reads the env limit.
        app = create_app(
            dev_mode=False,
            oidc_issuer=_RSAKey.ISSUER,
            oidc_client_id=_RSAKey.AUDIENCE,
            oidc_jwks_uri="https://issuer.example.com/jwks",
            runner=_dummy_runner,
        )
        from effgen.server import auth as auth_mod

        monkeypatch.setattr(auth_mod, "_fetch_jwks", lambda uri: {rsa_key.KID: rsa_key.jwk})
        token = rsa_key.sign(rsa_key.valid_claims(roles=["researcher"]))
        big = "x" * 5000
        r = _client(app).post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "gpt-4", "messages": [{"role": "user", "content": big}]},
        )
        assert r.status_code == 413

    def test_oversized_embeddings_body_rejected(self, monkeypatch):
        # /v1/embeddings carries no RBAC/budget enforcement, but the body-size
        # cap must still apply — a client should not be able to make the
        # server buffer/process an unbounded body just because that route
        # isn't in RBACBudgetMiddleware's enforced-paths list.
        monkeypatch.setenv("EFFGEN_MAX_BODY_BYTES", "1024")
        app = create_app(api_key="test-key", dev_mode=False, runner=_dummy_runner)
        r = _client(app).post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "text-embedding-3-small", "input": "x" * 5000},
        )
        assert r.status_code == 413

    def test_normal_embeddings_body_still_served(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_MAX_BODY_BYTES", str(10 * 1024 * 1024))
        app = create_app(api_key="test-key", dev_mode=False, runner=_dummy_runner)
        r = _client(app).post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "text-embedding-3-small", "input": "hello world"},
        )
        assert r.status_code == 200
        assert "data" in r.json()


# ---------------------------------------------------------------------------
# 7b. Rate-limit client-IP resolution
# ---------------------------------------------------------------------------


class TestRateLimitProxyTrust:
    """X-Forwarded-For must only steer the per-IP rate-limit key when the
    deployment explicitly opts in — otherwise any direct caller can rotate
    the header to defeat its own limit."""

    def _app(self, *, trust_proxy=None):
        return create_app(
            api_key="test-key",
            dev_mode=False,
            rate_limit_per_minute=3,
            trust_proxy=trust_proxy,
            runner=_dummy_runner,
        )

    def test_spoofed_xff_does_not_bypass_limit_by_default(self):
        app = self._app()
        client = _client(app)
        statuses = [
            client.get(
                "/v1/models",
                headers={
                    "Authorization": "Bearer test-key",
                    "X-Forwarded-For": f"10.0.0.{i}",
                },
            ).status_code
            for i in range(6)
        ]
        assert 429 in statuses, f"expected a 429 among {statuses}"

    def test_spoofed_xff_bypasses_limit_when_trust_proxy_enabled(self):
        app = self._app(trust_proxy=True)
        client = _client(app)
        statuses = [
            client.get(
                "/v1/models",
                headers={
                    "Authorization": "Bearer test-key",
                    "X-Forwarded-For": f"10.0.0.{i}",
                },
            ).status_code
            for i in range(6)
        ]
        assert 429 not in statuses, f"unexpected 429 among {statuses}"

    def test_env_var_opts_in_when_param_not_given(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_TRUST_PROXY", "1")
        app = self._app(trust_proxy=None)
        client = _client(app)
        statuses = [
            client.get(
                "/v1/models",
                headers={
                    "Authorization": "Bearer test-key",
                    "X-Forwarded-For": f"10.0.0.{i}",
                },
            ).status_code
            for i in range(6)
        ]
        assert 429 not in statuses, f"unexpected 429 among {statuses}"


class TestServeUvicornProxyTrust:
    """``effgen serve`` must launch uvicorn with proxy-header trust that
    matches the rate limiter's own decision. uvicorn rewrites the socket peer
    (``scope["client"]``) from ``X-Forwarded-For`` for any address in
    ``forwarded_allow_ips`` (default ``127.0.0.1``); left at that default, a
    loopback/same-host caller could set the rate-limit client IP even though
    ``trust_proxy`` defaults to off. The command must disable that rewriting
    unless the deployment opts into trusting a proxy."""

    def _run(self, monkeypatch, *, trust_proxy=None, env=None):
        from types import SimpleNamespace

        import effgen.cli._main as _main

        for var in ("EFFGEN_API_KEY", "EFFGEN_DEV_MODE", "EFFGEN_TRUST_PROXY"):
            monkeypatch.delenv(var, raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)

        captured = {}
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: captured.update(k))

        args = SimpleNamespace(
            host="127.0.0.1", port=8000, rate_limit=None,
            verbose=False, trust_proxy=trust_proxy,
        )
        rc = _main.CLIInterface().serve_api(args)
        assert rc == 0
        return captured

    def test_default_disables_uvicorn_xff_rewrite(self, monkeypatch):
        pytest.importorskip("fastapi")
        captured = self._run(monkeypatch)
        assert captured.get("forwarded_allow_ips") == []

    def test_trust_proxy_flag_leaves_uvicorn_default(self, monkeypatch):
        pytest.importorskip("fastapi")
        captured = self._run(monkeypatch, trust_proxy=True)
        assert "forwarded_allow_ips" not in captured

    def test_env_opts_in_leaves_uvicorn_default(self, monkeypatch):
        pytest.importorskip("fastapi")
        captured = self._run(monkeypatch, env={"EFFGEN_TRUST_PROXY": "1"})
        assert "forwarded_allow_ips" not in captured


# ---------------------------------------------------------------------------
# 8. Version sourced from metadata
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_not_hardcoded_literal(self):
        import effgen
        from effgen.server.app import _server_version

        assert _server_version() == effgen.__version__

    def test_app_and_health_report_package_version(self):
        import effgen

        app = create_app(dev_mode=True, runner=_dummy_runner)
        assert app.version == effgen.__version__
        c = _client(app)
        assert c.get("/health").json()["version"] == effgen.__version__
