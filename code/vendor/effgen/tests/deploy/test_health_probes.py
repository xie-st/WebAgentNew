"""Server liveness/readiness probe aliases + opt-in request rate limit.

These exercise the converged secure app (``create_app``) directly with
Starlette's TestClient — no live provider needed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


def _stub_runner(prompt, *, model, tools=None, stream=False, **_):
    return f"stub::{model}"


def _client(**kw):
    from fastapi.testclient import TestClient

    from effgen.server.app import create_app

    app = create_app(runner=_stub_runner, **kw)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Probe aliases: /healthz,/livez,/readyz,/ready were allowlisted
# but had no route (404). They must all respond 200 and stay public (no token).
# ---------------------------------------------------------------------------

LIVENESS_PATHS = ["/health", "/healthz", "/livez"]
READINESS_PATHS = ["/readyz", "/ready"]
ALL_PROBES = LIVENESS_PATHS + READINESS_PATHS


@pytest.mark.parametrize("path", ALL_PROBES)
def test_probe_public_in_secure_mode(path):
    """Every probe responds 200 without a token, even with auth configured."""
    c = _client(dev_mode=False, api_key="k-secret")
    r = c.get(path)
    assert r.status_code == 200, (path, r.status_code, r.text)
    assert "version" in r.json()


@pytest.mark.parametrize("path", LIVENESS_PATHS)
def test_liveness_status_ok(path):
    c = _client(dev_mode=True)
    assert c.get(path).json()["status"] == "ok"


@pytest.mark.parametrize("path", READINESS_PATHS)
def test_readiness_status_ready(path):
    c = _client(dev_mode=True)
    assert c.get(path).json()["status"] == "ready"


def test_protected_route_still_requires_auth():
    """The probes being public must not weaken auth on real routes."""
    c = _client(dev_mode=False, api_key="k-secret")
    assert c.get("/whoami").status_code == 401
    ok = c.get("/whoami", headers={"Authorization": "Bearer k-secret"})
    assert ok.status_code == 200


# ---------------------------------------------------------------------------
# Opt-in request rate limit (EFFGEN_RATE_LIMIT). Previously documented but a
# no-op; now a per-client fixed-window limiter returning 429.
# ---------------------------------------------------------------------------


def test_rate_limit_disabled_by_default():
    c = _client(dev_mode=True)
    # Far more than any sane limit; with the limiter off none are throttled.
    codes = [c.get("/whoami").status_code for _ in range(30)]
    assert 429 not in codes


def test_rate_limit_enforced_and_returns_retry_after():
    limit = 4
    c = _client(dev_mode=True, rate_limit_per_minute=limit)
    codes = [c.get("/whoami").status_code for _ in range(limit + 3)]
    assert codes[:limit] == [200] * limit
    assert codes[limit:] == [429] * 3
    # 429 carries a Retry-After hint.
    blocked = c.get("/whoami")
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1


def test_rate_limit_exempts_probes():
    """A flood of probe requests is never throttled (K8s polls them often)."""
    c = _client(dev_mode=True, rate_limit_per_minute=2)
    codes = [c.get("/livez").status_code for _ in range(20)]
    assert codes == [200] * 20


def test_rate_limit_env_var(monkeypatch):
    """The limit is honoured from EFFGEN_RATE_LIMIT when no param is passed."""
    monkeypatch.setenv("EFFGEN_RATE_LIMIT", "3")
    c = _client(dev_mode=True)
    codes = [c.get("/whoami").status_code for _ in range(6)]
    assert codes == [200, 200, 200, 429, 429, 429]


def test_rate_limit_invalid_env_is_disabled(monkeypatch):
    monkeypatch.setenv("EFFGEN_RATE_LIMIT", "not-a-number")
    c = _client(dev_mode=True)
    codes = [c.get("/whoami").status_code for _ in range(10)]
    assert 429 not in codes
