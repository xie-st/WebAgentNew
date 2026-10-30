"""API server tests using FastAPI TestClient.

The FastAPI app is constructed inline inside cli.serve_api(), so we
replicate the minimal app setup here for isolated testing.
"""

import time

import pytest

# Try to import FastAPI and create the test app
pytestmark = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("fastapi", "starlette")),
    reason="fastapi not installed",
)


def _create_test_app(api_key=None, rate_limit=60):
    """Create a minimal FastAPI app matching the CLI server endpoints."""

    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.security import APIKeyHeader

    _metrics = {"requests": 0, "errors": 0, "total_time": 0.0}
    _rate_buckets = {}
    _rate_limit = rate_limit

    def _check_rate(client_ip):
        now = time.time()
        bucket = _rate_buckets.setdefault(client_ip, [])
        _rate_buckets[client_ip] = bucket = [t for t in bucket if now - t < 60]
        if len(bucket) >= _rate_limit:
            return False
        bucket.append(now)
        return True

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_api_key(key: str | None = Depends(api_key_header)):
        if api_key and key != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    app = FastAPI(title="effGen API Test")

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        _metrics["requests"] += 1
        start = time.time()
        response = await call_next(request)
        _metrics["total_time"] += time.time() - start
        return response

    @app.get("/health")
    async def health():
        from effgen import __version__
        return {"status": "healthy", "version": __version__}

    @app.get("/metrics", dependencies=[Depends(verify_api_key)])
    async def metrics():
        avg_time = (
            _metrics["total_time"] / _metrics["requests"]
            if _metrics["requests"] > 0
            else 0
        )
        return {
            "requests_total": _metrics["requests"],
            "errors_total": _metrics["errors"],
            "avg_response_time_seconds": round(avg_time, 4),
        }

    return app


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self):
        from starlette.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_status(self):
        from starlette.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestMetricsEndpoint:
    """Test /metrics endpoint."""

    def test_metrics_returns_200_no_auth(self):
        from starlette.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_contains_fields(self):
        from starlette.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        data = resp.json()
        assert "requests_total" in data
        assert "errors_total" in data
        assert "avg_response_time_seconds" in data


class TestAuthentication:
    """Test API key authentication."""

    def test_auth_required_no_key(self):
        from starlette.testclient import TestClient
        app = _create_test_app(api_key="secret123")
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 401

    def test_auth_required_wrong_key(self):
        from starlette.testclient import TestClient
        app = _create_test_app(api_key="secret123")
        client = TestClient(app)
        resp = client.get("/metrics", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_auth_correct_key(self):
        from starlette.testclient import TestClient
        app = _create_test_app(api_key="secret123")
        client = TestClient(app)
        resp = client.get("/metrics", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200

    def test_health_no_auth_required(self):
        """Health endpoint should work even with auth enabled."""
        from starlette.testclient import TestClient
        app = _create_test_app(api_key="secret123")
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200


class TestRateLimiting:
    """Test rate limiting."""

    def test_rate_limit_exceeded(self):
        from starlette.testclient import TestClient
        app = _create_test_app(rate_limit=3)
        client = TestClient(app)
        # Make requests up to the limit
        for _ in range(3):
            resp = client.get("/health")
            assert resp.status_code == 200
        # Next request should be rate limited
        resp = client.get("/health")
        assert resp.status_code == 429

    def test_rate_limit_returns_detail(self):
        from starlette.testclient import TestClient
        app = _create_test_app(rate_limit=1)
        client = TestClient(app)
        client.get("/health")
        resp = client.get("/health")
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["detail"].lower()
