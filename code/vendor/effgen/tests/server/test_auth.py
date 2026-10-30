"""Tests for effgen.server.auth — JWT verification, AuthMiddleware, dev mode.

All tests are offline — no network calls.  RSA keys are generated in-process
and the verify_jwt() function is called with an explicit jwks_uri pointing to
a mock (monkeypatched) endpoint.
"""
from __future__ import annotations

import json
import time
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures: RSA key pair + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair():
    """Generate an RSA-2048 key pair once for the whole module."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    return private_key, private_key.public_key()


@pytest.fixture(scope="module")
def jwk_dict(rsa_keypair):
    """Return a JWK dict for the RSA public key (for mocking JWKS endpoint)."""
    from jwt.algorithms import RSAAlgorithm

    _, public_key = rsa_keypair
    jwk = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk["kid"] = "test-key-1"
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    return jwk


def make_token(
    private_key: Any,
    *,
    sub: str = "test-user",
    iss: str = "https://example.com",
    aud: str = "effgen-client",
    roles: list[str] | None = None,
    email: str = "test@example.com",
    exp_offset: int = 3600,
    kid: str = "test-key-1",
) -> str:
    import jwt as pyjwt

    payload: dict[str, Any] = {
        "sub": sub,
        "iss": iss,
        "aud": aud,
        "exp": int(time.time()) + exp_offset,
        "iat": int(time.time()),
        "email": email,
    }
    if roles is not None:
        payload["roles"] = roles
    return pyjwt.encode(payload, private_key, algorithm="RS256", headers={"kid": kid})


# ---------------------------------------------------------------------------
# verify_jwt — happy path
# ---------------------------------------------------------------------------


class TestVerifyJWT:
    def test_valid_token_returns_payload(self, rsa_keypair, jwk_dict):
        """A valid RS256 JWT should decode to the correct TokenPayload."""
        from effgen.server.auth import _jwks_cache, verify_jwt

        # Populate JWKS cache so no HTTP call is made
        _jwks_cache.keys = {"test-key-1": jwk_dict}
        _jwks_cache.fetched_at = time.time() + 3600  # mark fresh

        private_key, _ = rsa_keypair
        token = make_token(
            private_key,
            iss="https://example.com",
            aud="effgen-client",
            roles=["researcher"],
        )

        payload = verify_jwt(
            token,
            issuer="https://example.com",
            client_id="effgen-client",
            jwks_uri="https://example.com/.well-known/jwks.json",
        )

        assert payload.sub == "test-user"
        assert payload.iss == "https://example.com"
        assert "researcher" in payload.roles
        assert payload.email == "test@example.com"

    def test_token_with_multiple_roles(self, rsa_keypair, jwk_dict):
        """Tokens can carry multiple roles."""
        from effgen.server.auth import _jwks_cache, verify_jwt

        _jwks_cache.keys = {"test-key-1": jwk_dict}
        _jwks_cache.fetched_at = time.time() + 3600

        private_key, _ = rsa_keypair
        token = make_token(
            private_key,
            roles=["admin", "researcher"],
            iss="https://example.com",
            aud="effgen-client",
        )

        payload = verify_jwt(
            token,
            issuer="https://example.com",
            client_id="effgen-client",
            jwks_uri="https://example.com/.well-known/jwks.json",
        )

        assert set(payload.roles) == {"admin", "researcher"}

    def test_expired_token_raises_auth_error(self, rsa_keypair, jwk_dict):
        """An expired JWT must raise AuthError."""
        from effgen.server.auth import AuthError, _jwks_cache, verify_jwt

        _jwks_cache.keys = {"test-key-1": jwk_dict}
        _jwks_cache.fetched_at = time.time() + 3600

        private_key, _ = rsa_keypair
        token = make_token(private_key, exp_offset=-3600)  # already expired

        with pytest.raises(AuthError, match="expired"):
            verify_jwt(
                token,
                issuer="https://example.com",
                client_id="effgen-client",
                jwks_uri="https://example.com/.well-known/jwks.json",
            )

    def test_wrong_audience_raises_auth_error(self, rsa_keypair, jwk_dict):
        """A JWT with wrong audience must raise AuthError."""
        from effgen.server.auth import AuthError, _jwks_cache, verify_jwt

        _jwks_cache.keys = {"test-key-1": jwk_dict}
        _jwks_cache.fetched_at = time.time() + 3600

        private_key, _ = rsa_keypair
        token = make_token(private_key, aud="wrong-audience")

        with pytest.raises(AuthError, match="audience"):
            verify_jwt(
                token,
                issuer="https://example.com",
                client_id="effgen-client",
                jwks_uri="https://example.com/.well-known/jwks.json",
            )

    def test_wrong_issuer_raises_auth_error(self, rsa_keypair, jwk_dict):
        """A JWT from an unexpected issuer must raise AuthError."""
        from effgen.server.auth import AuthError, _jwks_cache, verify_jwt

        _jwks_cache.keys = {"test-key-1": jwk_dict}
        _jwks_cache.fetched_at = time.time() + 3600

        private_key, _ = rsa_keypair
        token = make_token(
            private_key,
            iss="https://evil.example.com",
            aud="effgen-client",
        )

        with pytest.raises(AuthError, match="issuer"):
            verify_jwt(
                token,
                issuer="https://example.com",
                client_id="effgen-client",
                jwks_uri="https://example.com/.well-known/jwks.json",
            )

    def test_malformed_token_raises_auth_error(self, rsa_keypair, jwk_dict):
        """A garbage string should raise AuthError."""
        from effgen.server.auth import AuthError, _jwks_cache, verify_jwt

        _jwks_cache.keys = {"test-key-1": jwk_dict}
        _jwks_cache.fetched_at = time.time() + 3600

        with pytest.raises(AuthError):
            verify_jwt(
                "not.a.jwt",
                issuer="https://example.com",
                client_id="effgen-client",
                jwks_uri="https://example.com/.well-known/jwks.json",
            )


# ---------------------------------------------------------------------------
# AuthMiddleware — ASGI tests
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    @pytest.fixture()
    def _fresh_cache(self, jwk_dict):
        """Pre-populate the JWKS cache so tests don't hit the network."""
        from effgen.server import auth as _auth

        _auth._jwks_cache.keys = {"test-key-1": jwk_dict}
        _auth._jwks_cache.fetched_at = time.time() + 3600
        yield
        _auth._jwks_cache.keys = {}
        _auth._jwks_cache.fetched_at = 0.0

    @pytest.fixture()
    def _no_dev_mode(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_DEV_MODE", "0")

    async def _call_middleware(
        self,
        middleware,
        path: str,
        auth_header: str | None = None,
    ) -> tuple[int, bytes]:
        """Call middleware and capture (status_code, body)."""
        headers = []
        if auth_header:
            headers.append((b"authorization", auth_header.encode()))

        scope = {
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": b"",
            "headers": headers,
        }

        status = [200]
        body_parts: list[bytes] = []

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(event):
            if event["type"] == "http.response.start":
                status[0] = event["status"]
            elif event["type"] == "http.response.body":
                body_parts.append(event.get("body", b""))

        async def app(s, r, snd):
            await snd({"type": "http.response.start", "status": 200, "headers": []})
            await snd({"type": "http.response.body", "body": b'{"ok": true}'})

        await middleware(scope, receive, send)
        return status[0], b"".join(body_parts)

    @pytest.mark.asyncio
    async def test_public_health_path_no_auth_needed(self, _no_dev_mode, monkeypatch):
        """GET /health must pass without any Authorization header."""
        monkeypatch.setenv("EFFGEN_DEV_MODE", "0")
        from effgen.server.auth import AuthMiddleware

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"status":"ok"}'})

        mw = AuthMiddleware(inner)
        status, _ = await self._call_middleware(mw, "/health")
        assert status == 200

    @pytest.mark.asyncio
    async def test_unauthenticated_request_rejected(self, _no_dev_mode, monkeypatch):
        """A request without Authorization header to /secure should get 401."""
        monkeypatch.setenv("EFFGEN_DEV_MODE", "0")
        from effgen.server.auth import AuthMiddleware

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"ok": true}'})

        mw = AuthMiddleware(inner)
        status, body = await self._call_middleware(mw, "/secure")
        assert status == 401
        data = json.loads(body)
        # Auth rejections share the OpenAI error envelope so a client can branch
        # on err.type/err.code uniformly. The message stays a readable hint (it
        # spells out the header syntax) — not mangled by the secret scrubber.
        assert "error" in data and data["error"]["type"]
        assert "Authorization" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_valid_bearer_token_passes(
        self, _fresh_cache, _no_dev_mode, rsa_keypair, monkeypatch
    ):
        """A valid JWT must pass the middleware and populate state.user."""
        monkeypatch.setenv("EFFGEN_DEV_MODE", "0")
        monkeypatch.setenv("EFFGEN_OIDC_ISSUER", "https://example.com")
        monkeypatch.setenv("EFFGEN_OIDC_CLIENT_ID", "effgen-client")
        monkeypatch.setenv(
            "EFFGEN_OIDC_JWKS_URI", "https://example.com/.well-known/jwks.json"
        )

        private_key, _ = rsa_keypair
        token = make_token(
            private_key,
            iss="https://example.com",
            aud="effgen-client",
            roles=["researcher"],
        )

        user_captured: list[Any] = []

        async def inner(scope, receive, send):
            user_captured.append(scope.get("state", {}).get("user"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"ok": true}'})

        from effgen.server.auth import AuthMiddleware

        mw = AuthMiddleware(
            inner,
            issuer="https://example.com",
            client_id="effgen-client",
            jwks_uri="https://example.com/.well-known/jwks.json",
        )
        status, _ = await self._call_middleware(mw, "/secure", f"Bearer {token}")
        assert status == 200
        assert user_captured[0] is not None
        assert user_captured[0].sub == "test-user"

    @pytest.mark.asyncio
    async def test_dev_mode_bypasses_auth(self, monkeypatch):
        """EFFGEN_DEV_MODE=1 must bypass auth with a dev-user token."""
        monkeypatch.setenv("EFFGEN_DEV_MODE", "1")
        from effgen.server import auth as _auth

        user_captured: list[Any] = []

        async def inner(scope, receive, send):
            user_captured.append(scope.get("state", {}).get("user"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"ok": true}'})

        from effgen.server.auth import AuthMiddleware

        with pytest.warns(UserWarning, match="EFFGEN_DEV_MODE"):
            mw = AuthMiddleware(inner)

        _auth._DEV_MODE_WARNED = False  # reset so the call warning also fires
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status, _ = await self._call_middleware(mw, "/secure")

        assert status == 200
        assert user_captured[0] is not None
        assert user_captured[0].sub == "dev-user"
        assert "admin" in user_captured[0].roles


# ---------------------------------------------------------------------------
# Role matrix tests (JWT claims → roles)
# ---------------------------------------------------------------------------


class TestRoleMatrix:
    def _make_payload_with_roles(self, roles: list[str]):
        """Helper to create a TokenPayload with specific roles."""
        from effgen.server.auth import TokenPayload

        return TokenPayload(
            sub="u1",
            iss="test",
            aud="test",
            exp=int(time.time()) + 3600,
            iat=int(time.time()),
            roles=roles,
        )

    def test_scope_string_parsed_into_roles(self):
        """A space-separated scope string becomes a list of roles."""
        from effgen.server.auth import TokenPayload

        claims = {
            "sub": "u",
            "iss": "i",
            "aud": "a",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "scope": "admin researcher",
        }
        p = TokenPayload.from_claims(claims)
        assert set(p.roles) == {"admin", "researcher"}

    def test_roles_list_claim_preserved(self):
        """A JSON list roles claim is kept as-is."""
        from effgen.server.auth import TokenPayload

        claims = {
            "sub": "u",
            "iss": "i",
            "aud": "a",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "roles": ["viewer", "researcher"],
        }
        p = TokenPayload.from_claims(claims)
        assert set(p.roles) == {"viewer", "researcher"}
