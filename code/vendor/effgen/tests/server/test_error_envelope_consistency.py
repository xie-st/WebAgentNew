"""Every error the server returns uses one envelope, and one auth rule decides access.

The whole surface is swept here — the ``/v1`` routes, the ops and RBAC routes,
the dashboard and playground endpoints, plus the failures produced outside a
route (unknown URL, wrong method, unhandled exception) — so a client can branch
on ``error.type`` / ``error.code`` without special-casing an endpoint.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

API_KEY = "envelope-consistency-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
ENVELOPE_KEYS = {"message", "type", "param", "code"}


def _stub_runner(prompt, *, model=None, tools=None, stream=False, **kwargs):
    """Deterministic stand-in for a model so error mechanics stay the subject."""
    if stream:
        return iter(["stub ", "answer"])
    return "stub answer"


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from effgen.server.app import create_app

    monkeypatch.setenv("EFFGEN_API_KEY", API_KEY)
    monkeypatch.setenv("EFFGEN_NO_DOTENV", "1")
    monkeypatch.setenv("EFFGEN_DEV_MODE", "0")
    monkeypatch.delenv("EFFGEN_PUBLIC_DASHBOARD", raising=False)
    monkeypatch.delenv("EFFGEN_PUBLIC_PLAYGROUND", raising=False)
    # The spans endpoint is an open SSE stream; cap it so a test that reads it
    # to completion returns immediately.
    monkeypatch.setenv("EFFGEN_DASHBOARD_SSE_MAX_SECONDS", "0")
    app = create_app(api_key=API_KEY, runner=_stub_runner)
    return TestClient(app, raise_server_exceptions=False)


def assert_envelope(resp, *, status=None, code=None, err_type=None):
    """Assert *resp* carries the shared error envelope (and optional fields)."""
    if status is not None:
        assert resp.status_code == status, resp.text
    assert resp.headers["content-type"].startswith("application/json"), resp.text
    body = resp.json()
    assert set(body) == {"error"}, body
    assert set(body["error"]) == ENVELOPE_KEYS, body
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert isinstance(body["error"]["type"], str) and body["error"]["type"]
    assert body["error"]["param"] is None
    if code is not None:
        assert body["error"]["code"] == code, body
    if err_type is not None:
        assert body["error"]["type"] == err_type, body
    return body["error"]


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

# (method, path, json body) covering every mounted route plus the off-route
# failures. Each is exercised with no credentials, a wrong key, and the key.
SWEEP: list[tuple[str, str, object]] = [
    ("GET", "/health", None),
    ("GET", "/ready", None),
    ("GET", "/metrics", None),
    ("GET", "/slo", None),
    ("GET", "/whoami", None),
    ("GET", "/rbac/policy", None),
    ("GET", "/rbac/roles", None),
    ("GET", "/openapi.json", None),
    ("GET", "/v1/models", None),
    ("GET", "/v1/models/catalog", None),
    ("POST", "/v1/chat/completions", {"model": "m", "messages": []}),
    ("POST", "/v1/chat/completions", {"model": "m"}),
    ("POST", "/v1/completions", {"model": "m", "prompt": ""}),
    ("POST", "/v1/completions", {"model": "m"}),
    ("POST", "/v1/embeddings", {"model": "text-embedding-small", "input": []}),
    ("POST", "/v1/embeddings", {"model": "text-embedding-small"}),
    ("GET", "/dashboard", None),
    ("GET", "/dashboard/data.json", None),
    ("GET", "/dashboard/catalog.json", None),
    ("GET", "/dashboard/history.json", None),
    ("GET", "/dashboard/topology.json", None),
    ("GET", "/dashboard/no-such-asset.css", None),
    ("GET", "/playground", None),
    ("GET", "/playground/bootstrap", None),
    ("GET", "/playground/no-such-asset.css", None),
    ("GET", "/no-such-route", None),
    ("DELETE", "/health", None),
]


@pytest.mark.parametrize("method,path,body", SWEEP, ids=lambda v: str(v))
@pytest.mark.parametrize("creds", ["none", "wrong", "valid"])
def test_every_error_response_uses_the_shared_envelope(client, method, path, body, creds):
    headers = {
        "none": {},
        "wrong": {"Authorization": "Bearer not-the-key"},
        "valid": AUTH,
    }[creds]
    kwargs: dict = {"headers": headers}
    if body is not None:
        kwargs["json"] = body
    resp = client.request(method, path, **kwargs)
    if resp.status_code < 400:
        return
    assert_envelope(resp)


def test_unknown_url_and_wrong_method_are_typed(client):
    assert_envelope(
        client.get("/no-such-route", headers=AUTH),
        status=404,
        code="not_found",
        err_type="invalid_request_error",
    )
    # A 404 for a URL is not the "model_not_found" a 404 from a model route means.
    assert_envelope(
        client.request("DELETE", "/health"),
        status=405,
        code="method_not_allowed",
        err_type="invalid_request_error",
    )


def test_missing_web_asset_returns_the_envelope(client):
    for path in ("/dashboard/nope.css", "/playground/nope.css"):
        assert_envelope(client.get(path), status=404, code="not_found")


def test_unhandled_route_error_is_a_redacted_500_envelope(client, monkeypatch):
    import effgen.server.app as appmod

    def _boom():
        raise RuntimeError("dashboard blew up with key sk-abcdefghijklmnopqrstuvwxyz012345")

    monkeypatch.setattr(appmod, "_build_dashboard_data", _boom)
    err = assert_envelope(
        client.get("/dashboard/data.json", headers=AUTH), status=500, err_type="server_error"
    )
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in err["message"]
    # The marker names which pattern matched, so match the shape, not one word.
    assert "REDACTED" in err["message"]


def test_rbac_denial_reads_the_same_on_both_routes(client, monkeypatch):
    """An unknown role is reported identically by /rbac/policy and the model route."""
    from fastapi.testclient import TestClient

    from effgen.server.app import create_app

    monkeypatch.setenv("EFFGEN_API_KEY_ROLES", "no-such-role")
    denied = TestClient(
        create_app(api_key=API_KEY, runner=_stub_runner), raise_server_exceptions=False
    )
    policy = assert_envelope(
        denied.get("/rbac/policy", headers=AUTH), status=403, err_type="permission_error"
    )
    call = assert_envelope(
        denied.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        ),
        status=403,
        err_type="permission_error",
    )
    assert policy["message"] == call["message"]


# ---------------------------------------------------------------------------
# Content-free requests are refused before a billed call, on every /v1 route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,body,code",
    [
        ("/v1/chat/completions", {"model": "m", "messages": []}, "empty_messages"),
        (
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "  "}]},
            "empty_content",
        ),
        ("/v1/completions", {"model": "m", "prompt": ""}, "empty_prompt"),
        ("/v1/completions", {"model": "m", "prompt": "   "}, "empty_prompt"),
        ("/v1/completions", {"model": "m", "prompt": []}, "empty_prompt"),
        ("/v1/embeddings", {"model": "text-embedding-small", "input": ""}, "empty_input"),
        ("/v1/embeddings", {"model": "text-embedding-small", "input": []}, "empty_input"),
        (
            "/v1/embeddings",
            {"model": "text-embedding-small", "input": ["  ", ""]},
            "empty_input",
        ),
    ],
)
def test_content_free_request_is_rejected_with_400(client, path, body, code):
    assert_envelope(client.post(path, headers=AUTH, json=body), status=400, code=code)


def test_embeddings_still_serves_a_batch_holding_one_blank_entry(client):
    resp = client.post(
        "/v1/embeddings",
        headers=AUTH,
        json={"model": "text-embedding-small", "input": ["real text", ""]},
    )
    assert resp.status_code == 200, resp.text
    # Index alignment is preserved for the caller.
    assert len(resp.json()["data"]) == 2


# ---------------------------------------------------------------------------
# Streaming failures
# ---------------------------------------------------------------------------


STREAM_ROUTES = [
    (
        "/v1/chat/completions",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ),
    ("/v1/completions", {"model": "m", "prompt": "hi", "stream": True}),
]


def _stream_events(client, path, body):
    with client.stream("POST", path, headers=AUTH, json=body) as resp:
        return [
            json.loads(line[len("data: "):])
            for line in resp.iter_lines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]


def _streaming_client(monkeypatch, exc: Exception):
    from fastapi.testclient import TestClient

    from effgen.server.app import create_app

    monkeypatch.setenv("EFFGEN_API_KEY", API_KEY)
    monkeypatch.setenv("EFFGEN_DEV_MODE", "0")
    monkeypatch.setenv("EFFGEN_API_KEY_ROLES", "admin")

    def _failing(prompt, *, model=None, tools=None, stream=False, **kwargs):
        def _gen():
            yield "partial "
            raise exc

        return _gen() if stream else "x"

    return TestClient(
        create_app(api_key=API_KEY, runner=_failing), raise_server_exceptions=False
    )


@pytest.mark.parametrize("path,body", STREAM_ROUTES)
def test_midstream_failure_emits_a_terminal_error_event(monkeypatch, path, body):
    """A stream that dies mid-flight says so instead of truncating silently."""
    client = _streaming_client(monkeypatch, RuntimeError("upstream exploded"))
    events = _stream_events(client, path, body)
    assert events, "stream produced no events"
    last = events[-1]
    assert last["choices"][0]["finish_reason"] == "error"
    assert set(last["error"]) == ENVELOPE_KEYS
    assert "upstream exploded" in last["error"]["message"]


@pytest.mark.parametrize("path,body", STREAM_ROUTES)
def test_a_midstream_typeerror_is_reported_not_swallowed(monkeypatch, path, body):
    """A TypeError raised after the first chunk is a failure, not a non-iterable result.

    The non-iterable-runner fallback only applies before any chunk is produced;
    once the stream is under way a ``TypeError`` is reported like any other
    mid-stream failure.
    """
    client = _streaming_client(monkeypatch, TypeError("adapter returned the wrong shape"))
    events = _stream_events(client, path, body)
    assert events, "stream produced no events"
    last = events[-1]
    assert last["choices"][0]["finish_reason"] == "error", last
    assert set(last["error"]) == ENVELOPE_KEYS
    assert "wrong shape" in last["error"]["message"]


@pytest.mark.parametrize("path,body", STREAM_ROUTES)
def test_a_non_iterable_streaming_result_is_still_stringified(monkeypatch, path, body):
    """A runner that returns a bare object under ``stream=True`` still answers."""
    from fastapi.testclient import TestClient

    from effgen.server.app import create_app

    monkeypatch.setenv("EFFGEN_API_KEY", API_KEY)
    monkeypatch.setenv("EFFGEN_DEV_MODE", "0")

    class _Bare:
        def __str__(self) -> str:
            return "bare answer"

    def _runner(prompt, *, model=None, tools=None, stream=False, **kwargs):
        return _Bare()

    client = TestClient(
        create_app(api_key=API_KEY, runner=_runner), raise_server_exceptions=False
    )
    events = _stream_events(client, path, body)
    text = json.dumps(events)
    assert "bare answer" in text, events
    assert '"finish_reason": "error"' not in text


# ---------------------------------------------------------------------------
# One auth rule
# ---------------------------------------------------------------------------

# Endpoints reachable without credentials: liveness/readiness probes, the
# aggregate SLO status, the API schema, and the inert web shells.
PUBLIC = [
    "/health",
    "/healthz",
    "/livez",
    "/ready",
    "/readyz",
    "/slo",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/dashboard",
    "/dashboard/",
    "/dashboard/app.js",
    "/playground",
    "/playground/",
]

PROTECTED = [
    "/metrics",
    "/whoami",
    "/rbac/policy",
    "/rbac/roles",
    "/v1/models",
    "/v1/models/catalog",
    "/dashboard/data.json",
    "/dashboard/catalog.json",
    "/dashboard/history.json",
    "/dashboard/topology.json",
    "/dashboard/spans",
    "/playground/bootstrap",
]


@pytest.mark.parametrize("path", PUBLIC)
def test_public_paths_need_no_credentials(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", PROTECTED)
def test_protected_paths_reject_a_missing_or_wrong_key(client, path):
    for headers in ({}, {"Authorization": "Bearer not-the-key"}):
        assert_envelope(
            client.get(path, headers=headers),
            status=401,
            code="invalid_api_key",
        )
    assert client.get(path, headers=AUTH).status_code == 200


@pytest.mark.parametrize("path", ["/health/", "/healthz/", "/ready/", "/readyz/", "/slo/"])
def test_a_probe_path_written_with_a_trailing_slash_is_still_public(client, path):
    assert client.get(path).status_code == 200


def test_shutdown_drain_refusal_uses_the_envelope():
    """A request arriving after the drain started is refused in the same shape."""
    import warnings

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from effgen.api.middleware import _install_graceful_shutdown

    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _install_graceful_shutdown(app, timeout=0.1)
        with TestClient(app) as running:
            assert running.get("/ping").status_code == 200
        # The context manager ran the shutdown hook; the app is now draining.
        drained = TestClient(app)
        assert_envelope(
            drained.get("/ping"),
            status=503,
            code="server_shutting_down",
            err_type="upstream_unavailable",
        )


def test_an_unknown_path_is_not_reachable_without_credentials(client):
    """An unrouted URL must not reveal that it is unrouted before authenticating."""
    assert_envelope(client.get("/no-such-route"), status=401)


# ---------------------------------------------------------------------------
# One condition, one status: an absent credential is a configuration gap (503),
# a rejected one is an upstream failure (502) — on every provider.
# ---------------------------------------------------------------------------
# The exact sentences the provider adapters raise when nothing is configured.
# They word it two ways ("not found" / "not provided"); both mean the same
# thing, so both must classify the same way.
MISSING_KEY_MESSAGES = [
    "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key",
    "Google API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key",
    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key",
    "Cerebras API key not found. Set the CEREBRAS_API_KEY environment variable",
    "Fireworks API key not found. Set the FIREWORKS_API_KEY environment variable",
    "Groq API key not found. Set the GROQ_API_KEY environment variable",
    "Together API key not found. Set the TOGETHER_API_KEY environment variable",
]

REJECTED_KEY_MESSAGES = [
    "openai error: Incorrect API key provided: sk-proj-***. You can find your API key at ...",
    "groq error: Invalid API Key",
    "gemini error: API key not valid. Please pass a valid API key.",
]


@pytest.mark.parametrize("message", MISSING_KEY_MESSAGES)
def test_an_absent_provider_key_is_a_503(message):
    from effgen.api.openai_compat import _classify_http
    from effgen.models.errors import ModelAuthError

    status, err_type, code = _classify_http(ModelAuthError(message))
    assert (status, code) == (503, "upstream_key_missing"), (status, err_type, code)


@pytest.mark.parametrize("message", REJECTED_KEY_MESSAGES)
def test_a_rejected_provider_key_is_a_502(message):
    from effgen.api.openai_compat import _classify_http
    from effgen.models.errors import ModelAuthError

    status, err_type, code = _classify_http(ModelAuthError(message))
    assert (status, code) == (502, "upstream_auth_failed"), (status, err_type, code)
