"""``/v1/embeddings`` reporting contract.

The endpoint must (a) resolve a ``provider:`` prefix the same way the chat
endpoint does, so ``openai:text-embedding-3-small`` reaches the real neural
model exactly like the bare alias; and (b) never silently serve degraded
lexical TF-IDF vectors under a neural model's name — when the neural backend
cannot load it warns once and reflects the fallback to the caller (body
extension + ``x-effgen-embedding-backend`` header), or fails closed (503) in
strict mode.

The neural backend is *simulated as unavailable* by replacing the
``SentenceTransformerEmbedder`` with one that raises — this exercises the
fallback-reporting logic deterministically and offline (it does not mock any
live model output, which the project forbids).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("fastapi", "starlette")),
    reason="fastapi not installed",
)


def _broken_engine(monkeypatch):
    """An EmbeddingEngine whose neural backend always fails to load."""
    import effgen.api.embeddings as emb

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("sentence-transformers unavailable (simulated)")

    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", _Boom)
    return emb.EmbeddingEngine(persistent_cache=False)


def test_resolve_model_strips_provider_prefix():
    from effgen.api.embeddings import MODEL_ALIASES, EmbeddingEngine

    eng = EmbeddingEngine(persistent_cache=False)
    bare = eng._resolve_model("text-embedding-3-small")
    prefixed = eng._resolve_model("openai:text-embedding-3-small")
    assert bare == MODEL_ALIASES["text-embedding-3-small"]
    assert prefixed == bare  # the prefix must not change the resolved model
    # An hf: prefix on a concrete repo id resolves to that repo (prefix stripped).
    assert eng._resolve_model("hf:some-org/some-model") == "some-org/some-model"


def test_fallback_is_flagged_on_engine(monkeypatch):
    eng = _broken_engine(monkeypatch)
    vecs = eng.embed(["hello world"], model="text-embedding-3-small")
    assert len(vecs) == 1
    # The neural backend could not load → lexical fallback, and it is recorded.
    assert eng.is_fallback("text-embedding-3-small") is True


def test_route_reflects_fallback(monkeypatch):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from effgen.api.embeddings import TFIDF_FALLBACK_ID, create_embeddings_router

    eng = _broken_engine(monkeypatch)
    app = FastAPI()
    app.include_router(create_embeddings_router(engine=eng))
    c = TestClient(app)

    r = c.post("/v1/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert r.headers["x-effgen-embedding-backend"] == TFIDF_FALLBACK_ID
    assert body["effgen"]["degraded"] is True
    assert body["effgen"]["backend"] == TFIDF_FALLBACK_ID
    # Vectors are still returned at the expected dim (just lexical, not neural).
    assert len(body["data"][0]["embedding"]) > 0


def test_strict_mode_fails_closed(monkeypatch):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from effgen.api.embeddings import create_embeddings_router

    monkeypatch.setenv("EFFGEN_EMBEDDINGS_STRICT", "1")
    eng = _broken_engine(monkeypatch)
    app = FastAPI()
    app.include_router(create_embeddings_router(engine=eng))
    c = TestClient(app)

    r = c.post("/v1/embeddings", json={"model": "text-embedding-3-small", "input": "hi"})
    assert r.status_code == 503  # refuse to serve degraded vectors under a neural id
    # Same {"error": {message, type, param, code}} envelope every other /v1
    # route uses, not FastAPI's default {"detail": ...}.
    body = r.json()
    assert "detail" not in body
    err = body["error"]
    assert err["message"]
    assert err["type"] == "upstream_unavailable"
    assert err["code"] == "embedding_backend_unavailable"


def test_empty_input_error_envelope():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from effgen.api.embeddings import EmbeddingEngine, create_embeddings_router

    app = FastAPI()
    app.include_router(
        create_embeddings_router(engine=EmbeddingEngine(persistent_cache=False))
    )
    c = TestClient(app)

    r = c.post("/v1/embeddings", json={"model": "tfidf", "input": []})
    assert r.status_code == 400
    body = r.json()
    assert "detail" not in body
    err = body["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "empty_input"
    assert "empty" in err["message"]


def test_unexpected_embedding_failure_error_envelope(monkeypatch):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import effgen.api.embeddings as emb

    eng = emb.EmbeddingEngine(persistent_cache=False)
    monkeypatch.setattr(
        eng,
        "embed",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    app = FastAPI()
    app.include_router(emb.create_embeddings_router(engine=eng))
    c = TestClient(app)

    r = c.post("/v1/embeddings", json={"model": "tfidf", "input": "hi"})
    assert r.status_code == 500
    body = r.json()
    assert "detail" not in body
    err = body["error"]
    assert err["type"] == "server_error"
    assert "boom" in err["message"]


def test_explicit_tfidf_not_marked_degraded():
    """Asking for the lexical embedder by name is intended use, not a downgrade."""
    from effgen.api.embeddings import EmbeddingEngine

    eng = EmbeddingEngine(persistent_cache=False)
    vecs = eng.embed(["hello world"], model="tfidf")
    assert len(vecs) == 1
    assert eng.is_fallback("tfidf") is False
