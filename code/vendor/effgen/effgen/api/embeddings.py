"""OpenAI-compatible local embeddings API.

Provides a ``/v1/embeddings`` endpoint that works out of the box without
paid APIs. Uses ``sentence-transformers`` when available, otherwise falls
back to an in-process TF-IDF hashing embedder so the endpoint remains
functional in minimal installs.

Caching:
    * In-memory LRU cache for repeated texts.
    * Optional SQLite-backed persistent cache keyed by (model, text_hash).

Usage:
    >>> from fastapi import FastAPI
    >>> from effgen.api.embeddings import create_embeddings_router
    >>> app = FastAPI()
    >>> app.include_router(create_embeddings_router())
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sqlite3
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

# Model aliases — OpenAI-style names mapped to local models.
MODEL_ALIASES = {
    "text-embedding-small": "sentence-transformers/all-MiniLM-L6-v2",
    "text-embedding-ada-002": "sentence-transformers/all-MiniLM-L6-v2",
    "text-embedding-3-small": "sentence-transformers/all-MiniLM-L6-v2",
    "text-embedding-3-large": "sentence-transformers/all-mpnet-base-v2",
}
DEFAULT_DIM = 384  # TF-IDF fallback dimension

# Provider prefixes an OpenAI-compatible client may send (the same ``provider:``
# prefix the chat endpoint accepts). They are stripped before the alias lookup so
# that, e.g., ``openai:text-embedding-3-small`` resolves to the real neural model
# exactly as the bare ``text-embedding-3-small`` does — instead of falling
# through to the lexical fallback because the prefixed id matched no alias.
_PROVIDER_PREFIXES = (
    "openai",
    "hf",
    "huggingface",
    "sentence-transformers",
    "together",
    "fireworks",
    "cohere",
    "voyage",
)

# Identifier reported on the response (and ``x-effgen-embedding-backend`` header)
# when the requested neural backend could not load and the lexical TF-IDF
# embedder served the request instead. Never silently substituted.
TFIDF_FALLBACK_ID = "tfidf-fallback"


def _strict_mode() -> bool:
    """Whether to fail closed (503) rather than serve degraded TF-IDF vectors."""
    return os.getenv("EFFGEN_EMBEDDINGS_STRICT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


class EmbeddingBackendUnavailable(RuntimeError):
    """The requested neural embedding backend could not be loaded (strict mode)."""


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class LRUCache:
    """Thread-safe LRU cache for embeddings keyed by (model, text)."""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
        self._data: "OrderedDict[str, list[float]]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, text: str) -> str:
        """Return the cache key for *model* and *text* (SHA-256 of the text)."""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{model}:{h}"

    def get(self, model: str, text: str) -> list[float] | None:
        """Return the cached vector for (*model*, *text*), or ``None`` on a miss."""
        k = self.key(model, text)
        with self._lock:
            val = self._data.get(k)
            if val is not None:
                self._data.move_to_end(k)
            return val

    def put(self, model: str, text: str, vec: list[float]) -> None:
        """Store *vec* for (*model*, *text*), evicting the oldest entries past capacity."""
        k = self.key(model, text)
        with self._lock:
            self._data[k] = vec
            self._data.move_to_end(k)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)


class SQLiteCache:
    """Persistent embedding cache backed by SQLite."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.path.expanduser("~/.effgen/embeddings_cache.db")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  model TEXT NOT NULL,"
            "  text_hash TEXT NOT NULL,"
            "  vector BLOB NOT NULL,"
            "  dim INTEGER NOT NULL,"
            "  PRIMARY KEY (model, text_hash)"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        """Return the stored vector for (*model*, *text*), or ``None`` on a miss."""
        with self._lock:
            row = self._conn.execute(
                "SELECT vector, dim FROM embeddings WHERE model=? AND text_hash=?",
                (model, self._hash(text)),
            ).fetchone()
        if not row:
            return None
        blob, dim = row
        import struct

        return list(struct.unpack(f"{dim}f", blob))

    def put(self, model: str, text: str, vec: list[float]) -> None:
        """Persist *vec* for (*model*, *text*), replacing any existing row."""
        import struct

        blob = struct.pack(f"{len(vec)}f", *vec)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, vector, dim)"
                " VALUES (?, ?, ?, ?)",
                (model, self._hash(text), blob, len(vec)),
            )
            self._conn.commit()


# ---------------------------------------------------------------------------
# Embedder backends
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class TFIDFEmbedder:
    """Hashing TF-IDF embedder — dependency-free fallback.

    Produces fixed-dimension L2-normalized sparse-to-dense vectors using a
    hashing trick. Not as good as a neural embedder, but deterministic and
    works without any additional packages.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _WORD_RE.findall(text)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized hashed TF vector per input text."""
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = self._tokenize(text)
            if not tokens:
                out.append(vec)
                continue
            # term frequency with hashing
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 32) & 1 else -1.0
                vec[idx] += sign
            # IDF-ish log dampening
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence-transformers`` when installed."""

    def __init__(self, model_name: str) -> None:
        from effgen.utils.embedding_backend import load_sentence_transformer

        self.model_name = model_name
        self.model = load_sentence_transformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized sentence-transformer vector per input text."""
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class EmbeddingEngine:
    """Main embedding engine: resolves models, caches, dispatches."""

    def __init__(
        self,
        lru_size: int = 1024,
        persistent_cache: bool = True,
        cache_path: str | None = None,
    ) -> None:
        self.lru = LRUCache(max_size=lru_size)
        self.sqlite: SQLiteCache | None
        try:
            self.sqlite = SQLiteCache(cache_path) if persistent_cache else None
        except Exception:
            self.sqlite = None
        self._backends: dict = {}
        self._backend_lock = threading.Lock()
        # Resolved model ids that fell back to the lexical TF-IDF embedder
        # because their neural backend could not load — surfaced to callers.
        self._fallback_models: set[str] = set()

    def _resolve_model(self, model: str) -> str:
        # Strip a known ``provider:`` prefix so a prefixed id resolves like the
        # bare alias (``openai:text-embedding-3-small`` -> the MiniLM model).
        bare = model
        prefix, sep, rest = model.partition(":")
        if sep and prefix.lower() in _PROVIDER_PREFIXES and rest:
            bare = rest
        return MODEL_ALIASES.get(bare, bare)

    def is_fallback(self, model: str) -> bool:
        """True when *model* is being served by the lexical TF-IDF fallback."""
        return self._resolve_model(model) in self._fallback_models

    def _get_backend(self, model: str) -> Any:
        resolved = self._resolve_model(model)
        with self._backend_lock:
            if resolved in self._backends:
                return self._backends[resolved]
            backend: Any
            if isinstance(resolved, str) and resolved.lower().startswith("tfidf"):
                # The caller explicitly asked for the lexical embedder.
                backend = TFIDFEmbedder()
            else:
                try:
                    backend = SentenceTransformerEmbedder(resolved)
                except Exception as exc:  # noqa: BLE001 - neural backend optional
                    if _strict_mode():
                        # Fail closed instead of quietly degrading the vectors.
                        raise EmbeddingBackendUnavailable(
                            f"embedding backend for {resolved!r} could not load "
                            f"({exc}). Install the neural backend "
                            "(`pip install 'effgen[rag]'`) and check the model "
                            "id, or set EFFGEN_EMBEDDINGS_STRICT=0 to allow the "
                            "lexical TF-IDF fallback."
                        ) from exc
                    # Non-spammy fallback: warn once per model, record it
                    # so the response can tell the caller the vectors are lexical.
                    self._fallback_models.add(resolved)
                    logger.warning(
                        "Embedding backend for %r could not load (%s); serving "
                        "lexical TF-IDF fallback vectors. Install/repair "
                        "sentence-transformers for neural embeddings, or set "
                        "EFFGEN_EMBEDDINGS_STRICT=1 to fail closed instead.",
                        resolved, exc,
                    )
                    backend = TFIDFEmbedder()
            self._backends[resolved] = backend
            return backend

    def embed(self, texts: list[str], model: str = "text-embedding-small") -> list[list[float]]:
        """Embed *texts* with *model*, serving repeats from the LRU/SQLite caches."""
        if not texts:
            return []
        resolved = self._resolve_model(model)

        # Lookup caches
        results: list[list[float] | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = self.lru.get(resolved, t)
            if cached is None and self.sqlite is not None:
                cached = self.sqlite.get(resolved, t)
                if cached is not None:
                    self.lru.put(resolved, t, cached)
            if cached is not None:
                results[i] = cached
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        if missing_texts:
            backend = self._get_backend(model)
            new_vecs = backend.embed(missing_texts)
            for idx, text, vec in zip(missing_idx, missing_texts, new_vecs):
                results[idx] = vec
                self.lru.put(resolved, text, vec)
                if self.sqlite is not None:
                    try:
                        self.sqlite.put(resolved, text, vec)
                    except Exception:
                        pass

        return [r for r in results if r is not None]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FastAPI router (optional dependency)
# ---------------------------------------------------------------------------
# Imported at module level so the ``response: Response`` route annotation
# resolves under ``from __future__ import annotations`` (FastAPI reads type hints
# from module globals; a function-local import would not be found and the param
# would be misread as a query field).
try:
    from fastapi import Response  # type: ignore
except Exception:  # pragma: no cover - fastapi optional
    Response = None  # type: ignore


try:
    from pydantic import BaseModel as _PydBaseModel  # type: ignore
    from pydantic import Field as _PydField

    class EmbeddingRequest(_PydBaseModel):
        """Request body for ``POST /v1/embeddings``."""

        model: str = _PydField(default="text-embedding-small")
        input: str | list[str]

    class EmbeddingItem(_PydBaseModel):
        """One embedding vector in the response ``data`` list."""

        object: str = "embedding"
        index: int
        embedding: list[float]

    class EmbeddingUsage(_PydBaseModel):
        """Token usage counters for an embeddings request."""

        prompt_tokens: int = 0
        total_tokens: int = 0

    class EmbeddingResponse(_PydBaseModel):
        """Response body for ``POST /v1/embeddings``."""

        object: str = "list"
        data: list[EmbeddingItem]
        model: str
        usage: EmbeddingUsage
        # Non-standard extension (OpenAI clients ignore unknown keys): records
        # the resolved model and flags when the lexical TF-IDF fallback served
        # the request, so a caller building semantic search/RAG can tell the
        # vectors are degraded rather than neural.
        effgen: dict[str, Any] | None = None
except Exception:  # pragma: no cover - pydantic optional
    EmbeddingRequest = None  # type: ignore
    EmbeddingItem = None  # type: ignore
    EmbeddingUsage = None  # type: ignore
    EmbeddingResponse = None  # type: ignore


def create_embeddings_router(engine: EmbeddingEngine | None = None) -> Any:
    """Create a FastAPI router exposing ``/v1/embeddings``.

    Requires ``fastapi`` and ``pydantic`` to be installed. Otherwise import
    of this function will raise.
    """
    from fastapi import APIRouter, Body  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore

    from effgen.api.openai_compat import error_envelope

    eng = engine or EmbeddingEngine()

    router = APIRouter()

    @router.post("/v1/embeddings", response_model=EmbeddingResponse)
    def create_embeddings(
        response: Response, req: EmbeddingRequest = Body(...)
    ) -> Any:
        # A JSONResponse built from the shared `{"error": {message, type, param,
        # code}}` envelope (the same one every other /v1 route uses) bypasses
        # the declared `response_model` and is returned as-is, matching every
        # other error path on this server instead of FastAPI's default
        # `{"detail": ...}` shape.
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        # An empty list — or one holding nothing but blank strings — has no text
        # to embed. Reject it with a 400 instead of returning vectors for
        # nothing, the same rule /v1/chat/completions applies to content-free
        # messages. A blank entry alongside real text is still embedded, so a
        # batch caller keeps index alignment.
        if not texts or all(not str(t).strip() for t in texts):
            return JSONResponse(
                status_code=400,
                content=error_envelope(400, "input must not be empty", code="empty_input"),
            )
        try:
            # Resolve the backend first so the fallback status is known even when
            # every vector comes from the cache (a cache hit skips backend
            # creation, so without this the accuracy flag could go stale).
            eng._get_backend(req.model)
            vecs = eng.embed(texts, model=req.model)
        except EmbeddingBackendUnavailable as e:
            # Strict mode: refuse to serve degraded vectors under a neural id.
            return JSONResponse(
                status_code=503,
                content=error_envelope(503, str(e), code="embedding_backend_unavailable"),
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content=error_envelope(500, f"embedding failed: {e}"),
            )
        total_tokens = sum(len(t.split()) for t in texts)
        resolved = eng._resolve_model(req.model)
        degraded = eng.is_fallback(req.model)
        backend = TFIDF_FALLBACK_ID if degraded else "sentence-transformers"
        # Reflect the real backend on a header *and* the body extension so the
        # caller is never fooled into thinking lexical hash vectors are neural.
        response.headers["x-effgen-embedding-backend"] = backend
        return EmbeddingResponse(
            data=[EmbeddingItem(index=i, embedding=v) for i, v in enumerate(vecs)],
            model=req.model,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
            effgen={
                "requested_model": req.model,
                "resolved_model": resolved,
                "backend": backend,
                "degraded": degraded,
            },
        )

    return router


__all__ = [
    "MODEL_ALIASES",
    "TFIDF_FALLBACK_ID",
    "LRUCache",
    "SQLiteCache",
    "TFIDFEmbedder",
    "SentenceTransformerEmbedder",
    "EmbeddingBackendUnavailable",
    "EmbeddingEngine",
    "create_embeddings_router",
]
