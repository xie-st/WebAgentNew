"""Retrieval breadth: diversity re-ranking and configurable defaults.

Deterministic and offline — a hand-built embedding provider returns fixed
vectors so the ranking is exact, with no model download.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from effgen.tools.builtin.retrieval import EmbeddingProvider, Retrieval


class _FixedEmbedding(EmbeddingProvider):
    """Return a preset vector per text; unknown text embeds to zeros."""

    def __init__(self, table: dict[str, list[float]]):
        self._table = {k: np.asarray(v, dtype=float) for k, v in table.items()}
        self._dim = len(next(iter(table.values())))

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([
            self._table.get(t, np.zeros(self._dim)) for t in texts
        ])


# Three near-identical "topic A" chunks and one "topic B" chunk. The query is
# closest to A but B is also relevant. Pure relevance fills top-2 with A chunks;
# diversity trades a redundant A for the distinct B.
_A1 = "topic A passage one"
_A2 = "topic A passage two"
_A3 = "topic A passage three"
_B = "topic B passage"
_QUERY = "query about A and B"

_TABLE = {
    _A1: [1.0, 0.0, 0.0],
    _A2: [0.98, 0.2, 0.0],
    _A3: [0.96, 0.28, 0.0],
    _B: [0.0, 0.0, 1.0],
    _QUERY: [0.9, 0.0, 0.44],
}


def _build() -> Retrieval:
    t = Retrieval(embedding_provider=_FixedEmbedding(_TABLE), enable_hybrid_search=False)
    t.add_documents(
        [
            {"content": _A1, "id": "a1"},
            {"content": _A2, "id": "a2"},
            {"content": _A3, "id": "a3"},
            {"content": _B, "id": "b"},
        ],
        chunk=False,
    )
    return t


def _run(tool: Retrieval, **kw):
    return asyncio.new_event_loop().run_until_complete(tool._execute(**kw))


def test_pure_relevance_is_default_and_unchanged():
    tool = _build()
    r = _run(tool, query=_QUERY, top_k=2)
    ids = [x["id"] for x in r["results"]]
    # Default diversity 0: the two highest-relevance (both topic A) chunks.
    assert ids == ["a1", "a2"]


def test_diversity_surfaces_distinct_topic():
    tool = _build()
    r = _run(tool, query=_QUERY, top_k=2, diversity=0.7)
    ids = [x["id"] for x in r["results"]]
    # The redundant second A chunk is traded for the distinct B chunk.
    assert ids[0] == "a1"
    assert "b" in ids


def test_default_top_k_is_configurable():
    tool = Retrieval(
        embedding_provider=_FixedEmbedding(_TABLE),
        enable_hybrid_search=False,
        default_top_k=3,
    )
    tool.add_documents(
        [{"content": c, "id": i} for c, i in [(_A1, "a1"), (_A2, "a2"), (_A3, "a3"), (_B, "b")]],
        chunk=False,
    )
    r = _run(tool, query=_QUERY)  # no top_k passed
    assert len(r["results"]) == 3


def test_configure_updates_defaults_and_schema():
    tool = _build()
    tool.configure(default_top_k=4, diversity=0.5)
    assert tool.default_top_k == 4
    assert tool.diversity == 0.5
    specs = {p.name: p for p in tool.metadata.parameters}
    assert specs["top_k"].default == 4
    assert specs["diversity"].default == 0.5


def test_diversity_clamped_to_unit_interval():
    tool = _build()
    r = _run(tool, query=_QUERY, top_k=2, diversity=5.0)
    assert len(r["results"]) == 2  # out-of-range value does not crash


@pytest.mark.parametrize("div", [0.0, 0.3, 1.0])
def test_result_count_never_exceeds_top_k(div):
    tool = _build()
    r = _run(tool, query=_QUERY, top_k=2, diversity=div)
    assert len(r["results"]) <= 2
