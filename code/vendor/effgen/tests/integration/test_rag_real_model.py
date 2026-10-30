"""
End-to-end RAG pipeline tests with a REAL small language model.

**IMPORTANT: These are REAL-MODEL integration tests — NOT mock tests.**

They load an actual SLM (Qwen/Qwen2.5-3B-Instruct) onto a GPU via the
session-scoped `real_model` fixture defined in `tests/integration/conftest.py`,
and verify the full RAG pipeline (ingest → hybrid search → rerank →
context building → grounded generation → citation tracking) using
real model inference.

For fast MOCK-MODEL unit tests of the same pipeline, see
`tests/unit/test_rag.py`.

Uses the session-scoped `real_model` fixture (Qwen/Qwen2.5-3B-Instruct).
Exercises:
  1. DocumentIngester → HybridSearchEngine → LLMReranker → ContextBuilder
  2. Real LLM answering a question grounded in retrieved context
  3. RAG preset agent with knowledge_base auto-ingestion
  4. Citations + source attribution end-to-end

These are slow (model load). Skipped automatically if no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effgen.rag import (
    ContextBuilder,
    DocumentIngester,
    HybridSearchEngine,
    LLMReranker,
    RuleBasedReranker,
)
from effgen.rag.attribution import CitationTracker

# -----------------------------------------------------------------------------
# Fixture: a knowledge base with distinctive content we can ground-truth on
# -----------------------------------------------------------------------------

@pytest.fixture
def knowledge_base(tmp_path_factory) -> Path:
    kb = tmp_path_factory.mktemp("kb")

    (kb / "architecture.md").write_text(
        "# effGen Architecture\n\n"
        "## Scaling\n\n"
        "effGen scales horizontally using stateless agent workers. "
        "Each worker can process requests independently. "
        "The sharding key is the session id.\n\n"
        "## Storage\n\n"
        "Persistent state is stored in SQLite by default, "
        "with optional Postgres for multi-node deployments."
    )
    (kb / "tools.md").write_text(
        "# Built-in Tools\n\n"
        "effGen ships with 14 built-in tools including Calculator, "
        "WebSearch, PythonREPL, and Retrieval. "
        "All tools extend the BaseTool class."
    )
    (kb / "models.md").write_text(
        "# Supported Models\n\n"
        "effGen supports Qwen, Llama, Phi, and Gemma models "
        "via the Transformers and vLLM backends. "
        "The minimum recommended model size is 1B parameters."
    )
    (kb / "unrelated.md").write_text(
        "# Cafeteria Menu\n\n"
        "The company cafeteria serves lunch from 11am to 2pm. "
        "Today's special is lasagna."
    )
    return kb


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestRagRealModel:
    """Real-model RAG pipeline tests."""

    def test_pipeline_retrieves_correct_chunk(self, knowledge_base, real_model):
        """Ingest → search → the top hit for a scaling query must come from architecture.md."""
        chunks = DocumentIngester(show_progress=False).ingest(knowledge_base)
        assert len(chunks) >= 4

        engine = HybridSearchEngine(chunks)
        results = engine.search("How does effGen scale horizontally?", top_k=3)

        assert len(results) == 3
        assert all(r.relevance_score > 0 for r in results)
        # Top result must be from architecture.md — NOT the cafeteria file
        assert "architecture.md" in results[0].source
        assert "unrelated.md" not in results[0].source

    def test_llm_reranker_with_real_model(self, knowledge_base, real_model):
        """LLMReranker with a real SLM should prefer relevant passages."""
        chunks = DocumentIngester(show_progress=False).ingest(knowledge_base)
        engine = HybridSearchEngine(chunks)
        results = engine.search("cafeteria lunch", top_k=4)

        reranker = LLMReranker(real_model)
        ranked = reranker.rerank("cafeteria lunch", results)

        assert len(ranked) == 4
        # All scores should be valid floats in [0, 1]
        assert all(0.0 <= r.relevance_score <= 1.0 for r in ranked)
        # The unrelated.md cafeteria chunk should now be in the top-2
        top2_sources = [r.source for r in ranked[:2]]
        assert any("unrelated.md" in s for s in top2_sources), (
            f"Expected cafeteria chunk in top-2, got: {top2_sources}"
        )

    def test_rag_answer_with_grounded_context(self, knowledge_base, real_model):
        """
        Full pipeline: retrieve context, inject into prompt, have the real
        SLM answer. The answer must reference material only in the KB.
        """
        # 1. Ingest + search
        chunks = DocumentIngester(show_progress=False).ingest(knowledge_base)
        engine = HybridSearchEngine(chunks)
        results = engine.search(
            "What database does effGen use for persistent state?", top_k=3
        )

        # 2. Build context with citations
        built = ContextBuilder(
            max_tokens=600,
            per_source_limit=0,
            include_citations=True,
        ).build(results)
        assert built.text
        assert len(built.citations) >= 1

        # 3. Ask the real model with grounded context
        prompt = (
            "Answer the question using ONLY the provided context. "
            "Cite sources with [N] markers.\n\n"
            f"Context:\n{built.text}\n\n"
            "Question: What database does effGen use for persistent state?\n"
            "Answer:"
        )

        from effgen.models.base import GenerationConfig

        cfg = GenerationConfig(max_tokens=150, temperature=0.1)
        out = real_model.generate(prompt, config=cfg)
        answer = out if isinstance(out, str) else getattr(out, "text", str(out))

        # 4. Verify grounding: SQLite appears in KB, the model should mention it
        assert "sqlite" in answer.lower() or "SQLite" in answer, (
            f"Answer did not mention SQLite. Answer: {answer!r}"
        )

        # 5. Citation tracker extracts markers from the answer
        tracker = CitationTracker(citations=built.citations)
        used_indices = tracker.extract_used_indices(answer)
        # If the model followed instructions, it should have cited something
        if used_indices:
            assert all(1 <= i <= len(built.citations) for i in used_indices)

    def test_rag_preset_agent_ingests_and_retrieves(self, knowledge_base, real_model):
        """create_agent('rag', model, knowledge_base=...) wires everything up."""
        from effgen.presets import create_agent

        agent = create_agent(
            "rag",
            real_model,
            knowledge_base=str(knowledge_base),
            max_iterations=3,
        )

        # Retrieval tool must be populated
        retrieval = next(
            (t for t in agent.tools.values() if t.metadata.name == "retrieval"),
            None,
        )
        assert retrieval is not None
        assert retrieval.num_documents >= 4

        # Direct retrieval call must return architecture material for a
        # scaling query — this verifies the KB was actually indexed.
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            res = loop.run_until_complete(
                retrieval._execute(query="horizontal scaling", top_k=2)
            )
        finally:
            loop.close()
        assert res["total_found"] >= 1
        top = res["results"][0]
        assert "architecture.md" in top["metadata"].get("source", "")

    def test_rule_reranker_composes_with_real_search(self, knowledge_base, real_model):
        """RuleBasedReranker should re-order results while preserving all of them."""
        chunks = DocumentIngester(show_progress=False).ingest(knowledge_base)
        engine = HybridSearchEngine(chunks)
        results = engine.search("supported models", top_k=4)

        original_ids = {r.chunk_id for r in results}
        reranker = RuleBasedReranker()
        ranked = reranker.rerank("supported models Qwen Llama", results)

        assert {r.chunk_id for r in ranked} == original_ids
        # The models.md chunk should rise to the top via keyword boost
        assert "models.md" in ranked[0].source
