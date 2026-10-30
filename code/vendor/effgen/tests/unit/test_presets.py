"""Tests for preset registry and create_agent factory."""


import pytest

from effgen.presets import registry as _preset_registry
from effgen.presets.registry import (
    PRESETS,
    PresetConfig,
    create_agent,
    get_preset,
    list_presets,
    preset_tool_overhead,
)
from tests.fixtures.mock_models import MockModel

EXPECTED_PRESETS = {
    "math",
    "research",
    "coding",
    "general",
    "rag",
    "minimal",
    "media",
    "notify",
    "multimodal",
}


class TestListPresets:
    """Test list_presets()."""

    def test_returns_all_presets(self):
        presets = list_presets()
        assert set(presets.keys()) == EXPECTED_PRESETS

    def test_returns_descriptions(self):
        presets = list_presets()
        for _name, desc in presets.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestGetPreset:
    """Test get_preset()."""

    def test_valid_preset(self):
        cfg = get_preset("math")
        assert isinstance(cfg, PresetConfig)
        assert cfg.name == "math"

    def test_invalid_preset_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("invalid_preset_name")

    def test_math_has_calculator(self):
        cfg = get_preset("math")
        assert "calculator" in cfg.tool_names

    def test_minimal_has_no_tools(self):
        cfg = get_preset("minimal")
        assert cfg.tool_names == []


class TestCreateAgent:
    """Test create_agent() factory."""

    def _mock_model(self):
        return MockModel(responses=["Thought: done\nFinal Answer: ok"])

    def test_create_math_agent(self):
        model = self._mock_model()
        agent = create_agent("math", model)
        assert agent is not None
        assert agent.config.name == "math-agent"

    def test_create_minimal_agent(self):
        model = self._mock_model()
        agent = create_agent("minimal", model)
        assert agent.config.tools == []

    def test_invalid_preset_raises(self):
        model = self._mock_model()
        with pytest.raises(KeyError, match="Unknown preset"):
            create_agent("invalid_preset_name", model)

    def test_custom_agent_name(self):
        model = self._mock_model()
        agent = create_agent("minimal", model, agent_name="custom-name")
        assert agent.config.name == "custom-name"

    def test_custom_max_iterations(self):
        model = self._mock_model()
        agent = create_agent("minimal", model, max_iterations=99)
        assert agent.config.max_iterations == 99

    def test_create_all_presets(self):
        """Verify all presets can be created without error."""
        model = self._mock_model()
        for name in EXPECTED_PRESETS:
            # The rag preset requires knowledge_base= (see TestRagKnowledgeBase);
            # every other preset needs only a model.
            kwargs = {"knowledge_base": "placeholder text."} if name == "rag" else {}
            agent = create_agent(name, model, **kwargs)
            assert agent is not None

    def test_unknown_preset_reported_before_missing_model(self):
        """create_agent('legal') (an invalid preset) must report the bad preset
        up front, not first demand a model for a preset that can't exist."""
        with pytest.raises(KeyError, match="Unknown preset"):
            create_agent("legal")  # no model passed on purpose

    def test_valid_preset_no_model_still_asks_for_model(self):
        with pytest.raises(ValueError, match="needs a model"):
            create_agent("math")

    def test_no_preset_and_no_domain_raises(self):
        with pytest.raises(TypeError, match="preset name"):
            create_agent(model=self._mock_model())


class TestToolSchemaOverhead:
    """A tool-heavy preset carries a fixed per-call schema cost; it is
    discoverable and flagged so a user does not hit it by surprise on a
    small-context or rate-limited model."""

    def test_overhead_reports_count_and_tokens(self):
        n_tools, approx = preset_tool_overhead("general")
        assert n_tools >= 20  # the broad "kitchen sink" preset
        assert approx > 4000  # its schemas dominate a small model's budget

    def test_minimal_has_no_overhead(self):
        n_tools, approx = preset_tool_overhead("minimal")
        assert n_tools == 0
        assert approx == 0

    def test_lean_preset_reports_modest_overhead(self):
        n_tools, approx = preset_tool_overhead("math")
        assert 0 < n_tools < 5
        assert approx < 4000

    def test_heads_up_fires_once_for_heavy_preset(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(_preset_registry, "_tool_overhead_warned", set())
        tools = _preset_registry._instantiate_tools(get_preset("general").tool_names)
        with caplog.at_level(logging.WARNING, logger=_preset_registry.logger.name):
            _preset_registry._warn_tool_schema_overhead("general", tools)
            _preset_registry._warn_tool_schema_overhead("general", tools)
        hits = [r for r in caplog.records if "tool schema" in r.getMessage()]
        assert len(hits) == 1
        assert "31" in hits[0].getMessage() or "tools" in hits[0].getMessage()

    def test_heads_up_silent_for_lean_preset(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(_preset_registry, "_tool_overhead_warned", set())
        tools = _preset_registry._instantiate_tools(get_preset("math").tool_names)
        with caplog.at_level(logging.WARNING, logger=_preset_registry.logger.name):
            _preset_registry._warn_tool_schema_overhead("math", tools)
        assert not [r for r in caplog.records if "tool schema" in r.getMessage()]


class TestDomainAgentBridge:
    """A Domain must have one obvious on-ramp to a runnable agent."""

    def _mock_model(self):
        return MockModel(responses=["Thought: done\nFinal Answer: ok"])

    def test_domain_to_agent_wires_prompt_tools_guardrails(self):
        from effgen.domains import LegalDomain

        agent = LegalDomain().to_agent(self._mock_model())
        assert agent.config.name == "legal-agent"
        assert agent.config.system_prompt.startswith("You are a legal")
        names = [t.metadata.name for t in agent.config.tools]
        assert "web_search" in names and "wikipedia" in names
        # The domain's guardrails ("standard") get their first real consumer.
        assert agent._guardrail_chain is not None

    def test_create_agent_domain_kwarg_equivalent(self):
        from effgen.domains import LegalDomain
        from effgen.presets import create_agent as ca

        d = LegalDomain()
        agent = ca(domain=d, model=self._mock_model())
        assert agent.config.system_prompt == d.system_prompt

    def test_domain_without_guardrails_has_no_chain(self):
        from effgen.domains import TechDomain

        agent = TechDomain().to_agent(self._mock_model())
        assert agent._guardrail_chain is None

    def test_to_agent_overrides_apply(self):
        from effgen.domains import LegalDomain

        agent = LegalDomain().to_agent(
            self._mock_model(), temperature=0.1, max_iterations=3,
            system_prompt="custom",
        )
        assert agent.config.temperature == 0.1
        assert agent.config.max_iterations == 3
        assert agent.config.system_prompt == "custom"

    def test_to_agent_extra_tools_by_name(self):
        from effgen.domains import LegalDomain

        agent = LegalDomain().to_agent(self._mock_model(), extra_tools=["calculator"])
        names = [t.metadata.name for t in agent.config.tools]
        assert "calculator" in names

    def test_preset_and_domain_together_raises(self):
        from effgen.domains import LegalDomain
        from effgen.presets import create_agent as ca

        with pytest.raises(TypeError, match="both"):
            ca("math", self._mock_model(), domain=LegalDomain())

    def test_domain_kwarg_rejects_non_domain(self):
        from effgen.presets import create_agent as ca

        with pytest.raises(TypeError, match="Domain instance"):
            ca(domain="legal", model=self._mock_model())

    def test_domain_no_model_raises_domain_aware(self):
        from effgen.domains import LegalDomain

        with pytest.raises(ValueError, match="to_agent"):
            LegalDomain().to_agent()


class TestPresetCitationMarkers:
    """A preset that asks for inline markers must also turn them on.

    Markers are opt-in per run, and a run that did not ask for them has any it
    produced removed from the answer. A preset whose system prompt orders the
    model to write ``[1]``, ``[2]`` therefore pays for markers and then loses
    them, and the caller has no way to keep them.
    """

    def _mock_model(self):
        return MockModel(responses=["Thought: done\nFinal Answer: ok"])

    def test_the_rag_preset_asks_for_markers_and_enables_them(self):
        assert PRESETS["rag"].cite_sources is True
        assert "[1]" in PRESETS["rag"].system_prompt
        agent = create_agent(
            "rag", self._mock_model(), knowledge_base="Canberra is the capital."
        )
        assert agent.config.cite_sources is True

    def test_the_rag_prompt_leaves_the_wording_to_the_question(self):
        assert "in your own words" not in PRESETS["rag"].system_prompt.lower()

    @pytest.mark.parametrize(
        "name", [n for n in PRESETS if n != "rag"]
    )
    def test_every_other_preset_leaves_markers_off(self, name):
        assert PRESETS[name].cite_sources is False

    def test_an_explicit_setting_still_wins(self):
        agent = create_agent(
            "rag", self._mock_model(), knowledge_base="Canberra is the capital.",
            cite_sources=False,
        )
        assert agent.config.cite_sources is False


class TestRagKnowledgeBase:
    """create_agent('rag', knowledge_base=...) must work or fail loudly."""

    def _mock_model(self):
        return MockModel(responses=["Thought: done\nFinal Answer: ok"])

    def _retrieval_tool(self, agent):
        return next(
            (t for t in agent.config.tools if t.metadata.name == "retrieval"),
            None,
        )

    def test_raw_text_list_is_indexed(self):
        """The obvious call — raw text strings — populates the index."""
        import asyncio

        agent = create_agent(
            "rag",
            self._mock_model(),
            knowledge_base=[
                "The Eiffel Tower is 330 meters tall and located in Paris.",
                "The Great Wall of China is over 21,000 kilometers long.",
            ],
        )
        rt = self._retrieval_tool(agent)
        assert rt is not None
        res = asyncio.run(
            rt.execute(operation="search", query="How tall is the Eiffel Tower?", top_k=2)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert results, "expected the inline knowledge base to be retrievable"
        assert any("330 meters" in r["content"] for r in results)

    def test_single_raw_string_is_indexed(self):
        import asyncio

        agent = create_agent(
            "rag",
            self._mock_model(),
            knowledge_base="The capital of Australia is Canberra.",
        )
        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="capital of Australia", top_k=1)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Canberra" in r["content"] for r in results)

    def test_file_path_still_works(self, tmp_path):
        import asyncio

        p = tmp_path / "facts.txt"
        p.write_text("Mercury is the closest planet to the Sun.", encoding="utf-8")
        agent = create_agent(
            "rag", self._mock_model(), knowledge_base=str(p)
        )
        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="closest planet to the Sun", top_k=1)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Mercury" in r["content"] for r in results)

    def test_pdf_knowledge_base_is_indexed(self, tmp_path):
        """A PDF knowledge base ingests out of the box (pypdf/pdfplumber),
        not only when pymupdf happens to be installed."""
        import asyncio

        pytest.importorskip("reportlab")
        pytest.importorskip("pypdf")
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = tmp_path / "facts.pdf"
        c = canvas.Canvas(str(pdf), pagesize=letter)
        c.drawString(72, 720, "Neptune is the farthest planet from the Sun.")
        c.save()

        agent = create_agent("rag", self._mock_model(), knowledge_base=str(pdf))
        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="farthest planet", top_k=1)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Neptune" in r["content"] for r in results)

    def test_unparseable_file_surfaces_skip_reason(self, tmp_path):
        """When nothing indexes, the error names *why* each file was skipped
        instead of a bare "0 documents"."""
        pytest.importorskip("pypdf")
        bad = tmp_path / "broken.pdf"
        bad.write_text("not a real pdf")
        with pytest.raises(ValueError, match="Skipped:.*broken.pdf"):
            create_agent("rag", self._mock_model(), knowledge_base=str(bad))

    def test_partial_ingestion_warns_about_skipped_file(self, tmp_path):
        """A good file plus a corrupt one must still index the good file, but
        the skip must not be silent — a caller querying an incomplete corpus
        needs to know before it gets a confidently incomplete answer."""
        pytest.importorskip("pypdf")
        import asyncio

        good = tmp_path / "facts.txt"
        good.write_text("Mercury is the closest planet to the Sun.", encoding="utf-8")
        bad = tmp_path / "broken.pdf"
        bad.write_text("not a real pdf")

        with pytest.warns(RuntimeWarning, match="1 file.*skipped.*broken.pdf"):
            agent = create_agent(
                "rag", self._mock_model(), knowledge_base=[str(good), str(bad)]
            )

        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="closest planet to the Sun", top_k=1)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Mercury" in r["content"] for r in results)

    def test_full_ingestion_success_emits_no_warning(self, tmp_path):
        """No files skipped -> no warning noise on the common happy path."""
        import warnings as _warnings

        good = tmp_path / "facts.txt"
        good.write_text("Mercury is the closest planet to the Sun.", encoding="utf-8")

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            create_agent("rag", self._mock_model(), knowledge_base=str(good))

    def test_empty_knowledge_base_fails_loud(self):
        """Blank entries must raise, never silently build an empty index."""
        with pytest.raises(ValueError, match="0 documents"):
            create_agent(
                "rag", self._mock_model(), knowledge_base=["", "   "]
            )

    def test_missing_knowledge_base_fails_as_loud_as_empty(self):
        """Omitting `knowledge_base=` entirely is as plausible a mistake as
        passing an empty one, and must fail the same way — never build a
        retrieval agent with zero documents indexed."""
        with pytest.raises(ValueError, match="knowledge_base"):
            create_agent("rag", self._mock_model())

    def test_missing_knowledge_base_error_points_at_cli_file_flag(self):
        """A CLI user must be pointed at the working `--file` path, not only the
        Python `knowledge_base=` argument."""
        with pytest.raises(ValueError, match="--preset rag --file"):
            create_agent("rag", self._mock_model())

    def test_typod_path_fails_loud_not_indexed_literally(self):
        """A path-like entry that doesn't exist is a typo, not a document:
        it must raise, never index the literal path string as a 1-doc index."""
        with pytest.raises(ValueError, match="looks like a file"):
            create_agent(
                "rag", self._mock_model(),
                knowledge_base=["/nonexistent/path/xyz.txt"],
            )

    def test_single_word_inline_text_is_not_treated_as_path(self):
        """A bare word with no separator/extension stays inline (not a path)."""
        agent = create_agent(
            "rag", self._mock_model(), knowledge_base="Canberra"
        )
        rt = self._retrieval_tool(agent)
        assert rt is not None

    def test_empty_string_does_not_ingest_cwd(self):
        """A "" entry must not resolve to Path('.') and ingest the whole tree."""
        with pytest.raises(ValueError, match="0 documents"):
            create_agent("rag", self._mock_model(), knowledge_base="")

    def test_ingest_text_chunks_directly(self):
        from effgen.rag import DocumentIngester

        chunks = DocumentIngester(show_progress=False).ingest_text(
            "Hello world. This is an inline document."
        )
        assert chunks
        assert all(c.content.strip() for c in chunks)
        assert chunks[0].source.startswith("inline:")

    def test_ingest_text_blank_returns_empty(self):
        from effgen.rag import DocumentIngester

        assert DocumentIngester(show_progress=False).ingest_text("   ") == []

    def test_prebuilt_vector_memory_store_is_indexed(self):
        """A pre-built VectorMemoryStore connects straight to a RAG agent —
        the memory/ and rag/ retrieval subsystems link up."""
        pytest.importorskip("faiss")
        import asyncio

        from effgen.memory import VectorMemoryStore

        store = VectorMemoryStore()
        store.add(
            "The Eiffel Tower is 330 meters tall and located in Paris.",
            metadata={"topic": "landmark"},
        )
        store.add("The Great Wall of China is over 21,000 kilometers long.")

        agent = create_agent("rag", self._mock_model(), knowledge_base=store)
        rt = self._retrieval_tool(agent)
        assert rt is not None
        res = asyncio.run(
            rt.execute(operation="search", query="How tall is the Eiffel Tower?", top_k=2)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("330 meters" in r["content"] for r in results)

    def test_vector_store_mixed_with_text(self):
        """A VectorMemoryStore can be mixed with raw text in one list."""
        pytest.importorskip("faiss")
        import asyncio

        from effgen.memory import VectorMemoryStore

        store = VectorMemoryStore()
        store.add("Mercury is the closest planet to the Sun.")
        agent = create_agent(
            "rag",
            self._mock_model(),
            knowledge_base=[store, "Neptune is the farthest planet from the Sun."],
        )
        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="closest planet", top_k=2)
        )
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Mercury" in r["content"] for r in results)

    def test_empty_vector_store_fails_loud(self):
        """An empty VectorMemoryStore must raise, never build an empty index."""
        pytest.importorskip("faiss")
        from effgen.memory import VectorMemoryStore

        with pytest.raises(ValueError, match="0 documents"):
            create_agent("rag", self._mock_model(), knowledge_base=VectorMemoryStore())
