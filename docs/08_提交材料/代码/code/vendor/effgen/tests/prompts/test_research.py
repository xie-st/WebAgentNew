"""Tests for research domain prompt templates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._harness.live_eval import assert_live_eval_passed

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "research"
GOLDENS_DIR = Path(__file__).parent / "goldens"


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def get_registry():
    from effgen.prompts.library.registry import PromptRegistry
    r = PromptRegistry()
    # Force import of research domain
    import effgen.prompts.library.domains.research  # noqa: F401
    r._discovered = True
    # Re-register by importing individual modules
    from effgen.prompts.library.domains.research.citation_extract_v1 import citation_extract_v1
    from effgen.prompts.library.domains.research.literature_review_v1 import (
        lit_review_cot,
        lit_review_zero_shot,
    )
    from effgen.prompts.library.domains.research.methodology_critique_v1 import (
        methodology_critique_v1,
    )
    from effgen.prompts.library.domains.research.paper_summary_v1 import paper_summary_v1
    r.register(lit_review_zero_shot)
    r.register(lit_review_cot)
    r.register(paper_summary_v1)
    r.register(citation_extract_v1)
    r.register(methodology_critique_v1)
    return r


# ---------------------------------------------------------------------------
# Fixtures dir
# ---------------------------------------------------------------------------

class TestFixtures:
    def test_fixtures_exist(self):
        assert FIXTURES_DIR.exists(), f"Fixtures dir missing: {FIXTURES_DIR}"
        txt_files = list(FIXTURES_DIR.glob("*.txt"))
        assert len(txt_files) >= 2, "Expected at least 2 fixture files"

    def test_attention_fixture(self):
        p = FIXTURES_DIR / "attention_abstract.txt"
        assert p.exists()
        content = p.read_text()
        assert "Attention" in content
        assert "Transformer" in content

    def test_diffusion_fixture(self):
        p = FIXTURES_DIR / "diffusion_abstract.txt"
        assert p.exists()
        content = p.read_text()
        assert "Diffusion" in content or "diffusion" in content


# ---------------------------------------------------------------------------
# Template registration
# ---------------------------------------------------------------------------

class TestResearchRegistration:
    def test_global_registry_has_research_prompts(self):
        from effgen.prompts.library.registry import registry
        prompts = registry.search(domain="research")
        assert len(prompts) >= 4, f"Expected ≥4 research prompts, got {len(prompts)}"

    def test_all_required_names_present(self):
        from effgen.prompts.library.registry import registry
        names = {p.name for p in registry.search(domain="research")}
        required = {
            "research.literature_review.v1.zero_shot",
            "research.literature_review.v1.cot",
            "research.paper_summary.v1",
            "research.citation_extract.v1",
            "research.methodology_critique.v1",
        }
        missing = required - names
        assert not missing, f"Missing research prompts: {missing}"

    def test_variants(self):
        from effgen.prompts.library.registry import registry
        prompts = {p.name: p for p in registry.search(domain="research")}
        assert prompts["research.literature_review.v1.zero_shot"].variant == "zero_shot"
        assert prompts["research.literature_review.v1.cot"].variant == "cot"
        assert prompts["research.paper_summary.v1"].variant == "structured"
        assert prompts["research.citation_extract.v1"].variant == "tool"
        assert prompts["research.methodology_critique.v1"].variant == "cot"


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

class TestLiteratureReview:
    def test_zero_shot_renders(self):
        from effgen.prompts.library.domains.research.literature_review_v1 import (
            lit_review_zero_shot,
        )
        rendered = lit_review_zero_shot.render_fixture()
        assert "diffusion models" in rendered.lower()
        assert "2022-2024" in rendered
        assert "10" in rendered

    def test_cot_renders(self):
        from effgen.prompts.library.domains.research.literature_review_v1 import lit_review_cot
        rendered = lit_review_cot.render_fixture()
        assert "step" in rendered.lower()
        assert "diffusion models" in rendered.lower()

    def test_cot_has_numbered_steps(self):
        from effgen.prompts.library.domains.research.literature_review_v1 import lit_review_cot
        rendered = lit_review_cot.render_fixture()
        assert "1." in rendered
        assert "2." in rendered
        assert "3." in rendered

    def test_live_shape_requires_three_candidate_paper_lines(self):
        from effgen.prompts.library.domains.research.literature_review_v1 import (
            lit_review_zero_shot,
        )
        from effgen.prompts.library.eval import PromptEval

        output = (
            "1. Ho et al. (2020), Denoising Diffusion Probabilistic Models.\n"
            "2. Dhariwal and Nichol (2021), Diffusion Models Beat GANs.\n"
            "3. Rombach et al. (2022), High-Resolution Image Synthesis with Latent Diffusion Models."
        )
        evaluator = PromptEval()
        ok, msg = evaluator._check_shape(output, lit_review_zero_shot.expected_shape)
        assert ok, msg

        ok, msg = evaluator._check_shape("Overview without concrete papers.", lit_review_zero_shot.expected_shape)
        assert not ok
        assert "3 candidate" in msg

    def test_custom_inputs(self):
        from effgen.prompts.library.domains.research.literature_review_v1 import (
            lit_review_zero_shot,
        )
        rendered = lit_review_zero_shot.render(
            topic="protein folding", years_range="2020-2023", max_papers=5
        )
        assert "protein folding" in rendered
        assert "2020-2023" in rendered
        assert "5" in rendered


class TestPaperSummary:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.research.paper_summary_v1 import paper_summary_v1
        rendered = paper_summary_v1.render_fixture()
        assert "Attention Is All You Need" in rendered
        assert "abstract_summary" in rendered
        assert "key_findings" in rendered
        assert "limitations" in rendered
        assert "future_work" in rendered

    def test_output_schema_has_required_keys(self):
        from effgen.prompts.library.domains.research.paper_summary_v1 import paper_summary_v1
        assert paper_summary_v1.expected_shape is not None
        assert paper_summary_v1.expected_shape["type"] == "json"
        required = paper_summary_v1.expected_shape["schema"]["required"]
        assert "abstract_summary" in required
        assert "key_findings" in required
        assert "limitations" in required
        assert "future_work" in required

    def test_no_markdown_fences_in_prompt(self):
        from effgen.prompts.library.domains.research.paper_summary_v1 import paper_summary_v1
        rendered = paper_summary_v1.render_fixture()
        assert "no markdown fences" in rendered.lower()

    def test_author_optional(self):
        from effgen.prompts.library.domains.research.paper_summary_v1 import paper_summary_v1
        rendered = paper_summary_v1.render(
            title="Test Paper",
            abstract="This paper studies X.",
        )
        assert "Test Paper" in rendered


class TestCitationExtract:
    def test_renders_without_error(self):
        from effgen.prompts.library.domains.research.citation_extract_v1 import citation_extract_v1
        rendered = citation_extract_v1.render_fixture()
        assert "Query:" in rendered
        assert "transformer" in rendered.lower()
        assert "Tool use required" in rendered

    def test_contains_source_label(self):
        from effgen.prompts.library.domains.research.citation_extract_v1 import citation_extract_v1
        rendered = citation_extract_v1.render_fixture()
        assert "ARXIV" in rendered or "arxiv" in rendered.lower()

    def test_pubmed_source(self):
        from effgen.prompts.library.domains.research.citation_extract_v1 import citation_extract_v1
        rendered = citation_extract_v1.render(
            query="CRISPR gene editing", source="pubmed", max_results=2
        )
        assert "PUBMED" in rendered or "pubmed" in rendered.lower()
        assert "PubMedTool" in rendered

    def test_variant_is_tool(self):
        from effgen.prompts.library.domains.research.citation_extract_v1 import citation_extract_v1
        assert citation_extract_v1.variant == "tool"


class TestMethodologyCritique:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.research.methodology_critique_v1 import (
            methodology_critique_v1,
        )
        rendered = methodology_critique_v1.render_fixture()
        assert "educational technology" in rendered
        assert "sampling" in rendered.lower()
        assert "design" in rendered.lower()

    def test_has_numbered_dimensions(self):
        from effgen.prompts.library.domains.research.methodology_critique_v1 import (
            methodology_critique_v1,
        )
        rendered = methodology_critique_v1.render_fixture()
        for dim in ["Study Design", "Sampling", "Measurement", "Analysis"]:
            assert dim in rendered

    def test_custom_field(self):
        from effgen.prompts.library.domains.research.methodology_critique_v1 import (
            methodology_critique_v1,
        )
        rendered = methodology_critique_v1.render(
            methodology_text="RCT with 200 participants.",
            field="clinical medicine",
        )
        assert "clinical medicine" in rendered


# ---------------------------------------------------------------------------
# Golden eval tests
# ---------------------------------------------------------------------------

class TestGoldenEval:
    def test_all_goldens_pass(self):
        from effgen.prompts.library.eval import PromptEval
        from effgen.prompts.library.registry import registry

        evaluator = PromptEval()
        prompts = registry.search(domain="research")
        assert len(prompts) >= 4

        report = evaluator.eval_all_golden(prompts)
        failed = report.failed()
        assert not failed, f"Golden failures: {[r.name + ': ' + r.message for r in failed]}"


# ---------------------------------------------------------------------------
# Live eval (skipped unless CEREBRAS_API_KEY is set)
# ---------------------------------------------------------------------------

class TestLiveEval:
    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_literature_review_live_has_three_candidate_papers(self):
        """literature_review_v1 live output must include at least 3 paper-like lines."""
        from effgen.prompts.library.domains.research.literature_review_v1 import (
            lit_review_zero_shot,
        )
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(lit_review_zero_shot, model="gpt-oss-120b")
        assert_live_eval_passed(result)

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_paper_summary_live_json(self):
        """paper_summary_v1 must produce valid JSON with all 4 required keys."""
        from effgen.prompts.library.domains.research.paper_summary_v1 import paper_summary_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(paper_summary_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        # Also assert JSON parseable with required keys
        output = result.model_output.strip()
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", output, re.DOTALL)
            assert m, f"No JSON found in output: {output[:300]}"
            parsed = json.loads(m.group(0))
        for key in ["abstract_summary", "key_findings", "limitations", "future_work"]:
            assert key in parsed, f"Missing key '{key}' in output"
