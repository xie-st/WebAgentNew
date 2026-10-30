"""Tests for legal domain prompt templates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._harness.live_eval import assert_live_eval_passed

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "legal"
GOLDENS_DIR = Path(__file__).parent / "goldens"


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

class TestFixtures:
    def test_fixtures_exist(self):
        assert FIXTURES_DIR.exists(), f"Fixtures dir missing: {FIXTURES_DIR}"
        txt_files = list(FIXTURES_DIR.glob("*.txt"))
        assert len(txt_files) >= 2, "Expected at least 2 legal fixture files"

    def test_service_agreement_fixture(self):
        p = FIXTURES_DIR / "service_agreement.txt"
        assert p.exists()
        content = p.read_text()
        assert "Acme Corp" in content
        assert "TechVendor" in content

    def test_limitation_clause_fixture(self):
        p = FIXTURES_DIR / "limitation_clause.txt"
        assert p.exists()
        content = p.read_text()
        assert "liable" in content.lower()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestLegalRegistration:
    def test_global_registry_has_legal_prompts(self):
        from effgen.prompts.library.registry import registry
        prompts = registry.search(domain="legal")
        assert len(prompts) == 3, f"Expected exactly 3 legal prompts, got {len(prompts)}"

    def test_all_required_names_present(self):
        from effgen.prompts.library.registry import registry
        names = {p.name for p in registry.search(domain="legal")}
        required = {
            "legal.contract_summarize.v1",
            "legal.clause_classify.v1",
            "legal.legal_research_brief.v1",
        }
        missing = required - names
        assert not missing, f"Missing legal prompts: {missing}"

    def test_variants(self):
        from effgen.prompts.library.registry import registry
        prompts = {p.name: p for p in registry.search(domain="legal")}
        assert prompts["legal.contract_summarize.v1"].variant == "structured"
        assert prompts["legal.clause_classify.v1"].variant == "zero_shot"
        assert prompts["legal.legal_research_brief.v1"].variant == "tool"


# ---------------------------------------------------------------------------
# Disclaimer invariant
# ---------------------------------------------------------------------------

class TestLegalDisclaimerInvariant:
    """Every legal prompt render must contain DISCLAIMERS.legal verbatim."""

    def test_contract_summarize_has_disclaimer(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        rendered = contract_summarize_v1.render_fixture()
        assert DISCLAIMERS.legal in rendered, (
            "DISCLAIMERS.legal not found verbatim in contract_summarize_v1 render"
        )

    def test_clause_classify_has_disclaimer(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.legal.clause_classify_v1 import clause_classify_v1
        rendered = clause_classify_v1.render_fixture()
        assert DISCLAIMERS.legal in rendered, (
            "DISCLAIMERS.legal not found verbatim in clause_classify_v1 render"
        )

    def test_legal_research_brief_has_disclaimer(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.legal.legal_research_brief_v1 import (
            legal_research_brief_v1,
        )
        rendered = legal_research_brief_v1.render_fixture()
        assert DISCLAIMERS.legal in rendered, (
            "DISCLAIMERS.legal not found verbatim in legal_research_brief_v1 render"
        )

    def test_all_legal_prompts_have_disclaimer(self):
        """Registry-level check: every legal prompt render contains the disclaimer."""
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.registry import registry
        failures = []
        for p in registry.search(domain="legal"):
            rendered = p.render_fixture()
            if DISCLAIMERS.legal not in rendered:
                failures.append(p.name)
        assert not failures, f"Missing disclaimer in: {failures}"


# ---------------------------------------------------------------------------
# contract_summarize_v1
# ---------------------------------------------------------------------------

class TestContractSummarize:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        rendered = contract_summarize_v1.render_fixture()
        assert "Acme Corp" in rendered
        assert "TechVendor" in rendered

    def test_json_output_instruction(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        rendered = contract_summarize_v1.render_fixture()
        assert '"parties"' in rendered
        assert '"obligations"' in rendered
        assert '"risks"' in rendered

    def test_no_markdown_fences_instruction(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        rendered = contract_summarize_v1.render_fixture()
        assert "no markdown fences" in rendered.lower()

    def test_variant_is_structured(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        assert contract_summarize_v1.variant == "structured"

    def test_expected_shape_is_callable(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        assert contract_summarize_v1.expected_shape is not None
        assert contract_summarize_v1.expected_shape["type"] == "callable"

    def test_schema_validates_valid_output(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_output = json.dumps({
            "parties": ["Acme Corp", "TechVendor Inc"],
            "term": "12 months from January 1, 2024",
            "obligations": ["Vendor provides software development", "Client pays $10,000/month"],
            "termination": "30 days written notice; immediate for cause",
            "risks": ["Limitation of liability clause", "No specified IP ownership"],
        })
        ok, msg = evaluator._check_shape(valid_output, contract_summarize_v1.expected_shape)
        assert ok, msg

    def test_schema_fails_for_missing_parties(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid = json.dumps({
            "term": "12 months",
            "obligations": ["pay"],
            "termination": "30 days",
            "risks": [],
        })
        ok, _ = evaluator._check_shape(invalid, contract_summarize_v1.expected_shape)
        assert not ok

    def test_custom_contract_renders(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        rendered = contract_summarize_v1.render(contract_text="LEASE AGREEMENT between Landlord LLC and Tenant Inc.")
        assert "LEASE AGREEMENT" in rendered


# ---------------------------------------------------------------------------
# clause_classify_v1
# ---------------------------------------------------------------------------

class TestClauseClassify:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.legal.clause_classify_v1 import clause_classify_v1
        rendered = clause_classify_v1.render_fixture()
        assert "liable" in rendered.lower()
        assert "SaaS" in rendered

    def test_clause_types_listed(self):
        from effgen.prompts.library.domains.legal.clause_classify_v1 import clause_classify_v1
        rendered = clause_classify_v1.render_fixture()
        assert "limitation-of-liability" in rendered
        assert "indemnification" in rendered

    def test_context_included_when_provided(self):
        from effgen.prompts.library.domains.legal.clause_classify_v1 import clause_classify_v1
        rendered = clause_classify_v1.render(
            clause_text="Vendor shall indemnify Client for any third-party claims.",
            context="Enterprise software contract",
        )
        assert "Enterprise software contract" in rendered

    def test_context_omitted_when_empty(self):
        from effgen.prompts.library.domains.legal.clause_classify_v1 import clause_classify_v1
        rendered = clause_classify_v1.render(
            clause_text="Vendor shall indemnify Client.",
            context="",
        )
        assert "Contract context:" not in rendered

    def test_expected_shape_is_regex(self):
        from effgen.prompts.library.domains.legal.clause_classify_v1 import clause_classify_v1
        assert clause_classify_v1.expected_shape["type"] == "regex"

    def test_variant_is_zero_shot(self):
        from effgen.prompts.library.domains.legal.clause_classify_v1 import clause_classify_v1
        assert clause_classify_v1.variant == "zero_shot"


# ---------------------------------------------------------------------------
# legal_research_brief_v1
# ---------------------------------------------------------------------------

class TestLegalResearchBrief:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.legal.legal_research_brief_v1 import (
            legal_research_brief_v1,
        )
        rendered = legal_research_brief_v1.render_fixture()
        assert "non-compete" in rendered.lower()
        assert "California" in rendered

    def test_structure_instructions_present(self):
        from effgen.prompts.library.domains.legal.legal_research_brief_v1 import (
            legal_research_brief_v1,
        )
        rendered = legal_research_brief_v1.render_fixture()
        for section in ("Issue", "Applicable Law", "Analysis", "Conclusion", "Open Questions"):
            assert section in rendered, f"Missing section: {section}"

    def test_sources_block_included_when_provided(self):
        from effgen.prompts.library.domains.legal.legal_research_brief_v1 import (
            legal_research_brief_v1,
        )
        rendered = legal_research_brief_v1.render_fixture()
        assert "§ 16600" in rendered

    def test_sources_block_note_when_empty(self):
        from effgen.prompts.library.domains.legal.legal_research_brief_v1 import (
            legal_research_brief_v1,
        )
        rendered = legal_research_brief_v1.render(
            topic="Force majeure clauses",
            jurisdiction="US federal",
            retrieved_sources="",
        )
        assert "No pre-retrieved sources" in rendered

    def test_variant_is_tool(self):
        from effgen.prompts.library.domains.legal.legal_research_brief_v1 import (
            legal_research_brief_v1,
        )
        assert legal_research_brief_v1.variant == "tool"

    def test_expected_shape_is_regex(self):
        from effgen.prompts.library.domains.legal.legal_research_brief_v1 import (
            legal_research_brief_v1,
        )
        assert legal_research_brief_v1.expected_shape["type"] == "regex"


# ---------------------------------------------------------------------------
# Golden eval
# ---------------------------------------------------------------------------

class TestGoldenEval:
    def test_all_goldens_pass(self):
        from effgen.prompts.library.eval import PromptEval
        from effgen.prompts.library.registry import registry

        evaluator = PromptEval()
        prompts = registry.search(domain="legal")
        assert len(prompts) >= 3

        report = evaluator.eval_all_golden(prompts)
        failed = report.failed()
        assert not failed, f"Golden failures: {[r.name + ': ' + r.message for r in failed]}"


# ---------------------------------------------------------------------------
# Live eval
# ---------------------------------------------------------------------------

class TestLiveEval:
    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_contract_summarize_live_json_schema_valid(self):
        from effgen.prompts.library.domains.legal.contract_summarize_v1 import contract_summarize_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(contract_summarize_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        parsed = PromptEval._extract_json_object(result.model_output)
        assert parsed is not None, f"No JSON found: {result.model_output[:300]}"
        assert "parties" in parsed
        assert "obligations" in parsed

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_clause_classify_live_mentions_type(self):
        from effgen.prompts.library.domains.legal.clause_classify_v1 import clause_classify_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(clause_classify_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_legal_research_brief_live_has_sections(self):
        from effgen.prompts.library.domains.legal.legal_research_brief_v1 import (
            legal_research_brief_v1,
        )
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(legal_research_brief_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
