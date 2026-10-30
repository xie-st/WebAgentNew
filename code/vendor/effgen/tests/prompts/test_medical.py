"""Tests for medical domain prompt templates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._harness.live_eval import assert_live_eval_passed

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "medical"
GOLDENS_DIR = Path(__file__).parent / "goldens"

_TRANSIENT_LIVE_MARKERS = (
    "429",
    "high traffic",
    "queue_exceeded",
    "rate limit",
    "request_quota_exceeded",
    "too_many_requests",
)


def _skip_transient_live_error(message: str) -> None:
    lowered = message.lower()
    if any(marker in lowered for marker in _TRANSIENT_LIVE_MARKERS):
        pytest.skip(f"Transient live provider pressure: {message}")


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

class TestFixtures:
    def test_fixtures_exist(self):
        assert FIXTURES_DIR.exists(), f"Fixtures dir missing: {FIXTURES_DIR}"
        txt_files = list(FIXTURES_DIR.glob("*.txt"))
        assert len(txt_files) >= 2, "Expected at least 2 medical fixture files"

    def test_symptom_vignette_fixture(self):
        p = FIXTURES_DIR / "symptom_vignette.txt"
        assert p.exists()
        content = p.read_text()
        assert "fever" in content.lower() or "symptoms" in content.lower()

    def test_drug_list_fixture(self):
        p = FIXTURES_DIR / "drug_list.txt"
        assert p.exists()
        content = p.read_text()
        assert "warfarin" in content.lower()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestMedicalRegistration:
    def test_global_registry_has_medical_prompts(self):
        from effgen.prompts.library.registry import registry
        prompts = registry.search(domain="medical")
        assert len(prompts) == 3, f"Expected exactly 3 medical prompts, got {len(prompts)}"

    def test_all_required_names_present(self):
        from effgen.prompts.library.registry import registry
        names = {p.name for p in registry.search(domain="medical")}
        required = {
            "medical.symptom_triage.v1",
            "medical.drug_interaction_query.v1",
            "medical.medical_literature.v1",
        }
        missing = required - names
        assert not missing, f"Missing medical prompts: {missing}"

    def test_variants(self):
        from effgen.prompts.library.registry import registry
        prompts = {p.name: p for p in registry.search(domain="medical")}
        assert prompts["medical.symptom_triage.v1"].variant == "structured"
        assert prompts["medical.drug_interaction_query.v1"].variant == "structured"
        assert prompts["medical.medical_literature.v1"].variant == "tool"


# ---------------------------------------------------------------------------
# Disclaimer invariant
# ---------------------------------------------------------------------------

class TestMedicalDisclaimerInvariant:
    """Every medical prompt render must contain DISCLAIMERS.medical verbatim."""

    def test_symptom_triage_has_disclaimer(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        rendered = symptom_triage_v1.render_fixture()
        assert DISCLAIMERS.medical in rendered, (
            "DISCLAIMERS.medical not found verbatim in symptom_triage_v1 render"
        )

    def test_drug_interaction_has_disclaimer(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        rendered = drug_interaction_query_v1.render_fixture()
        assert DISCLAIMERS.medical in rendered, (
            "DISCLAIMERS.medical not found verbatim in drug_interaction_query_v1 render"
        )

    def test_medical_literature_has_disclaimer(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.medical.medical_literature_v1 import (
            medical_literature_v1,
        )
        rendered = medical_literature_v1.render_fixture()
        assert DISCLAIMERS.medical in rendered, (
            "DISCLAIMERS.medical not found verbatim in medical_literature_v1 render"
        )

    def test_all_medical_prompts_have_disclaimer(self):
        """Registry-level check: every medical prompt render contains the disclaimer."""
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.registry import registry
        failures = []
        for p in registry.search(domain="medical"):
            rendered = p.render_fixture()
            if DISCLAIMERS.medical not in rendered:
                failures.append(p.name)
        assert not failures, f"Missing disclaimer in: {failures}"


# ---------------------------------------------------------------------------
# symptom_triage_v1
# ---------------------------------------------------------------------------

class TestSymptomTriage:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        rendered = symptom_triage_v1.render_fixture()
        assert "fever" in rendered.lower() or "38.5" in rendered
        assert "urgency_level" in rendered

    def test_disclaimer_key_in_json_instructions(self):
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        rendered = symptom_triage_v1.render_fixture()
        assert '"disclaimer"' in rendered
        assert '"see_doctor_if"' in rendered

    def test_context_included_when_provided(self):
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        rendered = symptom_triage_v1.render(
            symptoms="headache and nausea",
            patient_context="Young adult, no medications",
        )
        assert "Young adult" in rendered

    def test_context_omitted_when_empty(self):
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        rendered = symptom_triage_v1.render(symptoms="headache", patient_context="")
        assert "Patient context:" not in rendered

    def test_variant_is_structured(self):
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        assert symptom_triage_v1.variant == "structured"

    def test_expected_shape_is_callable(self):
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        assert symptom_triage_v1.expected_shape["type"] == "callable"

    def test_schema_validates_valid_output(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_output = json.dumps({
            "assessment": "Likely viral upper respiratory infection",
            "urgency_level": "routine",
            "see_doctor_if": ["Difficulty breathing", "Persistent fever > 3 days"],
            "self_care": ["Rest", "Stay hydrated", "OTC fever reducers"],
            "disclaimer": DISCLAIMERS.medical,
        })
        ok, msg = evaluator._check_shape(valid_output, symptom_triage_v1.expected_shape)
        assert ok, msg

    def test_schema_fails_for_invalid_urgency(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid_output = json.dumps({
            "assessment": "Likely viral infection",
            "urgency_level": "low",  # not in enum
            "see_doctor_if": ["High fever"],
            "self_care": [],
            "disclaimer": DISCLAIMERS.medical,
        })
        ok, _ = evaluator._check_shape(invalid_output, symptom_triage_v1.expected_shape)
        assert not ok

    def test_schema_fails_for_missing_disclaimer_key(self):
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid_output = json.dumps({
            "assessment": "Likely viral infection",
            "urgency_level": "routine",
            "see_doctor_if": ["High fever"],
            "self_care": [],
            # missing "disclaimer"
        })
        ok, _ = evaluator._check_shape(invalid_output, symptom_triage_v1.expected_shape)
        assert not ok


# ---------------------------------------------------------------------------
# drug_interaction_query_v1
# ---------------------------------------------------------------------------

class TestDrugInteractionQuery:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        rendered = drug_interaction_query_v1.render_fixture()
        assert "warfarin" in rendered.lower()
        assert "aspirin" in rendered.lower()

    def test_json_output_instruction(self):
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        rendered = drug_interaction_query_v1.render_fixture()
        assert '"interactions"' in rendered
        assert '"severity_overall"' in rendered
        assert '"recommendations"' in rendered

    def test_patient_notes_included(self):
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        rendered = drug_interaction_query_v1.render_fixture()
        assert "atrial fibrillation" in rendered.lower()

    def test_notes_omitted_when_empty(self):
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        rendered = drug_interaction_query_v1.render(drugs="aspirin, ibuprofen", patient_notes="")
        assert "Patient notes:" not in rendered

    def test_variant_is_structured(self):
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        assert drug_interaction_query_v1.variant == "structured"

    def test_schema_validates_valid_output(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_output = json.dumps({
            "interactions": [
                {
                    "drug_pair": "warfarin + aspirin",
                    "severity": "major",
                    "description": "Increased bleeding risk due to additive anticoagulation effects.",
                },
                {
                    "drug_pair": "warfarin + ibuprofen",
                    "severity": "major",
                    "description": "NSAIDs inhibit platelet function and may displace warfarin from protein binding.",
                },
            ],
            "severity_overall": "major",
            "recommendations": ["Avoid ibuprofen; use paracetamol instead", "Monitor INR closely"],
            "disclaimer": DISCLAIMERS.medical,
        })
        ok, msg = evaluator._check_shape(valid_output, drug_interaction_query_v1.expected_shape)
        assert ok, msg

    def test_schema_fails_for_invalid_severity(self):
        from effgen.prompts.library.disclaimers import DISCLAIMERS
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid_output = json.dumps({
            "interactions": [
                {"drug_pair": "a + b", "severity": "severe", "description": "bad"},  # 'severe' not in enum
            ],
            "severity_overall": "major",
            "recommendations": ["check"],
            "disclaimer": DISCLAIMERS.medical,
        })
        ok, _ = evaluator._check_shape(invalid_output, drug_interaction_query_v1.expected_shape)
        assert not ok


# ---------------------------------------------------------------------------
# medical_literature_v1
# ---------------------------------------------------------------------------

class TestMedicalLiterature:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.medical.medical_literature_v1 import (
            medical_literature_v1,
        )
        rendered = medical_literature_v1.render_fixture()
        assert "metformin" in rendered.lower()
        assert "PMID" in rendered

    def test_structure_instructions_present(self):
        from effgen.prompts.library.domains.medical.medical_literature_v1 import (
            medical_literature_v1,
        )
        rendered = medical_literature_v1.render_fixture()
        for section in ("Bottom Line", "Key Evidence", "Evidence Quality", "Clinical Implications"):
            assert section in rendered, f"Missing section: {section}"

    def test_abstracts_block_included_when_provided(self):
        from effgen.prompts.library.domains.medical.medical_literature_v1 import (
            medical_literature_v1,
        )
        rendered = medical_literature_v1.render_fixture()
        assert "UKPDS" in rendered

    def test_abstracts_note_when_empty(self):
        from effgen.prompts.library.domains.medical.medical_literature_v1 import (
            medical_literature_v1,
        )
        rendered = medical_literature_v1.render(
            clinical_question="Is aspirin effective for primary prevention?",
            retrieved_abstracts="",
        )
        assert "No abstracts retrieved" in rendered

    def test_variant_is_tool(self):
        from effgen.prompts.library.domains.medical.medical_literature_v1 import (
            medical_literature_v1,
        )
        assert medical_literature_v1.variant == "tool"

    def test_expected_shape_is_regex(self):
        from effgen.prompts.library.domains.medical.medical_literature_v1 import (
            medical_literature_v1,
        )
        assert medical_literature_v1.expected_shape["type"] == "regex"


# ---------------------------------------------------------------------------
# Golden eval
# ---------------------------------------------------------------------------

class TestGoldenEval:
    def test_all_goldens_pass(self):
        from effgen.prompts.library.eval import PromptEval
        from effgen.prompts.library.registry import registry

        evaluator = PromptEval()
        prompts = registry.search(domain="medical")
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
    def test_symptom_triage_live_json_schema_valid(self):
        from effgen.prompts.library.domains.medical.symptom_triage_v1 import symptom_triage_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(symptom_triage_v1, model="gpt-oss-120b")
        if not result.passed:
            _skip_transient_live_error(result.message)
        assert_live_eval_passed(result)
        parsed = PromptEval._extract_json_object(result.model_output)
        assert parsed is not None, f"No JSON found: {result.model_output[:300]}"
        assert "urgency_level" in parsed
        assert "see_doctor_if" in parsed

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_drug_interaction_live_json_schema_valid(self):
        from effgen.prompts.library.domains.medical.drug_interaction_query_v1 import (
            drug_interaction_query_v1,
        )
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(drug_interaction_query_v1, model="gpt-oss-120b")
        if not result.passed:
            _skip_transient_live_error(result.message)
        assert_live_eval_passed(result)
        parsed = PromptEval._extract_json_object(result.model_output)
        assert parsed is not None, f"No JSON found: {result.model_output[:300]}"
        assert "interactions" in parsed

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_medical_literature_live_has_sections(self):
        from effgen.prompts.library.domains.medical.medical_literature_v1 import (
            medical_literature_v1,
        )
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(medical_literature_v1, model="gpt-oss-120b")
        if not result.passed:
            _skip_transient_live_error(result.message)
        assert_live_eval_passed(result)
