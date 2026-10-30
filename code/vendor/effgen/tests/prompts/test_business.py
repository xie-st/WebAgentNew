"""Tests for business domain prompt templates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._harness.live_eval import assert_live_eval_passed

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "business"
GOLDENS_DIR = Path(__file__).parent / "goldens"


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

class TestFixtures:
    def test_fixtures_exist(self):
        assert FIXTURES_DIR.exists(), f"Fixtures dir missing: {FIXTURES_DIR}"
        txt_files = list(FIXTURES_DIR.glob("*.txt"))
        assert len(txt_files) >= 3, "Expected at least 3 business fixture files"

    def test_meeting_transcript_fixture(self):
        p = FIXTURES_DIR / "q3_meeting_transcript.txt"
        assert p.exists()
        content = p.read_text()
        assert "Alice" in content
        assert "Q3" in content

    def test_elevator_pitch_fixture(self):
        p = FIXTURES_DIR / "contractiq_pitch.txt"
        assert p.exists()
        content = p.read_text()
        assert "ContractIQ" in content


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestBusinessRegistration:
    def test_global_registry_has_business_prompts(self):
        from effgen.prompts.library.registry import registry
        prompts = registry.search(domain="business")
        assert len(prompts) == 5, f"Expected exactly 5 business prompts, got {len(prompts)}"

    def test_all_required_names_present(self):
        from effgen.prompts.library.registry import registry
        names = {p.name for p in registry.search(domain="business")}
        required = {
            "business.meeting_summary.v1",
            "business.email_draft.v1",
            "business.okr_generate.v1",
            "business.swot_analysis.v1",
            "business.elevator_pitch.v1",
        }
        missing = required - names
        assert not missing, f"Missing business prompts: {missing}"

    def test_variants(self):
        from effgen.prompts.library.registry import registry
        prompts = {p.name: p for p in registry.search(domain="business")}
        assert prompts["business.meeting_summary.v1"].variant == "structured"
        assert prompts["business.email_draft.v1"].variant == "few_shot"
        assert prompts["business.okr_generate.v1"].variant == "cot"
        assert prompts["business.swot_analysis.v1"].variant == "structured"
        assert prompts["business.elevator_pitch.v1"].variant == "zero_shot"


# ---------------------------------------------------------------------------
# meeting_summary_v1
# ---------------------------------------------------------------------------

class TestMeetingSummary:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        rendered = meeting_summary_v1.render_fixture()
        assert "Q3 Product Roadmap Review" in rendered
        assert "Alice" in rendered

    def test_json_output_keys_in_instructions(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        rendered = meeting_summary_v1.render_fixture()
        assert '"decisions"' in rendered
        assert '"action_items"' in rendered
        assert '"risks"' in rendered

    def test_no_markdown_fences_instruction(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        rendered = meeting_summary_v1.render_fixture()
        assert "no markdown fences" in rendered.lower()

    def test_variant_is_structured(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        assert meeting_summary_v1.variant == "structured"

    def test_expected_shape_is_callable(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        assert meeting_summary_v1.expected_shape["type"] == "callable"

    def test_schema_validates_valid_output(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_output = json.dumps({
            "decisions": ["Prioritize onboarding flow first", "Bob to assess auth layer"],
            "action_items": [
                {"owner": "Carol", "item": "Fast-track onboarding mockups", "due": "July 10th"},
                {"owner": "Bob", "item": "Auth layer analysis", "due": "Friday"},
            ],
            "risks": ["Auth refactor may cause August 15th deadline slip"],
        })
        ok, msg = evaluator._check_shape(valid_output, meeting_summary_v1.expected_shape)
        assert ok, msg

    def test_schema_fails_missing_required_keys(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid = json.dumps({"decisions": ["Something was decided"]})
        ok, _ = evaluator._check_shape(invalid, meeting_summary_v1.expected_shape)
        assert not ok

    def test_action_item_missing_owner_fails(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid = json.dumps({
            "decisions": ["Decided X"],
            "action_items": [{"item": "Do something", "due": "Friday"}],
            "risks": [],
        })
        ok, _ = evaluator._check_shape(invalid, meeting_summary_v1.expected_shape)
        assert not ok


# ---------------------------------------------------------------------------
# email_draft_v1
# ---------------------------------------------------------------------------

class TestEmailDraft:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.business.email_draft_v1 import email_draft_v1
        rendered = email_draft_v1.render_fixture()
        assert "formal" in rendered.lower()
        assert "extension" in rendered.lower()

    def test_both_tone_examples_present(self):
        from effgen.prompts.library.domains.business.email_draft_v1 import email_draft_v1
        rendered = email_draft_v1.render_fixture()
        assert "Formal" in rendered
        assert "Casual" in rendered

    def test_casual_tone_render(self):
        from effgen.prompts.library.domains.business.email_draft_v1 import email_draft_v1
        rendered = email_draft_v1.render(
            purpose="announce a product launch",
            recipient="the team",
            key_points=["launch on July 15th", "big win for everyone"],
            tone="casual",
        )
        assert "casual" in rendered.lower()

    def test_key_points_included(self):
        from effgen.prompts.library.domains.business.email_draft_v1 import email_draft_v1
        rendered = email_draft_v1.render_fixture()
        assert "unexpected technical complexity" in rendered

    def test_variant_is_few_shot(self):
        from effgen.prompts.library.domains.business.email_draft_v1 import email_draft_v1
        assert email_draft_v1.variant == "few_shot"


# ---------------------------------------------------------------------------
# okr_generate_v1
# ---------------------------------------------------------------------------

class TestOkrGenerate:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.business.okr_generate_v1 import okr_generate_v1
        rendered = okr_generate_v1.render_fixture()
        assert "Platform Engineering" in rendered
        assert "Q3 2025" in rendered

    def test_step_instructions_present(self):
        from effgen.prompts.library.domains.business.okr_generate_v1 import okr_generate_v1
        rendered = okr_generate_v1.render_fixture()
        assert "Step 1" in rendered
        assert "Step 5" in rendered

    def test_priorities_in_render(self):
        from effgen.prompts.library.domains.business.okr_generate_v1 import okr_generate_v1
        rendered = okr_generate_v1.render_fixture()
        assert "downtime" in rendered.lower()
        assert "kubernetes" in rendered.lower()

    def test_num_objectives_respected(self):
        from effgen.prompts.library.domains.business.okr_generate_v1 import okr_generate_v1
        rendered = okr_generate_v1.render(
            team_name="Sales",
            mission="Grow revenue.",
            strategic_priorities=["increase pipeline", "improve win rate"],
            timeframe="Q4 2025",
            num_objectives=2,
        )
        assert "2" in rendered

    def test_variant_is_cot(self):
        from effgen.prompts.library.domains.business.okr_generate_v1 import okr_generate_v1
        assert okr_generate_v1.variant == "cot"


# ---------------------------------------------------------------------------
# swot_analysis_v1
# ---------------------------------------------------------------------------

class TestSwotAnalysis:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.business.swot_analysis_v1 import swot_analysis_v1
        rendered = swot_analysis_v1.render_fixture()
        assert "investor" in rendered.lower()
        assert "ContractIQ" in rendered or "contract review" in rendered.lower()

    def test_json_output_keys_in_instructions(self):
        from effgen.prompts.library.domains.business.swot_analysis_v1 import swot_analysis_v1
        rendered = swot_analysis_v1.render_fixture()
        for key in ("strengths", "weaknesses", "opportunities", "threats", "strategic_insights"):
            assert f'"{key}"' in rendered, f"Missing key instruction for '{key}'"

    def test_variant_is_structured(self):
        from effgen.prompts.library.domains.business.swot_analysis_v1 import swot_analysis_v1
        assert swot_analysis_v1.variant == "structured"

    def test_schema_validates_valid_output(self):
        from effgen.prompts.library.domains.business.swot_analysis_v1 import swot_analysis_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_output = json.dumps({
            "strengths": ["Strong ARR growth", "Enterprise traction"],
            "weaknesses": ["Small team", "Limited integrations"],
            "opportunities": ["Expanding AI legal market", "International expansion"],
            "threats": ["Well-funded competitor", "LLM commoditization"],
            "strategic_insights": [
                "Use strong growth metrics to raise a Series A before competitor gains market share.",
                "Deepen integrations as moat to counteract commoditization threat.",
            ],
        })
        ok, msg = evaluator._check_shape(valid_output, swot_analysis_v1.expected_shape)
        assert ok, msg

    def test_schema_fails_missing_quadrant(self):
        from effgen.prompts.library.domains.business.swot_analysis_v1 import swot_analysis_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid = json.dumps({
            "strengths": ["Fast growth"],
            "weaknesses": ["Small team"],
            # missing opportunities, threats, strategic_insights
        })
        ok, _ = evaluator._check_shape(invalid, swot_analysis_v1.expected_shape)
        assert not ok


# ---------------------------------------------------------------------------
# elevator_pitch_v1
# ---------------------------------------------------------------------------

class TestElevatorPitch:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.business.elevator_pitch_v1 import elevator_pitch_v1
        rendered = elevator_pitch_v1.render_fixture()
        assert "ContractIQ" in rendered
        assert "150" in rendered

    def test_word_count_constraint_in_instructions(self):
        from effgen.prompts.library.domains.business.elevator_pitch_v1 import elevator_pitch_v1
        rendered = elevator_pitch_v1.render_fixture()
        assert "150" in rendered
        assert "word" in rendered.lower()

    def test_variant_is_zero_shot(self):
        from effgen.prompts.library.domains.business.elevator_pitch_v1 import elevator_pitch_v1
        assert elevator_pitch_v1.variant == "zero_shot"

    def test_expected_shape_is_callable(self):
        from effgen.prompts.library.domains.business.elevator_pitch_v1 import elevator_pitch_v1
        assert elevator_pitch_v1.expected_shape["type"] == "callable"

    def test_word_count_check_passes_short_pitch(self):
        from effgen.prompts.library.domains.business.elevator_pitch_v1 import elevator_pitch_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        short_pitch = " ".join(["word"] * 120)
        ok, msg = evaluator._check_shape(short_pitch, elevator_pitch_v1.expected_shape)
        assert ok, msg

    def test_word_count_check_fails_long_pitch(self):
        from effgen.prompts.library.domains.business.elevator_pitch_v1 import elevator_pitch_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        long_pitch = " ".join(["word"] * 200)
        ok, msg = evaluator._check_shape(long_pitch, elevator_pitch_v1.expected_shape)
        assert not ok
        assert "151" in msg or "200" in msg or "≤150" in msg

    def test_word_count_exactly_150(self):
        from effgen.prompts.library.domains.business.elevator_pitch_v1 import elevator_pitch_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        exact_pitch = " ".join(["word"] * 150)
        ok, _ = evaluator._check_shape(exact_pitch, elevator_pitch_v1.expected_shape)
        assert ok


# ---------------------------------------------------------------------------
# Golden eval
# ---------------------------------------------------------------------------

class TestGoldenEval:
    def test_all_goldens_pass(self):
        from effgen.prompts.library.eval import PromptEval
        from effgen.prompts.library.registry import registry

        evaluator = PromptEval()
        prompts = registry.search(domain="business")
        assert len(prompts) >= 5

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
    def test_meeting_summary_live_json_schema_valid(self):
        from effgen.prompts.library.domains.business.meeting_summary_v1 import meeting_summary_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(meeting_summary_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        parsed = PromptEval._extract_json_object(result.model_output)
        assert parsed is not None, f"No JSON found: {result.model_output[:300]}"
        assert "decisions" in parsed
        assert "action_items" in parsed
        assert "risks" in parsed

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_elevator_pitch_live_word_count(self):
        from effgen.prompts.library.domains.business.elevator_pitch_v1 import elevator_pitch_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(elevator_pitch_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        word_count = len(result.model_output.split())
        assert word_count <= 150, (
            f"Elevator pitch is {word_count} words — must be ≤150\n"
            f"Output: {result.model_output}"
        )

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_swot_analysis_live_json_schema_valid(self):
        from effgen.prompts.library.domains.business.swot_analysis_v1 import swot_analysis_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(swot_analysis_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_okr_generate_live_has_objectives(self):
        from effgen.prompts.library.domains.business.okr_generate_v1 import okr_generate_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(okr_generate_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
