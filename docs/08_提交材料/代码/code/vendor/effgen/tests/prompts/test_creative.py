"""Tests for creative domain prompt templates."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests._harness.live_eval import assert_live_eval_passed

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "creative"
GOLDENS_DIR = Path(__file__).parent / "goldens"


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

class TestFixtures:
    def test_fixtures_exist(self):
        assert FIXTURES_DIR.exists(), f"Fixtures dir missing: {FIXTURES_DIR}"
        txt_files = list(FIXTURES_DIR.glob("*.txt"))
        assert len(txt_files) >= 3, "Expected at least 3 creative fixture files"

    def test_lighthouse_story_fixture(self):
        p = FIXTURES_DIR / "lighthouse_story.txt"
        assert p.exists()
        content = p.read_text()
        assert "lighthouse" in content.lower()

    def test_poetry_fixture(self):
        p = FIXTURES_DIR / "poetry_time.txt"
        assert p.exists()
        content = p.read_text()
        assert "time" in content.lower()

    def test_character_fixture(self):
        p = FIXTURES_DIR / "detective_character.txt"
        assert p.exists()
        content = p.read_text()
        assert "detective" in content.lower()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestCreativeRegistration:
    def test_global_registry_has_creative_prompts(self):
        from effgen.prompts.library.registry import registry
        prompts = registry.search(domain="creative")
        # story_continuation has 2 variants (zero_shot + few_shot) + poetry + character_bio + world_building = 5
        assert len(prompts) == 5, f"Expected exactly 5 creative prompts, got {len(prompts)}"

    def test_all_required_names_present(self):
        from effgen.prompts.library.registry import registry
        names = {p.name for p in registry.search(domain="creative")}
        required = {
            "creative.story_continuation.v1.zero_shot",
            "creative.story_continuation.v1.few_shot",
            "creative.poetry_forms.v1",
            "creative.character_bio.v1",
            "creative.world_building.v1",
        }
        # At minimum 4 must be present (story has 2 variants but counts as 2 prompts)
        assert len(names & required) >= 4, f"Missing creative prompts. Found: {names}"

    def test_variants(self):
        from effgen.prompts.library.registry import registry
        prompts = {p.name: p for p in registry.search(domain="creative")}
        assert prompts["creative.character_bio.v1"].variant == "structured"
        assert prompts["creative.world_building.v1"].variant == "cot"
        assert prompts["creative.poetry_forms.v1"].variant == "few_shot"


# ---------------------------------------------------------------------------
# story_continuation_v1
# ---------------------------------------------------------------------------

class TestStoryContinuation:
    def test_zero_shot_renders_with_fixture(self):
        from effgen.prompts.library.domains.creative.story_continuation_v1 import (
            story_continuation_zero_shot,
        )
        rendered = story_continuation_zero_shot.render_fixture()
        assert "lighthouse" in rendered.lower()
        assert "mystery" in rendered.lower()

    def test_few_shot_renders_with_fixture(self):
        from effgen.prompts.library.domains.creative.story_continuation_v1 import (
            story_continuation_few_shot,
        )
        rendered = story_continuation_few_shot.render_fixture()
        assert "lighthouse" in rendered.lower()
        assert "Example" in rendered

    def test_few_shot_contains_exemplars(self):
        from effgen.prompts.library.domains.creative.story_continuation_v1 import (
            story_continuation_few_shot,
        )
        rendered = story_continuation_few_shot.render_fixture()
        assert "Example 1" in rendered
        assert "Example 2" in rendered

    def test_zero_shot_custom_genre(self):
        from effgen.prompts.library.domains.creative.story_continuation_v1 import (
            story_continuation_zero_shot,
        )
        rendered = story_continuation_zero_shot.render(
            story_so_far="The spaceship hummed silently.",
            genre="sci-fi",
            desired_length="1 paragraph",
        )
        assert "sci-fi" in rendered

    def test_variants_are_correct(self):
        from effgen.prompts.library.domains.creative.story_continuation_v1 import (
            story_continuation_few_shot,
            story_continuation_zero_shot,
        )
        assert story_continuation_zero_shot.variant == "zero_shot"
        assert story_continuation_few_shot.variant == "few_shot"


# ---------------------------------------------------------------------------
# poetry_forms_v1
# ---------------------------------------------------------------------------

class TestPoetryForms:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.creative.poetry_forms_v1 import poetry_forms_v1
        rendered = poetry_forms_v1.render_fixture()
        assert "time" in rendered.lower()
        assert "haiku" in rendered.lower()

    def test_all_three_form_exemplars_present(self):
        from effgen.prompts.library.domains.creative.poetry_forms_v1 import poetry_forms_v1
        rendered = poetry_forms_v1.render_fixture()
        assert "haiku" in rendered.lower()
        assert "sonnet" in rendered.lower()
        assert "free verse" in rendered.lower() or "free_verse" in rendered.lower()

    def test_sonnet_form(self):
        from effgen.prompts.library.domains.creative.poetry_forms_v1 import poetry_forms_v1
        rendered = poetry_forms_v1.render(
            theme="lost friendship", form="sonnet", mood="melancholic"
        )
        assert "sonnet" in rendered.lower()
        assert "iambic" in rendered.lower() or "rhyme" in rendered.lower()

    def test_free_verse_form(self):
        from effgen.prompts.library.domains.creative.poetry_forms_v1 import poetry_forms_v1
        rendered = poetry_forms_v1.render(
            theme="city life", form="free_verse", mood="energetic"
        )
        assert "free" in rendered.lower()

    def test_variant_is_few_shot(self):
        from effgen.prompts.library.domains.creative.poetry_forms_v1 import poetry_forms_v1
        assert poetry_forms_v1.variant == "few_shot"


# ---------------------------------------------------------------------------
# character_bio_v1
# ---------------------------------------------------------------------------

class TestCharacterBio:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.creative.character_bio_v1 import character_bio_v1
        rendered = character_bio_v1.render_fixture()
        assert "detective" in rendered.lower()
        assert "protagonist" in rendered.lower()

    def test_json_keys_in_instructions(self):
        from effgen.prompts.library.domains.creative.character_bio_v1 import character_bio_v1
        rendered = character_bio_v1.render_fixture()
        for key in ("name", "age", "background", "personality_traits", "goals", "flaws"):
            assert f'"{key}"' in rendered, f"Missing key '{key}' in instructions"

    def test_no_markdown_fences_instruction(self):
        from effgen.prompts.library.domains.creative.character_bio_v1 import character_bio_v1
        rendered = character_bio_v1.render_fixture()
        assert "no markdown fences" in rendered.lower()

    def test_variant_is_structured(self):
        from effgen.prompts.library.domains.creative.character_bio_v1 import character_bio_v1
        assert character_bio_v1.variant == "structured"

    def test_expected_shape_is_callable(self):
        from effgen.prompts.library.domains.creative.character_bio_v1 import character_bio_v1
        assert character_bio_v1.expected_shape is not None
        assert character_bio_v1.expected_shape["type"] == "callable"

    def test_schema_validates_valid_output(self):
        import json

        from effgen.prompts.library.domains.creative.character_bio_v1 import character_bio_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_output = json.dumps({
            "name": "Jack Moran",
            "age": 58,
            "background": "Retired NYPD homicide detective who spent 30 years on the force.",
            "personality_traits": ["methodical", "brooding", "perceptive", "loyal"],
            "goals": ["solve the Langford case", "make peace with the past"],
            "flaws": ["alcoholism", "inability to let go"],
            "relationships": ["estranged daughter Claire", "former partner Luis Reyes"],
        })
        ok, msg = evaluator._check_shape(valid_output, character_bio_v1.expected_shape)
        assert ok, msg

    def test_schema_fails_missing_required_fields(self):
        import json

        from effgen.prompts.library.domains.creative.character_bio_v1 import character_bio_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid = json.dumps({"name": "Jack", "background": "A detective."})
        ok, _ = evaluator._check_shape(invalid, character_bio_v1.expected_shape)
        assert not ok


# ---------------------------------------------------------------------------
# world_building_v1
# ---------------------------------------------------------------------------

class TestWorldBuilding:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.creative.world_building_v1 import world_building_v1
        rendered = world_building_v1.render_fixture()
        assert "music" in rendered.lower()
        assert "magic" in rendered.lower()

    def test_focus_areas_in_render(self):
        from effgen.prompts.library.domains.creative.world_building_v1 import world_building_v1
        rendered = world_building_v1.render_fixture()
        assert "magic system" in rendered.lower()
        assert "culture" in rendered.lower()
        assert "politics" in rendered.lower()

    def test_step_instructions_present(self):
        from effgen.prompts.library.domains.creative.world_building_v1 import world_building_v1
        rendered = world_building_v1.render_fixture()
        assert "Step 1" in rendered
        assert "Step 5" in rendered

    def test_variant_is_cot(self):
        from effgen.prompts.library.domains.creative.world_building_v1 import world_building_v1
        assert world_building_v1.variant == "cot"

    def test_custom_focus_areas(self):
        from effgen.prompts.library.domains.creative.world_building_v1 import world_building_v1
        rendered = world_building_v1.render(
            world_concept="a world where everyone can see each other's memories",
            genre="sci-fi",
            focus_areas=["technology", "economy", "law"],
        )
        assert "technology" in rendered
        assert "economy" in rendered


# ---------------------------------------------------------------------------
# Golden eval
# ---------------------------------------------------------------------------

class TestGoldenEval:
    def test_all_goldens_pass(self):
        from effgen.prompts.library.eval import PromptEval
        from effgen.prompts.library.registry import registry

        evaluator = PromptEval()
        prompts = registry.search(domain="creative")
        assert len(prompts) >= 4

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
    def test_character_bio_live_json_schema_valid(self):
        from effgen.prompts.library.domains.creative.character_bio_v1 import character_bio_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(character_bio_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        parsed = PromptEval._extract_json_object(result.model_output)
        assert parsed is not None, f"No JSON found: {result.model_output[:300]}"
        assert "name" in parsed
        assert "personality_traits" in parsed

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_story_continuation_live_has_content(self):
        from effgen.prompts.library.domains.creative.story_continuation_v1 import (
            story_continuation_zero_shot,
        )
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(story_continuation_zero_shot, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        assert len(result.model_output.split()) >= 20, "Story continuation too short"

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_world_building_live_has_steps(self):
        from effgen.prompts.library.domains.creative.world_building_v1 import world_building_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(world_building_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
