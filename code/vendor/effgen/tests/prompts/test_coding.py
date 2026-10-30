"""Tests for coding domain prompt templates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._harness.live_eval import assert_live_eval_passed

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "coding"
GOLDENS_DIR = Path(__file__).parent / "goldens"


# ---------------------------------------------------------------------------
# Fixtures dir
# ---------------------------------------------------------------------------

class TestFixtures:
    def test_fixtures_exist(self):
        assert FIXTURES_DIR.exists(), f"Fixtures dir missing: {FIXTURES_DIR}"
        py_files = list(FIXTURES_DIR.glob("*.py"))
        assert len(py_files) >= 3, "Expected at least 3 fixture files"

    def test_clamp_fixture(self):
        p = FIXTURES_DIR / "clamp.py"
        assert p.exists()
        content = p.read_text()
        assert "clamp" in content

    def test_buggy_fibonacci_fixture(self):
        p = FIXTURES_DIR / "buggy_fibonacci.py"
        assert p.exists()
        content = p.read_text()
        assert "fibonacci" in content


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestCodingRegistration:
    def test_global_registry_has_coding_prompts(self):
        from effgen.prompts.library.registry import registry
        prompts = registry.search(domain="coding")
        assert len(prompts) == 5, f"Expected exactly 5 coding prompts, got {len(prompts)}"

    def test_all_required_names_present(self):
        from effgen.prompts.library.registry import registry
        names = {p.name for p in registry.search(domain="coding")}
        required = {
            "coding.code_review.v1",
            "coding.bug_diagnose.v1",
            "coding.refactor_plan.v1",
            "coding.test_generate.v1",
            "coding.docstring_fill.v1",
        }
        missing = required - names
        assert not missing, f"Missing coding prompts: {missing}"

    def test_variants(self):
        from effgen.prompts.library.registry import registry
        prompts = {p.name: p for p in registry.search(domain="coding")}
        assert prompts["coding.code_review.v1"].variant == "structured"
        assert prompts["coding.bug_diagnose.v1"].variant == "cot"
        assert prompts["coding.refactor_plan.v1"].variant == "tool"
        assert prompts["coding.test_generate.v1"].variant == "few_shot"
        assert prompts["coding.docstring_fill.v1"].variant == "zero_shot"


# ---------------------------------------------------------------------------
# code_review_v1
# ---------------------------------------------------------------------------

class TestCodeReview:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        rendered = code_review_v1.render_fixture()
        assert "issues" in rendered
        assert "severity" in rendered
        assert "location" in rendered
        assert "suggestion" in rendered

    def test_contains_code_snippet(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        rendered = code_review_v1.render_fixture()
        assert "divide" in rendered
        assert "get_user" in rendered

    def test_no_markdown_fences_instruction(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        rendered = code_review_v1.render_fixture()
        assert "no markdown fences" in rendered.lower()

    def test_expected_schema_has_issues(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        assert code_review_v1.expected_shape is not None
        assert code_review_v1.expected_shape["type"] == "json"
        schema = code_review_v1.expected_shape["schema"]
        assert "issues" in schema["required"]

    def test_schema_validation_passes_for_valid_output(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_output = json.dumps({
            "issues": [
                {
                    "severity": "critical",
                    "location": "get_user() line 3",
                    "suggestion": "Use parameterized queries to prevent SQL injection.",
                }
            ]
        })
        ok, msg = evaluator._check_shape(valid_output, code_review_v1.expected_shape)
        assert ok, msg

    def test_schema_validation_fails_for_missing_field(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid_output = json.dumps({
            "issues": [
                {
                    "severity": "high",
                    "location": "line 1",
                    # missing 'suggestion'
                }
            ]
        })
        ok, _ = evaluator._check_shape(invalid_output, code_review_v1.expected_shape)
        assert not ok

    def test_schema_validation_fails_for_extra_field(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid_output = json.dumps({
            "issues": [
                {
                    "severity": "high",
                    "location": "line 1",
                    "suggestion": "Fix the issue.",
                    "confidence": 0.9,
                }
            ]
        })
        ok, _ = evaluator._check_shape(invalid_output, code_review_v1.expected_shape)
        assert not ok

    def test_focus_clause_in_security_mode(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        rendered = code_review_v1.render(
            code="x = 1",
            language="Python",
            focus="security",
        )
        assert "security" in rendered.lower()

    def test_custom_language(self):
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        rendered = code_review_v1.render(
            code="function foo() { return 1; }",
            language="JavaScript",
            focus="all",
        )
        assert "JavaScript" in rendered


# ---------------------------------------------------------------------------
# bug_diagnose_v1
# ---------------------------------------------------------------------------

class TestBugDiagnose:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.coding.bug_diagnose_v1 import bug_diagnose_v1
        rendered = bug_diagnose_v1.render_fixture()
        assert "KeyError" in rendered
        assert "fibonacci" in rendered

    def test_has_numbered_steps(self):
        from effgen.prompts.library.domains.coding.bug_diagnose_v1 import bug_diagnose_v1
        rendered = bug_diagnose_v1.render_fixture()
        for i in range(1, 7):
            assert f"{i}." in rendered

    def test_repro_steps_included_when_provided(self):
        from effgen.prompts.library.domains.coding.bug_diagnose_v1 import bug_diagnose_v1
        rendered = bug_diagnose_v1.render_fixture()
        assert "Reproduction steps" in rendered

    def test_repro_steps_omitted_when_empty(self):
        from effgen.prompts.library.domains.coding.bug_diagnose_v1 import bug_diagnose_v1
        rendered = bug_diagnose_v1.render(
            code="def foo(): pass",
            error_message="NameError: foo not defined",
            repro_steps="",
        )
        assert "Reproduction steps" not in rendered

    def test_variant_is_cot(self):
        from effgen.prompts.library.domains.coding.bug_diagnose_v1 import bug_diagnose_v1
        assert bug_diagnose_v1.variant == "cot"


# ---------------------------------------------------------------------------
# refactor_plan_v1
# ---------------------------------------------------------------------------

class TestRefactorPlan:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.coding.refactor_plan_v1 import refactor_plan_v1
        rendered = refactor_plan_v1.render_fixture()
        assert "refactor" in rendered.lower()
        assert "risk" in rendered.lower()
        assert "test" in rendered.lower()

    def test_has_five_sections(self):
        from effgen.prompts.library.domains.coding.refactor_plan_v1 import refactor_plan_v1
        rendered = refactor_plan_v1.render_fixture()
        assert "1." in rendered
        assert "2." in rendered
        assert "5." in rendered

    def test_tool_instruction_when_no_snippet(self):
        from effgen.prompts.library.domains.coding.refactor_plan_v1 import refactor_plan_v1
        rendered = refactor_plan_v1.render(
            file_path="src/main.py",
            goals="Remove duplication",
            language="Python",
            code_snippet="",
        )
        assert "file_operations" in rendered
        assert "operation='read'" in rendered

    def test_code_snippet_renders_when_provided(self):
        from effgen.prompts.library.domains.coding.refactor_plan_v1 import refactor_plan_v1
        rendered = refactor_plan_v1.render_fixture()
        assert "calc_area_circle" in rendered

    def test_variant_is_tool(self):
        from effgen.prompts.library.domains.coding.refactor_plan_v1 import refactor_plan_v1
        assert refactor_plan_v1.variant == "tool"


# ---------------------------------------------------------------------------
# test_generate_v1
# ---------------------------------------------------------------------------

class TestTestGenerate:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import test_generate_v1
        rendered = test_generate_v1.render_fixture()
        assert "clamp" in rendered
        assert "pytest" in rendered.lower()

    def test_has_two_exemplars(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import test_generate_v1
        rendered = test_generate_v1.render_fixture()
        assert "Example 1" in rendered
        assert "Example 2" in rendered

    def test_no_markdown_fences_instruction(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import test_generate_v1
        rendered = test_generate_v1.render_fixture()
        assert "no markdown fences" in rendered.lower()

    def test_ast_parse_check_passes_valid_python(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import _ast_parse_check
        code = (
            "import pytest\n\n"
            "def test_clamp_normal():\n"
            "    assert clamp(5, 0, 10) == 5\n\n"
            "def test_clamp_low():\n"
            "    assert clamp(-1, 0, 10) == 0\n"
        )
        result = _ast_parse_check(code)
        assert result is True

    def test_ast_parse_check_fails_on_syntax_error(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import _ast_parse_check
        result = _ast_parse_check("def bad(:\n    pass")
        assert result is not True
        assert "SyntaxError" in str(result)

    def test_ast_parse_check_strips_markdown_fences(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import _ast_parse_check
        code = "```python\ndef test_foo():\n    assert 1 == 1\n```"
        result = _ast_parse_check(code)
        assert result is True

    def test_function_name_clause_included(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import test_generate_v1
        rendered = test_generate_v1.render(
            code="def foo(x): return x",
            framework="pytest",
            function_name="foo",
        )
        assert "foo" in rendered
        assert "Focus on the function" in rendered

    def test_variant_is_few_shot(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import test_generate_v1
        assert test_generate_v1.variant == "few_shot"

    def test_shape_check_via_eval(self):
        from effgen.prompts.library.domains.coding.test_generate_v1 import test_generate_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_code = (
            "import pytest\n\n"
            "def test_clamp_below_lo():\n"
            "    assert clamp(-5, 0, 10) == 0\n\n"
            "def test_clamp_above_hi():\n"
            "    assert clamp(15, 0, 10) == 10\n"
        )
        ok, msg = evaluator._check_shape(valid_code, test_generate_v1.expected_shape)
        assert ok, msg


# ---------------------------------------------------------------------------
# docstring_fill_v1
# ---------------------------------------------------------------------------

class TestDocstringFill:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.coding.docstring_fill_v1 import docstring_fill_v1
        rendered = docstring_fill_v1.render_fixture()
        assert "merge_sorted" in rendered
        assert "Google" in rendered

    def test_google_style_example_in_prompt(self):
        from effgen.prompts.library.domains.coding.docstring_fill_v1 import docstring_fill_v1
        rendered = docstring_fill_v1.render(code="def foo(x): pass", style="google")
        assert "Args:" in rendered

    def test_numpy_style_example_in_prompt(self):
        from effgen.prompts.library.domains.coding.docstring_fill_v1 import docstring_fill_v1
        rendered = docstring_fill_v1.render(code="def foo(x): pass", style="numpy")
        assert "Parameters" in rendered

    def test_sphinx_style_example_in_prompt(self):
        from effgen.prompts.library.domains.coding.docstring_fill_v1 import docstring_fill_v1
        rendered = docstring_fill_v1.render(code="def foo(x): pass", style="sphinx")
        assert ":param" in rendered

    def test_no_markdown_fences_instruction(self):
        from effgen.prompts.library.domains.coding.docstring_fill_v1 import docstring_fill_v1
        rendered = docstring_fill_v1.render_fixture()
        assert "no markdown fences" in rendered.lower()

    def test_variant_is_zero_shot(self):
        from effgen.prompts.library.domains.coding.docstring_fill_v1 import docstring_fill_v1
        assert docstring_fill_v1.variant == "zero_shot"


# ---------------------------------------------------------------------------
# Golden eval
# ---------------------------------------------------------------------------

class TestGoldenEval:
    def test_all_goldens_pass(self):
        from effgen.prompts.library.eval import PromptEval
        from effgen.prompts.library.registry import registry

        evaluator = PromptEval()
        prompts = registry.search(domain="coding")
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
    def test_test_generate_live_ast_parse(self):
        """test_generate_v1 live output must be syntactically valid Python (ast.parse passes)."""
        from effgen.prompts.library.domains.coding.test_generate_v1 import test_generate_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(test_generate_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_code_review_live_json(self):
        """code_review_v1 live output must produce schema-valid JSON."""
        from effgen.prompts.library.domains.coding.code_review_v1 import code_review_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(code_review_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        output = result.model_output.strip()
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", output, re.DOTALL)
            assert m, f"No JSON found in output: {output[:300]}"
            parsed = json.loads(m.group(0))
        assert "issues" in parsed, "Missing 'issues' key in output"
        assert isinstance(parsed["issues"], list)
        assert len(parsed["issues"]) >= 1
        for issue in parsed["issues"]:
            assert "severity" in issue
            assert "location" in issue
            assert "suggestion" in issue
