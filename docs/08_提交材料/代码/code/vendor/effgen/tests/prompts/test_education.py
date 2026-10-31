"""Tests for education domain prompt templates (tutor / lesson / quiz / explain)."""

from __future__ import annotations

import pytest

from effgen.prompts.library.registry import registry

_EXPECTED = [
    "education.socratic_tutor.v1",
    "education.lesson_plan.v1",
    "education.quiz_generate.v1",
    "education.explain_simply.v1",
]


class TestRegistration:
    def test_education_domain_present(self):
        assert "education" in registry.domains()

    @pytest.mark.parametrize("name", _EXPECTED)
    def test_template_registered_and_renders(self, name):
        prompt = registry.get(name)
        assert prompt is not None
        assert prompt.domain == "education"
        assert len(prompt.render_fixture()) > 50


class TestGoldenEval:
    def test_all_goldens_pass(self):
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        prompts = registry.search(domain="education")
        assert len(prompts) >= 4

        report = evaluator.eval_all_golden(prompts)
        failed = report.failed()
        assert not failed, f"Golden failures: {[r.name + ': ' + r.message for r in failed]}"
