"""The prompt library ships an education domain for course creators.

Pins that the four education templates register, render from their fixtures, and
expose a usable input schema — so an instructor has ready-made starting points
(Socratic tutoring, lesson plans, quizzes, plain-language explanations).
"""

from __future__ import annotations

import pytest

from effgen.prompts.library.registry import registry

_EXPECTED = [
    "education.socratic_tutor.v1",
    "education.lesson_plan.v1",
    "education.quiz_generate.v1",
    "education.explain_simply.v1",
]


def test_education_domain_is_discovered():
    assert "education" in registry.domains()


@pytest.mark.parametrize("name", _EXPECTED)
def test_education_template_registered_and_renders(name):
    prompt = registry.get(name)
    assert prompt is not None, f"{name} not registered"
    assert prompt.domain == "education"
    rendered = prompt.render_fixture()
    assert isinstance(rendered, str) and len(rendered) > 50
    assert prompt.input_schema.get("type") == "object"


def test_socratic_tutor_guides_without_revealing():
    """The Socratic template instructs the model to ask, not tell."""
    text = registry.get("education.socratic_tutor.v1").render_fixture().lower()
    assert "do not give it" in text or "never" in text
    # Its output check insists on a question.
    fn = registry.get("education.socratic_tutor.v1").expected_shape["fn"]
    assert fn("What do you get when you isolate x?") is True
    assert isinstance(fn("The answer is 4."), str)  # no '?' -> failure reason
