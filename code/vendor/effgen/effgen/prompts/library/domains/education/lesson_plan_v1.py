"""
Education prompt: Lesson Plan — a structured, time-boxed plan for one class.

Inputs: topic (str), grade_level (str), duration_minutes (int),
        learning_objective (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "The lesson topic, e.g. 'the water cycle'.",
            "minLength": 3,
        },
        "grade_level": {
            "type": "string",
            "description": "Audience, e.g. '5th grade', 'intro undergraduate'.",
            "minLength": 2,
        },
        "duration_minutes": {
            "type": "integer",
            "description": "Total class length in minutes.",
            "minimum": 5,
        },
        "learning_objective": {
            "type": "string",
            "description": "What students should be able to do after the lesson.",
            "minLength": 5,
        },
    },
    "required": ["topic", "grade_level", "duration_minutes", "learning_objective"],
}

_FIXTURE = {
    "topic": "the water cycle",
    "grade_level": "5th grade",
    "duration_minutes": 45,
    "learning_objective": (
        "students can name and describe evaporation, condensation, and precipitation"
    ),
}


def _lesson_plan(
    topic: str,
    grade_level: str,
    duration_minutes: int,
    learning_objective: str,
) -> str:
    return (
        f"You are an experienced {grade_level} teacher. Write a clear, time-boxed "
        f"lesson plan for a {duration_minutes}-minute class on \"{topic}\".\n\n"
        f"Learning objective: {learning_objective}\n\n"
        "Structure the plan with these sections, each with a time estimate that "
        f"adds up to {duration_minutes} minutes:\n"
        "  1. Hook / warm-up\n"
        "  2. Direct instruction (key ideas, in plain language)\n"
        "  3. Guided practice (an activity students do together)\n"
        "  4. Independent practice / check for understanding\n"
        "  5. Wrap-up and a quick formative assessment\n\n"
        "Also list the materials needed and one differentiation tip for students "
        "who need extra support. Write the lesson plan now:"
    )


lesson_plan_v1 = LibraryPrompt(
    name="education.lesson_plan.v1",
    domain="education",
    variant="zero_shot",
    description=(
        "Time-boxed lesson plan (hook → instruction → practice → assessment) for a "
        "single class. Inputs: topic, grade_level, duration_minutes, learning_objective."
    ),
    template=_lesson_plan,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape=None,
    tags=["education", "lesson-plan", "teaching", "zero_shot"],
)

registry.register(lesson_plan_v1)
