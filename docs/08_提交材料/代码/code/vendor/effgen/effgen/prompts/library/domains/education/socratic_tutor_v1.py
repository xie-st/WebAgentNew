"""Education prompt: guide a learner to the answer with questions, not the answer itself.

Inputs: subject (str), question (str), student_level (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "Subject area, e.g. 'algebra', 'cell biology'.",
            "minLength": 2,
        },
        "question": {
            "type": "string",
            "description": "The student's question or the problem they are stuck on.",
            "minLength": 5,
        },
        "student_level": {
            "type": "string",
            "description": "Who the learner is, e.g. 'a 9th grader', 'an adult beginner'.",
            "minLength": 2,
        },
    },
    "required": ["subject", "question", "student_level"],
}

_FIXTURE = {
    "subject": "algebra",
    "question": "How do I solve 2x + 6 = 14?",
    "student_level": "a 9th grader new to equations",
}


def _socratic_tutor(subject: str, question: str, student_level: str) -> str:
    return (
        f"You are a patient Socratic tutor in {subject}, working with {student_level}.\n\n"
        f"The student asks:\n{question}\n\n"
        "Guide them to the answer — do NOT give it away. Specifically:\n"
        "  - Respond with ONE clear, encouraging guiding question that moves them "
        "a single step forward.\n"
        "  - Never state the final answer or do the decisive step for them.\n"
        "  - Keep it short, warm, and free of jargon they would not know yet.\n"
        "  - If they seem stuck, offer a small hint framed as a question.\n\n"
        "Ask your guiding question now:"
    )


def _socratic_no_answer_check(output: str) -> bool | str:
    """Heuristic: a good Socratic turn ends by asking, not telling."""
    if "?" not in output:
        return "Socratic reply should pose a guiding question (no '?' found)"
    return True


socratic_tutor_v1 = LibraryPrompt(
    name="education.socratic_tutor.v1",
    domain="education",
    variant="zero_shot",
    description=(
        "Socratic tutor that guides a learner with one question at a time instead "
        "of revealing the answer. Inputs: subject, question, student_level."
    ),
    template=_socratic_tutor,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _socratic_no_answer_check,
    },
    tags=["education", "tutor", "socratic", "zero_shot"],
)

registry.register(socratic_tutor_v1)
