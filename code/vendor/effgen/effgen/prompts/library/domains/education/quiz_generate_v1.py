"""
Education prompt: Quiz Generator — multiple-choice questions with an answer key.

Inputs: topic (str), num_questions (int), difficulty (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "What the quiz covers, e.g. 'photosynthesis'.",
            "minLength": 3,
        },
        "num_questions": {
            "type": "integer",
            "description": "How many questions to generate.",
            "minimum": 1,
            "maximum": 20,
        },
        "difficulty": {
            "type": "string",
            "description": "Difficulty band: 'easy', 'medium', or 'hard'.",
            "enum": ["easy", "medium", "hard"],
        },
    },
    "required": ["topic", "num_questions", "difficulty"],
}

_FIXTURE = {
    "topic": "photosynthesis",
    "num_questions": 5,
    "difficulty": "medium",
}


def _quiz_generate(topic: str, num_questions: int, difficulty: str) -> str:
    return (
        f"Create a {difficulty} {num_questions}-question multiple-choice quiz on "
        f"\"{topic}\" for students.\n\n"
        "Requirements:\n"
        f"  - Exactly {num_questions} questions, numbered.\n"
        "  - Each question has four options labeled A, B, C, D, with exactly one "
        "correct answer and three plausible distractors.\n"
        "  - Keep wording clear and unambiguous; avoid trick questions.\n"
        "  - After all questions, add an 'Answer Key' section listing the correct "
        "letter and a one-sentence explanation for each.\n\n"
        "Write the quiz now:"
    )


quiz_generate_v1 = LibraryPrompt(
    name="education.quiz_generate.v1",
    domain="education",
    variant="zero_shot",
    description=(
        "Multiple-choice quiz generator with an explained answer key. "
        "Inputs: topic, num_questions, difficulty (easy|medium|hard)."
    ),
    template=_quiz_generate,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape=None,
    tags=["education", "quiz", "assessment", "zero_shot"],
)

registry.register(quiz_generate_v1)
