"""Education domain prompts: Socratic tutoring, lesson plans, quizzes, plain-language explanations."""

from .explain_simply_v1 import explain_simply_v1
from .lesson_plan_v1 import lesson_plan_v1
from .quiz_generate_v1 import quiz_generate_v1
from .socratic_tutor_v1 import socratic_tutor_v1

__all__ = [
    "socratic_tutor_v1",
    "lesson_plan_v1",
    "quiz_generate_v1",
    "explain_simply_v1",
]
