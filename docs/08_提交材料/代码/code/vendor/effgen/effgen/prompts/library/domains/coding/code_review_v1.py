"""
Coding prompt: Code Review — structured-output variant.

Returns JSON: {issues: [{severity, location, suggestion}]}

Inputs: code (str), language (str), focus (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "minLength": 1},
        "language": {"type": "string", "minLength": 1},
        "focus": {
            "type": "string",
            "description": "Review focus: 'all' | 'security' | 'performance' | 'style'",
        },
    },
    "required": ["code", "language"],
}

_FIXTURE = {
    "code": (
        "def divide(a, b):\n"
        "    return a / b\n\n"
        "def process_items(items):\n"
        "    result = []\n"
        "    for i in range(len(items)):\n"
        "        result.append(items[i] * 2)\n"
        "    return result\n\n"
        "def get_user(user_id):\n"
        "    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
        "    return query\n"
    ),
    "language": "Python",
    "focus": "all",
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low", "info"],
                    },
                    "location": {"type": "string", "minLength": 1},
                    "suggestion": {"type": "string", "minLength": 1},
                },
                "required": ["severity", "location", "suggestion"],
                "additionalProperties": False,
            },
            "minItems": 1,
        }
    },
    "required": ["issues"],
    "additionalProperties": False,
}


def _code_review_structured(code: str, language: str, focus: str = "all") -> str:
    focus_clause = (
        "Focus on all aspects: correctness, security, performance, and style."
        if focus == "all"
        else f"Focus specifically on {focus} concerns."
    )
    return (
        f"You are an expert {language} code reviewer.\n\n"
        f"Review the following {language} code. {focus_clause}\n\n"
        f"```{language.lower()}\n{code}\n```\n\n"
        "Identify every issue found. For each issue provide:\n"
        "  severity  — one of: critical, high, medium, low, info\n"
        "  location  — function name or line reference (e.g. 'divide() line 2')\n"
        "  suggestion — a concise, actionable fix description\n\n"
        "Return a JSON object with a single key 'issues' containing an array of issue objects.\n"
        "Each issue object must have exactly the three fields above.\n"
        "Output only the JSON object, no markdown fences, no extra text."
    )


code_review_v1 = LibraryPrompt(
    name="coding.code_review.v1",
    domain="coding",
    variant="structured",
    description=(
        "Structured code review returning JSON {issues: [{severity, location, suggestion}]}."
    ),
    template=_code_review_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "json",
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["coding", "review", "structured", "json", "security"],
)

registry.register(code_review_v1)
