"""
Business prompt: SWOT Analysis — structured output variant.

Returns JSON: {strengths, weaknesses, opportunities, threats, strategic_insights}

Inputs: subject (str), context (str), perspective (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.eval import PromptEval
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "The business, product, or decision to analyze.",
            "minLength": 3,
        },
        "context": {
            "type": "string",
            "description": "Relevant background: industry, competitors, market conditions.",
            "minLength": 10,
        },
        "perspective": {
            "type": "string",
            "description": "Analytical lens: 'investor', 'operator', 'competitor', 'customer'.",
            "minLength": 1,
        },
    },
    "required": ["subject", "context", "perspective"],
}

_FIXTURE = {
    "subject": "a B2B SaaS startup offering AI-powered contract review",
    "context": (
        "The company has 50 enterprise customers, $2M ARR, growing 15% MoM. "
        "Main competitors are legacy legal software providers and a well-funded startup "
        "backed by a16z. The market is estimated at $5B and growing with AI adoption."
    ),
    "perspective": "investor",
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "opportunities": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "threats": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "strategic_insights": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["strengths", "weaknesses", "opportunities", "threats", "strategic_insights"],
}


def _swot_analysis_structured(subject: str, context: str, perspective: str) -> str:
    return (
        f"You are a strategic business analyst. Conduct a thorough SWOT analysis of the "
        f"following subject from the perspective of a {perspective}.\n\n"
        f"Subject: {subject}\n\n"
        f"Context:\n{context}\n\n"
        f"Return a JSON object with exactly these keys:\n"
        f'  "strengths"          — array of 3-5 internal advantages (strings)\n'
        f'  "weaknesses"         — array of 3-5 internal vulnerabilities (strings)\n'
        f'  "opportunities"      — array of 3-5 external growth opportunities (strings)\n'
        f'  "threats"            — array of 3-5 external risks or competitive pressures (strings)\n'
        f'  "strategic_insights" — array of 2-3 cross-quadrant strategic insights that '
        f"combine S/W/O/T to suggest concrete strategic moves (strings)\n\n"
        f"Be specific and analytical — avoid generic platitudes. "
        f"Output only the JSON object, no markdown fences, no extra text."
    )


def _json_schema_check(output: str) -> bool | str:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema import exceptions as jsonschema_exceptions
    except ImportError:
        return "jsonschema required"

    parsed = PromptEval._extract_json_object(output)
    if parsed is None:
        return "output is not valid JSON"
    try:
        Draft202012Validator(_EXPECTED_SCHEMA).validate(parsed)
    except jsonschema_exceptions.ValidationError as exc:
        return f"JSON schema validation failed: {exc.message}"
    return True


swot_analysis_v1 = LibraryPrompt(
    name="business.swot_analysis.v1",
    domain="business",
    variant="structured",
    description=(
        "Structured SWOT analysis returning JSON with strengths, weaknesses, opportunities, "
        "threats, and strategic insights. Perspective-aware."
    ),
    template=_swot_analysis_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _json_schema_check,
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["business", "swot", "strategy", "structured", "json"],
)

registry.register(swot_analysis_v1)
