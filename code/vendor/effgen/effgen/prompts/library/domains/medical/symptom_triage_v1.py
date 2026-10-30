"""
Medical prompt: Symptom Triage — structured-output variant.

Returns JSON: {assessment, urgency_level, see_doctor_if, self_care, disclaimer}
The 'disclaimer' key contains DISCLAIMERS.medical verbatim.
Every render also includes DISCLAIMERS.medical in the system prompt.

Inputs: symptoms (str), patient_context (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.disclaimers import DISCLAIMERS
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "symptoms": {
            "type": "string",
            "description": "Description of the patient's symptoms.",
            "minLength": 5,
        },
        "patient_context": {
            "type": "string",
            "description": (
                "Optional de-identified context: age group, known conditions, "
                "duration of symptoms, etc."
            ),
            "default": "",
        },
    },
    "required": ["symptoms"],
}

_FIXTURE = {
    "symptoms": (
        "Fever of 38.5°C (101.3°F) for 2 days, dry cough, mild sore throat, "
        "fatigue, and mild headache. No shortness of breath."
    ),
    "patient_context": "Adult, no known chronic conditions, vaccinated against flu.",
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string", "minLength": 5},
        "urgency_level": {
            "type": "string",
            "enum": ["routine", "soon", "urgent", "emergency"],
        },
        "see_doctor_if": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "self_care": {
            "type": "array",
            "items": {"type": "string"},
        },
        "disclaimer": {"type": "string", "minLength": 20},
    },
    "required": ["assessment", "urgency_level", "see_doctor_if", "self_care", "disclaimer"],
}


def _json_schema_check(output: str) -> bool | str:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema import exceptions as jsonschema_exceptions
    except ImportError:
        return "jsonschema required"

    from effgen.prompts.library.eval import PromptEval

    parsed = PromptEval._extract_json_object(output)
    if parsed is None:
        return "output is not valid JSON"
    try:
        Draft202012Validator(_EXPECTED_SCHEMA).validate(parsed)
    except jsonschema_exceptions.ValidationError as exc:
        return f"JSON schema validation failed: {exc.message}"
    return True


def _symptom_triage_structured(symptoms: str, patient_context: str = "") -> str:
    context_block = (
        f"\nPatient context: {patient_context}" if patient_context.strip() else ""
    )
    return (
        f"{DISCLAIMERS.medical}\n\n"
        "You are a medical triage assistant. Based on the reported symptoms, provide "
        "a lay-language triage assessment to help the person decide on next steps. "
        "You are NOT diagnosing or prescribing — only helping prioritize care.\n"
        f"\nSymptoms: {symptoms}{context_block}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '  "assessment"    — brief plain-language assessment (string)\n'
        '  "urgency_level" — one of: "routine" | "soon" | "urgent" | "emergency"\n'
        '  "see_doctor_if" — array of strings: red-flag symptoms that warrant immediate care\n'
        '  "self_care"     — array of self-care suggestions (empty array if not appropriate)\n'
        f'  "disclaimer"    — include this exact string: "{DISCLAIMERS.medical}"\n\n'
        "Output only the JSON object, no markdown fences, no extra text."
    )


symptom_triage_v1 = LibraryPrompt(
    name="medical.symptom_triage.v1",
    domain="medical",
    variant="structured",
    description=(
        "Triage reported symptoms into urgency level with see_doctor_if conditions. "
        "Structured JSON with mandatory disclaimer field. Includes mandatory medical disclaimer."
    ),
    template=_symptom_triage_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _json_schema_check,
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["medical", "triage", "structured", "json", "disclaimer"],
)

registry.register(symptom_triage_v1)
