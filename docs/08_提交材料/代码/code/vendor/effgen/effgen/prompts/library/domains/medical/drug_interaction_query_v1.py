"""
Medical prompt: Drug Interaction Query — structured-output variant.

Returns JSON: {interactions, severity_overall, recommendations, disclaimer}
Every render includes DISCLAIMERS.medical verbatim.

Inputs: drugs (list as comma-separated str or list), patient_notes (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.disclaimers import DISCLAIMERS
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "drugs": {
            "type": "string",
            "description": "Comma-separated list of drug names to check for interactions.",
            "minLength": 3,
        },
        "patient_notes": {
            "type": "string",
            "description": "Optional de-identified notes (age group, conditions, renal/hepatic status).",
            "default": "",
        },
    },
    "required": ["drugs"],
}

_FIXTURE = {
    "drugs": "warfarin, aspirin, ibuprofen",
    "patient_notes": "Elderly adult, atrial fibrillation, no known renal impairment.",
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "interactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "drug_pair": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["minor", "moderate", "major", "contraindicated"],
                    },
                    "description": {"type": "string"},
                },
                "required": ["drug_pair", "severity", "description"],
            },
        },
        "severity_overall": {
            "type": "string",
            "enum": ["minor", "moderate", "major", "contraindicated"],
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "disclaimer": {"type": "string", "minLength": 20},
    },
    "required": ["interactions", "severity_overall", "recommendations", "disclaimer"],
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


def _drug_interaction_structured(drugs: str, patient_notes: str = "") -> str:
    context_block = (
        f"\nPatient notes: {patient_notes}" if patient_notes.strip() else ""
    )
    return (
        f"{DISCLAIMERS.medical}\n\n"
        "You are a clinical pharmacology assistant. Identify known drug interactions "
        "between the listed medications based on established pharmacological knowledge. "
        "You are providing general information — not patient-specific advice.\n"
        f"\nDrugs: {drugs}{context_block}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '  "interactions"     — array of objects, each with:\n'
        '                       "drug_pair": "<DrugA> + <DrugB>"\n'
        '                       "severity":  "minor" | "moderate" | "major" | "contraindicated"\n'
        '                       "description": plain-language explanation\n'
        '  "severity_overall" — highest severity across all interactions\n'
        '  "recommendations"  — array of action strings (e.g., "Monitor INR closely")\n'
        f'  "disclaimer"       — include this exact string: "{DISCLAIMERS.medical}"\n\n'
        "If no interactions are known, return an empty interactions array. "
        "Output only the JSON object, no markdown fences, no extra text."
    )


drug_interaction_query_v1 = LibraryPrompt(
    name="medical.drug_interaction_query.v1",
    domain="medical",
    variant="structured",
    description=(
        "Query known drug interactions for a list of medications. "
        "Structured JSON with severity levels and recommendations. "
        "Includes mandatory medical disclaimer."
    ),
    template=_drug_interaction_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _json_schema_check,
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["medical", "pharmacology", "drug-interaction", "structured", "json"],
)

registry.register(drug_interaction_query_v1)
