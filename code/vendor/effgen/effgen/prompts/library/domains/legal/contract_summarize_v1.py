"""
Legal prompt: Contract Summarize — structured-output variant.

Returns JSON: {parties, term, obligations, termination, risks}
Every render includes DISCLAIMERS.legal verbatim.

Inputs: contract_text (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.disclaimers import DISCLAIMERS
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "contract_text": {
            "type": "string",
            "description": "Full text of the contract to summarize.",
            "minLength": 10,
        },
    },
    "required": ["contract_text"],
}

_FIXTURE = {
    "contract_text": (
        "SERVICE AGREEMENT\n\n"
        "This Service Agreement ('Agreement') is entered into as of January 1, 2024, "
        "by and between Acme Corp ('Client') and TechVendor Inc ('Vendor').\n\n"
        "1. SERVICES. Vendor shall provide software development services as described "
        "in Schedule A attached hereto.\n\n"
        "2. TERM. This Agreement shall commence on January 1, 2024 and continue for "
        "twelve (12) months, unless earlier terminated.\n\n"
        "3. PAYMENT. Client shall pay Vendor $10,000 per month within 30 days of invoice.\n\n"
        "4. CONFIDENTIALITY. Each party agrees to maintain the confidentiality of the "
        "other party's proprietary information.\n\n"
        "5. TERMINATION. Either party may terminate this Agreement upon 30 days' written "
        "notice. Client may terminate immediately for cause.\n\n"
        "6. LIMITATION OF LIABILITY. Neither party shall be liable for indirect, "
        "consequential, or punitive damages.\n\n"
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date "
        "first written above."
    ),
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "parties": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "term": {"type": "string", "minLength": 1},
        "obligations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "termination": {"type": "string", "minLength": 1},
        "risks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["parties", "term", "obligations", "termination", "risks"],
}


def _contract_summarize_structured(contract_text: str) -> str:
    return (
        f"{DISCLAIMERS.legal}\n\n"
        "You are a legal document analyst. Your task is to summarize the following "
        "contract in a structured, objective manner. Focus on extracting key facts — "
        "do not offer legal advice or opinions.\n\n"
        "Contract text:\n"
        f"---\n{contract_text}\n---\n\n"
        "Return a JSON object with exactly these keys:\n"
        '  "parties"      — array of party names (strings)\n'
        '  "term"         — contract duration/dates as a string\n'
        '  "obligations"  — array of strings, one per major obligation\n'
        '  "termination"  — termination conditions as a string\n'
        '  "risks"        — array of strings identifying notable risk clauses\n\n'
        "Output only the JSON object, no markdown fences, no extra text."
    )


def _json_schema_check(output: str) -> bool | str:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema import exceptions as jsonschema_exceptions
    except ImportError:
        return "jsonschema required; install effgen[prompts-legal]"

    from effgen.prompts.library.eval import PromptEval

    parsed = PromptEval._extract_json_object(output)
    if parsed is None:
        return "output is not valid JSON"
    try:
        Draft202012Validator(_EXPECTED_SCHEMA).validate(parsed)
    except jsonschema_exceptions.ValidationError as exc:
        return f"JSON schema validation failed: {exc.message}"
    return True


contract_summarize_v1 = LibraryPrompt(
    name="legal.contract_summarize.v1",
    domain="legal",
    variant="structured",
    description=(
        "Summarize a contract into structured JSON with parties, term, obligations, "
        "termination, and risks. Includes mandatory legal disclaimer."
    ),
    template=_contract_summarize_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _json_schema_check,
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["legal", "contract", "structured", "json"],
)

registry.register(contract_summarize_v1)
