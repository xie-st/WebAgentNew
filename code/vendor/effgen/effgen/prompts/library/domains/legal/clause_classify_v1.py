"""
Legal prompt: Clause Classify — zero-shot variant.

Classifies a contract clause by type and flags notable characteristics.
Every render includes DISCLAIMERS.legal verbatim.

Inputs: clause_text (str), context (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.disclaimers import DISCLAIMERS
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "clause_text": {
            "type": "string",
            "description": "The contract clause to classify.",
            "minLength": 10,
        },
        "context": {
            "type": "string",
            "description": "Optional context about the contract type (e.g. 'SaaS subscription').",
            "default": "",
        },
    },
    "required": ["clause_text"],
}

_FIXTURE = {
    "clause_text": (
        "Neither party shall be liable to the other for any indirect, incidental, "
        "special, consequential, or punitive damages arising out of or related to "
        "this Agreement, even if such party has been advised of the possibility of "
        "such damages. Each party's total cumulative liability shall not exceed the "
        "amounts paid or payable under this Agreement in the twelve months preceding "
        "the claim."
    ),
    "context": "SaaS software license agreement",
}

_CLAUSE_TYPES = (
    "indemnification | limitation-of-liability | confidentiality | ip-ownership | "
    "termination | payment | warranty | dispute-resolution | force-majeure | "
    "non-compete | non-solicitation | governing-law | other"
)


def _clause_classify_zero_shot(clause_text: str, context: str = "") -> str:
    context_block = f"\nContract context: {context}" if context.strip() else ""
    return (
        f"{DISCLAIMERS.legal}\n\n"
        "You are a legal document analyst. Classify the following contract clause "
        "by its type and note any characteristics that a reviewing attorney should "
        "be aware of.\n"
        f"{context_block}\n\n"
        "Clause:\n"
        f'"""\n{clause_text}\n"""\n\n'
        f"Clause types (pick the best match): {_CLAUSE_TYPES}\n\n"
        "Respond with:\n"
        "1. Clause Type: <type>\n"
        "2. Plain-language summary (1–2 sentences)\n"
        "3. Notable characteristics or risk flags (bullet list, or 'None')\n\n"
        "Be concise and factual. Do not give legal advice."
    )


clause_classify_v1 = LibraryPrompt(
    name="legal.clause_classify.v1",
    domain="legal",
    variant="zero_shot",
    description=(
        "Classify a contract clause by type and flag notable characteristics. "
        "Zero-shot. Includes mandatory legal disclaimer."
    ),
    template=_clause_classify_zero_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"clause\s+type\s*:",
    },
    tags=["legal", "clause", "classification", "zero_shot"],
)

registry.register(clause_classify_v1)
