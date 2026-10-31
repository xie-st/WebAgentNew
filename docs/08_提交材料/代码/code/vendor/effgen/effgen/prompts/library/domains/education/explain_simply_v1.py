"""Education prompt: explain a concept for a given audience using an everyday analogy.

Inputs: concept (str), audience (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "concept": {
            "type": "string",
            "description": "The idea to explain, e.g. 'recursion', 'inflation'.",
            "minLength": 2,
        },
        "audience": {
            "type": "string",
            "description": "Who it is for, e.g. 'a 10-year-old', 'a non-technical manager'.",
            "minLength": 2,
        },
    },
    "required": ["concept", "audience"],
}

_FIXTURE = {
    "concept": "recursion",
    "audience": "a 10-year-old",
}


def _explain_simply(concept: str, audience: str) -> str:
    return (
        f"Explain \"{concept}\" to {audience}.\n\n"
        "Requirements:\n"
        f"  - Use plain words {audience} already knows; define any new term in passing.\n"
        "  - Anchor the explanation in ONE concrete everyday analogy.\n"
        "  - Keep it to a short paragraph (3-5 sentences).\n"
        "  - End with a single sentence that checks understanding "
        "(\"Does that make sense, or should I try a different example?\").\n\n"
        "Write the explanation now:"
    )


explain_simply_v1 = LibraryPrompt(
    name="education.explain_simply.v1",
    domain="education",
    variant="zero_shot",
    description=(
        "Explain a concept for a specific audience with one everyday analogy and a "
        "comprehension check. Inputs: concept, audience."
    ),
    template=_explain_simply,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape=None,
    tags=["education", "explain", "analogy", "zero_shot"],
)

registry.register(explain_simply_v1)
