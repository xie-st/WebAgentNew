"""
Research prompt: Paper Summary — structured-output variant.

Returns JSON with keys: abstract_summary, key_findings, limitations, future_work.

Inputs: title (str), abstract (str), authors (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "abstract": {"type": "string", "minLength": 1},
        "authors": {"type": "string"},
    },
    "required": ["title", "abstract"],
}

_FIXTURE = {
    "title": "Attention Is All You Need",
    "abstract": (
        "The dominant sequence transduction models are based on complex recurrent or "
        "convolutional neural networks that include an encoder and a decoder. The best "
        "performing models also connect the encoder and decoder through an attention "
        "mechanism. We propose a new simple network architecture, the Transformer, based "
        "solely on attention mechanisms, dispensing with recurrence and convolutions "
        "entirely. Experiments on two machine translation tasks show these models to be "
        "superior in quality while being more parallelizable and requiring significantly "
        "less time to train."
    ),
    "authors": "Vaswani et al.",
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "abstract_summary": {"type": "string", "minLength": 1},
        "key_findings": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
        "future_work": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
    },
    "required": ["abstract_summary", "key_findings", "limitations", "future_work"],
    "additionalProperties": False,
}


def _paper_summary_structured(title: str, abstract: str, authors: str = "") -> str:
    if authors:
        trimmed = authors.rstrip(".")
        author_line = f" by {trimmed}"
    else:
        author_line = ""
    return (
        f'Summarize the following academic paper{author_line}.\n\n'
        f"Title: {title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Return a JSON object with exactly these four keys (and no others):\n"
        f"  abstract_summary  — 2–3 sentence plain-language summary\n"
        f"  key_findings      — list of 3–5 main contributions or results\n"
        f"  limitations       — list of 2–4 noted limitations or caveats\n"
        f"  future_work       — list of 2–3 suggested future research directions\n\n"
        f"Do NOT include any additional keys (e.g. no 'methodology', 'authors', "
        f"'conclusion', 'introduction'). Output only the JSON object — no "
        f"markdown fences, no extra text, no commentary."
    )


paper_summary_v1 = LibraryPrompt(
    name="research.paper_summary.v1",
    domain="research",
    variant="structured",
    description=(
        "Structured paper summary returning JSON with abstract_summary, key_findings, "
        "limitations, and future_work."
    ),
    template=_paper_summary_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "json",
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["research", "summary", "structured", "json"],
)

registry.register(paper_summary_v1)
