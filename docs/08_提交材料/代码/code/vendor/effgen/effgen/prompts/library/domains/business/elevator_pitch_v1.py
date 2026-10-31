"""
Business prompt: Elevator Pitch — zero-shot variant with ≤150 word constraint.

Inputs: product_name (str), target_audience (str), problem (str),
        solution (str), differentiator (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {
            "type": "string",
            "description": "Name of the product, service, or company.",
            "minLength": 1,
        },
        "target_audience": {
            "type": "string",
            "description": "Primary customer segment being addressed.",
            "minLength": 3,
        },
        "problem": {
            "type": "string",
            "description": "The core problem being solved.",
            "minLength": 10,
        },
        "solution": {
            "type": "string",
            "description": "How the product/service solves the problem.",
            "minLength": 10,
        },
        "differentiator": {
            "type": "string",
            "description": "What makes this unique versus alternatives.",
            "minLength": 5,
        },
    },
    "required": ["product_name", "target_audience", "problem", "solution", "differentiator"],
}

_FIXTURE = {
    "product_name": "ContractIQ",
    "target_audience": "legal teams at mid-size companies",
    "problem": (
        "Contract review takes 3-5 days per agreement and legal teams are drowning in "
        "volume, missing risk clauses and delaying deals."
    ),
    "solution": (
        "ContractIQ uses AI to review contracts in under 2 minutes, flagging risky clauses, "
        "summarizing obligations, and suggesting redlines."
    ),
    "differentiator": (
        "Unlike generic AI tools, ContractIQ is trained on 10M+ enterprise contracts "
        "and integrates directly into CLM platforms like DocuSign and Ironclad."
    ),
}


def _elevator_pitch_word_count_check(output: str) -> bool | str:
    words = output.split()
    if len(words) <= 150:
        return True
    return f"pitch is {len(words)} words — must be ≤150"


def _elevator_pitch_zero_shot(
    product_name: str,
    target_audience: str,
    problem: str,
    solution: str,
    differentiator: str,
) -> str:
    return (
        f"You are a startup pitch coach. Write a compelling elevator pitch for the following "
        f"product. The pitch MUST be 150 words or fewer — count carefully.\n\n"
        f"Product: {product_name}\n"
        f"Target Audience: {target_audience}\n"
        f"Problem: {problem}\n"
        f"Solution: {solution}\n"
        f"Key Differentiator: {differentiator}\n\n"
        f"Requirements:\n"
        f"  - Open with a hook that resonates with {target_audience}\n"
        f"  - Clearly state the problem and solution\n"
        f"  - Highlight what makes {product_name} uniquely better\n"
        f"  - End with a memorable call to action or vision statement\n"
        f"  - STRICT LIMIT: 150 words or fewer (this is non-negotiable)\n\n"
        f"Write the elevator pitch now:"
    )


elevator_pitch_v1 = LibraryPrompt(
    name="business.elevator_pitch.v1",
    domain="business",
    variant="zero_shot",
    description=(
        "Zero-shot elevator pitch generator with strict ≤150 word constraint. "
        "Inputs: product name, target audience, problem, solution, differentiator."
    ),
    template=_elevator_pitch_zero_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _elevator_pitch_word_count_check,
    },
    tags=["business", "pitch", "startup", "zero_shot", "word-limit"],
)

registry.register(elevator_pitch_v1)
