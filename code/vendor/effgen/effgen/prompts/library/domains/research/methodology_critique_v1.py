"""
Research prompt: Methodology Critique — CoT variant.

Inputs: methodology_text (str), field (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "methodology_text": {"type": "string", "minLength": 10},
        "field": {"type": "string", "minLength": 1},
    },
    "required": ["methodology_text", "field"],
}

_FIXTURE = {
    "methodology_text": (
        "We collected survey responses from 150 undergraduate students at a single "
        "university and asked them to rate their agreement with 12 Likert-scale items "
        "about AI tool adoption. Data were analyzed using descriptive statistics and "
        "multiple linear regression without adjusting for multiple comparisons."
    ),
    "field": "educational technology",
}


def _methodology_critique_cot(methodology_text: str, field: str) -> str:
    return (
        f"You are an expert methodologist in {field}. Critically evaluate the following "
        f"research methodology using chain-of-thought reasoning.\n\n"
        f"Methodology description:\n{methodology_text}\n\n"
        f"Reason through each of these dimensions in order:\n\n"
        f"1. **Study Design**: Is the design appropriate for the stated goals? "
        f"Identify the design type and any fundamental flaws.\n\n"
        f"2. **Sampling**: Assess sampling strategy, sample size adequacy, and "
        f"representativeness. Are there selection biases?\n\n"
        f"3. **Measurement**: Evaluate the measurement instruments, operationalizations, "
        f"and potential construct validity issues.\n\n"
        f"4. **Analysis**: Critique the statistical or qualitative analysis approach. "
        f"Are assumptions tested? Are there Type I/II error concerns?\n\n"
        f"5. **Generalizability**: To what populations or contexts can findings be "
        f"generalized? What limits external validity?\n\n"
        f"6. **Overall Verdict**: Summarize the three most critical weaknesses and "
        f"suggest concrete improvements for each.\n\n"
        f"Show your reasoning before each verdict."
    )


methodology_critique_v1 = LibraryPrompt(
    name="research.methodology_critique.v1",
    domain="research",
    variant="cot",
    description=(
        "Chain-of-thought methodology critique covering design, sampling, "
        "measurement, analysis, and generalizability."
    ),
    template=_methodology_critique_cot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(design|sampling|measurement|analysis|validity|weakness|limitation)",
    },
    tags=["research", "methodology", "critique", "cot"],
)

registry.register(methodology_critique_v1)
