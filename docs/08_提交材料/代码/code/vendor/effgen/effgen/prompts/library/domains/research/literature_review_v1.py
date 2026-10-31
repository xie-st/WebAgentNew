"""
Research prompt: Literature Review — zero-shot and CoT variants.

Inputs: topic (str), years_range (str), max_papers (int)
"""

from __future__ import annotations

import re

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "minLength": 1},
        "years_range": {"type": "string", "minLength": 1},
        "max_papers": {"type": "integer", "minimum": 1, "maximum": 100},
    },
    "required": ["topic", "years_range", "max_papers"],
}

_FIXTURE = {
    "topic": "diffusion models for image generation",
    "years_range": "2022-2024",
    "max_papers": 10,
}


def _has_three_candidate_paper_lines(output: str) -> bool | str:
    """Check for at least three candidate citation-like lines in live output."""
    paper_lines = []
    for line in output.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        has_year = re.search(r"\b(?:19|20)\d{2}\b", normalized)
        has_citation_marker = re.search(
            r"(\bet al\.?\b|\barxiv\b|\bdoi\b|\([A-Z][^)]+,\s*(?:19|20)\d{2}\)|^\s*(?:[-*]|\d+[.)]))",
            normalized,
            re.IGNORECASE,
        )
        if has_year and has_citation_marker:
            paper_lines.append(normalized)

    if len(paper_lines) >= 3:
        return True
    return f"expected at least 3 candidate paper-like lines, found {len(paper_lines)}"


# ---------------------------------------------------------------------------
# Zero-shot variant
# ---------------------------------------------------------------------------

def _lit_review_zero_shot(topic: str, years_range: str, max_papers: int) -> str:
    return (
        f"You are an expert research assistant. Conduct a literature review on "
        f'"{topic}" covering the period {years_range}. Identify up to {max_papers} '
        f"key papers, summarize their contributions, and highlight the major themes, "
        f"research gaps, and open challenges in the field. Present your findings in "
        f"a structured format with sections: Overview, Key Papers, Major Themes, "
        f"Research Gaps."
    )


lit_review_zero_shot = LibraryPrompt(
    name="research.literature_review.v1.zero_shot",
    domain="research",
    variant="zero_shot",
    description="Zero-shot literature review covering a topic and year range.",
    template=_lit_review_zero_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _has_three_candidate_paper_lines,
    },
    tags=["research", "literature", "survey", "zero_shot"],
)

registry.register(lit_review_zero_shot)


# ---------------------------------------------------------------------------
# Chain-of-thought variant
# ---------------------------------------------------------------------------

def _lit_review_cot(topic: str, years_range: str, max_papers: int) -> str:
    return (
        f"You are an expert research assistant conducting a literature review on "
        f'"{topic}" for the period {years_range}.\n\n'
        f"Think step by step:\n"
        f"1. First, identify the key sub-topics and research directions within this field.\n"
        f"2. For each direction, name up to {max_papers} representative papers with year, "
        f"authors, and a one-sentence contribution summary.\n"
        f"3. Synthesize the major findings: what is well-established vs. actively debated?\n"
        f"4. Identify at least three research gaps or open problems.\n"
        f"5. Conclude with a brief paragraph on promising future directions.\n\n"
        f"Show your reasoning at each step before writing the final structured review."
    )


lit_review_cot = LibraryPrompt(
    name="research.literature_review.v1.cot",
    domain="research",
    variant="cot",
    description="Chain-of-thought literature review with step-by-step reasoning.",
    template=_lit_review_cot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _has_three_candidate_paper_lines,
    },
    tags=["research", "literature", "survey", "cot"],
)

registry.register(lit_review_cot)
