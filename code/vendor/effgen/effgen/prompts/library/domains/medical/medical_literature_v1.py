"""
Medical prompt: Medical Literature Summary — tool-augmented variant.

Summarizes retrieved PubMed abstracts into a clinical evidence brief.
Every render includes DISCLAIMERS.medical verbatim.

Inputs: clinical_question (str), retrieved_abstracts (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.disclaimers import DISCLAIMERS
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "clinical_question": {
            "type": "string",
            "description": "The clinical question to answer from the literature.",
            "minLength": 10,
        },
        "retrieved_abstracts": {
            "type": "string",
            "description": (
                "Pre-retrieved PubMed abstracts or excerpts (tool output piped in). "
                "Include PMID and title where available."
            ),
            "default": "",
        },
    },
    "required": ["clinical_question"],
}

_FIXTURE = {
    "clinical_question": (
        "What is the evidence for metformin use in reducing cardiovascular risk "
        "in type 2 diabetes patients?"
    ),
    "retrieved_abstracts": (
        "[PMID 28866983] UK Prospective Diabetes Study (UKPDS 34): Metformin decreased "
        "the risk of any diabetes-related endpoint (RR 0.68, 95% CI 0.53-0.87) and "
        "all-cause mortality (RR 0.64) compared with diet alone in overweight type 2 "
        "diabetes patients.\n\n"
        "[PMID 26052981] EMPA-REG OUTCOME (reference): While this trial focused on "
        "empagliflozin, the comparator arm including metformin showed baseline "
        "cardiovascular event rates consistent with established cardiovascular risk.\n\n"
        "[PMID 30811742] A 2019 meta-analysis (n=18 RCTs) found metformin reduced "
        "major adverse cardiovascular events (MACE) by approximately 20% vs. placebo "
        "in type 2 diabetes (pooled HR 0.80, 95% CI 0.72-0.89)."
    ),
}


def _medical_literature_tool(
    clinical_question: str,
    retrieved_abstracts: str = "",
) -> str:
    abstracts_block = (
        f"\nRetrieved abstracts:\n{retrieved_abstracts}\n"
        if retrieved_abstracts.strip()
        else "\n(No abstracts retrieved — summarize based on general medical knowledge.)\n"
    )
    return (
        f"{DISCLAIMERS.medical}\n\n"
        "You are a medical literature analyst. Synthesize the following research "
        "abstracts to answer the clinical question. Cite PMIDs where provided.\n\n"
        f"Clinical question: {clinical_question}\n"
        f"{abstracts_block}\n"
        "Structure your response as follows:\n"
        "1. Bottom Line — one-sentence evidence summary\n"
        "2. Key Evidence — bullet points citing each abstract with PMID\n"
        "3. Evidence Quality — overall quality assessment (e.g., RCT / meta-analysis / observational)\n"
        "4. Limitations — gaps, heterogeneity, or biases in the evidence\n"
        "5. Clinical Implications — what this means for practice (non-prescriptive)\n\n"
        "Be precise. Do not fabricate PMIDs or study results."
    )


medical_literature_v1 = LibraryPrompt(
    name="medical.medical_literature.v1",
    domain="medical",
    variant="tool",
    description=(
        "Synthesize retrieved PubMed abstracts into a structured clinical evidence brief. "
        "Tool-augmented. Includes mandatory medical disclaimer."
    ),
    template=_medical_literature_tool,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(Bottom Line|Key Evidence|Evidence Quality|Clinical Implications)",
    },
    tags=["medical", "literature", "pubmed", "evidence", "tool"],
)

registry.register(medical_literature_v1)
