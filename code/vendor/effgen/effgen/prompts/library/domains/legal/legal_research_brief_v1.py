"""
Legal prompt: Legal Research Brief — tool-augmented variant.

Produces a concise legal research brief on a topic, incorporating
retrieved case law / statute references supplied as tool context.
Every render includes DISCLAIMERS.legal verbatim.

Inputs: topic (str), jurisdiction (str), retrieved_sources (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.disclaimers import DISCLAIMERS
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "Legal research topic or question.",
            "minLength": 5,
        },
        "jurisdiction": {
            "type": "string",
            "description": "Jurisdiction (e.g. 'US federal', 'California', 'UK').",
            "default": "US federal",
        },
        "retrieved_sources": {
            "type": "string",
            "description": (
                "Pre-retrieved case law, statutes, or regulatory text "
                "to ground the brief (tool output piped in)."
            ),
            "default": "",
        },
    },
    "required": ["topic"],
}

_FIXTURE = {
    "topic": "Enforceability of non-compete agreements for remote software engineers",
    "jurisdiction": "California",
    "retrieved_sources": (
        "[Source 1] Cal. Bus. & Prof. Code § 16600: Every contract by which anyone is "
        "restrained from engaging in a lawful profession, trade, or business of any kind "
        "is to that extent void.\n"
        "[Source 2] Edwards v. Arthur Andersen LLP (2008) 44 Cal.4th 937: California "
        "courts strictly enforce § 16600 and reject the 'rule of reason' approach.\n"
        "[Source 3] AMN Healthcare v. Aya Healthcare (2018): Non-solicitation clauses "
        "that effectively restrain competition are also void under § 16600."
    ),
}


def _legal_research_brief_tool(
    topic: str,
    jurisdiction: str = "US federal",
    retrieved_sources: str = "",
) -> str:
    sources_block = (
        f"\nRetrieved sources (use these to ground your brief):\n{retrieved_sources}\n"
        if retrieved_sources.strip()
        else "\n(No pre-retrieved sources provided — rely on general legal knowledge.)\n"
    )
    return (
        f"{DISCLAIMERS.legal}\n\n"
        "You are a legal research assistant. Produce a concise research brief on the "
        "following topic for an attorney's review.\n\n"
        f"Topic: {topic}\n"
        f"Jurisdiction: {jurisdiction}\n"
        f"{sources_block}\n"
        "Structure your brief as follows:\n"
        "1. Issue — one-sentence statement of the legal question\n"
        "2. Applicable Law — relevant statutes and key cases (cite sources where provided)\n"
        "3. Analysis — how the law applies to the topic\n"
        "4. Conclusion — one-sentence bottom line\n"
        "5. Open Questions — any unresolved issues or gaps in the sources\n\n"
        "Be precise. Cite specific statutes/cases from the sources when available. "
        "Do not fabricate citations."
    )


legal_research_brief_v1 = LibraryPrompt(
    name="legal.legal_research_brief.v1",
    domain="legal",
    variant="tool",
    description=(
        "Produce a structured legal research brief grounded in pre-retrieved sources. "
        "Tool-augmented. Includes mandatory legal disclaimer."
    ),
    template=_legal_research_brief_tool,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(Issue|Applicable Law|Analysis|Conclusion)",
    },
    tags=["legal", "research", "brief", "tool"],
)

registry.register(legal_research_brief_v1)
