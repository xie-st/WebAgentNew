"""
Research prompt: Citation Extraction — tool-augmented variant.

Renders a deterministic, tool-aware prompt that instructs an agent to use the
ArXiv and PubMed tools before extracting citation metadata.

Inputs:
    query (str)         — search query for ArXiv/PubMed
    source (str)        — "arxiv" | "pubmed" | "auto"
    max_results (int)   — max papers to fetch (default 5)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "source": {"type": "string", "enum": ["arxiv", "pubmed", "auto"]},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
        "citation_context": {"type": "string"},
    },
    "required": ["query"],
}

_FIXTURE = {
    "query": "transformer neural network attention mechanism",
    "source": "arxiv",
    "max_results": 3,
    "citation_context": "",
}


def _citation_extract_tool(
    query: str,
    source: str = "arxiv",
    max_results: int = 3,
    citation_context: str = "",
) -> str:
    src = source.lower()
    if src == "auto":
        search_instruction = (
            "Use ArXivTool first for computer science, physics, mathematics, and "
            "preprint-heavy topics; use PubMedTool first for biomedical topics. "
            "If the topic overlaps both areas, query both tools."
        )
    elif src == "pubmed":
        search_instruction = "Use PubMedTool with operation='search'."
    else:
        src = "arxiv"
        search_instruction = "Use ArXivTool with operation='search'."

    context_block = (
        citation_context.strip()
        if citation_context.strip()
        else "No pre-fetched citation context was provided. Retrieve live metadata with the selected tool before writing the final answer."
    )

    return (
        "You are a research assistant specializing in academic citation analysis.\n\n"
        "Tool use required before final answer:\n"
        f"- Query: {query}\n"
        f"- Source preference: {src.upper()}\n"
        f"- Maximum results: {max_results}\n"
        f"- Tool routing: {search_instruction}\n\n"
        "Available tools:\n"
        "- ArXivTool: search scholarly preprints and return title, authors, arxiv_id, published date, URL, and summary.\n"
        "- PubMedTool: search biomedical literature and return title, authors, PMID, journal, publication date, and URL.\n\n"
        "Pre-fetched citation context:\n"
        f"{context_block}\n\n"
        "After retrieving metadata, extract and format each citation. For each paper, identify:\n"
        "1. The full citation in APA style when metadata is available.\n"
        "2. The main claim or contribution of the paper.\n"
        "3. Whether it is an empirical, theoretical, methods, or survey paper.\n\n"
        "Return a structured citation list followed by a brief synthesis of how the papers relate to each other and to the query topic."
    )


citation_extract_v1 = LibraryPrompt(
    name="research.citation_extract.v1",
    domain="research",
    variant="tool",
    description=(
        "Tool-augmented citation extraction using ArXiv/PubMed APIs. "
        "Instructs agents to retrieve live paper metadata before answering."
    ),
    template=_citation_extract_tool,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(citation|paper|author|reference|arxiv|pubmed)",
    },
    tags=["research", "citation", "tool", "arxiv", "pubmed"],
)

registry.register(citation_extract_v1)
