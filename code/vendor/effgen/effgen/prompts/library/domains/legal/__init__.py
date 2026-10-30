"""Legal domain prompt templates."""

from .clause_classify_v1 import clause_classify_v1
from .contract_summarize_v1 import contract_summarize_v1
from .legal_research_brief_v1 import legal_research_brief_v1

__all__ = [
    "contract_summarize_v1",
    "clause_classify_v1",
    "legal_research_brief_v1",
]
