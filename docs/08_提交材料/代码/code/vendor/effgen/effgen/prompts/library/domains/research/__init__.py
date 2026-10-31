"""Research domain prompt templates."""

from .citation_extract_v1 import citation_extract_v1
from .literature_review_v1 import lit_review_cot, lit_review_zero_shot
from .methodology_critique_v1 import methodology_critique_v1
from .paper_summary_v1 import paper_summary_v1

__all__ = [
    "lit_review_zero_shot",
    "lit_review_cot",
    "paper_summary_v1",
    "citation_extract_v1",
    "methodology_critique_v1",
]
