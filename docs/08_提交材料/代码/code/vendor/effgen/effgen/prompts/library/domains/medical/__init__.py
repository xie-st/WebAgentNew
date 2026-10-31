"""Medical domain prompt templates."""

from .drug_interaction_query_v1 import drug_interaction_query_v1
from .medical_literature_v1 import medical_literature_v1
from .symptom_triage_v1 import symptom_triage_v1

__all__ = [
    "symptom_triage_v1",
    "drug_interaction_query_v1",
    "medical_literature_v1",
]
