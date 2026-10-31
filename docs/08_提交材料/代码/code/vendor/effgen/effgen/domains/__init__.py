"""
effGen Domains — Knowledge domain definitions with keyword expansion.

Provides domain-specific configurations (keywords, prompts, tools, guardrails)
and a KeywordExpander that grows N seed keywords to 10N+ using templates,
optional LLM-based expansion, and opt-in WordNet synonyms.

Usage:
    from effgen.domains import TechDomain, KeywordExpander, Domain

    domain = TechDomain(keywords=["Python", "machine learning"])
    expanded = domain.expand_keywords(factor=10)
"""

from effgen.domains.base import Domain
from effgen.domains.expander import KeywordExpander
from effgen.domains.presets import (
    FinanceDomain,
    HealthDomain,
    LegalDomain,
    ScienceDomain,
    TechDomain,
)

__all__ = [
    "Domain",
    "KeywordExpander",
    "TechDomain",
    "ScienceDomain",
    "FinanceDomain",
    "HealthDomain",
    "LegalDomain",
]
