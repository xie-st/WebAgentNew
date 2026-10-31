"""
Built-in domain presets for effGen.

Each preset provides curated seed keywords, a domain-specific system prompt,
and recommended tool names for a particular knowledge area.

Usage:
    from effgen.domains import TechDomain, ScienceDomain

    domain = TechDomain()
    expanded = domain.expand_keywords(factor=10)
"""

from __future__ import annotations

from typing import Any

from .base import Domain

# Domain-appropriate query templates. The base ``Domain`` default templates are
# tech how-to oriented ("{kw} tutorial", "{kw} for beginners", "best {kw} tools"),
# which read as junk for a professional non-tech domain. Each set below produces
# query variants a practitioner in that field would actually search for. The LLM
# strategy (``expand_keywords(use_llm=True, model=...)``) remains the highest-
# quality path; these make the offline template default sensible too.
_LEGAL_TEMPLATES = [
    "{kw} clause",
    "{kw} obligations",
    "{kw} regulation",
    "{kw} compliance requirements",
    "{kw} liability",
    "{kw} case law",
    "{kw} statute",
    "{kw} legal definition",
    "{kw} vs {alt}",
]

_FINANCE_TEMPLATES = [
    "{kw} analysis",
    "{kw} risk",
    "{kw} regulation",
    "{kw} valuation",
    "{kw} strategy",
    "{kw} market trends",
    "{kw} metrics",
    "{kw} forecast",
    "{kw} vs {alt}",
]

_HEALTH_TEMPLATES = [
    "{kw} symptoms",
    "{kw} treatment",
    "{kw} diagnosis",
    "{kw} guidelines",
    "{kw} risk factors",
    "{kw} prevention",
    "{kw} causes",
    "{kw} research",
    "{kw} vs {alt}",
]

_SCIENCE_TEMPLATES = [
    "{kw} theory",
    "{kw} experiment",
    "{kw} mechanism",
    "{kw} research",
    "{kw} applications",
    "{kw} principles",
    "{kw} equation",
    "{kw} examples",
    "{kw} vs {alt}",
]


def TechDomain(**overrides: Any) -> Domain:  # noqa: N802
    """Software engineering, programming, and DevOps domain."""
    defaults = {
        "name": "tech",
        "description": "Software engineering, programming languages, DevOps, and cloud computing.",
        "keywords": [
            "Python", "JavaScript", "TypeScript", "Rust", "Go",
            "machine learning", "deep learning", "Docker", "Kubernetes",
            "REST API", "microservices", "CI/CD", "Git", "SQL",
            "cloud computing", "Linux", "algorithms", "data structures",
            "web development", "cybersecurity",
        ],
        "system_prompt": (
            "You are a technology expert specializing in software engineering, "
            "programming, DevOps, and cloud computing. Provide accurate, "
            "up-to-date technical information with code examples when relevant."
        ),
        "tool_names": ["code_executor", "python_repl", "web_search", "bash"],
    }
    defaults.update(overrides)
    return Domain(**defaults)


def ScienceDomain(**overrides: Any) -> Domain:  # noqa: N802
    """Physics, chemistry, biology, and general science domain."""
    defaults = {
        "name": "science",
        "description": "Physics, chemistry, biology, astronomy, and general sciences.",
        "keywords": [
            "quantum mechanics", "organic chemistry", "molecular biology",
            "genetics", "astrophysics", "thermodynamics", "evolution",
            "neuroscience", "ecology", "particle physics",
            "biochemistry", "geology", "climate science", "microbiology",
            "electromagnetism", "relativity", "cell biology",
            "periodic table", "DNA", "photosynthesis",
        ],
        "system_prompt": (
            "You are a science expert covering physics, chemistry, biology, "
            "and related fields. Explain concepts clearly, reference established "
            "theories, and use formulas or diagrams when helpful."
        ),
        "tool_names": ["calculator", "python_repl", "web_search", "wikipedia"],
        "templates": list(_SCIENCE_TEMPLATES),
    }
    defaults.update(overrides)
    return Domain(**defaults)


def FinanceDomain(**overrides: Any) -> Domain:  # noqa: N802
    """Markets, banking, crypto, and personal finance domain."""
    defaults = {
        "name": "finance",
        "description": "Financial markets, banking, cryptocurrency, and personal finance.",
        "keywords": [
            "stock market", "cryptocurrency", "blockchain", "banking",
            "investment", "portfolio management", "risk analysis",
            "financial modeling", "interest rates", "inflation",
            "derivatives", "bonds", "mutual funds", "ETF",
            "fintech", "insurance", "accounting", "taxation",
            "real estate", "venture capital",
        ],
        "system_prompt": (
            "You are a finance expert covering markets, banking, cryptocurrency, "
            "and personal finance. Provide factual analysis, explain financial "
            "concepts clearly, and note that your responses are not financial advice."
        ),
        "tool_names": ["calculator", "web_search", "python_repl"],
        "templates": list(_FINANCE_TEMPLATES),
    }
    defaults.update(overrides)
    return Domain(**defaults)


def HealthDomain(**overrides: Any) -> Domain:  # noqa: N802
    """Medical, wellness, and nutrition domain.

    This is a consumer health-information assistant. It uses the ``standard``
    guardrails, which redact well-formed and labeled identifiers but do not
    guarantee removal of every free-text identifier. It is not a de-identification
    tool. To strip personal identifiers before sending records to a model, pass
    ``guardrails="phi"`` (or ``guardrails=phi_guardrails(strict=True)`` to fail
    closed), and extend ``PIIGuardrail`` with ``custom_patterns``/``custom_terms``
    for site-specific identifiers.
    """
    defaults = {
        "name": "health",
        "description": "Medical science, wellness, nutrition, and public health.",
        "keywords": [
            "nutrition", "exercise", "mental health", "cardiology",
            "immunology", "pharmacology", "epidemiology", "oncology",
            "diabetes", "vaccines", "public health", "anatomy",
            "physical therapy", "dermatology", "pediatrics",
            "sleep science", "stress management", "preventive medicine",
            "clinical trials", "telemedicine",
        ],
        "system_prompt": (
            "You are a health and medical information expert. Provide accurate, "
            "evidence-based health information. Always note that your responses "
            "are for informational purposes and not a substitute for professional "
            "medical advice."
        ),
        "tool_names": ["web_search", "wikipedia", "calculator"],
        "guardrails": "standard",
        "templates": list(_HEALTH_TEMPLATES),
    }
    defaults.update(overrides)
    return Domain(**defaults)


def LegalDomain(**overrides: Any) -> Domain:  # noqa: N802
    """Law, regulations, and compliance domain."""
    defaults = {
        "name": "legal",
        "description": "Law, regulations, compliance, and legal concepts.",
        "keywords": [
            "contract law", "intellectual property", "copyright",
            "patent law", "criminal law", "civil rights", "tort law",
            "corporate law", "employment law", "data privacy",
            "GDPR", "antitrust", "securities regulation", "tax law",
            "constitutional law", "international law", "arbitration",
            "compliance", "litigation", "legal ethics",
        ],
        "system_prompt": (
            "You are a legal information expert. Explain legal concepts, "
            "regulations, and compliance requirements clearly. Always note "
            "that your responses are for informational purposes and not "
            "legal advice. Recommend consulting a licensed attorney for "
            "specific legal matters."
        ),
        "tool_names": ["web_search", "wikipedia"],
        "guardrails": "standard",
        "templates": list(_LEGAL_TEMPLATES),
    }
    defaults.update(overrides)
    return Domain(**defaults)
