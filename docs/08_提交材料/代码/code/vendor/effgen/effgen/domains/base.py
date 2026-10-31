"""
Domain base class for effGen.

A Domain bundles seed keywords, tools, system prompts, and guardrails
for a particular knowledge area. The ``expand_keywords`` method
delegates to :class:`~effgen.domains.expander.KeywordExpander` to
grow N seed keywords into a larger set of related **search-query
variants** (a query expander, not a synonym thesaurus).

Usage:
    from effgen.domains.base import Domain

    domain = Domain(
        name="tech",
        keywords=["Python", "machine learning"],
        system_prompt="You are a technology expert.",
    )
    expanded = domain.expand_keywords(factor=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Domain:
    """Configurable knowledge domain.

    Attributes:
        name: Domain identifier.
        keywords: Seed keywords for this domain.
        description: Human-readable description of the domain.
        system_prompt: System prompt tailored to the domain.
        tool_names: Names of tools relevant to this domain.
        guardrails: Optional guardrail preset name or GuardrailChain.
        templates: Query templates used by the template-based expander.
        metadata: Arbitrary extra metadata.
    """

    name: str
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    system_prompt: str = "You are a helpful AI assistant."
    tool_names: list[str] = field(default_factory=list)
    guardrails: Any = None
    # Default templates are tech how-to oriented. The built-in non-tech presets
    # (Legal/Finance/Health/Science) override these with field-appropriate query
    # variants; for a custom non-tech Domain, pass your own ``templates=`` or use
    # ``expand_keywords(use_llm=True, model=...)`` for the highest-quality terms.
    templates: list[str] = field(default_factory=lambda: [
        "{kw} tutorial",
        "{kw} examples",
        "{kw} guide",
        "what is {kw}",
        "how to use {kw}",
        "learn {kw}",
        "best {kw} tools",
        "{kw} vs {alt}",
        "{kw} for beginners",
    ])
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Keyword expansion
    # ------------------------------------------------------------------

    def expand_keywords(
        self,
        factor: int = 10,
        *,
        use_wordnet: bool = False,
        use_templates: bool = True,
        use_llm: bool = False,
        model: Any = None,
    ) -> list[str]:
        """Expand seed keywords by the given factor.

        Combines multiple strategies (templates, optional LLM-based
        generation, optional WordNet synonyms) and deduplicates.

        Args:
            factor: Target multiplier — aim for ``len(keywords) * factor``
                expanded terms.
            use_wordnet: Enable WordNet synonym expansion (requires nltk).
                Off by default — it has no word-sense disambiguation and
                drifts off-domain for polysemous seeds.
            use_templates: Enable template-based expansion.
            use_llm: Enable LLM-based expansion (requires *model*).
            model: An effgen BaseModel instance for LLM expansion.

        Returns:
            Deduplicated list of expanded keyword strings.
        """
        from .expander import KeywordExpander

        expander = KeywordExpander(
            use_wordnet=use_wordnet,
            use_templates=use_templates,
            use_llm=use_llm,
            model=model,
            templates=self.templates,
        )
        expanded = expander.expand(self.keywords, factor=factor)
        logger.info(
            "Domain '%s': expanded %d seed keywords to %d terms (factor=%d)",
            self.name, len(self.keywords), len(expanded), factor,
        )
        return expanded

    # ------------------------------------------------------------------
    # Build a runnable agent
    # ------------------------------------------------------------------

    def to_agent(self, model: Any = None, **overrides: Any) -> Any:
        """Build a runnable :class:`~effgen.core.agent.Agent` from this domain.

        Wires the domain's ``system_prompt``, ``tool_names`` (resolved through
        the tool registry, skipping any that can't load), and ``guardrails`` into
        an agent — the one obvious on-ramp from a domain to something you can
        ``.run(...)``.

        Args:
            model: A loaded model instance or a model-id string (e.g.
                ``"gpt-5-nano"`` or ``"Qwen/Qwen2.5-1.5B-Instruct"``). Required
                unless ``EFFGEN_DEFAULT_MODEL`` is set.
            **overrides: Any keyword accepted by
                :func:`~effgen.presets.registry.create_agent` — e.g.
                ``extra_tools=``, ``system_prompt=``, ``guardrails=``,
                ``temperature=``, ``max_iterations=``, ``engine=`` (for a local
                model id). The domain's own ``guardrails`` are used unless you
                override them.

        Returns:
            A configured ``Agent`` ready to ``run(...)``.

        Example:
            >>> from effgen.domains import LegalDomain
            >>> agent = LegalDomain().to_agent("gpt-5-nano")
            >>> agent.run("Summarize the obligations in a standard NDA.")
        """
        from effgen.presets.registry import create_agent

        return create_agent(domain=self, model=model, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "name": self.name,
            "keywords": self.keywords,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tool_names": self.tool_names,
        }
