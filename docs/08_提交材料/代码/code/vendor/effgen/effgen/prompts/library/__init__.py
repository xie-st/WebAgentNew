"""A curated, domain-organized catalog of prompt templates with an evaluation harness.
Usage:
    from effgen.prompts.library import registry, LibraryPrompt
    prompts = registry.search(domain="research")
"""

from .base import LibraryPrompt
from .eval import PromptEval
from .registry import PromptRegistry, registry

__all__ = [
    "LibraryPrompt",
    "PromptRegistry",
    "PromptEval",
    "registry",
]
