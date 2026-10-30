"""effGen caching subsystem.

Provides prompt-prefix caching and result caching for tools and agents.
All components are pure-Python and have no required external dependencies.
"""
from __future__ import annotations

from .prompt_cache import PromptCache, PromptCacheEntry
from .result_cache import ResultCache, ResultCacheEntry

__all__ = [
    "PromptCache",
    "PromptCacheEntry",
    "ResultCache",
    "ResultCacheEntry",
]
