"""
Clarification request system for effGen agents.

Allows agents to detect ambiguous queries and request clarification
from the user before proceeding with execution.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ClarificationRequest:
    """
    A request for clarification from the user.

    Attributes:
        question: The clarification question to ask.
        options: Possible answer options (empty = free-text).
        default: Default option index (for options) or default text.
        context: Additional context about why clarification is needed.
    """
    question: str
    options: list[str] = field(default_factory=list)
    default: int | str = 0
    context: str = ""

    def ask(
        self,
        choice_callback: Callable[[str, list[str]], int] | None = None,
        input_callback: Callable[[str], str] | None = None,
    ) -> str:
        """
        Ask the clarification question via callbacks.

        Uses choice_callback if options are provided, otherwise input_callback.

        Returns:
            The user's response as a string.
        """
        if self.options:
            if choice_callback is not None:
                idx = choice_callback(self.question, self.options)
                if isinstance(idx, int) and 0 <= idx < len(self.options):
                    return self.options[idx]
                # Fall through to default
            default_idx = self.default if isinstance(self.default, int) else 0
            if 0 <= default_idx < len(self.options):
                return self.options[default_idx]
            return self.options[0] if self.options else ""
        else:
            if input_callback is not None:
                return input_callback(self.question)
            return str(self.default) if self.default else ""


class ClarificationDetector:
    """
    Detects ambiguous queries that may need clarification.

    Uses heuristic analysis to determine if a query is too vague,
    matches multiple tools, or could have conflicting interpretations.
    """

    # Queries shorter than this (in words) are considered potentially vague
    MIN_CLEAR_QUERY_WORDS: int = 3

    # Words that signal ambiguity
    AMBIGUOUS_WORDS: set[str] = {
        "something", "stuff", "thing", "things", "it",
        "that", "those", "whatever", "somehow",
    }

    def __init__(
        self,
        choice_callback: Callable[[str, list[str]], int] | None = None,
        input_callback: Callable[[str], str] | None = None,
    ) -> None:
        self.choice_callback = choice_callback
        self.input_callback = input_callback

    def detect_ambiguity(
        self,
        query: str,
        available_tools: dict[str, Any] | None = None,
    ) -> ClarificationRequest | None:
        """
        Check if a query is ambiguous and needs clarification.

        Args:
            query: The user's query.
            available_tools: Dict of tool_name -> tool_object for matching.

        Returns:
            A ClarificationRequest if ambiguity detected, else None.
        """
        reasons: list[str] = []

        # Check 1: Very short query
        words = query.strip().split()
        if len(words) < self.MIN_CLEAR_QUERY_WORDS:
            reasons.append("query is very short")

        # Check 2: Contains ambiguous words
        query_lower = query.lower()
        found_ambiguous = [w for w in self.AMBIGUOUS_WORDS if w in query_lower.split()]
        if found_ambiguous:
            reasons.append(f"contains vague terms: {', '.join(found_ambiguous)}")

        # Check 3: Multiple tools match the query
        if available_tools:
            matching_tools = self._find_matching_tools(query_lower, available_tools)
            if len(matching_tools) > 2:
                reasons.append(f"matches {len(matching_tools)} tools")
                return ClarificationRequest(
                    question="Your query could use multiple tools. Which would you prefer?",
                    options=[f"{name}: {getattr(t, 'description', name)}"
                             for name, t in matching_tools[:5]],
                    context="; ".join(reasons),
                )

        if not reasons:
            return None

        return ClarificationRequest(
            question="Could you provide more details about what you'd like to do?",
            context="; ".join(reasons),
        )

    def request_clarification(
        self,
        query: str,
        available_tools: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Detect ambiguity and request clarification if needed.

        Returns:
            The user's clarification response, or None if no ambiguity.
        """
        request = self.detect_ambiguity(query, available_tools)
        if request is None:
            return None

        logger.debug("Requesting clarification: %s (reason: %s)", request.question, request.context)
        return request.ask(
            choice_callback=self.choice_callback,
            input_callback=self.input_callback,
        )

    @staticmethod
    def _find_matching_tools(
        query: str,
        tools: dict[str, Any],
    ) -> list[tuple[str, Any]]:
        """Find tools whose name or description matches words in the query."""
        query_words = set(query.lower().split())
        matches = []
        for name, tool in tools.items():
            name_words = set(name.lower().replace("_", " ").split())
            desc = getattr(tool, 'description', '').lower()
            desc_words = set(desc.split())
            overlap = query_words & (name_words | desc_words)
            if overlap:
                matches.append((name, tool))
        return matches
