"""
Tool fallback chain for the effGen framework.

When a tool fails, the fallback chain tries alternative tools in a defined
order, so a single tool failure does not end the run.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ToolFallbackChain:
    """
    Define fallback chains: if tool A fails, try tool B, then C.

    Comes with sensible defaults for built-in tools and allows
    users to register custom chains.
    """

    # Default fallback chains for built-in tools
    DEFAULT_CHAINS: dict[str, list[str]] = {
        "calculator": ["python_repl", "code_executor"],
        "python_repl": ["code_executor"],
        "code_executor": ["python_repl"],
        "web_search": ["agentic_search"],
    }

    def __init__(self, custom_chains: dict[str, list[str]] | None = None) -> None:
        """
        Initialize with default chains, optionally merged with custom ones.

        Args:
            custom_chains: Optional user-defined fallback chains that override
                           or extend the defaults.
        """
        self.chains: dict[str, list[str]] = dict(self.DEFAULT_CHAINS)
        if custom_chains:
            self.chains.update(custom_chains)

    def register_chain(self, tool_name: str, fallbacks: list[str]) -> None:
        """
        Register or replace a fallback chain for a tool.

        Args:
            tool_name: Primary tool name.
            fallbacks: Ordered list of fallback tool names.
        """
        self.chains[tool_name] = list(fallbacks)
        logger.debug(f"Registered fallback chain: {tool_name} -> {fallbacks}")

    def get_fallbacks(self, tool_name: str) -> list[str]:
        """
        Get ordered fallback tools for a given tool.

        Args:
            tool_name: The tool that failed.

        Returns:
            List of fallback tool names (may be empty).
        """
        return list(self.chains.get(tool_name, []))

    def has_fallbacks(self, tool_name: str) -> bool:
        """Check whether a tool has any registered fallbacks."""
        return bool(self.chains.get(tool_name))
