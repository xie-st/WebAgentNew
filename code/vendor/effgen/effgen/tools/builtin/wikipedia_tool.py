"""
Wikipedia tool using the free Wikipedia API.

No API key required. Searches articles, retrieves summaries,
and extracts sections from Wikipedia.
"""

import json
import logging
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urlencode

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

# Wikipedia serves per-language API hosts (en/de/fr/….wikipedia.org); pin the
# family (the shared SSRF guard also blocks internal targets).
_ALLOWED_HOSTS = frozenset({"*.wikipedia.org"})


def _get_user_agent(tool_name: str = "") -> str:
    """Build User-Agent string with current effGen version."""
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    suffix = f" ({tool_name})" if tool_name else ""
    return f"effGen/{__version__}{suffix}"

logger = logging.getLogger(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


class WikipediaTool(BaseTool):
    """
    Wikipedia information retrieval tool.

    Uses the free Wikipedia API (no key needed) to:
    - Search for articles
    - Get article summaries
    - Extract specific sections
    - Get page metadata

    Features:
    - Free, no API key required
    - Search and summary modes
    - Configurable language
    - Content length control
    """

    def __init__(self, language: str = "en") -> None:
        """
        Initialize Wikipedia tool.

        Args:
            language: Wikipedia language code (default: 'en')
        """
        self.language = language
        self._api_url = f"https://{language}.wikipedia.org/w/api.php"

        super().__init__(
            metadata=ToolMetadata(
                name="wikipedia",
                description=(
                    "Search Wikipedia and get article summaries. "
                    "Free, no API key needed. Use this for factual "
                    "information about topics, people, places, events, etc."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query or article title",
                        required=True,
                        min_length=1,
                        max_length=300,
                    ),
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'search' to find articles, 'summary' to get article summary",
                        required=False,
                        default="summary",
                        enum=["search", "summary"],
                    ),
                    ParameterSpec(
                        name="sentences",
                        type=ParameterType.INTEGER,
                        description="Number of sentences for summary (default: 5)",
                        required=False,
                        default=5,
                        min_value=1,
                        max_value=20,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
                timeout_seconds=15,
                tags=["wikipedia", "knowledge", "encyclopedia", "free", "search"],
                examples=[
                    {
                        "query": "Python programming language",
                        "operation": "summary",
                        "output": {
                            "title": "Python (programming language)",
                            "summary": "Python is a high-level programming language...",
                        },
                    },
                    {
                        "query": "machine learning",
                        "operation": "search",
                        "output": {"results": ["Machine learning", "Deep learning", ...]},
                    },
                ],
            )
        )

    def _api_request(self, params: dict) -> dict:
        """Make a request to the Wikipedia API."""
        params["format"] = "json"
        url = f"{self._api_url}?{urlencode(params)}"
        headers = {"User-Agent": _get_user_agent("Wikipedia Tool")}
        try:
            with safe_urlopen(
                url, headers=headers, timeout=10, allowed_hosts=_ALLOWED_HOSTS
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            raise ConnectionError(f"Wikipedia API error: {e}")

    def _search(self, query: str, limit: int = 5) -> list[dict]:
        """Search Wikipedia for articles."""
        data = self._api_request({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet|titlesnippet",
        })

        results = []
        for item in data.get("query", {}).get("search", []):
            # Clean HTML from snippets
            snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
            results.append({
                "title": item["title"],
                "snippet": snippet,
                "url": f"https://{self.language}.wikipedia.org/wiki/{quote(item['title'].replace(' ', '_'))}",
            })
        return results

    def _get_summary(self, title: str, sentences: int = 5) -> dict:
        """Get article summary using the REST API."""
        # Use the extracts API for summary
        data = self._api_request({
            "action": "query",
            "titles": title,
            "prop": "extracts|info",
            "exintro": "true",
            "explaintext": "true",
            "exsentences": sentences,
            "inprop": "url",
            "redirects": "1",
        })

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return {"title": title, "summary": "Article not found.", "found": False}

        page = next(iter(pages.values()))
        if "missing" in page:
            # Try searching for it
            search_results = self._search(title, limit=1)
            if search_results:
                return self._get_summary(search_results[0]["title"], sentences)
            return {"title": title, "summary": "Article not found.", "found": False}

        page_title = page.get("title", title)
        extract = page.get("extract", "").strip()

        return {
            "title": page_title,
            "summary": extract if extract else "No summary available.",
            "url": f"https://{self.language}.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}",
            "page_id": page.get("pageid"),
            "found": True,
        }

    async def _execute(
        self,
        query: str,
        operation: str = "summary",
        sentences: int = 5,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute Wikipedia operation."""

        if operation == "search":
            results = self._search(query)
            return {
                "query": query,
                "results": results,
                "total": len(results),
            }

        elif operation == "summary":
            result = self._get_summary(query, sentences)
            return result

        raise ValueError(f"Unknown operation: {operation}")
