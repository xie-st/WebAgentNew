"""
Knowledge tools for the effGen framework.

Provides access to arXiv, Stack Overflow, GitHub, and (optionally)
Wolfram Alpha using only free, unauthenticated endpoints where possible.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

logger = logging.getLogger(__name__)

# Fixed third-party API endpoints this module talks to (host-pinned; a redirect
# may not leave these hosts, and the shared SSRF guard blocks internal targets).
_ALLOWED_HOSTS = frozenset({
    "api.stackexchange.com",
    "api.github.com",
    "api.wolframalpha.com",
})


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"


def _api_error_detail(body: bytes) -> str:
    """Return what an error body says, when it is one of the documented shapes.

    The Stack Exchange API reports a rate limit as HTTP 400 with the reason in
    the body (``throttle_violation`` and how long to wait); GitHub does the same
    with a ``message``. Without the body the caller sees only "HTTP 400 Bad
    Request" and cannot tell a throttle from a malformed query.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    parts = [
        str(payload[key])
        for key in ("error_name", "error_message", "message")
        if payload.get(key)
    ]
    backoff = payload.get("backoff")
    if backoff:
        parts.append(f"retry after {backoff}s")
    return " - ".join(parts)


def _fetch(url: str, timeout: int = 15, accept: str = "application/json") -> bytes:
    headers = {"User-Agent": _user_agent(), "Accept": accept}
    try:
        with safe_urlopen(
            url, headers=headers, timeout=timeout, allowed_hosts=_ALLOWED_HOSTS
        ) as resp:
            return resp.read()
    except HTTPError as e:
        detail = ""
        try:
            detail = _api_error_detail(e.read())
        except Exception:  # noqa: BLE001 - the body is a courtesy; the status is the answer
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise ConnectionError(f"HTTP {e.code} from {url}: {e.reason}{suffix}")
    except URLError as e:
        raise ConnectionError(f"Network error fetching {url}: {e.reason}")


def _fetch_json(url: str, timeout: int = 15) -> Any:
    return json.loads(_fetch(url, timeout=timeout).decode("utf-8"))


# `ArxivTool` was promoted to a dedicated module (`effgen.tools.builtin.arxiv`)
# in v0.2.5 with richer operations (search / fetch / download_pdf). The legacy
# name remains importable from this module for backwards compatibility.
from .arxiv import ArXivTool as ArxivTool  # noqa: F401,E402


class StackOverflowTool(BaseTool):
    """Search Stack Overflow via the free Stack Exchange API (no key)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="stackoverflow",
                description=(
                    "Search Stack Overflow for programming questions. Returns title, "
                    "score, answer count, tags, and link. Uses the free Stack Exchange API."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query",
                        required=True,
                        min_length=1,
                        max_length=300,
                    ),
                    ParameterSpec(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Maximum number of results",
                        required=False,
                        default=5,
                        min_value=1,
                        max_value=20,
                    ),
                ],
                timeout_seconds=20,
                tags=["knowledge", "stackoverflow", "programming", "free"],
                examples=[{"query": "python asyncio gather", "max_results": 3}],
            )
        )

    async def _execute(
        self,
        query: str,
        max_results: int = 5,
        **kwargs,
    ) -> dict[str, Any]:
        params = urlencode({
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "site": "stackoverflow",
            "pagesize": max_results,
        })
        url = f"https://api.stackexchange.com/2.3/search/advanced?{params}"
        data = _fetch_json(url)
        items = data.get("items", [])
        results = [
            {
                "title": it.get("title"),
                "score": it.get("score"),
                "answer_count": it.get("answer_count"),
                "is_answered": it.get("is_answered"),
                "tags": it.get("tags", []),
                "link": it.get("link"),
            }
            for it in items
        ]
        return {"query": query, "count": len(results), "results": results}


class GitHubTool(BaseTool):
    """Search GitHub public repos and issues via the free unauthenticated API."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="github",
                description=(
                    "Search GitHub for public repositories or issues using the free "
                    "GitHub search API (no authentication required for public data)."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query",
                        required=True,
                        min_length=1,
                        max_length=256,
                    ),
                    ParameterSpec(
                        name="kind",
                        type=ParameterType.STRING,
                        description="Search kind",
                        required=False,
                        default="repositories",
                        enum=["repositories", "issues", "code"],
                    ),
                    ParameterSpec(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Maximum results",
                        required=False,
                        default=5,
                        min_value=1,
                        max_value=20,
                    ),
                ],
                timeout_seconds=20,
                tags=["knowledge", "github", "code", "free"],
                examples=[{"query": "agentic framework language=python"}],
            )
        )

    async def _execute(
        self,
        query: str,
        kind: str = "repositories",
        max_results: int = 5,
        **kwargs,
    ) -> dict[str, Any]:
        params = urlencode({"q": query, "per_page": max_results})
        url = f"https://api.github.com/search/{kind}?{params}"
        data = _fetch_json(url)
        items = data.get("items", [])[:max_results]

        results: list[dict[str, Any]] = []
        if kind == "repositories":
            for it in items:
                results.append({
                    "full_name": it.get("full_name"),
                    "description": it.get("description"),
                    "stars": it.get("stargazers_count"),
                    "forks": it.get("forks_count"),
                    "language": it.get("language"),
                    "url": it.get("html_url"),
                })
        elif kind == "issues":
            for it in items:
                results.append({
                    "title": it.get("title"),
                    "state": it.get("state"),
                    "comments": it.get("comments"),
                    "url": it.get("html_url"),
                    "repository": (it.get("repository_url") or "").replace(
                        "https://api.github.com/repos/", ""
                    ),
                })
        else:  # code
            for it in items:
                results.append({
                    "name": it.get("name"),
                    "path": it.get("path"),
                    "repository": (it.get("repository") or {}).get("full_name"),
                    "url": it.get("html_url"),
                })

        return {
            "query": query,
            "kind": kind,
            "total_count": data.get("total_count", 0),
            "count": len(results),
            "results": results,
        }


class WolframAlphaTool(BaseTool):
    """
    Wolfram Alpha Short Answers API.

    Requires a free API key from https://developer.wolframalpha.com/.
    The key can be passed to the constructor or set via the
    ``WOLFRAM_ALPHA_APPID`` environment variable.
    """

    def __init__(self, app_id: str | None = None) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="wolfram_alpha",
                description=(
                    "Query Wolfram Alpha for a short answer. Requires a free API "
                    "key (WOLFRAM_ALPHA_APPID env var)."
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Question or query",
                        required=True,
                        min_length=1,
                        max_length=500,
                    ),
                ],
                requires_api_key=True,
                timeout_seconds=20,
                tags=["knowledge", "wolfram", "math", "optional"],
                examples=[{"query": "distance from earth to mars"}],
            )
        )
        self.app_id = app_id or os.environ.get("WOLFRAM_ALPHA_APPID")
        if not self.app_id:
            # Construction-time note only \u2014 listing/discovery instantiates every
            # tool, so this must not log at WARNING (it would spam clean tables).
            # The missing key is reported as an actionable error at execution.
            logger.debug(
                "WolframAlphaTool has no API key. Set WOLFRAM_ALPHA_APPID "
                "or pass app_id=... to the constructor. Get a free key at "
                "https://developer.wolframalpha.com/"
            )

    async def _execute(self, query: str, **kwargs) -> dict[str, Any]:
        if not self.app_id:
            raise RuntimeError(
                "Wolfram Alpha API key missing. Set WOLFRAM_ALPHA_APPID env var."
            )
        params = urlencode({"appid": self.app_id, "i": query, "units": "metric"})
        url = f"https://api.wolframalpha.com/v1/result?{params}"
        try:
            body = _fetch(url, accept="text/plain").decode("utf-8", errors="replace")
        except ConnectionError as e:
            raise ConnectionError(f"Wolfram Alpha query failed: {e}")
        return {"query": query, "answer": body.strip(), "source": "wolfram_alpha"}
