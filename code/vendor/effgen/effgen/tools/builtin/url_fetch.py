"""
URL Fetch tool for retrieving webpage content.

Fetches and extracts text from web pages using requests + BeautifulSoup
(both free/open source). Falls back to stdlib urllib if packages unavailable.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import (
    _METADATA_IPS,  # noqa: F401  (back-compat re-export)
    BlockedURLError,
    check_url_safe,
    safe_requests_get,
    safe_urlopen,
)
from ._net import BlockedURLError as _BlockedURLError  # noqa: F401  (back-compat re-export)
from ._net import is_blocked_ip as _is_blocked_ip  # noqa: F401  (back-compat re-export)

# "scheme://" at the start of a URL, per RFC 3986 scheme syntax.
_HAS_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _get_user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__} (URL Fetch Tool)"

logger = logging.getLogger(__name__)


class _SimpleHTMLTextExtractor(HTMLParser):
    """Simple HTML to text converter using stdlib."""

    SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag.lower() in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._text_parts.append(data)

    def get_text(self) -> str:
        text = " ".join(self._text_parts)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


class URLFetchTool(BaseTool):
    """
    URL content fetching tool.

    Fetches webpage content and extracts readable text.

    Features:
    - Fetch and extract text from web pages
    - Configurable max content length
    - Domain allowlist/blocklist
    - Timeout control
    - Uses requests + beautifulsoup4 if available, falls back to urllib

    Note:
        This tool makes HTTP requests to external websites.
        Configure allowed_domains to restrict access.
    """

    def __init__(
        self,
        allowed_domains: set[str] | None = None,
        blocked_domains: set[str] | None = None,
        max_content_length: int = 10000,
        timeout: int = 15,
        allow_private: bool = False,
        max_redirects: int = 5,
    ) -> None:
        """
        Initialize the URL Fetch tool.

        Args:
            allowed_domains: If set, only fetch from these domains.
            blocked_domains: Domains to never fetch from.
            max_content_length: Max characters to return (default: 10000).
            timeout: Request timeout in seconds (default: 15).
            allow_private: Allow fetching private/loopback/link-local/metadata
                addresses. Default ``False`` blocks them (SSRF protection); set
                ``True`` only when you intentionally fetch internal services.
            max_redirects: Maximum number of redirects to follow; each hop's
                destination is re-validated against the SSRF rules.
        """
        super().__init__(
            metadata=ToolMetadata(
                name="url_fetch",
                description=(
                    "Fetch a webpage and extract its text content. "
                    "Returns the readable text from a URL. "
                    "Use this to get information from specific web pages."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="url",
                        type=ParameterType.STRING,
                        description="The URL to fetch (must start with http:// or https://)",
                        required=True,
                        min_length=8,
                    ),
                    ParameterSpec(
                        name="extract_links",
                        type=ParameterType.BOOLEAN,
                        description="Also extract links from the page",
                        required=False,
                        default=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
                timeout_seconds=timeout,
                tags=["url", "web", "fetch", "http", "scraping"],
                examples=[
                    {
                        "url": "https://example.com",
                        "output": {"title": "Example Domain", "text": "This domain is for use in..."},
                    },
                ],
            )
        )

        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains or set()
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.allow_private = allow_private
        self.max_redirects = max_redirects

        # Construction-time note only \u2014 tool discovery instantiates every tool,
        # so keep this at DEBUG to avoid polluting `tools list` and agent setup.
        logger.debug(
            "URL Fetch tool makes HTTP requests to external websites. "
            "Private/loopback/link-local/metadata addresses are blocked by "
            "default; set allow_private=True to override."
        )

    def _check_address_safe(self, url: str) -> None:
        """Resolve the host and reject non-public addresses (SSRF guard).

        Delegates to the shared :func:`effgen.tools.builtin._net.check_url_safe`
        so every URL-taking tool applies an identical guard. Called for the
        original URL and for each redirect hop.
        """
        check_url_safe(url, allow_private=self.allow_private)

    def _validate_url(self, url: str) -> str:
        """Validate and normalize URL (scheme, domain lists, SSRF guard)."""
        url = url.strip()
        # A bare host ("example.com/page") gets https://; a URL that already
        # names a scheme keeps it, so `check_url_safe` can reject a non-http(s)
        # one by name. Prepending to "file:///etc/passwd" would turn the scheme
        # into the hostname and report a DNS failure instead.
        if not _HAS_SCHEME.match(url):
            url = "https://" + url

        # A malformed URL raises at the parse (an unclosed IPv6 literal) or only
        # when the authority is read (a non-numeric port). Both are reported as
        # a refusal naming the URL, the same wording the shared guard uses.
        try:
            parsed = urlparse(url)
            domain = parsed.hostname or ""
        except ValueError as e:
            raise BlockedURLError(
                f"Refusing to fetch unparseable URL '{url}': {e}. Pass a "
                "complete http(s) URL — check the scheme, the host, and that "
                "any IPv6 literal is bracketed."
            ) from e

        if self.allowed_domains and domain not in self.allowed_domains:
            raise ValueError(
                f"Domain '{domain}' is not in this tool's allowed domains "
                f"({sorted(self.allowed_domains)}). Request one of those, or "
                "add this domain to allowed_domains."
            )
        if domain in self.blocked_domains:
            raise ValueError(
                f"Domain '{domain}' is blocked for this tool. Request a "
                "different host, or remove it from blocked_domains."
            )

        self._check_address_safe(url)
        return url

    def _extract_with_beautifulsoup(self, html: str) -> tuple:
        """Extract text using BeautifulSoup."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)

        links = []
        for a in soup.find_all("a", href=True):
            links.append({"text": a.get_text(strip=True), "href": a["href"]})

        return title, text, links

    def _extract_with_stdlib(self, html: str) -> tuple:
        """Extract text using stdlib HTML parser."""
        extractor = _SimpleHTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()

        # Try to find title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # Extract links
        links = []
        for m in re.finditer(r'<a\s+[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
            links.append({"text": re.sub(r'<[^>]+>', '', m.group(2)).strip(), "href": m.group(1)})

        return title, text, links

    def _fetch_with_requests(self, requests, url: str) -> str:
        """Fetch via the shared SSRF-safe requests helper (redirects re-validated)."""
        resp = safe_requests_get(
            requests,
            url,
            headers={"User-Agent": _get_user_agent()},
            timeout=self.timeout,
            allow_private=self.allow_private,
            max_redirects=self.max_redirects,
        )
        resp.raise_for_status()
        return resp.text

    def _fetch_with_urllib(self, url: str) -> str:
        """Stdlib fallback via the shared SSRF-safe urlopen (redirects re-validated)."""
        with safe_urlopen(
            url,
            headers={"User-Agent": _get_user_agent()},
            method="GET",
            timeout=self.timeout,
            allow_private=self.allow_private,
            max_redirects=self.max_redirects,
        ) as resp:
            return resp.read().decode("utf-8", errors="replace")

    async def _execute(
        self,
        url: str,
        extract_links: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Fetch URL and extract text."""
        url = self._validate_url(url)

        # Try requests first, fall back to urllib. Redirects are followed
        # manually so every hop is re-checked against the SSRF guard (a public
        # URL must not be able to bounce us onto an internal address).
        html = ""
        try:
            import requests
            html = self._fetch_with_requests(requests, url)
        except ImportError:
            html = self._fetch_with_urllib(url)

        # Extract text
        try:
            title, text, links = self._extract_with_beautifulsoup(html)
        except ImportError:
            title, text, links = self._extract_with_stdlib(html)

        # Truncate
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "\n... (content truncated)"

        result = {
            "url": url,
            "title": title,
            "text": text,
            "content_length": len(text),
        }

        if extract_links:
            result["links"] = links[:50]  # Limit links

        return result
