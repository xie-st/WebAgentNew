"""Whether a generated page would fetch anything when opened offline.

Several surfaces promise self-containment — the battle report, the run card,
the dashboard, the playground — and each used to check it by asserting that
``"https://"`` appears nowhere in the output. That is a proxy, not the
property, and it is wrong in the direction that matters: a provider whose
error text names its billing console ("...retry today at
https://console.groq.com/settings/billing") puts a URL in the page as *escaped
text*, the page still fetches nothing, and the anchor goes red anyway. A green
anchor that fails on an ordinary rate limit trains its reader to ignore it.

What is asserted here instead is the property: no URL sits in a position the
browser would fetch.
"""

from __future__ import annotations

import re

#: Positions a browser fetches from. Text content is deliberately not one.
_FETCHING_POSITIONS = (
    # attribute="url" — src, href, srcset, poster, data, action, formaction
    re.compile(
        r"""\b(?:src|href|srcset|poster|data|action|formaction)\s*=\s*["']?\s*"""
        r"""(https?://[^"'\s>]+)""",
        re.IGNORECASE,
    ),
    # CSS url(...) and @import
    re.compile(r"""url\(\s*["']?\s*(https?://[^"')\s]+)""", re.IGNORECASE),
    re.compile(r"""@import\s+["']\s*(https?://[^"']+)""", re.IGNORECASE),
    # Script-driven fetches
    re.compile(r"""\bfetch\(\s*["'`](https?://[^"'`]+)""", re.IGNORECASE),
    # A socket is a fetch too, and it carries its own scheme.
    re.compile(
        r"""\b(?:XMLHttpRequest|WebSocket|EventSource)\b[^;]{0,120}?"""
        r"""["'`]((?:https?|wss?)://[^"'`]+)""",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"""\.open\(\s*["'][A-Z]+["']\s*,\s*["'`](https?://[^"'`]+)"""),
)


def external_references(page: str) -> list[str]:
    """Return every URL the page would actually fetch.

    Args:
        page: The rendered HTML, CSS or JS.

    Returns:
        The fetched URLs, in the order found, without duplicates. A URL that
        appears only as visible text — a provider's error message quoting its
        console, say — is not one of them.
    """
    found: list[str] = []
    for pattern in _FETCHING_POSITIONS:
        for match in pattern.finditer(page):
            url = match.group(1)
            if url not in found:
                found.append(url)
    return found


def assert_self_contained(page: str, what: str = "page") -> None:
    """Fail if *page* would fetch anything when opened without a network.

    Args:
        page: The rendered output to check.
        what: Name used in the failure message.

    Raises:
        AssertionError: The page carries at least one fetched external URL.
    """
    references = external_references(page)
    assert not references, (
        f"{what} would fetch {len(references)} external reference(s) when "
        f"opened offline: {references[:5]}"
    )
