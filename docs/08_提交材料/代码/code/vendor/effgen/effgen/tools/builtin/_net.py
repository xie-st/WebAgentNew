"""Shared SSRF-safe HTTP helpers for the built-in tools.

Every built-in tool that fetches a URL must go through this module instead of
calling :func:`urllib.request.urlopen` / ``requests`` directly. Centralising the
guard here means the private/loopback/link-local/metadata block (and the
per-redirect re-validation) is applied uniformly — an agent cannot reach an
internal endpoint just by switching from ``url_fetch`` to ``rss`` or a webhook.

The guard:

* rejects non-http(s) schemes;
* for a bare IP literal, classifies it directly;
* otherwise DNS-resolves the host and classifies **every** returned address, so
  a public name that maps to an internal address is still blocked;
* re-validates **each redirect hop** (a public first hop cannot bounce onto an
  internal address);
* optionally pins the request to a fixed set of hosts (``allowed_hosts``) for
  fixed-endpoint API tools;
* can be opted out per call/tool via ``allow_private=True`` for intentional
  access to internal services.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

# Well-known cloud metadata endpoints (also covered by the range checks below,
# listed explicitly so the failure message is unambiguous).
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254", "100.100.100.200"}

# Redirect status codes we follow manually (so each hop is re-validated).
_REDIRECT_CODES = (301, 302, 303, 307, 308)


class BlockedURLError(ValueError):
    """Raised for a URL targeting a non-public address or a pinned-host mismatch.

    Subclasses ValueError so existing tool error handling catches it.
    """


def is_blocked_ip(addr: ipaddress._BaseAddress) -> bool:
    """Return True for any non-public address an SSRF would target."""
    # Normalise IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) to its IPv4 form.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if str(addr) in _METADATA_IPS:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _host_matches(host: str, allowed_hosts: set[str]) -> bool:
    """Case-insensitive host match supporting ``*.suffix`` wildcard entries."""
    host = host.lower().rstrip(".")
    for entry in allowed_hosts:
        entry = entry.lower().rstrip(".")
        if entry.startswith("*."):
            suffix = entry[1:]  # ".example.com"
            if host == entry[2:] or host.endswith(suffix):
                return True
        elif host == entry:
            return True
    return False


def check_url_safe(
    url: str,
    *,
    allow_private: bool = False,
    allowed_hosts: set[str] | None = None,
) -> None:
    """Validate scheme, optional host pin, and SSRF rules for a single URL.

    Raises :class:`BlockedURLError` if the URL must not be fetched.

    Args:
        url: The URL to validate.
        allow_private: Permit private, loopback and link-local addresses.
        allowed_hosts: When given, the only hosts that may be reached.
    """
    # A malformed URL raises either here (an unclosed IPv6 literal) or only when
    # the authority is read below (a non-numeric port), so both are turned into
    # a refusal naming the URL instead of a bare urllib ValueError.
    try:
        parsed = urlparse(url)
    except ValueError as e:
        raise BlockedURLError(
            f"Refusing to fetch unparseable URL '{url}': {e}. "
            "Pass a complete http(s) URL — check the scheme, the host, and "
            "that any IPv6 literal is bracketed."
        ) from e
    if parsed.scheme not in ("http", "https"):
        raise BlockedURLError(
            f"Refusing to fetch non-http(s) URL '{url}' (scheme "
            f"'{parsed.scheme or 'none'}'). Pass an http:// or https:// URL; "
            "this tool does not read other schemes."
        )
    try:
        host = parsed.hostname
    except ValueError as e:
        raise BlockedURLError(
            f"Refusing to fetch unparseable URL '{url}': {e}. "
            "Pass a complete http(s) URL — check the scheme, the host, and "
            "that any IPv6 literal is bracketed."
        ) from e
    if not host:
        raise BlockedURLError(
            f"URL '{url}' has no host. Pass an absolute URL including the "
            "host, such as https://example.com/page."
        )

    if allowed_hosts is not None and not _host_matches(host, allowed_hosts):
        raise BlockedURLError(
            f"Refusing to fetch host '{host}': not in the allowed host list "
            f"for this tool ({sorted(allowed_hosts)}). Request one of those "
            "hosts, or add this one to the tool's allowed_hosts."
        )

    if allow_private:
        return

    # A bare IP literal can be checked directly without DNS.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if is_blocked_ip(literal):
            raise BlockedURLError(
                f"Refusing to fetch internal address '{host}' (SSRF "
                "protection). Pass allow_private=True to override."
            )
        return

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as e:
        raise BlockedURLError(
            f"Refusing to fetch unparseable URL '{url}': {e}. "
            "Pass a complete http(s) URL — check the scheme, the host, and "
            "that any IPv6 literal is bracketed."
        ) from e
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise BlockedURLError(
            f"Cannot resolve host '{host}': {e}. Check the spelling and that "
            "this machine has working DNS for it."
        )
    for info in infos:
        ip_str = info[4][0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if is_blocked_ip(addr):
            raise BlockedURLError(
                f"Refusing to fetch '{host}': it resolves to non-public address "
                f"{ip_str} (SSRF protection). Pass allow_private=True to override."
            )


def _build_no_redirect_opener():
    """A urllib opener that surfaces 3xx as HTTPError instead of following them,
    so each hop can be re-validated by the caller."""
    from urllib.error import HTTPError
    from urllib.request import HTTPRedirectHandler, build_opener

    class _NoRedirect(HTTPRedirectHandler):
        def http_error_301(self, req, fp, code, msg, headers):
            raise HTTPError(req.full_url, code, msg, headers, fp)

        http_error_302 = http_error_301
        http_error_303 = http_error_301
        http_error_307 = http_error_301
        http_error_308 = http_error_301

    return build_opener(_NoRedirect)


def safe_urlopen(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    method: str | None = None,
    timeout: float = 30,
    allow_private: bool = False,
    allowed_hosts: set[str] | None = None,
    max_redirects: int = 5,
) -> Any:
    """SSRF-safe drop-in for ``urllib.request.urlopen``.

    Validates the target (and every redirect hop) before connecting and returns
    the final, still-open response object (usable as a context manager). On a
    blocked target raises :class:`BlockedURLError`.

    Args:
        url: The URL to fetch.
        data: Request body, which makes the call a POST unless *method* says
            otherwise.
        headers: Extra request headers.
        method: HTTP method to use.
        timeout: Seconds to wait for the response.
        allow_private: Permit private, loopback and link-local addresses.
        allowed_hosts: When given, the only hosts that may be reached.
        max_redirects: Redirect hops to follow, each re-validated.
    """
    from urllib.error import HTTPError
    from urllib.request import Request

    current = url
    current_data = data
    current_method = method
    for _ in range(max_redirects + 1):
        check_url_safe(current, allow_private=allow_private, allowed_hosts=allowed_hosts)
        req = Request(
            current,
            data=current_data,
            headers=headers or {},
            method=current_method,
        )
        opener = _build_no_redirect_opener()
        try:
            return opener.open(req, timeout=timeout)
        except HTTPError as e:
            if e.code in _REDIRECT_CODES:
                location = e.headers.get("Location")
                if not location:
                    raise BlockedURLError(
                        "Redirect without a Location header. The server sent "
                        "a redirect status with nowhere to go; request the "
                        "final URL directly."
                    )
                current = urljoin(current, location)
                # 301/302/303 downgrade to GET without a body (HTTP semantics);
                # 307/308 preserve method and body.
                if e.code in (301, 302, 303):
                    current_data = None
                    current_method = "GET" if current_method not in (None, "GET") else current_method
                continue
            raise
    raise BlockedURLError(
        f"Too many redirects (>{max_redirects}) fetching '{url}'. Request the "
        "final URL directly, or raise max_redirects if the chain is genuinely "
        "that long."
    )


def safe_requests_get(
    requests_module: Any,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
    allow_private: bool = False,
    allowed_hosts: set[str] | None = None,
    max_redirects: int = 5,
    **kwargs: Any,
) -> Any:
    """SSRF-safe ``requests.get`` with manual, re-validated redirect following.

    ``requests_module`` is the imported ``requests`` module (passed in so this
    helper never hard-depends on it). Returns the final ``requests.Response``.

    Args:
        requests_module: The imported ``requests`` module to call through.
        url: The URL to fetch.
        headers: Extra request headers.
        timeout: Seconds to wait for the response.
        allow_private: Permit private, loopback and link-local addresses.
        allowed_hosts: When given, the only hosts that may be reached.
        max_redirects: Redirect hops to follow, each re-validated.
        **kwargs: Extra keyword arguments forwarded to ``requests.get``.
    """
    current = url
    with requests_module.Session() as session:
        for _ in range(max_redirects + 1):
            check_url_safe(
                current, allow_private=allow_private, allowed_hosts=allowed_hosts
            )
            resp = session.get(
                current,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                **kwargs,
            )
            if resp.status_code in _REDIRECT_CODES:
                location = resp.headers.get("Location")
                if not location:
                    return resp
                current = urljoin(current, location)
                continue
            return resp
    raise BlockedURLError(
        f"Too many redirects (>{max_redirects}) fetching '{url}'. Request the "
        "final URL directly, or raise max_redirects if the chain is genuinely "
        "that long."
    )
