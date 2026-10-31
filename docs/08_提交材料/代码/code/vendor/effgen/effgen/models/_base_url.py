"""Resolution of the endpoint URL for OpenAI-protocol adapters.

One place decides where an OpenAI-protocol call is sent, so the adapter, the
model loader and the CLI all agree. An explicit ``base_url=`` argument always
wins; with none given the environment is consulted in the order below.
"""

from __future__ import annotations

import os

#: Environment variables consulted for the endpoint, in precedence order.
#: ``EFFGEN_BASE_URL`` is effGen's own and comes first so it can point effGen at
#: a server without redirecting every other OpenAI client on the machine. The
#: two ``OPENAI_*`` names are the conventional ones the OpenAI SDK and most
#: tooling already read.
BASE_URL_ENV_VARS: tuple[str, ...] = (
    "EFFGEN_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)

#: The api_key sent when a server needs the field populated but checks nothing.
#: vLLM, SGLang, TGI, llama.cpp and Ollama all accept any non-empty string.
PLACEHOLDER_API_KEY = "EMPTY"

#: OpenAI's own endpoint — where a call goes when nothing overrides it.
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def resolve_base_url(explicit: str | None = None) -> str | None:
    """Return the endpoint URL to call, or None to use the provider's own.

    Args:
        explicit: A URL passed by the caller. Used as-is when non-empty.

    Returns:
        The URL with any trailing slash removed, or None when neither an
        argument nor any of :data:`BASE_URL_ENV_VARS` supplies one.

    Raises:
        ValueError: The URL carries no ``http://`` or ``https://`` scheme. The
            message names where the value came from, because a URL taken from
            the environment is the case where the caller is least likely to
            know it was applied at all.
    """
    candidate = explicit
    source = "the base_url argument"
    if not candidate:
        for var in BASE_URL_ENV_VARS:
            value = os.getenv(var)
            if value and value.strip():
                candidate = value
                source = f"the {var} environment variable"
                break
    if not candidate or not candidate.strip():
        return None
    url = candidate.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        # Without this the HTTP client raises "Connection error" several
        # retries later, and the remediation it carries points at the
        # provider's status page — the wrong place entirely when the address
        # came from this machine's own environment.
        raise ValueError(
            f"The endpoint URL from {source} is missing an http:// or "
            f"https:// scheme: {url!r}. Give the full URL, for example "
            f"http://{url}."
        )
    return url


def openai_client_base_url(resolved: str | None = None) -> str:
    """Return the URL to hand the OpenAI SDK, so one resolution decides.

    The SDK reads ``OPENAI_BASE_URL`` itself whenever no ``base_url`` argument
    is given, and it does not agree with :func:`resolve_base_url` about a
    variable that is present but empty: the resolver reads that as "no
    override", the SDK reads it as an address and sends every request to
    ``''``. What comes back is a connection error advising the caller to check
    the provider's status page, which is the wrong place — the address came
    from this machine.

    A blank value is the ordinary state rather than a corner case: effGen's own
    project template names these variables with nothing after the ``=`` for the
    user to fill in. Passing this value explicitly settles the disagreement, so
    the SDK never re-reads the environment behind the resolver's back.

    Args:
        resolved: The endpoint :func:`resolve_base_url` decided on, or None.

    Returns:
        That URL, or :data:`OPENAI_DEFAULT_BASE_URL` when there is no override.
    """
    return resolved or OPENAI_DEFAULT_BASE_URL


def describe_endpoint(base_url: str | None) -> str:
    """Return a short label for the endpoint, for logs and error messages."""
    return base_url if base_url else "the OpenAI API"
