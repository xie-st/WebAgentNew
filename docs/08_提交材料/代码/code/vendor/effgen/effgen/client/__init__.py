"""effGen Python client SDK.

Provides sync and async clients for connecting to an effGen API server.

Example:
    >>> from effgen.client import EffGenClient
    >>> client = EffGenClient(base_url="http://localhost:8000", api_key="...")
    >>> result = client.chat("What is 2+2?", tools=["calculator"])
    >>> print(result.content)

Async / streaming:
    >>> import asyncio
    >>> async def main():
    ...     async for chunk in client.chat_stream("Tell me a story"):
    ...         print(chunk, end="")
    >>> asyncio.run(main())
"""
from __future__ import annotations

from effgen.client.client import (
    ChatResponse,
    EffGenClient,
    HealthStatus,
)
from effgen.client.exceptions import (
    EffGenAPIError,
    EffGenAuthError,
    EffGenClientError,
    EffGenConnectionError,
    EffGenRateLimitError,
    EffGenServerError,
    EffGenTimeoutError,
)

__all__ = [
    "EffGenClient",
    "ChatResponse",
    "HealthStatus",
    "EffGenClientError",
    "EffGenAPIError",
    "EffGenAuthError",
    "EffGenConnectionError",
    "EffGenRateLimitError",
    "EffGenServerError",
    "EffGenTimeoutError",
]
