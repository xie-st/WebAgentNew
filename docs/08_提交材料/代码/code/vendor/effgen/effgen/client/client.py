"""effGen client — sync (requests) and async (httpx) HTTP client.

The client speaks the effGen API server protocol. For OpenAI-compatible
endpoints it will POST JSON and parse responses. Streaming uses Server-Sent
Events (SSE) for chat_stream.

Retry strategy: on connection errors, timeouts, and 5xx/429 responses the
client retries with exponential backoff up to ``max_retries`` attempts.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from effgen.client.exceptions import (
    EffGenAPIError,
    EffGenAuthError,
    EffGenConnectionError,
    EffGenRateLimitError,
    EffGenServerError,
    EffGenTimeoutError,
)

# requests/httpx are imported lazily so the SDK is importable even if only
# one of the two is installed.


@dataclass
class ChatResponse:
    """Response from a chat call."""

    content: str
    model: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Server health information."""

    status: str
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the reported status reads healthy (ok/healthy/up/ready)."""
        return self.status.lower() in ("ok", "healthy", "up", "ready")


class EffGenClient:
    """Client for an effGen API server.

    Parameters
    ----------
    base_url:
        Base URL of the effGen API server (e.g. ``http://localhost:8000``).
    api_key:
        Optional API key sent as ``Authorization: Bearer <key>``.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Maximum retry attempts for connection errors, timeouts, 429, and 5xx.
    backoff_base:
        Base for exponential backoff (seconds). Delay is
        ``backoff_base * 2**attempt`` with jitter.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            headers.update(extra)
        return headers

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _backoff(self, attempt: int) -> float:
        delay = self.backoff_base * (2 ** attempt)
        return delay + random.uniform(0, delay * 0.2)

    def _raise_for_status(self, status: int, payload: Any) -> None:
        if 200 <= status < 300:
            return
        msg = f"HTTP {status}"
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                # OpenAI-style envelope: {"error": {"message": ..., "type": ...}}.
                msg = err.get("message") or err.get("code") or msg
            elif isinstance(err, str):
                msg = err
            else:
                msg = payload.get("message") or msg
        if status in (401, 403):
            raise EffGenAuthError(msg, status_code=status, payload=payload)
        if status == 429:
            raise EffGenRateLimitError(msg, status_code=status, payload=payload)
        if 500 <= status < 600:
            raise EffGenServerError(msg, status_code=status, payload=payload)
        raise EffGenAPIError(msg, status_code=status, payload=payload)

    def _is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, EffGenConnectionError | EffGenTimeoutError):
            return True
        if isinstance(exc, EffGenServerError | EffGenRateLimitError):
            return True
        return False

    # ------------------------------------------------------------------
    # Sync request primitive (requests)
    # ------------------------------------------------------------------
    def _request_sync(self, method: str, path: str, json_body: dict | None = None) -> Any:
        try:
            import requests  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise EffGenClientError(  # noqa: F821
                "The 'requests' package is required for sync EffGenClient calls. "
                "Install it with `pip install requests`, or use "
                "AsyncEffGenClient."
            ) from e

        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.request(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    json=json_body,
                    timeout=self.timeout,
                )
                try:
                    payload = resp.json() if resp.content else None
                except Exception:
                    payload = resp.text
                self._raise_for_status(resp.status_code, payload)
                return payload
            except requests.exceptions.ConnectionError as e:
                last_exc = EffGenConnectionError(str(e))
            except requests.exceptions.Timeout as e:
                last_exc = EffGenTimeoutError(str(e))
            except (EffGenServerError, EffGenRateLimitError) as e:
                last_exc = e
            except EffGenAPIError:
                raise  # non-retryable
            if attempt < self.max_retries and self._is_retryable(last_exc):  # type: ignore[arg-type]
                time.sleep(self._backoff(attempt))
                continue
            break
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Async request primitive (httpx)
    # ------------------------------------------------------------------
    async def _request_async(
        self, method: str, path: str, json_body: dict | None = None
    ) -> Any:
        try:
            import httpx  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise EffGenClientError(  # noqa: F821
                "The 'httpx' package is required for async EffGenClient calls. "
                "Install it with `pip install httpx`, or use the sync "
                "EffGenClient."
            ) from e

        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as hx:
                    resp = await hx.request(
                        method,
                        self._url(path),
                        headers=self._headers(),
                        json=json_body,
                    )
                try:
                    payload = resp.json() if resp.content else None
                except Exception:
                    payload = resp.text
                self._raise_for_status(resp.status_code, payload)
                return payload
            except httpx.ConnectError as e:
                last_exc = EffGenConnectionError(str(e))
            except httpx.TimeoutException as e:
                last_exc = EffGenTimeoutError(str(e))
            except (EffGenServerError, EffGenRateLimitError) as e:
                last_exc = e
            except EffGenAPIError:
                raise
            if attempt < self.max_retries and self._is_retryable(last_exc):  # type: ignore[arg-type]
                await asyncio.sleep(self._backoff(attempt))
                continue
            break
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------
    def chat(
        self,
        message: str,
        tools: list[str | dict] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a single-turn chat message and return the response.

        ``tools`` accepts registered tool names (``["calculator"]``) or full
        OpenAI tool specs; names are expanded to the server's tool-spec shape.
        The server runs the tool and returns the final answer.

        Args:
            message: The user message to send.
            tools: Registered tool names or full OpenAI tool specs.
            model: The model id to use, defaulting to the server's.
            **kwargs: Extra fields forwarded in the request body.
        """
        body: dict = {
            "model": model or "effgen-default",
            "messages": [{"role": "user", "content": message}],
        }
        if tools is not None:
            body["tools"] = _tools_to_payload(tools)
        body.update(kwargs)
        payload = self._request_sync("POST", "/v1/chat/completions", body)
        return self._parse_chat(payload)

    def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """Compute embeddings for a list of input texts."""
        payload = self._request_sync(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": texts},
        )
        if not isinstance(payload, dict) or "data" not in payload:
            raise EffGenAPIError("Malformed embeddings response", payload=payload)
        return [item["embedding"] for item in payload["data"]]

    def health(self) -> HealthStatus:
        """Return server health status."""
        payload = self._request_sync("GET", "/health")
        if isinstance(payload, dict):
            return HealthStatus(status=str(payload.get("status", "unknown")), details=payload)
        return HealthStatus(status=str(payload))

    def chat_stream_sync(self, message: str, model: str | None = None) -> Iterator[str]:
        """Synchronous streaming chat — yields text chunks."""
        try:
            import requests  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise EffGenClientError(  # noqa: F821
                "The 'requests' package is required for sync streaming. "
                "Install it with `pip install requests`, or stream with "
                "AsyncEffGenClient."
            ) from e

        body = {
            "model": model or "effgen-default",
            "messages": [{"role": "user", "content": message}],
            "stream": True,
        }
        with requests.post(
            self._url("/v1/chat/completions"),
            headers=self._headers({"Accept": "text/event-stream"}),
            json=body,
            timeout=self.timeout,
            stream=True,
        ) as resp:
            if resp.status_code >= 400:
                self._raise_for_status(resp.status_code, resp.text)
            for line in resp.iter_lines(decode_unicode=True):
                chunk = _parse_sse_line(line)
                if chunk is not None:
                    yield chunk

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------
    async def achat(
        self,
        message: str,
        tools: list[str | dict] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Asynchronous ``chat()``: send one user message and return the response.

        Args:
            message: The user message to send.
            tools: Registered tool names or full OpenAI tool specs.
            model: The model id to use, defaulting to the server's.
            **kwargs: Extra fields forwarded in the request body.
        """
        body: dict = {
            "model": model or "effgen-default",
            "messages": [{"role": "user", "content": message}],
        }
        if tools is not None:
            body["tools"] = _tools_to_payload(tools)
        body.update(kwargs)
        payload = await self._request_async("POST", "/v1/chat/completions", body)
        return self._parse_chat(payload)

    async def aembed(
        self, texts: list[str], model: str = "text-embedding-3-small"
    ) -> list[list[float]]:
        """Asynchronous ``embed()``: return one embedding vector per input text."""
        payload = await self._request_async(
            "POST", "/v1/embeddings", {"model": model, "input": texts}
        )
        if not isinstance(payload, dict) or "data" not in payload:
            raise EffGenAPIError("Malformed embeddings response", payload=payload)
        return [item["embedding"] for item in payload["data"]]

    async def ahealth(self) -> HealthStatus:
        """Asynchronous ``health()``: return the server's health status."""
        payload = await self._request_async("GET", "/health")
        if isinstance(payload, dict):
            return HealthStatus(status=str(payload.get("status", "unknown")), details=payload)
        return HealthStatus(status=str(payload))

    async def chat_stream(
        self, message: str, model: str | None = None
    ) -> AsyncIterator[str]:
        """Asynchronous streaming chat — yields text chunks."""
        try:
            import httpx  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise EffGenClientError(  # noqa: F821
                "The 'httpx' package is required for async streaming. Install "
                "it with `pip install httpx`, or stream with the sync "
                "EffGenClient."
            ) from e

        body = {
            "model": model or "effgen-default",
            "messages": [{"role": "user", "content": message}],
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as hx:
            async with hx.stream(
                "POST",
                self._url("/v1/chat/completions"),
                headers=self._headers({"Accept": "text/event-stream"}),
                json=body,
            ) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    self._raise_for_status(resp.status_code, text.decode("utf-8", errors="ignore"))
                async for line in resp.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is not None:
                        yield chunk

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_chat(payload: Any) -> ChatResponse:
        if not isinstance(payload, dict):
            raise EffGenAPIError("Malformed chat response", payload=payload)
        content = ""
        tool_calls: list[Any] = []
        choices = payload.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
        return ChatResponse(
            content=content,
            model=payload.get("model"),
            tool_calls=tool_calls,
            usage=payload.get("usage") or {},
            raw=payload,
        )


def _tools_to_payload(tools: list[Any]) -> list[dict]:
    """Normalize tool references into the server's OpenAI tool-spec shape.

    A registered tool may be named by a plain string (``"calculator"``); the
    server expects each tool as ``{"type": "function", "function": {"name":
    ...}}``. A dict is passed through unchanged so callers can send a full spec.
    """
    payload: list[dict] = []
    for tool in tools:
        if isinstance(tool, str):
            payload.append({"type": "function", "function": {"name": tool}})
        elif isinstance(tool, dict):
            payload.append(tool)
        else:
            raise EffGenAPIError(
                f"Unsupported tool reference {tool!r}: pass a registered tool "
                "name (str) or an OpenAI tool spec (dict)"
            )
    return payload


def _parse_sse_line(line: str) -> str | None:
    """Parse a single SSE line and return the text delta, or None."""
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return data
    # OpenAI-style streaming chunk
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if choices:
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            return content
    return None


# Backfill an import for exceptions at module level to avoid NameError above.
from effgen.client.exceptions import EffGenClientError  # noqa: E402
