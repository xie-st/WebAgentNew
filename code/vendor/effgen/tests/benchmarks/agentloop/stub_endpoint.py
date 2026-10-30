"""An endpoint whose timing and token counts are known in advance.

This is not a stand-in for a model — nothing here says anything about how an
agent behaves, and every claim about that is checked against a real model. It is
a measuring standard. A clock can only be trusted once it has been made to
recover a number somebody planted, and a real model cannot plant one: it cannot
be told to take exactly 250 ms, or to report exactly 137 prompt tokens, or to
answer on the Responses API instead of Chat Completions.

The handler serves ``/v1/models`` and whichever completion route it was built
for, answers from a scripted list, sleeps for a stated time before replying, and
reports a stated usage block.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class Reply:
    """One scripted answer."""

    content: str = "Answer: A"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 100
    completion_tokens: int = 10
    finish_reason: str = "stop"


@dataclass
class StubConfig:
    model: str = "stub-model"
    #: Seconds the endpoint sleeps before answering, per completion.
    delay_s: float = 0.0
    #: Answers, in order. The last one repeats once the list runs out.
    replies: list[Reply] = field(default_factory=lambda: [Reply()])
    #: Serve completions at ``/v1/responses`` in the Responses API's shape,
    #: whose usage block names the same two numbers differently.
    responses_api: bool = False
    #: Answer with a server-sent-event body.
    stream: bool = False


class _Handler(BaseHTTPRequestHandler):
    config: StubConfig
    calls: list[dict[str, Any]]
    lock: threading.Lock

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep the test output readable
        pass

    # ------------------------------------------------------------- routing

    def do_GET(self) -> None:  # noqa: N802 - the name http.server requires
        if self.path.rstrip("/").endswith("/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.config.model, "object": "model"}],
                },
            )
            return
        self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(body)
        except ValueError:
            request = {}

        with self.lock:
            index = len(self.calls)
            self.calls.append({"path": self.path, "request": request})

        if self.config.delay_s:
            time.sleep(self.config.delay_s)

        replies = self.config.replies
        reply = replies[min(index, len(replies) - 1)]

        if self.config.stream:
            self._stream(reply)
        elif self.config.responses_api:
            self._json(200, self._responses_payload(reply))
        else:
            self._json(200, self._chat_payload(reply))

    # ------------------------------------------------------------ payloads

    def _chat_payload(self, reply: Reply) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments") or {}),
                    },
                }
                for call in reply.tool_calls
            ]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": (
                        "tool_calls" if reply.tool_calls else reply.finish_reason
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            },
        }

    def _responses_payload(self, reply: Reply) -> dict[str, Any]:
        return {
            "id": f"resp-{uuid.uuid4().hex[:8]}",
            "object": "response",
            "model": self.config.model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": reply.content}],
                }
            ],
            # The Responses API names the same two numbers differently. A reader
            # that knows only the Chat Completions names records this call as
            # free.
            "usage": {
                "input_tokens": reply.prompt_tokens,
                "output_tokens": reply.completion_tokens,
                "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            },
        }

    def _stream(self, reply: Reply) -> None:
        chunks = [
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "model": self.config.model,
                "choices": [{"index": 0, "delta": {"content": reply.content}}],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "model": self.config.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        payload = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StubEndpoint:
    """A stub endpoint running on a port of its own, as a context manager."""

    def __init__(self, config: StubConfig | None = None) -> None:
        self.config = config or StubConfig()
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        handler = type(
            "_BoundHandler",
            (_Handler,),
            {"config": self.config, "calls": self.calls, "lock": self._lock},
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "StubEndpoint":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def reset(self) -> None:
        """Forget the calls so far, so the script starts again from the top."""
        with self._lock:
            self.calls.clear()

    @property
    def completion_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if "completion" in c["path"] or "responses" in c["path"]]


__all__ = ["Reply", "StubConfig", "StubEndpoint"]
