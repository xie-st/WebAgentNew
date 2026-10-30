"""The generation path must not hold the event loop.

A completion can take tens of seconds. While one is running, the operational
endpoints — the liveness probe, the metrics scrape, the dashboard aggregation —
must keep answering, or a Kubernetes probe fails and a monitor freezes exactly
when there is activity worth watching.

These drive a real uvicorn server over a loopback socket and probe it from
another thread, because an in-process ASGI transport cannot measure a blocked
event loop: the probe's own timer would not start until the loop was free
again. The runner stands in for the model — a plain blocking ``sleep``
reproduces the property under test (a synchronous callable occupying a thread)
without spending a live model call or making the timing depend on a provider.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

RUNNER_BLOCK_S = 3.0
# A responsive endpoint answers in milliseconds; this bound is loose enough for
# a loaded CI box yet far below RUNNER_BLOCK_S.
RESPONSIVE_S = 1.0


def _free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def server(monkeypatch):
    """A real server whose runner blocks its calling thread for RUNNER_BLOCK_S."""
    import uvicorn

    monkeypatch.setenv("EFFGEN_DEV_MODE", "1")
    monkeypatch.setenv("EFFGEN_PUBLIC_DASHBOARD", "1")
    monkeypatch.setenv("EFFGEN_PUBLIC_METRICS", "1")

    from effgen.api.openai_compat import RunnerResult
    from effgen.server.app import create_app

    def blocking_runner(prompt, *, model, tools=None, stream=False, **_):
        time.sleep(RUNNER_BLOCK_S)
        if stream:
            return iter(["slow ", "answer"])
        return RunnerResult(
            text="slow answer", prompt_tokens=3, completion_tokens=2, resolved_model=model
        )

    port = _free_port()
    config = uvicorn.Config(
        create_app(runner=blocking_runner),
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    try:
        for _ in range(300):
            if instance.started:
                break
            time.sleep(0.1)
        assert instance.started, "server did not start"
        yield f"http://127.0.0.1:{port}"
    finally:
        instance.should_exit = True
        thread.join(timeout=20)


def _get(url: str, timeout: float = 30.0) -> tuple[int, float, str]:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
            return response.status, time.monotonic() - started, body
    except urllib.error.HTTPError as exc:
        return exc.code, time.monotonic() - started, ""


def _post_completion(base: str, *, stream: bool = False, path: str = "/v1/chat/completions"):
    payload: dict = {"model": "openai:gpt-5-nano"}
    if path.endswith("/completions") and "chat" not in path:
        payload["prompt"] = "hi"
    else:
        payload["messages"] = [{"role": "user", "content": "hi"}]
    if stream:
        payload["stream"] = True
    request = urllib.request.Request(  # noqa: S310
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.status, time.monotonic() - started, response.read().decode()


@pytest.mark.parametrize("path", ["/health", "/metrics", "/dashboard/data.json"])
def test_ops_endpoints_answer_during_a_completion(server, path):
    idle_status, idle_elapsed, _ = _get(f"{server}{path}")
    assert idle_status == 200

    with ThreadPoolExecutor(max_workers=2) as pool:
        completion = pool.submit(_post_completion, server)
        time.sleep(0.5)  # let the generation get under way
        status, elapsed, _ = _get(f"{server}{path}")
        chat_status, chat_elapsed, body = completion.result()

    assert status == 200, f"{path} returned {status} during a generation"
    assert elapsed < RESPONSIVE_S, (
        f"{path} took {elapsed:.2f}s while a generation was in flight "
        f"(idle: {idle_elapsed:.3f}s) — the event loop was blocked"
    )
    assert chat_status == 200
    assert chat_elapsed >= RUNNER_BLOCK_S, "the generation did not actually run"
    assert json.loads(body)["choices"][0]["message"]["content"] == "slow answer"


def test_text_completions_path_also_stays_responsive(server):
    assert _get(f"{server}/health")[0] == 200
    with ThreadPoolExecutor(max_workers=2) as pool:
        completion = pool.submit(_post_completion, server, path="/v1/completions")
        time.sleep(0.5)
        status, elapsed, _ = _get(f"{server}/health")
        assert completion.result()[0] == 200
    assert status == 200
    assert elapsed < RESPONSIVE_S, f"/health took {elapsed:.2f}s during a text completion"


def test_two_completions_overlap_instead_of_serialising(server):
    with ThreadPoolExecutor(max_workers=2) as pool:
        started = time.monotonic()
        results = [pool.submit(_post_completion, server) for _ in range(2)]
        statuses = [future.result()[0] for future in results]
        elapsed = time.monotonic() - started
    assert statuses == [200, 200]
    assert elapsed < 2 * RUNNER_BLOCK_S * 0.9, (
        f"two {RUNNER_BLOCK_S}s generations took {elapsed:.2f}s — they serialised"
    )


def test_streaming_still_streams(server):
    """Moving the runner call off the loop leaves the streamed path intact."""
    status, _, body = _post_completion(server, stream=True)
    assert status == 200
    assert "slow" in body and "answer" in body
    assert body.rstrip().endswith("data: [DONE]")
