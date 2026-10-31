"""Counting model calls, tokens and model time at the HTTP boundary.

An agent framework builds its own HTTP client, so there is no single place
inside it where calls could be counted. They all go through ``httpx`` though, so
this wraps ``httpx``'s two ``send`` methods once per process and reads the
``usage`` block off every completion response.

Wrapping the transport rather than the framework is deliberate. The framework is
what is being measured; a number it reports about itself cannot be used to judge
it. The HTTP boundary is outside it and is the one place where "time spent
waiting for the model" and "time spent in the code around the model" are cleanly
separated:

    wall            measured around one whole run
    model_wall      summed duration of the HTTP calls it made
    framework_wall  wall - model_wall
    framework_cpu   CPU this thread burned, which is almost all outside the call

Counters are per thread, so concurrent samples do not mix, and a thread a
framework starts inherits the bucket of the sample that started it — without
that, a framework that generates on its own thread is recorded as making no
calls at all.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

_local = threading.local()
_installed = False
_install_lock = threading.Lock()


@dataclass
class Usage:
    """What one sample's HTTP traffic came to."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Seconds spent inside completion calls, summed.
    model_wall_s: float = 0.0
    #: A streamed response was seen. ``send`` returns before a streamed body has
    #: been read, so ``model_wall_s`` is short by however long that took and the
    #: split must not be quoted as a number for this sample.
    streaming_seen: bool = False
    #: Requests that looked like completions but carried no usage block.
    uncounted: int = 0
    urls: list[str] = field(default_factory=list)

    def add(self, prompt: int, completion: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _current() -> Usage | None:
    return getattr(_local, "usage", None)


def _looks_like_completion(url: str) -> bool:
    """Whether a request is a generation call.

    ``/responses`` belongs here as much as ``/chat/completions``: a client that
    talks the Responses API is generating, and reading only the first shape once
    recorded a whole run as making no calls and costing nothing.
    """
    return (
        "/chat/completions" in url
        or url.endswith("/completions")
        or url.endswith("/responses")
        or "/responses/" in url
    )


def _record_usage(payload: dict) -> bool:
    bucket = _current()
    if bucket is None:
        return False
    usage = payload.get("usage") or {}
    # Chat Completions names these `prompt_tokens`/`completion_tokens`; the
    # Responses API names the same two numbers `input_tokens`/`output_tokens`.
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("input_tokens")
    completion = usage.get("completion_tokens")
    if completion is None:
        completion = usage.get("output_tokens")
    if prompt is None and completion is None:
        return False
    bucket.add(int(prompt or 0), int(completion or 0))
    return True


def _harvest(response, elapsed_s: float) -> None:
    """Read one finished response into the current sample's bucket."""
    bucket = _current()
    if bucket is None:
        return
    try:
        url = str(response.request.url)
        if not _looks_like_completion(url):
            return
        bucket.model_wall_s += elapsed_s
        bucket.urls.append(url)
        if response.status_code >= 400:
            return
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Reading the body here would consume it before the caller does, and
            # the call is not finished yet anyway. Flag it; the split is reported
            # as unreliable rather than as a number that is quietly short.
            bucket.streaming_seen = True
            bucket.uncounted += 1
            return
        if "json" not in content_type:
            bucket.uncounted += 1
            return
        # The body has to be read explicitly: the OpenAI SDK asks httpx for the
        # response with stream=True even for a JSON reply and decides per
        # response whether to buffer, so waiting for an already-read body misses
        # every call under an SDK that does. `read()` caches on the response, so
        # the caller still gets its body afterwards.
        body = getattr(response, "_content", None)
        if body is None:
            body = response.read()
        if not body:
            bucket.uncounted += 1
            return
        if not _record_usage(json.loads(body)):
            bucket.uncounted += 1
    except Exception:
        # Accounting must never break a run.
        pass


def install() -> None:
    """Wrap ``httpx`` once per process. Safe to call repeatedly."""
    global _installed
    with _install_lock:
        if _installed:
            return
        wrapped = 0
        for name in ("httpx", "httpx2"):
            try:
                module = __import__(name)
            except ImportError:
                continue
            _wrap(module)
            wrapped += 1
        if wrapped:
            _install_thread_inheritance()
        _installed = True


def _wrap(module) -> None:
    sync_send = module.Client.send
    async_send = module.AsyncClient.send

    def patched_send(self, request, **kwargs):
        started = time.perf_counter()
        response = sync_send(self, request, **kwargs)
        _harvest(response, time.perf_counter() - started)
        return response

    async def patched_asend(self, request, **kwargs):
        started = time.perf_counter()
        response = await async_send(self, request, **kwargs)
        _harvest(response, time.perf_counter() - started)
        return response

    module.Client.send = patched_send
    module.AsyncClient.send = patched_asend


def _install_thread_inheritance() -> None:
    """Let a thread a framework starts keep counting for the sample that started it.

    A thread inherits whatever bucket was current when it was *created*, which is
    the sample that created it. Threads created before a sample starts — a pool
    that was already warm — inherit nothing, which is what keeps concurrent
    samples separate.
    """
    original_init = threading.Thread.__init__
    original_run = threading.Thread.run

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._agentloop_usage = _current()

    def patched_run(self):
        bucket = getattr(self, "_agentloop_usage", None)
        if bucket is not None and _current() is None:
            _local.usage = bucket
        return original_run(self)

    threading.Thread.__init__ = patched_init  # type: ignore[method-assign]
    threading.Thread.run = patched_run  # type: ignore[method-assign]


@dataclass
class Timed:
    """One sample's clocks, filled in when the block ends."""

    usage: Usage
    wall_s: float = 0.0
    cpu_s: float = 0.0

    @property
    def model_wall_s(self) -> float:
        return self.usage.model_wall_s

    @property
    def framework_wall_s(self) -> float:
        """Wall time outside the model calls.

        At concurrency 1 this is the framework's own time. Above 1 it also holds
        contention for the interpreter and time the scheduler spent elsewhere, so
        ``framework_cpu_s`` is the stable one there. Both are always recorded and
        the concurrency is always in the header.
        """
        return max(self.wall_s - self.usage.model_wall_s, 0.0)

    @property
    def framework_cpu_s(self) -> float:
        return self.cpu_s


@contextmanager
def measure():
    """Count and time everything this thread sends while the block runs."""
    install()
    previous = getattr(_local, "usage", None)
    timed = Timed(usage=Usage())
    _local.usage = timed.usage
    wall0 = time.perf_counter()
    cpu0 = time.thread_time()
    try:
        yield timed
    finally:
        timed.wall_s = time.perf_counter() - wall0
        timed.cpu_s = time.thread_time() - cpu0
        _local.usage = previous


def note_usage(prompt_tokens: int, completion_tokens: int) -> None:
    """Record a call the transport hook could not see."""
    bucket = _current()
    if bucket is not None:
        bucket.add(int(prompt_tokens or 0), int(completion_tokens or 0))


__all__ = ["Timed", "Usage", "install", "measure", "note_usage"]
