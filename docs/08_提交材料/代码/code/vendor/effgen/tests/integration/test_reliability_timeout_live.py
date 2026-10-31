"""``with_timeout()`` must bound a real, slow cloud model call.

The OpenAI (and groq/cerebras/together) SDK client constructs its own
``max_retries``/``timeout``, independent of ``effgen.reliability``. A
one-shot SIGALRM-based timeout can be silently absorbed by the SDK's own
internal retry loop, letting the call run to its natural (much longer)
completion instead of the requested bound. This test drives a real,
naturally-slow completion through ``with_timeout()`` and confirms the call
is bounded to roughly the requested timeout, not the call's natural latency.
Skipped when no OpenAI key is configured.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)


def _has_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


_ESSAY_PROMPT = (
    "Write a detailed 500-word essay about the history and evolution of "
    "cloud computing, covering virtualization, the rise of major providers, "
    "containerization, and serverless architectures."
)


@pytest.mark.skipif(not _has_key(), reason="SKIPPED: OPENAI_API_KEY not set")
class TestTimeoutBoundsRealCall:
    def test_with_timeout_bounds_a_slow_openai_call(self):
        from effgen import load_model
        from effgen.reliability.timeouts import TimeoutError as EffGenTimeoutError
        from effgen.reliability.timeouts import with_timeout

        m = load_model("openai:gpt-5-nano")
        m.load()

        t0 = time.monotonic()
        with pytest.raises(Exception) as exc_info:  # noqa: PT011 - adapter may wrap or not
            with with_timeout(2.0, "model_call"):
                m.generate(_ESSAY_PROMPT)
        elapsed = time.monotonic() - t0

        # Bounded to a handful of retry ticks past the 2.0s deadline, not the
        # prompt's natural ~20-50s completion.
        assert elapsed < 10.0, f"with_timeout() did not bound the call: took {elapsed:.1f}s"

        exc = exc_info.value
        is_direct_timeout = isinstance(exc, EffGenTimeoutError)
        is_wrapped_timeout = "timeout" in str(exc).lower() or isinstance(
            getattr(exc, "error_context", None), dict
        ) and exc.error_context.get("category") == "timeout"
        assert is_direct_timeout or is_wrapped_timeout, (
            f"expected a timeout signal, got {type(exc).__name__}: {exc}"
        )

    def test_wrapped_timeout_is_retryable(self):
        """Whatever shape the adapter surfaces the timeout in (typed or a
        wrapped RuntimeError), is_transient_error() must classify it as
        worth retrying — the whole point of bounding the call is that a
        caller wrapping it in Retry can act on it."""
        from effgen import load_model
        from effgen.reliability.retry import is_transient_error
        from effgen.reliability.timeouts import with_timeout

        m = load_model("openai:gpt-5-nano")
        m.load()

        try:
            with with_timeout(2.0, "model_call"):
                m.generate(_ESSAY_PROMPT)
        except Exception as exc:  # noqa: BLE001
            assert is_transient_error(exc), (
                f"timeout exception not classified as transient: {type(exc).__name__}: {exc}"
            )
        else:
            pytest.fail("expected the call to be interrupted by the timeout")
