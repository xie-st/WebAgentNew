"""Reading a live prompt evaluation without reading the provider's mood.

``PromptEval.eval_live`` calls a real model, so its result carries two very
different kinds of "did not pass": the prompt produced the wrong thing, which is
what the test is about, and the provider refused to run it at all, which is not.
A spent free-tier quota or a full inference queue is the provider's answer, and
asserting on it reports the account's state as a defect in the prompt.
"""

from __future__ import annotations

from typing import Any

import pytest


def assert_live_eval_passed(result: Any, *, output_chars: int = 800) -> None:
    """Assert a live evaluation passed, skipping if the provider refused it.

    Args:
        result: The ``EvalResult`` from ``PromptEval.eval_live``.
        output_chars: How much of the model's output to quote on failure.

    Raises:
        AssertionError: The evaluation ran and did not pass.
    """
    if result.passed:
        return

    message = str(getattr(result, "message", "") or "")
    if message:
        from effgen.models.errors import classify_provider_error

        classified = classify_provider_error(RuntimeError(message))
        if classified.rate_limited:
            pytest.skip(f"the provider throttled this evaluation: {message[:200]}")

    raise AssertionError(
        f"Live eval failed: {message}\n"
        f"Output: {str(getattr(result, 'model_output', '') or '')[:output_chars]}"
    )
