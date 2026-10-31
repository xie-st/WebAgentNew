"""``with_timeout()`` around a local Transformers generate() call must
propagate a typed, classifiable timeout instead of a bare ``RuntimeError``.

``TransformersEngine.generate()``/``generate_stream()``/``generate_batch()``
each end in a blanket ``except Exception as e: raise RuntimeError(...) from e``
that used to flatten a real ``effgen.reliability.timeouts.TimeoutError`` into
an unclassifiable ``RuntimeError`` — ``is_transient_error()`` could no longer
tell it apart from an unrelated application bug. ``_reraise_if_classified()``
re-raises the typed timeout unwrapped before that blanket handler runs;
these tests cover it directly (no GPU/model load required). The end-to-end
behavior on a real loaded model is covered by the live timeout integration
test in ``tests/integration/test_reliability_timeout_live.py``.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from effgen.models.transformers_engine import _reraise_if_classified  # noqa: E402
from effgen.reliability.retry import is_transient_error  # noqa: E402
from effgen.reliability.timeouts import TimeoutError as EffGenTimeoutError  # noqa: E402


def test_reraises_effgen_timeout_unwrapped():
    exc = EffGenTimeoutError("local_model_call", 3.0)
    with pytest.raises(EffGenTimeoutError):
        _reraise_if_classified(exc)


def test_reraised_timeout_is_still_transient():
    exc = EffGenTimeoutError("local_model_call", 3.0)
    try:
        _reraise_if_classified(exc)
    except EffGenTimeoutError as caught:
        assert is_transient_error(caught)
    else:
        pytest.fail("expected EffGenTimeoutError to propagate")


def test_non_timeout_exception_is_not_reraised():
    """Anything else must fall through so the caller's blanket RuntimeError
    wrap still applies (unchanged existing behavior for real bugs)."""
    exc = ValueError("some other failure")
    _reraise_if_classified(exc)  # returns normally, does not raise


def test_generate_blanket_handler_preserves_typed_timeout():
    """Simulates the exact try/except shape in generate()/generate_stream()/
    generate_batch(): the typed timeout must survive the blanket wrap."""
    def _simulated_generate_body():
        raise EffGenTimeoutError("local_model_call", 3.0)

    def _simulated_generate():
        try:
            _simulated_generate_body()
        except Exception as e:  # noqa: BLE001 - mirrors the real handler
            _reraise_if_classified(e)
            raise RuntimeError(f"Generation failed: {e}") from e

    with pytest.raises(EffGenTimeoutError):
        _simulated_generate()
