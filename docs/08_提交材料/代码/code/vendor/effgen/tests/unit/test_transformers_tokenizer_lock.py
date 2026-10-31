"""Local batch must not trip the fast tokenizer's "Already borrowed".

HuggingFace "fast" (Rust) tokenizers are not thread-safe: two threads encoding
on the same tokenizer at once raise ``RuntimeError("Already borrowed")``. The
batch runner drives an agent at concurrency>1, so the engine must serialize
tokenizer access. These tests use an in-process fake tokenizer that *detects*
overlapping access and raises the same error — no real model is loaded, and no
live behaviour is mocked; they verify the engine's own locking.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")

from effgen.models.transformers_engine import TransformersEngine  # noqa: E402


class _RaceDetectingTokenizer:
    """Raises 'Already borrowed' if two threads are inside it at once."""

    def __init__(self) -> None:
        self._active = 0
        self._guard = threading.Lock()

    def _enter(self) -> None:
        with self._guard:
            if self._active:
                raise RuntimeError("Already borrowed")
            self._active += 1

    def _exit(self) -> None:
        with self._guard:
            self._active -= 1

    def encode(self, text, add_special_tokens=False):  # noqa: ANN001
        self._enter()
        try:
            time.sleep(0.005)  # widen the window for a real race to surface
            return list(range(len(text.split())))
        finally:
            self._exit()


def _make_engine() -> TransformersEngine:
    eng = TransformersEngine("fake/model")
    eng.tokenizer = _RaceDetectingTokenizer()
    eng._is_loaded = True
    return eng


def test_engine_has_tokenizer_lock():
    eng = _make_engine()
    assert isinstance(eng._tokenizer_lock, type(threading.RLock()))


def test_concurrent_count_tokens_does_not_raise_already_borrowed():
    eng = _make_engine()
    errors: list[Exception] = []

    def work(_):
        try:
            eng.count_tokens("one two three four")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(40)))

    assert errors == [], f"tokenizer race surfaced: {errors[:3]}"


def test_unlocked_tokenizer_would_race_control():
    # Control: without the lock, the same fake tokenizer DOES race — proving the
    # detector works and that the lock above is what prevents the failure.
    tok = _RaceDetectingTokenizer()
    errors: list[Exception] = []

    def work(_):
        try:
            tok.encode("one two three four")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(40)))

    assert any("Already borrowed" in str(e) for e in errors)
