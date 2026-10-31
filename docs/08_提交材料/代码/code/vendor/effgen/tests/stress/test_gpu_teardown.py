"""Unloading a local model returns its device memory to the GPU.

``torch.cuda.empty_cache()`` can only release an allocator segment that has no
live block left in it. cuBLAS keeps a per-stream workspace allocated through
the same allocator; when the weights were loaded into one large segment that
workspace sits inside it, and a few megabytes then hold the whole segment — so
the gigabytes a model occupied stayed reserved after ``unload()`` and the next
model to load could not use them.

Runs a real small model on a real GPU (skipped without one). No mocking of the
allocator behaviour under test.
"""

from __future__ import annotations

import gc
import os

import pytest

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch always present in this env
    _HAS_CUDA = False

from effgen.models import load_model

MODEL_NAME = "transformers:Qwen/Qwen2.5-1.5B-Instruct"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.expensive,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_CUDA, reason="requires a CUDA GPU"),
]


def _reserved_mb() -> float:
    """Memory this process has cached, summed over every visible device.

    The model does not necessarily land on the current device — a busy GPU 0
    sends it elsewhere — so a single-device reading can miss it entirely.
    """
    return sum(
        torch.cuda.memory_reserved(device) for device in range(torch.cuda.device_count())
    ) / 1024**2


def _allocated_mb() -> float:
    """Memory held by live tensors, summed over every visible device.

    Unlike the reserved figure this is unaffected by what the allocator has
    already cached, so it states whether the weights are resident whatever the
    process did before.
    """
    return sum(
        torch.cuda.memory_allocated(device) for device in range(torch.cuda.device_count())
    ) / 1024**2


class TestLocalModelTeardown:
    def test_unload_returns_reserved_memory(self):
        """After ``unload()`` the process holds no cached device memory."""
        gc.collect()
        torch.cuda.empty_cache()
        baseline = _reserved_mb()
        baseline_live = _allocated_mb()

        model = load_model(MODEL_NAME)
        result = model.generate("Reply with one word: hello", max_tokens=8)
        assert (result.text or "").strip(), "model produced no output"
        loaded = _reserved_mb()
        # memory_allocated (not reserved) for the precondition: when the process
        # already holds a large cache the weights are served from it, so the
        # reserved figure barely moves while the live figure grows by the model.
        loaded_live = _allocated_mb()
        assert loaded_live > baseline_live + 100, (
            f"expected the weights to be resident: {baseline_live}MB -> {loaded_live}MB"
        )

        model.unload()
        gc.collect()
        after = _reserved_mb()
        # A little slack for anything the test process itself allocated.
        assert after <= baseline + 32, (
            f"{after - baseline:.0f}MB still reserved after unload "
            f"(baseline {baseline:.0f}MB, loaded {loaded:.0f}MB)"
        )

    def test_repeated_load_unload_does_not_drift(self):
        """Load/unload cycles return to the same reserved footprint each time."""
        gc.collect()
        torch.cuda.empty_cache()
        baseline = _reserved_mb()

        for cycle in range(3):
            model = load_model(MODEL_NAME)
            result = model.generate("Reply with one word: hello", max_tokens=8)
            assert (result.text or "").strip(), f"no output on cycle {cycle}"
            model.unload()
            del model
            gc.collect()
            after = _reserved_mb()
            assert after <= baseline + 32, (
                f"cycle {cycle}: {after - baseline:.0f}MB still reserved"
            )


class TestReleaseCachedMemory:
    def test_probing_devices_does_not_create_contexts(self):
        """Listing devices with cached memory leaves untouched GPUs untouched."""
        from effgen.gpu.utils import _devices_with_cached_memory, get_device_count

        before = {d: torch.cuda.memory_reserved(d) for d in range(get_device_count())}
        devices = _devices_with_cached_memory()
        after = {d: torch.cuda.memory_reserved(d) for d in range(get_device_count())}
        assert before == after
        assert all(before[d] > 0 for d in devices)

    def test_release_is_safe_to_call_with_nothing_loaded(self):
        """Releasing with no model loaded is a no-op, not an error."""
        from effgen.gpu.utils import release_cached_memory

        gc.collect()
        release_cached_memory()
        release_cached_memory(0)


class TestVLLMTeardown:
    def test_unload_releases_reserved_memory(self):
        """The vLLM engine's teardown returns cached device memory.

        Drives ``unload()`` on an engine that was never loaded, so the release
        runs against real reserved memory on a machine where vLLM itself may
        not be importable (its wheel is built against a specific CUDA runtime).
        """
        from effgen.models.vllm_engine import VLLMEngine

        gc.collect()
        torch.cuda.empty_cache()
        baseline = _reserved_mb()

        # memory_allocated (not reserved) for the precondition: a small buffer
        # can be served from a segment that is already reserved.
        allocated = torch.cuda.memory_allocated()
        buffer = torch.empty(50_000_000 // 4, device="cuda")
        assert torch.cuda.memory_allocated() > allocated, "buffer did not allocate"

        engine = VLLMEngine("Qwen/Qwen2.5-1.5B-Instruct")
        del buffer
        engine.unload()

        assert engine._is_loaded is False
        assert _reserved_mb() <= baseline, (
            f"{_reserved_mb() - baseline:.0f}MB still reserved after unload"
        )
