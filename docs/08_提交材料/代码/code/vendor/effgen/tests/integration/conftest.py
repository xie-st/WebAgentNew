"""
Integration test fixtures requiring real GPU models.
"""

import os
import warnings

import pytest

warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _visible_cuda_indices() -> list[int] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    indices = []
    for raw in visible.split(","):
        raw = raw.strip()
        if raw.isdigit():
            indices.append(int(raw))
    return indices or None


def _find_free_gpu():
    """Find a GPU with enough free memory for the integration SLM."""
    min_free_gb = float(os.environ.get("EFFGEN_TEST_MIN_FREE_GPU_GB", "8"))
    min_free_bytes = int(min_free_gb * 1024**3)

    try:
        import pynvml

        pynvml.nvmlInit()
        visible = _visible_cuda_indices()
        candidates = visible if visible is not None else list(range(pynvml.nvmlDeviceGetCount()))
        best = None
        best_free = -1
        for visible_idx, physical_idx in enumerate(candidates):
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if info.free > best_free:
                best_free = info.free
                best = visible_idx if visible is not None else physical_idx
        if best is not None and best_free >= min_free_bytes:
            return best
        return None
    except Exception:
        pass

    try:
        import torch
        if not torch.cuda.is_available():
            return None
        best = None
        best_free = -1
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_bytes = props.total_memory - torch.cuda.memory_reserved(i)
            # Also check via nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_bytes = info.free
            except Exception:
                pass
            if free_bytes > best_free:
                best_free = free_bytes
                best = i
        if best is None or best_free < min_free_bytes:
            return None
        return best
    except ImportError:
        return None


@pytest.fixture(scope="session")
def gpu_id():
    """Session-scoped fixture for GPU ID."""
    return _find_free_gpu()


def _load_and_yield(gpu_id, quantization=None):
    """Load a model and clean up CUDA state on teardown."""
    from effgen import load_model
    load_kwargs = {"device_map": {"": int(gpu_id)}}
    if quantization:
        try:
            model = load_model(
                "Qwen/Qwen2.5-3B-Instruct",
                quantization=quantization,
                **load_kwargs,
            )
        except Exception:
            model = load_model("Qwen/Qwen2.5-3B-Instruct", **load_kwargs)
    else:
        model = load_model("Qwen/Qwen2.5-3B-Instruct", **load_kwargs)
    yield model
    try:
        model.unload()
    except Exception:
        pass
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


@pytest.fixture(scope="class")
def real_model(gpu_id):
    """Class-scoped fixture that loads a real model once per test class.

    Class scope isolates accelerate's device-dispatch hook state between test
    classes so a prior test's forward pass cannot corrupt Qwen2 RMSNorm on a
    subsequent class — an upstream torch/accelerate issue that otherwise
    manifests as a C-level abort. Loaded in fp16 (no bitsandbytes 4-bit).
    """
    if gpu_id is None:
        pytest.skip("No GPU available")
    yield from _load_and_yield(gpu_id, quantization=None)


@pytest.fixture(scope="class")
def streaming_model(gpu_id):
    """Class-scoped fixture for streaming tests — loaded without 4-bit quant.

    bitsandbytes 4-bit kernels leave CUDA stream state that causes
    TextIteratorStreamer.text_queue.get() to block indefinitely.  Loading
    the model in fp16 / bf16 avoids those kernels entirely. Class scope
    (not module / session) prevents cross-class state bleed.
    """
    if gpu_id is None:
        pytest.skip("No GPU available")
    yield from _load_and_yield(gpu_id, quantization=None)
