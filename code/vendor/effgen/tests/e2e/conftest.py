"""E2E test fixtures."""

import gc
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


def _cuda_cleanup():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _find_free_gpu() -> int | None:
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
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            free_bytes = props.total_memory - torch.cuda.memory_reserved(idx)
            if free_bytes > best_free:
                best_free = free_bytes
                best = idx
        if best is None or best_free < min_free_bytes:
            return None
        return best
    except ImportError:
        return None


@pytest.fixture(scope="class")
def real_model():
    """Class-scoped real model for e2e tests.

    Class scope (not module / session) isolates CUDA state between test
    classes so bitsandbytes / accelerate dispatch hooks cannot leak RMSNorm
    corruption into the next class's forward passes — a pre-existing
    upstream issue that showed up as sporadic full-suite failures.
    """
    gpu_id = _find_free_gpu()
    if gpu_id is None:
        pytest.skip("No GPU with enough free memory for e2e tests")

    from effgen import load_model

    model = load_model("Qwen/Qwen2.5-3B-Instruct", device_map={"": int(gpu_id)})
    yield model
    try:
        model.unload()
    except Exception:
        pass
    del model
    gc.collect()
    _cuda_cleanup()
