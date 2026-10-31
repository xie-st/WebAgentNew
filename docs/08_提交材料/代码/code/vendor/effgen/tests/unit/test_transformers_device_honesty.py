"""Device placement and offline-cache reporting for the Transformers engine.

These verify the engine reports where parameters actually reside (rather than
assuming CUDA whenever a GPU exists), and that a missing model in offline mode
reads as a local cache miss instead of a connectivity error. The device
resolution is exercised with lightweight fakes carrying an ``hf_device_map`` /
parameters, so no real model is loaded and no live behaviour is mocked.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from effgen.models.transformers_engine import (  # noqa: E402
    ModelNotCachedError,
    TransformersEngine,
    _is_cache_miss_error,
    _offline_mode_active,
)


class _FakeParam:
    def __init__(self, is_cuda: bool) -> None:
        self.is_cuda = is_cuda


class _FakeModel:
    """Stand-in for a loaded HF model with a device map / parameters."""

    def __init__(self, device_map=None, param_cuda=True) -> None:
        if device_map is not None:
            self.hf_device_map = device_map
        self._param = _FakeParam(param_cuda)

    def parameters(self):
        yield self._param


def _engine():
    return TransformersEngine.__new__(TransformersEngine)


def test_resolve_placement_all_gpu_reports_cuda():
    eng = _engine()
    eng.device = "cuda"
    eng.model = _FakeModel(device_map={"model.layers.0": 0, "model.layers.1": 0})
    assert eng._resolve_placement() == "cuda"


def test_resolve_placement_offloaded_reports_mixed():
    eng = _engine()
    eng.device = "cuda"
    eng.model = _FakeModel(device_map={"model.layers.0": 0, "model.layers.1": "cpu"})
    assert eng._resolve_placement() == "mixed"


def test_resolve_placement_all_cpu_reports_cpu():
    eng = _engine()
    eng.device = "cuda"
    eng.model = _FakeModel(device_map={"model.layers.0": "cpu", "lm_head": "disk"})
    assert eng._resolve_placement() == "cpu"


def test_resolve_placement_without_device_map_reads_params():
    eng = _engine()
    eng.device = "cuda"
    eng.model = _FakeModel(device_map=None, param_cuda=False)
    assert eng._resolve_placement() == "cpu"


def test_offline_mode_active_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert _offline_mode_active() is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert _offline_mode_active() is True


def test_cache_miss_error_detection():
    exc = OSError(
        "We couldn't connect to 'https://huggingface.co' to load the files, and "
        "couldn't find them in the cached files."
    )
    assert _is_cache_miss_error(exc) is True
    assert _is_cache_miss_error(ValueError("some other error")) is False


def test_offline_missing_model_raises_not_cached(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    eng = TransformersEngine(model_name="definitely/not-a-real-repo-xyz-123")
    with pytest.raises(ModelNotCachedError) as excinfo:
        eng.load()
    msg = str(excinfo.value)
    assert "not in the local HuggingFace cache" in msg
    assert "definitely/not-a-real-repo-xyz-123" in msg


def test_free_vram_gb_is_nonnegative():
    # Whether or not CUDA is present, the helper returns a finite, >= 0 figure.
    assert TransformersEngine._free_vram_gb() >= 0.0
