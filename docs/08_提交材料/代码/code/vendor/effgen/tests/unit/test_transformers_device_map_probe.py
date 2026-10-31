"""The device-map viability probe runs once per set of weights.

``device_map='auto'`` can shard a model in a way that yields invalid logits,
and sampling those triggers a CUDA device-side assert that poisons the context
for the rest of the process. The engine therefore checks a forward pass before
it samples, and pins to GPU 0 when that check fails.

The check is a full forward pass, so it has to be an answer the engine
remembers: asking again before every call would tax each generation on every
multi-GPU machine forever. These use lightweight fakes — no model is loaded.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from effgen.models.transformers_engine import TransformersEngine  # noqa: E402


def _engine(device="cuda", device_map="auto"):
    engine = TransformersEngine.__new__(TransformersEngine)
    engine.device = device
    engine.device_map = device_map
    engine.model = object()
    engine.tokenizer = object()
    engine._pin_device_map_for_cuda = False
    engine._retrying_after_cuda_assert = False
    engine._device_map_probe_passed = False
    return engine


def _counting(engine, result=True):
    calls = []

    def probe():
        calls.append(1)
        return result

    engine._probe_auto_device_map_logits = probe
    return calls


def test_a_passing_probe_is_not_repeated_before_every_call():
    """Ten generations ask the GPU once, not ten times."""
    engine = _engine()
    calls = _counting(engine)

    for _ in range(10):
        engine._ensure_device_map_viable_before_sampling()

    assert len(calls) == 1, f"the probe ran {len(calls)} times for ten calls"
    assert engine._device_map_probe_passed


def test_a_new_set_of_weights_is_probed_again():
    """Dropping the weights clears the answer, because it was about those weights."""
    engine = _engine()
    calls = _counting(engine)

    engine._ensure_device_map_viable_before_sampling()
    engine.model = None
    TransformersEngine._drop_model_weights(engine)
    assert not engine._device_map_probe_passed

    engine.model = object()
    engine._ensure_device_map_viable_before_sampling()
    assert len(calls) == 2


def test_a_failing_probe_pins_to_gpu_zero_and_stops_probing():
    """Once pinned, there is nothing left to decide."""
    engine = _engine()
    calls = _counting(engine, result=False)
    pinned = []
    engine._apply_cuda_device_map_pin_fallback = lambda *, reason: (
        pinned.append(reason), setattr(engine, "_pin_device_map_for_cuda", True)
    )

    engine._ensure_device_map_viable_before_sampling()
    engine._ensure_device_map_viable_before_sampling()

    assert pinned, "an invalid-logits probe did not pin the device map"
    assert len(calls) == 1, "the engine kept probing after pinning"


@pytest.mark.parametrize(
    "device, device_map",
    [("cpu", "auto"), ("cuda", {"": 0}), ("cpu", {"": 0})],
)
def test_only_cuda_with_auto_sharding_is_probed(device, device_map):
    """A single-device placement cannot produce the sharding fault at all."""
    engine = _engine(device=device, device_map=device_map)
    calls = _counting(engine)

    engine._ensure_device_map_viable_before_sampling()

    assert not calls, f"device={device!r} device_map={device_map!r} was probed"
