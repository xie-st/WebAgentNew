"""Local Transformers engine honors ``seed`` for reproducible sampling.

With a fixed ``seed`` and ``temperature>0``, two on-device generations of the
same prompt must return identical text; changing the seed must change it. Runs
a real small model on GPU when one is available (skipped otherwise). No mocking
of generation behavior.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from effgen.models import load_model
from effgen.models.base import GenerationConfig

MODEL_NAME = "transformers:Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Write one imaginative sentence about the ocean."
NEEDS_GB = 5


def _load(gpu_id):
    """Load the model on a device with room for it, or skip.

    The box these run on is shared, so a device that is merely present is not
    a device this can use; without the free-memory check a busy neighbour turns
    the test into an allocator error instead of a skip.
    """
    if gpu_id is None:
        pytest.skip("SKIPPED: no CUDA device with free memory for local model")
    import torch

    free, _total = torch.cuda.mem_get_info(int(gpu_id))
    if free < NEEDS_GB * 1024**3:
        pytest.skip(
            f"SKIPPED: device {gpu_id} has {free / 1024**3:.1f} GiB free, "
            f"needs {NEEDS_GB} GiB"
        )
    return load_model(MODEL_NAME, device_map={"": int(gpu_id)})


def test_local_engine_seed_makes_sampling_reproducible(gpu_id):
    model = _load(gpu_id)

    def _cfg(seed: int) -> GenerationConfig:
        return GenerationConfig(temperature=0.8, top_p=0.95, max_tokens=40, seed=seed)

    first = model.generate(PROMPT, _cfg(1234)).text
    again = model.generate(PROMPT, _cfg(1234)).text
    other = model.generate(PROMPT, _cfg(9999)).text

    assert first.strip()
    assert first == again, "same seed must reproduce identical sampled text"
    assert first != other, "a different seed must change the sampled text"


def test_local_engine_seed_honored_as_a_per_call_kwarg(gpu_id):
    """``seed=`` passed alongside the other sampling kwargs behaves like the config field."""
    model = _load(gpu_id)
    sampling = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 40}

    first = model.generate(PROMPT, seed=1234, **sampling).text
    again = model.generate(PROMPT, seed=1234, **sampling).text
    other = model.generate(PROMPT, seed=9999, **sampling).text

    assert first.strip()
    assert first == again
    assert first != other

    # A per-call seed supersedes the one on the config, like every other override.
    config = GenerationConfig(temperature=0.8, top_p=0.95, max_tokens=40, seed=1)
    assert model.generate(PROMPT, config, seed=1234).text == first


def test_local_engine_streaming_is_seeded_too(gpu_id):
    """A streamed generation reproduces from a fixed seed, same as a buffered one."""
    model = _load(gpu_id)

    def _stream(**kw) -> str:
        return "".join(model.generate_stream(PROMPT, **kw))

    sampling = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 40}
    first = _stream(seed=4242, **sampling)
    again = _stream(seed=4242, **sampling)
    other = _stream(seed=99, **sampling)

    assert first.strip()
    assert first == again
    assert first != other
    # The config field drives streaming identically to the kwarg.
    config = GenerationConfig(temperature=0.8, top_p=0.95, max_tokens=40, seed=4242)
    assert "".join(model.generate_stream(PROMPT, config)) == first
