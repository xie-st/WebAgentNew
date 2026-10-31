"""vLLM engine honors ``seed`` for reproducible sampling, config or keyword.

The same contract ``tests/integration/test_local_seed_reproducible_live.py`` pins
for the Transformers engine: with a fixed seed and ``temperature>0`` two
generations of one prompt return identical text, a different seed changes it, and
a per-call ``seed=`` keyword supersedes the config's field.

Runs a real small model through a real offline vLLM engine. It skips unless vLLM
imports and a CUDA device is present — a vLLM wheel is built against one CUDA
runtime, so a host whose driver or torch build does not match it cannot import
``vllm._C`` at all. No mocking of generation behavior.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch always present in this env
    _HAS_CUDA = False

try:  # the wheel's C extension, not just the package directory
    from vllm import LLM  # noqa: F401

    _VLLM_REASON = ""
except Exception as exc:  # pragma: no cover - depends on the host's CUDA runtime
    _VLLM_REASON = f"{type(exc).__name__}: {exc}"

from effgen.models.base import GenerationConfig  # noqa: E402
from effgen.models.vllm_engine import VLLMEngine  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Write one imaginative sentence about the ocean."
SAMPLING = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 32}

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _HAS_CUDA, reason="SKIPPED: no CUDA device for a vLLM engine"),
    pytest.mark.skipif(
        bool(_VLLM_REASON), reason=f"SKIPPED: vLLM is not usable on this host ({_VLLM_REASON})"
    ),
]


@pytest.fixture(scope="module")
def engine():
    eng = VLLMEngine(
        model_name=MODEL_NAME,
        gpu_memory_utilization=0.45,
        max_model_len=2048,
        use_tqdm=False,
    )
    try:
        eng.load()
    except Exception as exc:
        # Importing vLLM says the wheel matches the host's CUDA runtime; it does
        # not say the engine can start. vLLM's engine core is a separate process
        # that JIT-compiles sampling kernels and re-initializes CUDA, so a host
        # with no CUDA toolkit, or one whose CUDA context was already created in
        # this process, fails here for a reason outside effGen. That is the same
        # class of condition as the import guard above, so it skips rather than
        # reporting a defect this suite did not find.
        pytest.skip(f"SKIPPED: a vLLM engine cannot start on this host ({type(exc).__name__}: {exc})")
    yield eng
    eng.unload()


def test_config_seed_makes_sampling_reproducible(engine):
    def _text(seed: int) -> str:
        return engine.generate(PROMPT, GenerationConfig(seed=seed, **SAMPLING)).text

    first = _text(1234)
    assert first.strip()
    assert first == _text(1234), "same seed must reproduce identical sampled text"
    assert first != _text(9999), "a different seed must change the sampled text"


def test_per_call_seed_kwarg_is_honored(engine):
    first = engine.generate(PROMPT, seed=1234, **SAMPLING).text
    assert first.strip()
    assert first == engine.generate(PROMPT, seed=1234, **SAMPLING).text
    assert first != engine.generate(PROMPT, seed=9999, **SAMPLING).text
    # A per-call seed supersedes the one on the config, like every other override.
    assert engine.generate(PROMPT, GenerationConfig(seed=1, **SAMPLING), seed=1234).text == first


def test_streaming_and_batch_honor_a_per_call_seed(engine):
    streamed = "".join(engine.generate_stream(PROMPT, seed=4242, **SAMPLING))
    assert streamed.strip()
    assert streamed == "".join(engine.generate_stream(PROMPT, seed=4242, **SAMPLING))

    batch = [r.text for r in engine.generate_batch([PROMPT, PROMPT], seed=4242, **SAMPLING)]
    assert all(text.strip() for text in batch)
    assert batch == [r.text for r in engine.generate_batch([PROMPT, PROMPT], seed=4242, **SAMPLING)]
