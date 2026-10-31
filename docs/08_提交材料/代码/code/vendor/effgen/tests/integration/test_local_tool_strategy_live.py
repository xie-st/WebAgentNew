"""Small local models reach the calculator under both tool-calling modes.

A local chat template decides how tools reach the model, and the families differ:
Qwen2.5 and Llama 3.x render the definitions into the prompt, while gemma-2 and
Phi-3.5 accept a ``tools`` argument and discard it. Under the default mode a
model that never sees the tool answers the arithmetic itself and returns a
confident wrong number, which reads as success.

The fixture is the whole matrix — five models across four families, both modes —
because a fix that suits one family can break another. Runs real models on GPU
when one is available (skipped otherwise). No mocking of model behavior.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from effgen.core.agent import Agent, AgentConfig
from effgen.models import load_model
from effgen.tools import get_registry

TASK = "Use the calculator tool to compute 1367 * 89. Reply with just the number."
EXPECTED = "121663"

# Model id, how its chat template treats tool definitions, and the free device
# memory a run of it needs (measured on this engine's default quantized load,
# rounded up). The box these run on is shared, so a device that is merely
# present is not a device this can use.
MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "template", 5),
    ("Qwen/Qwen2.5-7B-Instruct", "template", 18),
    ("meta-llama/Llama-3.2-3B-Instruct", "template", 9),
    ("google/gemma-2-2b-it", "none", 7),
    ("microsoft/Phi-3.5-mini-instruct", "none", 10),
]


def _device_with(free_gb: int, gpu_id) -> int:
    """Index of a device with *free_gb* free, or skip the test.

    ``gpu_id`` is already the freest visible device; this only re-checks that
    what is free there is enough for the weights about to be loaded, so a
    busy shared box skips instead of failing with an allocator error.
    """
    if gpu_id is None:
        pytest.skip("SKIPPED: no CUDA device with free memory for local models")
    import torch

    free, _total = torch.cuda.mem_get_info(int(gpu_id))
    if free < free_gb * 1024**3:
        pytest.skip(
            f"SKIPPED: device {gpu_id} has {free / 1024**3:.1f} GiB free, "
            f"needs {free_gb} GiB"
        )
    return int(gpu_id)


@pytest.fixture(scope="module")
def calculator():
    return get_registry().get_tool_sync("calculator")


@pytest.mark.parametrize(
    "model_id,expected_support,needs_gb", MODELS, ids=[m[0] for m in MODELS]
)
@pytest.mark.parametrize("mode", ["auto", "react"])
def test_small_local_model_calls_the_calculator(
    model_id, expected_support, needs_gb, mode, calculator, gpu_id
):
    device = _device_with(needs_gb, gpu_id)
    model = load_model(f"transformers:{model_id}", device_map={"": device})
    try:
        assert model.tool_call_support() == expected_support
        agent = Agent(
            config=AgentConfig(
                name="strategy-matrix",
                model=model,
                tools=[calculator],
                tool_calling_mode=mode,
                temperature=0.0,
                max_tokens=512,
            )
        )
        response = agent.run(TASK)
        answer = str(response).replace(",", "").replace(" ", "")

        assert response.tool_calls >= 1, (
            f"{model_id} under {mode!r} answered without calling the tool: {str(response)!r}"
        )
        assert EXPECTED in answer, (
            f"{model_id} under {mode!r} returned {str(response)!r}, expected {EXPECTED}"
        )
    finally:
        model.unload()


@pytest.mark.parametrize(
    "model_id,needs_gb", [("Qwen/Qwen2.5-1.5B-Instruct", 5), ("google/gemma-2-2b-it", 7)]
)
def test_run_and_stream_resolve_the_same_action(model_id, needs_gb, calculator, gpu_id):
    """The two parsers must read one model output the same way."""
    from effgen.core.tool_calling import ReActStrategy

    device = _device_with(needs_gb, gpu_id)
    model = load_model(f"transformers:{model_id}", device_map={"": device})
    try:
        agent = Agent(
            config=AgentConfig(
                name="parser-parity",
                model=model,
                tools=[calculator],
                tool_calling_mode="react",
                temperature=0.0,
                max_tokens=512,
            )
        )
        text = (
            'Thought: I will use the calculator tool.\n\n'
            'Action: calculator | Action Input: {"expression": "1367 * 89"}'
        )
        assert (
            agent._parse_react_response(text)["action"]
            == ReActStrategy().parse_response(text).tool_name
            == "calculator"
        )
    finally:
        model.unload()
