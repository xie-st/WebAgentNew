"""Live proof that per-call generation kwargs work on modern OpenAI models.

``generate(prompt, temperature=0)`` — and ``max_tokens=`` / ``top_p=`` /
``stop=`` — must succeed on gpt-5/o-series, which reject the raw forms. These
tests make real API calls and are skipped if OPENAI_API_KEY is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set — skipping live OpenAI generation-kwargs tests",
)

MODEL = "gpt-5-nano"


@pytest.fixture(scope="module")
def adapter():
    from effgen.models.openai_adapter import OpenAIAdapter

    a = OpenAIAdapter(model_name=MODEL, api_key=OPENAI_API_KEY)
    a.load()
    yield a
    a.unload()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"max_tokens": 2000},
        {"top_p": 0.5},
        {"stop": ["\n"]},
        {"presence_penalty": 0.1},
        {"frequency_penalty": 0.1},
        {"seed": 7},
        {"temperature": 0, "max_tokens": 2000, "top_p": 0.9, "stop": "END"},
    ],
)
def test_per_call_kwargs_do_not_400(adapter, kwargs):
    # The whole point: none of these raise an "Unsupported parameter" 400.
    result = adapter.generate("Say hi in one word.", **kwargs)
    assert result is not None
    assert isinstance(result.text, str)


def test_stream_with_per_call_kwargs(adapter):
    chunks = list(adapter.generate_stream("Count 1 to 3.", temperature=0, max_tokens=2000))
    assert "".join(chunks).strip() != ""
