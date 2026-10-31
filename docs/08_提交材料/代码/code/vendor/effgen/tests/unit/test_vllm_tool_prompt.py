"""Tool definitions reach the vLLM chat template, not vLLM's ``generate()``.

The agent attaches its tool definitions to every model call. ``LLM.generate()``
has no ``tools`` parameter, so forwarding them raised ``got an unexpected
keyword argument 'tools'`` and **every** tool-using turn on the vLLM engine
failed — the answer a user saw was "Generation failed". Tools belong to the
chat template, which is where the Transformers engine has always put them.

These run without vLLM installed and without a GPU: the engine object is built
without loading anything and its tokenizer is replaced with a recorder, so what
is asserted is the routing rather than a model's answer.
"""

from __future__ import annotations

from typing import Any

import pytest

# The engine module reaches for the vLLM SDK, which is an optional extra and is
# not installed on the CI unit matrix. Skipping by name here states that plainly
# rather than leaving it to the absent-optional-dependency rule to infer from a
# traceback.
pytest.importorskip("vllm", reason="the vllm extra is not installed")


class _RecordingTokenizer:
    """Stands in for the HF tokenizer, recording what the template was given."""

    chat_template = "{% if tools %}TOOLS{% endif %}{{ messages }}"

    def __init__(self, *, accepts_tools: bool = True) -> None:
        self.accepts_tools = accepts_tools
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, **kwargs):
        if "tools" in kwargs and not self.accepts_tools:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'tools'")
        self.calls.append(dict(kwargs))
        return "FORMATTED"


TOOLS = [{"type": "function", "function": {"name": "calculator", "parameters": {}}}]


@pytest.fixture
def engine():
    from effgen.models.vllm_engine import VLLMEngine

    engine = VLLMEngine(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    engine._hf_tokenizer = _RecordingTokenizer()
    engine.apply_chat_template = True
    return engine


class TestToolsGoToTheTemplate:
    def test_the_template_is_given_the_tools(self, engine):
        engine._format_prompt_with_chat_template("hi", None, TOOLS)
        assert engine._hf_tokenizer.calls[-1]["tools"] == TOOLS

    def test_no_tools_means_no_tools_argument(self, engine):
        engine._format_prompt_with_chat_template("hi", None, None)
        assert "tools" not in engine._hf_tokenizer.calls[-1]

    def test_a_template_that_refuses_tools_still_renders(self, engine):
        """``tool_call_support() == "none"`` is a real answer, not a crash."""
        engine._hf_tokenizer = _RecordingTokenizer(accepts_tools=False)
        assert engine._format_prompt_with_chat_template("hi", None, TOOLS) == "FORMATTED"
        assert "tools" not in engine._hf_tokenizer.calls[-1]

    def test_generate_does_not_forward_tools_to_vllm(self, engine):
        """The regression itself: ``tools`` must not reach ``LLM.generate()``."""
        forwarded: dict[str, Any] = {}

        class _Output:
            text = "ok"
            token_ids = [1, 2]
            finish_reason = "stop"

        class _Result:
            outputs = [_Output()]
            prompt_token_ids = [1]

        class _LLM:
            def generate(self, prompts, sampling_params, **kwargs):
                if "tools" in kwargs:
                    raise TypeError(
                        "LLM.generate() got an unexpected keyword argument 'tools'"
                    )
                forwarded.update(kwargs)
                return [_Result()]

        engine.llm = _LLM()
        engine._is_loaded = True
        # Length validation counts tokens through the real tokenizer, which is
        # not what this is measuring.
        engine.validate_prompt = lambda *_a, **_k: None
        result = engine.generate("What is 6*7?", tools=TOOLS)

        assert str(result.text) == "ok"
        assert "tools" not in forwarded
        assert engine._hf_tokenizer.calls[-1]["tools"] == TOOLS


class TestChatTemplateKwargs:
    """The same ``chat_template_kwargs`` spelling as the Transformers engine.

    Switching engines should not change how a reasoning model's thinking is
    turned off.
    """

    def test_they_are_held_out_of_the_vllm_arguments(self):
        from effgen.models.vllm_engine import VLLMEngine

        engine = VLLMEngine(
            model_name="Qwen/Qwen3.5-2B",
            chat_template_kwargs={"enable_thinking": False},
        )
        assert engine.chat_template_kwargs == {"enable_thinking": False}
        assert "chat_template_kwargs" not in engine.additional_kwargs

    def test_they_reach_the_template(self, engine):
        engine.chat_template_kwargs = {"enable_thinking": False}
        engine._format_prompt_with_chat_template("hi", None, None)
        assert engine._hf_tokenizer.calls[-1]["enable_thinking"] is False
