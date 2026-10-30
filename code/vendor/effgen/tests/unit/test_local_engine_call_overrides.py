"""Per-call sampling keywords on the local engines.

``model.generate(prompt, seed=1234, temperature=0.8)`` — the bare keyword form,
without building a ``GenerationConfig`` — is honored by the Transformers and GGUF
engines (covered live in ``tests/integration/test_local_seed_reproducible_live.py``
and the GGUF suite). These tests pin the same contract for the two engines that
cannot run on a Linux/CUDA host: vLLM (needs a matching CUDA build plus a loaded
engine) and MLX (Apple Silicon only).

The provider entry points are stood in for, with the signatures the real ones
have, so a keyword the engine forwards by mistake fails here exactly as it would
against the real library: ``vllm.LLM.generate`` takes no ``seed`` and no
``**kwargs``, so a leaked sampling keyword raises ``TypeError``. What is asserted
is only what effGen builds and what it hands over — never model output.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from typing import Any

import pytest

from effgen.models._adapter_utils import (
    CALL_OVERRIDE_ALIASES,
    CALL_OVERRIDE_FIELDS,
    merge_call_overrides,
)
from effgen.models.base import GenerationConfig, TokenCount

# ---------------------------------------------------------------------------
# The shared merge helper
# ---------------------------------------------------------------------------


class TestMergeCallOverrides:
    """``merge_call_overrides`` resolves per-call keywords against the config."""

    def test_every_override_name_is_a_generation_config_field(self):
        fields = {f.name for f in dataclasses.fields(GenerationConfig)}
        assert set(CALL_OVERRIDE_FIELDS) <= fields
        assert set(CALL_OVERRIDE_ALIASES.values()) <= fields

    def test_per_call_value_supersedes_the_config_field(self):
        config = GenerationConfig(temperature=0.2, seed=1)
        kwargs: dict[str, Any] = {"seed": 1234, "temperature": 0.8}
        merged = merge_call_overrides(config, kwargs)
        assert (merged.seed, merged.temperature) == (1234, 0.8)
        # Consumed, so the caller cannot forward them to the provider call.
        assert kwargs == {}
        # The caller's config object is untouched.
        assert (config.seed, config.temperature) == (1, 0.2)

    def test_config_field_is_kept_when_no_keyword_is_given(self):
        config = GenerationConfig(seed=99)
        kwargs: dict[str, Any] = {}
        assert merge_call_overrides(config, kwargs) is config

    def test_unknown_keywords_are_left_for_the_caller(self):
        kwargs: dict[str, Any] = {"seed": 7, "lora_request": "adapter-a"}
        merged = merge_call_overrides(GenerationConfig(), kwargs)
        assert merged.seed == 7
        assert kwargs == {"lora_request": "adapter-a"}

    def test_stop_is_accepted_as_the_alias_for_stop_sequences(self):
        merged = merge_call_overrides(GenerationConfig(), {"stop": ["\n\n"]})
        assert merged.stop_sequences == ["\n\n"]

    @pytest.mark.parametrize("key", ["stop", "stop_sequences"])
    def test_a_bare_string_stop_becomes_a_one_element_list(self, key):
        """The backends iterate ``stop_sequences``, so a string must be wrapped.

        Left as a string it would be walked character by character and cut the
        generated text at the first matching letter instead of at the sequence.
        """
        merged = merge_call_overrides(GenerationConfig(), {key: "END"})
        assert merged.stop_sequences == ["END"]

    def test_the_sampling_keywords_match_the_cloud_adapters(self):
        """The local and cloud paths resolve the same keywords to the same fields.

        Both fold per-call sampling keywords into a copy of the config; a name
        one of them recognised and the other forwarded raw would behave
        differently depending on which model a caller happened to load.
        """
        from effgen.models.openai_adapter import (
            _GENERATION_KWARG_TO_FIELD,
            _fold_generation_kwargs,
        )

        local_names = set(CALL_OVERRIDE_FIELDS) | set(CALL_OVERRIDE_ALIASES)
        for name in local_names:
            assert name in _GENERATION_KWARG_TO_FIELD, name
            expected = CALL_OVERRIDE_ALIASES.get(name, name)
            assert _GENERATION_KWARG_TO_FIELD[name] == expected, name
        # ...including how each treats the API's bare-string ``stop``.
        assert (
            merge_call_overrides(GenerationConfig(), {"stop": "END"}).stop_sequences
            == _fold_generation_kwargs(GenerationConfig(), {"stop": "END"}).stop_sequences
        )


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

# Mirrors vllm.SamplingParams' accepted sampling fields (vllm 0.22).
_VLLM_SAMPLING_FIELDS = frozenset(
    {
        "n",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "max_tokens",
        "min_tokens",
        "logprobs",
        "detokenize",
        "skip_special_tokens",
    }
)


class FakeSamplingParams:
    """Stand-in for ``vllm.SamplingParams`` that records what it was asked for."""

    def __init__(self, **kw: Any) -> None:
        unexpected = sorted(set(kw) - _VLLM_SAMPLING_FIELDS)
        if unexpected:
            raise TypeError(f"SamplingParams got unexpected keyword(s): {unexpected}")
        self.requested = dict(kw)
        self.__dict__.update(kw)


class _FakeCompletion:
    text = "generated"
    token_ids = (1, 2, 3)
    finish_reason = "stop"


class _FakeRequestOutput:
    prompt_token_ids = (1, 2)
    outputs = (_FakeCompletion(),)


class FakeLLM:
    """Stand-in for ``vllm.LLM`` with the real ``generate`` signature.

    vLLM's entry point declares no ``seed`` and no ``**kwargs``, so a sampling
    keyword the engine forwards instead of consuming raises ``TypeError`` here,
    the same way it does against the installed library.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompts,
        sampling_params=None,
        *,
        use_tqdm=True,
        lora_request=None,
        priority=None,
        tokenization_kwargs=None,
        mm_processor_kwargs=None,
    ):
        self.calls.append(
            {
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "lora_request": lora_request,
                "priority": priority,
            }
        )
        count = 1 if isinstance(prompts, str) else len(prompts)
        return [_FakeRequestOutput() for _ in range(count)]


@pytest.fixture
def vllm_engine(monkeypatch):
    """A loaded ``VLLMEngine`` wired to the stand-in vLLM entry points."""
    pytest.importorskip("torch")  # the engine module imports torch at import time
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(SamplingParams=FakeSamplingParams, LLM=FakeLLM),
    )
    from effgen.models.vllm_engine import VLLMEngine

    engine = VLLMEngine(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        apply_chat_template=False,
        use_tqdm=False,
    )
    engine.llm = FakeLLM()
    engine._is_loaded = True
    engine._context_length = 4096
    monkeypatch.setattr(
        type(engine),
        "count_tokens",
        lambda self, text: TokenCount(count=len(text.split()), model_name=self.model_name),
    )
    return engine


class TestVLLMCallOverrides:
    """The vLLM engine builds SamplingParams from the call's keywords."""

    def test_config_seed_reaches_sampling_params(self, vllm_engine):
        vllm_engine.generate("hello", GenerationConfig(temperature=0.8, seed=1234))
        assert vllm_engine.llm.calls[0]["sampling_params"].seed == 1234

    def test_per_call_seed_reaches_sampling_params(self, vllm_engine):
        vllm_engine.generate("hello", seed=1234, temperature=0.8)
        params = vllm_engine.llm.calls[0]["sampling_params"]
        assert params.seed == 1234
        assert params.temperature == 0.8

    def test_per_call_seed_supersedes_the_config_seed(self, vllm_engine):
        vllm_engine.generate("hello", GenerationConfig(temperature=0.8, seed=1), seed=1234)
        assert vllm_engine.llm.calls[0]["sampling_params"].seed == 1234

    def test_no_sampling_keyword_is_forwarded_to_the_provider_call(self, vllm_engine):
        # A leaked keyword would raise TypeError from FakeLLM.generate.
        vllm_engine.generate(
            "hello",
            seed=1234,
            temperature=0.8,
            top_p=0.5,
            top_k=10,
            max_tokens=16,
            stop=["\n"],
            presence_penalty=0.1,
            frequency_penalty=0.2,
            repetition_penalty=1.1,
        )
        params = vllm_engine.llm.calls[0]["sampling_params"]
        assert params.requested["max_tokens"] == 16
        assert params.requested["stop"] == ["\n"]

    def test_streaming_honors_a_per_call_seed(self, vllm_engine):
        list(vllm_engine.generate_stream("hello", seed=4242, temperature=0.8))
        assert vllm_engine.llm.calls[0]["sampling_params"].seed == 4242

    def test_batch_honors_a_per_call_seed(self, vllm_engine):
        vllm_engine.generate_batch(["hello", "there"], seed=77, temperature=0.8)
        assert vllm_engine.llm.calls[0]["sampling_params"].seed == 77

    def test_engine_level_kwargs_still_reach_the_provider_call(self, vllm_engine):
        vllm_engine.generate("hello", seed=1, lora_request="adapter-a", priority=3)
        call = vllm_engine.llm.calls[0]
        assert call["lora_request"] == "adapter-a"
        assert call["priority"] == 3
        assert call["use_tqdm"] is False

    def test_per_call_use_tqdm_replaces_the_engine_default(self, vllm_engine):
        vllm_engine.generate("hello", use_tqdm=True)
        assert vllm_engine.llm.calls[0]["use_tqdm"] is True

    def test_stop_sequences_is_accepted_under_its_own_name(self, vllm_engine):
        vllm_engine.generate("hello", stop_sequences=["END"])
        assert vllm_engine.llm.calls[0]["sampling_params"].requested["stop"] == ["END"]

    def test_a_per_call_temperature_of_zero_selects_greedy_decoding(self, vllm_engine):
        """The greedy normalization reads the resolved value, not the config's."""
        vllm_engine.generate("hello", GenerationConfig(temperature=0.9), temperature=0)
        requested = vllm_engine.llm.calls[0]["sampling_params"].requested
        assert (requested["temperature"], requested["top_p"], requested["top_k"]) == (
            0.0,
            1.0,
            -1,
        )

    def test_the_callers_config_object_is_not_modified(self, vllm_engine):
        config = GenerationConfig(temperature=0.2, seed=5)
        vllm_engine.generate("hello", config, seed=99, temperature=0.7)
        assert (config.seed, config.temperature) == (5, 0.2)


# ---------------------------------------------------------------------------
# MLX
# ---------------------------------------------------------------------------


class _FakeMLXTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0] * max(1, len(text.split()))

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return "formatted"


@pytest.fixture
def mlx_engine(monkeypatch):
    """A loaded ``MLXEngine`` wired to stand-in mlx-lm entry points.

    The stand-ins record the keywords the engine builds; ``mlx_generate`` accepts
    only the shape the engine is expected to produce.
    """
    calls: list[dict[str, Any]] = []

    def mlx_generate(model, tokenizer, prompt=None, verbose=False, **kw):
        calls.append(dict(kw))
        return "generated"

    def mlx_stream_generate(model, tokenizer, prompt=None, **kw):
        calls.append(dict(kw))
        yield types.SimpleNamespace(text="generated")

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(generate=mlx_generate, stream_generate=mlx_stream_generate),
    )
    from effgen.models.mlx_engine import MLXEngine

    engine = MLXEngine(
        model_name="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        apply_chat_template=False,
    )
    engine.model = object()
    engine.tokenizer = _FakeMLXTokenizer()
    engine._is_loaded = True
    engine._context_length = 4096
    engine.calls = calls  # type: ignore[attr-defined]
    return engine


class TestMLXCallOverrides:
    """The MLX engine builds its mlx-lm keywords from the call's keywords."""

    def test_config_seed_reaches_mlx(self, mlx_engine):
        mlx_engine.generate("hello", GenerationConfig(temperature=0.8, seed=1234))
        assert mlx_engine.calls[0]["seed"] == 1234

    def test_per_call_seed_reaches_mlx(self, mlx_engine):
        mlx_engine.generate("hello", seed=1234, temperature=0.8)
        assert mlx_engine.calls[0]["seed"] == 1234
        assert mlx_engine.calls[0]["temp"] == 0.8

    def test_per_call_seed_supersedes_the_config_seed(self, mlx_engine):
        mlx_engine.generate("hello", GenerationConfig(temperature=0.8, seed=1), seed=1234)
        assert mlx_engine.calls[0]["seed"] == 1234

    def test_no_seed_is_sent_when_none_was_asked_for(self, mlx_engine):
        mlx_engine.generate("hello", GenerationConfig(temperature=0.8))
        assert "seed" not in mlx_engine.calls[0]

    def test_streaming_honors_a_per_call_seed(self, mlx_engine):
        list(mlx_engine.generate_stream("hello", seed=4242, temperature=0.8))
        assert mlx_engine.calls[0]["seed"] == 4242

    def test_batch_honors_a_per_call_seed(self, mlx_engine):
        mlx_engine.generate_batch(["hello", "there"], seed=77, temperature=0.8)
        assert [call["seed"] for call in mlx_engine.calls] == [77, 77]

    def test_a_per_call_repetition_penalty_reaches_mlx(self, mlx_engine):
        mlx_engine.generate("hello", temperature=0.8, repetition_penalty=1.3)
        assert mlx_engine.calls[0]["repetition_penalty"] == 1.3

    def test_a_per_call_stop_string_cuts_at_the_whole_sequence(self, mlx_engine):
        """A bare-string ``stop`` is one sequence, not a set of characters.

        The stand-in returns ``"generated"``: cutting at the sequence ``"ed"``
        leaves ``"generat"``, while walking the string character by character
        would match the leading ``"e"`` and leave ``"g"``.
        """
        result = mlx_engine.generate("hello", temperature=0.8, stop="ed")
        assert result.text == "generat"


# ---------------------------------------------------------------------------
# MLX-VLM (the image path seeds the MLX RNG directly)
# ---------------------------------------------------------------------------


@pytest.fixture
def mlx_vlm_engine(monkeypatch):
    """A loaded ``MLXVLMEngine`` whose image path records the seed it applies."""
    seeds: list[int] = []
    calls: list[dict[str, Any]] = []

    def vlm_generate(model, processor, prompt, images, **kw):
        calls.append(dict(kw))
        return "a cat on a table"

    mlx_core = types.SimpleNamespace(random=types.SimpleNamespace(seed=seeds.append))
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=mlx_core))
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.SimpleNamespace(generate=vlm_generate))
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        types.SimpleNamespace(apply_chat_template=lambda *a, **k: "formatted"),
    )
    from effgen.models.mlx_vlm_engine import MLXVLMEngine

    engine = MLXVLMEngine(
        model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        apply_chat_template=False,
    )
    engine.model = object()
    engine.processor = object()
    engine.tokenizer = _FakeMLXTokenizer()
    engine.vlm_config = {}
    engine._is_loaded = True
    engine._context_length = 4096
    engine.seeds = seeds  # type: ignore[attr-defined]
    engine.calls = calls  # type: ignore[attr-defined]
    return engine


class TestMLXVLMCallOverrides:
    """The image path resolves the seed the same way the text path does."""

    def test_config_seed_seeds_the_backend(self, mlx_vlm_engine):
        mlx_vlm_engine.generate(
            "describe this", GenerationConfig(temperature=0.8, seed=1234), images=["a.png"]
        )
        assert mlx_vlm_engine.seeds == [1234]

    def test_per_call_seed_seeds_the_backend(self, mlx_vlm_engine):
        mlx_vlm_engine.generate("describe this", images=["a.png"], seed=1234, temperature=0.8)
        assert mlx_vlm_engine.seeds == [1234]
        assert mlx_vlm_engine.calls[0]["temp"] == 0.8

    def test_per_call_seed_supersedes_the_config_seed(self, mlx_vlm_engine):
        mlx_vlm_engine.generate(
            "describe this",
            GenerationConfig(temperature=0.8, seed=1),
            images=["a.png"],
            seed=1234,
        )
        assert mlx_vlm_engine.seeds == [1234]

    def test_streaming_honors_a_per_call_seed(self, mlx_vlm_engine):
        list(mlx_vlm_engine.generate_stream("describe this", images=["a.png"], seed=555))
        assert mlx_vlm_engine.seeds == [555]

    def test_batch_honors_a_per_call_seed(self, mlx_vlm_engine):
        mlx_vlm_engine.generate_batch(
            ["describe this", "and this"], images_list=[["a.png"], ["b.png"]], seed=606
        )
        assert mlx_vlm_engine.seeds == [606, 606]

    def test_the_text_only_path_carries_the_seed_through(self, mlx_vlm_engine, mlx_engine):
        """With no images the call delegates to the text engine, seed included."""
        mlx_vlm_engine.generate("no image here", seed=808, temperature=0.8)
        assert mlx_engine.calls[-1]["seed"] == 808


# ---------------------------------------------------------------------------
# A bare string never becomes a list of characters
# ---------------------------------------------------------------------------
#
# ``stop_sequences`` is typed as a list and every consumer iterates it, so a
# string given directly was walked character by character: the text was cut at
# the first single letter that matched, not at the sequence. The OpenAI API
# accepts a bare string, so it is the shape a caller naturally writes.


class TestStopSequenceNormalization:
    def test_a_string_is_one_sequence_not_three_characters(self):
        from effgen.models._adapter_utils import normalize_stop_sequences

        assert normalize_stop_sequences("END") == ["END"]
        assert normalize_stop_sequences(["END"]) == ["END"]
        assert normalize_stop_sequences(None) is None
        assert normalize_stop_sequences(("A", "B")) == ["A", "B"]

    def test_a_value_that_cannot_be_a_sequence_is_refused_by_name(self):
        from effgen.models._adapter_utils import normalize_stop_sequences

        with pytest.raises(TypeError, match="string or a list of strings"):
            normalize_stop_sequences(7)
        with pytest.raises(TypeError, match="only strings"):
            normalize_stop_sequences(["ok", 7])

    def test_apply_stop_sequences_cuts_at_the_sequence_not_a_letter(self):
        """The text is chosen so the wrong and right answers differ.

        ``E`` occurs in ``DEEP`` before ``END`` occurs, so a character-wise walk
        cuts earlier and the two results cannot coincide.
        """
        from effgen.models._adapter_utils import apply_stop_sequences

        text = "The ocean is DEEP and ENDless."
        assert apply_stop_sequences(text, ["END"]) == "The ocean is DEEP and "
        assert apply_stop_sequences(text, "END") == "The ocean is DEEP and "

    def test_a_config_built_by_hand_carries_a_list(self):
        assert GenerationConfig(stop_sequences="END").stop_sequences == ["END"]
        assert GenerationConfig(stop_sequences=["END"]).stop_sequences == ["END"]
        assert GenerationConfig().stop_sequences is None

    def test_the_agent_boundary_does_not_split_a_string(self):
        """``agent.run(task, stop_sequences="END")`` reaches the model whole.

        The generation path builds its local-trim list from the caller's value
        directly, so this is the site the per-call helpers do not cover.
        """
        from effgen.core.agent import Agent, AgentConfig
        from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount

        seen: list = []

        class _Recorder(BaseModel):
            def __init__(self):
                super().__init__(model_name="recorder", model_type=ModelType.TRANSFORMERS)
                self._is_loaded = True

            def load(self):
                self._is_loaded = True

            def unload(self):
                self._is_loaded = False

            def generate(self, prompt, config=None, **kwargs):
                seen.append(config.stop_sequences if config else None)
                return GenerationResult(
                    text="The ocean is DEEP and ENDless.",
                    tokens_used=6,
                    finish_reason="stop",
                    model_name="recorder",
                    metadata={},
                )

            def generate_stream(self, prompt, config=None, **kwargs):
                yield "x"

            def count_tokens(self, text):
                return TokenCount(prompt_tokens=1, completion_tokens=0, total_tokens=1)

            def get_context_length(self):
                return 4096

        model = _Recorder()
        with Agent(
            AgentConfig(
                name="s", model=model, tools=[], raise_on_error=False,
                enable_sub_agents=False, enable_memory=False,
            )
        ) as agent:
            response = agent.run("say something", stop_sequences="END")

        assert seen and seen[0] == ["END"], seen
        assert response.output.startswith("The ocean is DEEP and ")


class TestTheTransformersEngineTakesPerCallOverrides:
    """A per-call sampling keyword must reach the local engine.

    The vLLM, MLX and Gemini paths fold per-call keywords into the config with
    ``merge_call_overrides``; the Transformers engine did not. It read the stop
    list off the config *before* looking at the keywords, so
    ``generate(prompt, stop_sequences="END")`` — the spelling the cloud adapters
    accept — was dropped without a word and the text came back uncut.
    """

    @staticmethod
    def _fold(config, kwargs):
        """The exact call both the generate and the stream path make."""
        from effgen.models._adapter_utils import apply_call_overrides

        return apply_call_overrides(config, kwargs)

    def test_a_bare_string_becomes_one_sequence(self):
        kwargs = {"stop_sequences": "END"}
        config = self._fold(None, kwargs)
        assert config.stop_sequences == ["END"]
        # Consumed, so it is not also forwarded to HuggingFace's generate().
        assert "stop_sequences" not in kwargs

    def test_a_list_is_taken_as_given(self):
        config = self._fold(None, {"stop_sequences": ["END", "STOP"]})
        assert config.stop_sequences == ["END", "STOP"]

    def test_a_per_call_value_supersedes_the_config(self):
        from effgen.models.base import GenerationConfig

        original = GenerationConfig(stop_sequences=["FROM_CONFIG"])
        config = self._fold(original, {"stop_sequences": "PER_CALL"})
        assert config.stop_sequences == ["PER_CALL"]
        # The caller's object is never mutated.
        assert original.stop_sequences == ["FROM_CONFIG"]

    def test_the_config_is_used_when_no_keyword_is_given(self):
        from effgen.models.base import GenerationConfig

        original = GenerationConfig(stop_sequences=["FROM_CONFIG"])
        config = self._fold(original, {})
        assert config.stop_sequences == ["FROM_CONFIG"]

    def test_other_sampling_keywords_come_through_too(self):
        kwargs = {"temperature": 0.0, "seed": 7, "top_p": 0.5}
        config = self._fold(None, kwargs)
        assert config.temperature == 0.0
        assert config.seed == 7
        assert config.top_p == 0.5

    def test_an_unrecognised_keyword_is_left_for_the_backend(self):
        kwargs = {"stop_sequences": "END", "some_hf_only_flag": True}
        self._fold(None, kwargs)
        assert kwargs == {"some_hf_only_flag": True}
