"""Per-call generation kwargs on the OpenAI adapter are model-aware.

A bare ``generate(prompt, temperature=0)`` (or ``max_tokens=`` / ``top_p=`` /
``stop=``) must travel the same model-aware request-building path as a
``GenerationConfig`` so the adapter can gate parameters per model family.
Previously these per-call kwargs were merged raw into the API request, which
made the most natural call fail with a 400 "Unsupported parameter" on gpt-5 and
the reasoning models (they require ``max_completion_tokens`` and reject
``temperature``/``top_p``/``stop``). These tests pin the normalization offline.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from effgen.models.base import GenerationConfig
from effgen.models.openai_adapter import OpenAIAdapter, _fold_generation_kwargs


def _make_adapter(model_name: str) -> OpenAIAdapter:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        adapter = OpenAIAdapter(model_name=model_name)
    adapter._is_loaded = True  # skip real load()
    return adapter


class TestFoldGenerationKwargs:
    def test_pops_known_leaves_unknown(self):
        kwargs = {"max_tokens": 5, "temperature": 0, "foo": "bar"}
        cfg = _fold_generation_kwargs(GenerationConfig(), kwargs)
        # Recognized generation kwargs are removed; genuinely-unknown ones stay.
        assert kwargs == {"foo": "bar"}
        assert cfg.max_tokens == 5
        assert cfg.temperature == 0

    def test_no_known_kwargs_returns_same_config(self):
        cfg_in = GenerationConfig()
        kwargs = {"foo": "bar"}
        cfg_out = _fold_generation_kwargs(cfg_in, kwargs)
        assert cfg_out is cfg_in
        assert kwargs == {"foo": "bar"}

    def test_does_not_mutate_input_config(self):
        cfg_in = GenerationConfig(temperature=0.7)
        kwargs = {"temperature": 0}
        cfg_out = _fold_generation_kwargs(cfg_in, kwargs)
        assert cfg_in.temperature == 0.7  # original untouched
        assert cfg_out.temperature == 0

    def test_stop_alias_string_normalized_to_list(self):
        kwargs = {"stop": "END"}
        cfg = _fold_generation_kwargs(GenerationConfig(), kwargs)
        assert cfg.stop_sequences == ["END"]
        assert kwargs == {}

    def test_stop_sequences_alias(self):
        kwargs = {"stop_sequences": ["A", "B"]}
        cfg = _fold_generation_kwargs(GenerationConfig(), kwargs)
        assert cfg.stop_sequences == ["A", "B"]

    def test_all_generation_kwargs_recognized(self):
        kwargs = {
            "max_tokens": 10,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 5,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "repetition_penalty": 1.1,
            "seed": 7,
            "stop": ["X"],
        }
        cfg = _fold_generation_kwargs(GenerationConfig(), kwargs)
        assert kwargs == {}
        assert cfg.max_tokens == 10
        assert cfg.seed == 7
        assert cfg.presence_penalty == 0.2


class TestGpt5KwargsGated:
    """gpt-5 must never receive raw max_tokens / temperature / top_p / stop."""

    def test_max_tokens_becomes_max_completion_tokens(self):
        adapter = _make_adapter("gpt-5-nano")
        kwargs = {"max_tokens": 5}
        cfg = _fold_generation_kwargs(GenerationConfig(), kwargs)
        params = adapter._build_request_params(adapter._create_messages("hi"), cfg)
        assert params["max_completion_tokens"] == 5
        assert "max_tokens" not in params

    def test_sampling_and_stop_dropped_for_gpt5(self):
        adapter = _make_adapter("gpt-5-nano")
        kwargs = {"temperature": 0, "top_p": 0.5, "stop": ["\n"]}
        cfg = _fold_generation_kwargs(GenerationConfig(), kwargs)
        params = adapter._build_request_params(adapter._create_messages("hi"), cfg)
        for unsupported in ("temperature", "top_p", "stop"):
            assert unsupported not in params

    def test_unknown_kwarg_not_folded(self):
        # A genuinely-unknown kwarg stays in kwargs to be forwarded raw.
        kwargs = {"logit_bias": {"1": 1}}
        _fold_generation_kwargs(GenerationConfig(), kwargs)
        assert kwargs == {"logit_bias": {"1": 1}}


class TestChatModelKwargsForwarded:
    """A regular chat model still receives the sampling params it supports."""

    def test_temperature_and_stop_built_for_chat_model(self):
        adapter = _make_adapter("gpt-4o-mini")
        kwargs = {"temperature": 0.0, "max_tokens": 12, "stop": ["END"]}
        cfg = _fold_generation_kwargs(GenerationConfig(), kwargs)
        params = adapter._build_request_params(adapter._create_messages("hi"), cfg)
        assert params["temperature"] == 0.0
        assert params["stop"] == ["END"]
        assert params["max_completion_tokens"] == 12
        assert "max_tokens" not in params


class TestNativeToolsKwargsGated:
    """The Responses-API native-tools path must also fold per-call kwargs.

    It uses a different parameter vocabulary (``max_output_tokens`` not
    ``max_tokens``) and 400s on a raw ``max_tokens``/``temperature`` for gpt-5,
    so the folding must happen before its manual param build too.
    """

    def _capture_params(self, model_name: str, kwargs: dict):
        adapter = _make_adapter(model_name)
        captured: dict = {}

        def fake_create(**params):
            captured.update(params)
            return SimpleNamespace(output=[], usage=None, status="completed", id="resp_test")

        adapter.client = MagicMock()
        adapter.client.responses.create.side_effect = fake_create
        adapter.generate_with_native_tools(
            "hi", [{"type": "web_search_preview"}], **kwargs
        )
        return captured

    def test_gpt5_no_raw_max_tokens_or_temperature(self):
        params = self._capture_params("gpt-5-nano", {"max_tokens": 5, "temperature": 0, "top_p": 0.5})
        # max_tokens is consumed into max_output_tokens; sampling params dropped.
        assert params["max_output_tokens"] == 5
        assert "max_tokens" not in params
        assert "temperature" not in params
        assert "top_p" not in params

    def test_unknown_kwarg_still_forwarded_raw(self):
        params = self._capture_params("gpt-5-nano", {"metadata": {"k": "v"}})
        assert params["metadata"] == {"k": "v"}
