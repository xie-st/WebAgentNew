"""A reasoning model that emits no visible token says so.

Some providers stream a reasoning model's chain and its answer through one
token stream: ``message.content`` is empty while ``message.reasoning`` carries
the chain, and the caller is billed for output they cannot see. Two shapes
reach the agent loop:

* the whole output budget went to the chain — ``finish_reason='length'``;
* generation ended at a stop sequence matched against the chain, before the
  first visible token — ``finish_reason='stop'``.

Both are deterministic at the same settings, so neither may be retried three
times, and the failure names the model, the cap in force and the reasoning
budget spent. The chain itself is never returned as the answer.
"""

from __future__ import annotations

import logging

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models._adapter_utils import (
    annotate_reasoning_only,
    apply_stop_sequences,
    extract_reasoning_text,
    extract_reasoning_tokens,
    reasoning_only_message,
    warn_empty_stream,
    warn_reasoning_only_stream,
)
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount

REASONING_CHAIN = "Thinking Process:\n1. Analyze the request.\n2. Compute.\n"


# ---------------------------------------------------------------------------
# Provider-response shapes (golden fixtures of what each SDK actually returns)
# ---------------------------------------------------------------------------

class _Message:
    def __init__(self, content=None, reasoning=None, reasoning_content=None):
        self.content = content
        if reasoning is not None:
            self.reasoning = reasoning
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content


class _Details:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class _Usage:
    def __init__(self, reasoning_tokens=None, nested=None):
        if reasoning_tokens is not None:
            self.reasoning_tokens = reasoning_tokens
        if nested is not None:
            self.completion_tokens_details = _Details(nested)


def test_reasoning_text_is_read_from_either_field_name():
    """Together/Groq/Cerebras send ``reasoning``; some deployments send
    ``reasoning_content``."""
    assert extract_reasoning_text(_Message(reasoning=REASONING_CHAIN)) == REASONING_CHAIN
    assert extract_reasoning_text(
        _Message(reasoning_content=REASONING_CHAIN)
    ) == REASONING_CHAIN
    assert extract_reasoning_text(_Message(content="hi")) == ""
    assert extract_reasoning_text(_Message(reasoning="   ")) == ""


class _ResponsesUsage:
    """The Responses API nests the count under ``output_tokens_details``."""

    def __init__(self, reasoning_tokens):
        self.output_tokens_details = _Details(reasoning_tokens)


def test_reasoning_tokens_are_read_from_either_usage_shape():
    """OpenAI/Groq/Cerebras nest the count; a few report it top level."""
    assert extract_reasoning_tokens(_Usage(nested=189)) == 189
    assert extract_reasoning_tokens(_Usage(reasoning_tokens=42)) == 42
    assert extract_reasoning_tokens(_ResponsesUsage(256)) == 256
    assert extract_reasoning_tokens(_Usage(nested=0)) == 0
    assert extract_reasoning_tokens(object()) == 0


# ---------------------------------------------------------------------------
# The adapter-side annotation
# ---------------------------------------------------------------------------

def _annotate(**overrides):
    kwargs = {
        "text": "",
        "reasoning_text": REASONING_CHAIN,
        "reasoning_tokens": 64,
        "model_name": "reasoner-1",
        "finish_reason": "length",
        "max_tokens": 64,
        "completion_tokens": 64,
    }
    kwargs.update(overrides)
    metadata: dict = {}
    flagged = annotate_reasoning_only(metadata, **kwargs)
    return flagged, metadata


def test_empty_content_beside_a_chain_is_flagged_and_explained():
    flagged, metadata = _annotate()
    assert flagged is True
    assert metadata["reasoning_only"] is True
    assert metadata["reasoning"] == REASONING_CHAIN
    assert metadata["reasoning_tokens"] == 64
    assert metadata["reasoning_chars"] == len(REASONING_CHAIN)
    reason = metadata["empty_response_reason"]
    assert "reasoner-1" in reason
    assert "64" in reason  # the cap and the reasoning budget both appear
    assert "no visible text" in reason


def test_a_reasoning_token_count_alone_is_enough():
    """OpenAI reports the reasoning budget but never the chain text."""
    flagged, metadata = _annotate(reasoning_text="", reasoning_tokens=24)
    assert flagged is True
    assert metadata["reasoning_chars"] == 0
    assert "reasoning" not in metadata  # no chain text to record
    assert "24 reasoning tokens" in metadata["empty_response_reason"]


def test_an_answer_is_never_reasoning_only():
    flagged, metadata = _annotate(text="132678")
    assert flagged is False
    assert "reasoning_only" not in metadata
    assert metadata["reasoning_tokens"] == 64  # still reported


def test_a_native_tool_call_is_a_complete_turn():
    """Empty text beside a tool call is the native-call shape, not a failure."""
    flagged, metadata = _annotate(tool_calls=[{"id": "1"}])
    assert flagged is False
    assert "reasoning_only" not in metadata


def test_a_streamed_tool_call_is_a_complete_turn(caplog):
    """The streamed twin exempts a tool turn the same way.

    A turn spent making a native call reaches the caller as an empty token
    stream beside a reasoning chain, which is the shape of a reasoning-only
    turn without being one. Warning about it tells the reader to raise a cap
    that is not the problem.
    """
    caplog.set_level(logging.WARNING)
    warn_reasoning_only_stream(
        model_name="reasoner-stream-1", yielded_text=False,
        reasoning_text=REASONING_CHAIN, reasoning_tokens=64,
        finish_reason="tool_calls", max_tokens=1024,
        tool_calls=[{"function": {"name": "calculator", "arguments": "{}"}}],
        logger=logging.getLogger("effgen.test.reasoning.stream"),
    )
    assert not [r for r in caplog.records if "no visible text" in r.message]


def test_a_streamed_turn_with_no_call_still_reports_why(caplog):
    """The exemption must not silence the signal it was built to carry."""
    caplog.set_level(logging.WARNING)
    warn_reasoning_only_stream(
        model_name="reasoner-stream-2", yielded_text=False,
        reasoning_text=REASONING_CHAIN, reasoning_tokens=64,
        finish_reason="length", max_tokens=64,
        logger=logging.getLogger("effgen.test.reasoning.stream"),
    )
    assert [r for r in caplog.records if "no visible text" in r.message]


def test_a_stream_that_sent_nothing_at_all_still_reports_why(caplog):
    """A provider can end the stream with no chunk, so no signal to key on.

    Gemini answers a request whose whole output budget went to thinking with an
    empty stream — no content, no usage block, no finish reason — which reaches
    the caller as an iterator that yielded nothing.
    """
    caplog.set_level(logging.WARNING)
    warn_empty_stream(
        model_name="reasoner-stream-3", yielded_text=False, max_tokens=32,
        logger=logging.getLogger("effgen.test.reasoning.stream"),
    )
    reports = [r for r in caplog.records if "streamed no tokens at all" in r.message]
    assert len(reports) == 1
    assert "max_tokens cap of 32" in reports[0].message


def test_an_empty_stream_report_fires_once_per_model(caplog):
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("effgen.test.reasoning.stream")
    for _ in range(3):
        warn_empty_stream(
            model_name="reasoner-stream-4", yielded_text=False, max_tokens=32,
            logger=logger,
        )
    assert sum("streamed no tokens at all" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize(
    ("yielded_text", "tool_calls"),
    [(True, None), (False, [{"function": {"name": "calculator", "arguments": "{}"}}])],
)
def test_a_stream_that_produced_something_is_not_reported(caplog, yielded_text, tool_calls):
    """A stream that yielded a token, or made a call, is a complete turn."""
    caplog.set_level(logging.WARNING)
    warn_empty_stream(
        model_name="reasoner-stream-5", yielded_text=yielded_text, max_tokens=32,
        tool_calls=tool_calls,
        logger=logging.getLogger("effgen.test.reasoning.stream"),
    )
    assert not [r for r in caplog.records if "streamed no tokens at all" in r.message]


def test_a_gemini_stream_with_no_chunks_reports_the_empty_turn(caplog):
    """The adapter wires the report to a stream the SDK ended with no chunk."""
    from unittest.mock import MagicMock

    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter

    caplog.set_level(logging.WARNING)
    adapter = GeminiAdapter(model_name="gemini-2.5-flash-empty-stream", api_key="test-key")
    adapter._is_loaded = True
    adapter._client = MagicMock()
    adapter._generate_with_retry = MagicMock(return_value=iter(()))

    chunks = list(adapter.generate_stream("Why is the sky blue?",
                                          config=GenerationConfig(max_tokens=32)))

    assert chunks == []
    reports = [r for r in caplog.records if "streamed no tokens at all" in r.message]
    assert len(reports) == 1
    assert "gemini-2.5-flash-empty-stream" in reports[0].message
    assert "max_tokens cap of 32" in reports[0].message


def test_no_reasoning_signal_is_not_flagged():
    flagged, metadata = _annotate(reasoning_text="", reasoning_tokens=0)
    assert flagged is False
    assert metadata == {}


def test_the_annotation_warns_once_per_model_and_finish_reason(caplog):
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("effgen.test.reasoning")
    for _ in range(3):
        annotate_reasoning_only(
            {}, text="", reasoning_text=REASONING_CHAIN, reasoning_tokens=64,
            model_name="reasoner-1", finish_reason="length", max_tokens=64,
            logger=logger,
        )
    assert sum("no visible text" in r.message for r in caplog.records) == 1


def test_the_stop_shape_names_the_stop_sequence_remedy():
    message = reasoning_only_message(
        "reasoner-1", finish_reason="stop", reasoning_tokens=90,
        reasoning_chars=367, max_tokens=4096,
    )
    assert "finish_reason='stop'" in message
    assert "4096" in message
    assert "90 reasoning tokens" in message
    assert "stop sequence" in message


# ---------------------------------------------------------------------------
# Applying stop sequences to the answer instead of sending them
# ---------------------------------------------------------------------------

def test_stop_sequences_cut_the_answer_at_the_earliest_match():
    text = "Action: calculator\nObservation: 42\nQuestion: next?"
    assert apply_stop_sequences(text, ["\nObservation:", "\nQuestion:"]) == (
        "Action: calculator"
    )
    assert apply_stop_sequences(text, []) == text
    assert apply_stop_sequences(text, ["\nNope:"]) == text
    assert apply_stop_sequences("", ["\nObservation:"]) == ""


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

class _ReasoningModel(BaseModel):
    """A model that answers only when its stop sequences are not sent upstream.

    Mirrors what Together's ``Qwen/Qwen3.5-9B`` does: the chain and the answer
    share one token stream, so a stop sequence matched against the chain ends
    generation before the first visible token.
    """

    def __init__(self, *, answer: str = "Final Answer: 132678",
                 recovers: bool = True, is_reasoning_model: bool = False,
                 finish_reason: str = "stop") -> None:
        super().__init__(model_name="reasoner-1", model_type=ModelType.OPENAI)
        self._answer = answer
        self._recovers = recovers
        self._finish_reason = finish_reason
        self._is_reasoning_model = is_reasoning_model
        self._provider = "fake"
        self.calls: list[list[str] | None] = []

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        stops = list(getattr(config, "stop_sequences", None) or []) or None
        self.calls.append(stops)
        if stops or not self._recovers:
            return GenerationResult(
                text="", tokens_used=90, finish_reason=self._finish_reason,
                model_name=self.model_name,
                metadata={
                    "reasoning_only": True,
                    "reasoning": REASONING_CHAIN,
                    "reasoning_chars": len(REASONING_CHAIN),
                    "reasoning_tokens": 90,
                    "empty_response_reason": (
                        f"Model '{self.model_name}' returned no visible text: it "
                        "produced 90 reasoning tokens and no answer."
                    ),
                },
            )
        return GenerationResult(
            text=self._answer + "\nObservation: hallucinated",
            tokens_used=30, finish_reason="stop", model_name=self.model_name,
            metadata={"reasoning_tokens": 40},
        )

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self._answer

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 4096

    def unload(self) -> None:  # pragma: no cover
        pass

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config) for p in prompts]


class _EmptyModel(_ReasoningModel):
    """Empty answers with no reasoning signal at all — the flaky-empty shape."""

    def generate(self, prompt, config=None, **kwargs):
        self.calls.append(list(getattr(config, "stop_sequences", None) or []) or None)
        return GenerationResult(
            text="", tokens_used=1, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )


def _agent(model):
    agent = Agent(config=AgentConfig(
        model="fake:reasoner-1", require_model=False, raise_on_error=False,
    ))
    agent.model = model
    agent._all_models = [model]
    return agent


def test_a_stop_collision_costs_one_extra_call_and_then_answers():
    """The collision is recovered by applying the stop sequences to the answer."""
    model = _ReasoningModel()
    result = _agent(model)._generate("Question: 234 * 567?")

    assert len(model.calls) == 2, model.calls
    assert model.calls[0] is not None  # first attempt sent them upstream
    assert model.calls[1] is None      # the retry held them back
    assert result["finish_reason"] != "error"
    # The answer is cut where the stop sequence would have cut it upstream.
    assert result["text"] == "Final Answer: 132678"


def test_the_recovery_is_learned_for_the_rest_of_the_process():
    """The wasted call happens at most once per model per process."""
    first = _ReasoningModel()
    _agent(first)._generate("Question: 234 * 567?")
    assert len(first.calls) == 2

    second = _ReasoningModel()
    _agent(second)._generate("Question: 234 * 567?")
    assert second.calls == [None], second.calls


def test_a_flagged_reasoning_model_never_pays_the_first_call():
    model = _ReasoningModel(is_reasoning_model=True)
    _agent(model)._generate("Question: 234 * 567?")
    assert model.calls == [None], model.calls


@pytest.mark.parametrize("finish_reason", ["stop", "unknown"])
def test_a_reasoning_only_turn_is_reported_not_retried_three_times(finish_reason):
    model = _ReasoningModel(recovers=False, finish_reason=finish_reason)
    result = _agent(model)._generate("Question: 234 * 567?")

    assert len(model.calls) == 2, model.calls  # never the 3-attempt empty loop
    assert result["finish_reason"] == "error"
    detail = result["metadata"]["error_detail"]
    assert detail["type"] == "ReasoningOnlyResponse"
    assert detail["category"] == "reasoning_only"
    assert detail["retryable"] is False
    assert detail["reasoning_tokens"] == 90
    assert "reasoner-1" in detail["message"]
    assert "90 reasoning tokens" in detail["message"]
    # The chain is reported as diagnosis, never as the answer.
    assert result["text"] == ""
    assert REASONING_CHAIN not in detail["message"]


def test_a_truncated_reasoning_turn_names_the_cap_and_the_budget():
    model = _ReasoningModel(recovers=False, is_reasoning_model=True,
                            finish_reason="length")
    result = _agent(model)._generate("Question: 234 * 567?", max_tokens=64)

    assert len(model.calls) == 1  # pinned budget: no escalation, no retry storm
    detail = result["metadata"]["error_detail"]
    assert detail["type"] == "TruncatedResponse"
    assert "64" in detail["message"]
    assert "90 reasoning tokens" in detail["message"]


def test_an_empty_answer_with_no_reasoning_still_retries():
    """Unchanged for every model that is not reasoning-only."""
    model = _EmptyModel()
    result = _agent(model)._generate("Question: 234 * 567?")

    assert len(model.calls) == 3
    assert result["metadata"]["error_detail"]["type"] == "EmptyResponse"


def test_the_deterministic_failure_is_logged_once(caplog):
    caplog.set_level(logging.WARNING, logger="effgen.core.agent_generation")
    model = _ReasoningModel(recovers=False)
    _agent(model)._generate("Question: 234 * 567?")
    failures = [r for r in caplog.records if r.message.startswith("Generation failed")]
    assert len(failures) == 1, [r.message for r in failures]


# ---------------------------------------------------------------------------
# The OpenAI Responses API: a status and a reason instead of a finish reason
# ---------------------------------------------------------------------------

class _Incomplete:
    def __init__(self, status="incomplete", reason=None):
        self.status = status
        self.incomplete_details = _Reason(reason) if reason else None


class _Reason:
    def __init__(self, reason):
        self.reason = reason


def test_a_responses_run_cut_off_at_the_cap_reads_as_truncation():
    """``incomplete`` alone would not trigger the budget escalation."""
    from effgen.models.openai_adapter import OpenAIAdapter

    finish = OpenAIAdapter._responses_finish_reason
    assert finish(_Incomplete(reason="max_output_tokens")) == "length"
    assert finish(_Incomplete(reason="content_filter")) == "content_filter"
    assert finish(_Incomplete(status="completed")) == "stop"


# ---------------------------------------------------------------------------
# Providers that report the chain under their own name
# ---------------------------------------------------------------------------

def test_a_gemini_turn_spent_on_thinking_is_reported():
    """Gemini reports the budget as ``thoughts_token_count``, not as tokens."""
    from unittest.mock import MagicMock

    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter(model_name="gemini-2.5-flash", api_key="test-key")
    adapter._is_loaded = True
    adapter._client = MagicMock()

    thought = MagicMock()
    thought.thought = True
    thought.text = REASONING_CHAIN
    thought.function_call = None
    thought.code_execution_result = None
    candidate = MagicMock()
    candidate.content.parts = [thought]
    candidate.finish_reason = "MAX_TOKENS"
    candidate.grounding_metadata = None
    response = MagicMock()
    response.candidates = [candidate]
    response.text = ""
    response.usage_metadata.prompt_token_count = 20
    response.usage_metadata.candidates_token_count = 0
    response.usage_metadata.total_token_count = 20
    response.usage_metadata.thoughts_token_count = 64
    adapter._generate_with_retry = MagicMock(return_value=response)

    result = adapter.generate("Question: 234 * 567?",
                              config=GenerationConfig(max_tokens=64))

    assert result.text == ""
    assert result.metadata["reasoning_only"] is True
    assert result.metadata["reasoning_tokens"] == 64
    assert "gemini-2.5-flash" in result.metadata["empty_response_reason"]
    assert "64" in result.metadata["empty_response_reason"]


def test_an_anthropic_turn_spent_on_thinking_is_reported():
    """Extended thinking is billed as output: an empty answer still cost."""
    from unittest.mock import MagicMock

    from effgen.models.anthropic_adapter import AnthropicAdapter
    from effgen.models.base import GenerationConfig

    adapter = AnthropicAdapter(model_name="claude-opus-4-7", api_key="test-key")
    adapter._is_loaded = True

    block = MagicMock()
    block.type = "thinking"
    block.thinking = REASONING_CHAIN
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "max_tokens"
    response.usage.input_tokens = 20
    response.usage.output_tokens = 96
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    adapter.client = MagicMock()
    adapter.client.messages.create.return_value = response
    adapter.client.messages.count_tokens.return_value.input_tokens = 8

    result = adapter.generate("Question: 234 * 567?",
                              config=GenerationConfig(max_tokens=96))

    assert result.text == ""
    assert result.metadata["reasoning_only"] is True
    assert result.metadata["reasoning_chars"] == len(REASONING_CHAIN)
    assert "96 reasoning tokens" in result.metadata["empty_response_reason"]


# ---------------------------------------------------------------------------
# The native-tool path, which does not go through the retry loop
# ---------------------------------------------------------------------------

_NATIVE_REASONING_ONLY = GenerationResult(
    text="", tokens_used=192, finish_reason="length", model_name="gpt-5-nano",
    metadata={
        "reasoning_only": True,
        "reasoning_tokens": 192,
        "native_tool_results": [],
        "empty_response_reason": (
            "Model 'gpt-5-nano' returned no visible text: it spent the whole "
            "output budget on internal reasoning and hit a max_tokens cap of 250."
        ),
    },
)


def _native_tool_agent(result: GenerationResult, **cfg):
    """An agent whose native-tool call is recorded instead of sent."""
    from effgen.models.openai_adapter import OpenAIAdapter
    from effgen.tools.builtin.openai_native import OpenAICodeInterpreterTool

    adapter = OpenAIAdapter(model_name="gpt-5-nano", api_key="test-key")
    adapter._is_loaded = True
    recorded: dict = {}

    def record(**kwargs):
        recorded.update(kwargs)
        return result

    adapter.generate_with_native_tools = record
    cfg.setdefault("raise_on_error", False)
    agent = Agent(config=AgentConfig(
        model=adapter, tools=[OpenAICodeInterpreterTool()], **cfg,
    ))
    return agent, recorded


def test_a_native_tool_turn_that_only_reasoned_names_the_cause():
    """That path reported only that there was no output, never why."""
    agent, _ = _native_tool_agent(_NATIVE_REASONING_ONLY)

    response = agent.run("Prove that sqrt(2) is irrational.")

    assert response.success is False
    assert "no output from native tools call" not in str(response)
    detail = response.metadata["error"]
    assert detail["type"] == "ReasoningOnlyResponse"
    assert detail["category"] == "reasoning_only"
    assert detail["retryable"] is False
    assert detail["reasoning_tokens"] == 192
    assert "max_tokens cap of 250" in detail["message"]
    agent.close()


def test_the_native_tool_call_carries_the_run_settings():
    """A reasoning model needs the budget the caller asked for on this path too."""
    answered = GenerationResult(
        text="It is irrational.", tokens_used=12, finish_reason="stop",
        model_name="gpt-5-nano", metadata={"native_tool_results": []},
    )
    agent, recorded = _native_tool_agent(answered, max_tokens=250, temperature=0.3)

    agent.run("Prove that sqrt(2) is irrational.")

    config = recorded["config"]
    assert config.max_tokens == 250
    assert config.temperature == 0.3
    agent.close()
