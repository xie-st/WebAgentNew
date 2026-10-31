"""A custom ``system_prompt`` persona must steer every generation path.

A configured persona is the whole product for use cases like a Socratic tutor
or a fixed-language assistant. These tests pin that the persona actually reaches
the model on the paths that previously dropped it — the no-tool direct path, the
no-tool streaming path (what ``effgen chat`` uses by default), and the
native/hybrid tool path — while a plain default agent's prompt is unchanged.

They use an in-process recording model (no network) to inspect the exact prompt
handed to the model. That is deterministic plumbing, not a mock of live API
behaviour, which the project forbids.
"""

from __future__ import annotations

from collections.abc import Iterator

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationConfig, GenerationResult, ModelType, TokenCount

PERSONA = "You always respond ONLY in French. Never use English."
_DEFAULT = "You are a helpful AI assistant."


class _RecordingModel(BaseModel):
    """Captures every prompt it is asked to generate from."""

    def __init__(self, *, supports_tools: bool = False, answer: str = "Final Answer: bonjour"):
        super().__init__(model_name="record-model", model_type=ModelType.TRANSFORMERS)
        self._is_loaded = True
        self._supports_tools = supports_tools
        self._answer = answer
        self.prompts: list[str] = []

    def load(self) -> None:
        self._is_loaded = True

    def supports_tool_calling(self) -> bool:
        return self._supports_tools

    def generate(self, prompt, config: GenerationConfig | None = None, **kwargs) -> GenerationResult:
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        return GenerationResult(
            text=self._answer, tokens_used=2, finish_reason="stop", model_name=self.model_name,
        )

    def generate_stream(self, prompt, config: GenerationConfig | None = None, **kwargs) -> Iterator[str]:
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        for tok in self._answer.split():
            yield tok + " "

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def unload(self) -> None:
        self._is_loaded = False


def test_persona_reaches_direct_no_tool_path():
    model = _RecordingModel()
    agent = Agent(config=AgentConfig(name="t", model=model, system_prompt=PERSONA))
    agent.run("What is the capital of France?")
    assert model.prompts, "model was never called"
    assert PERSONA in model.prompts[0]


def test_persona_reaches_streaming_no_tool_path():
    model = _RecordingModel()
    agent = Agent(config=AgentConfig(name="t", model=model, system_prompt=PERSONA))
    list(agent.stream("What is the capital of France?"))
    assert model.prompts and PERSONA in model.prompts[0]


def test_persona_reaches_native_hybrid_tool_path():
    from effgen.tools.builtin.calculator import Calculator

    model = _RecordingModel(supports_tools=True)
    agent = Agent(config=AgentConfig(
        name="t", model=model, system_prompt=PERSONA,
        tools=[Calculator()], tool_calling_mode="hybrid",
    ))
    assert agent._tool_calling_strategy.name in ("native", "hybrid")
    agent.run("What is the capital of France?")
    assert model.prompts and PERSONA in model.prompts[0]


def test_default_persona_not_prepended_to_direct_prompt():
    """A plain default agent's prompt is byte-for-byte unchanged (no persona prefix)."""
    model = _RecordingModel()
    agent = Agent(config=AgentConfig(name="t", model=model))  # default system_prompt
    agent.run("What is the capital of France?")
    assert model.prompts
    assert not model.prompts[0].startswith(_DEFAULT)
    assert model.prompts[0].startswith("Answer this question directly")


def test_persona_owns_contract_no_competing_boilerplate():
    """A custom persona leads the prompt and the framework's own
    "answer directly / Answer:" framing is dropped, so the persona is not fought
    by a contradictory instruction (it literally says "Answer:", which defeats a
    Socratic "never give the answer" tutor and overrides the persona on the
    smallest models)."""
    model = _RecordingModel()
    agent = Agent(config=AgentConfig(name="t", model=model, system_prompt=PERSONA))
    agent.run("What is the capital of France?")
    prompt = model.prompts[0]
    assert prompt.startswith(PERSONA)
    assert "Answer this question directly and concisely" not in prompt
    assert not prompt.rstrip().endswith("Answer:")


def test_chat_repl_wires_system_prompt_into_config():
    """`effgen chat --system-prompt/--persona` actually builds an agent with it."""
    from types import SimpleNamespace

    from effgen.cli.chat import ChatREPL

    class _CLI:
        console = None
        tool_registry = None

        def _animate(self, args):
            return False

    args = SimpleNamespace(model="record", system_prompt=PERSONA, preset=None,
                           session_id=None, temperature=None, provider=None, no_sub_agents=True)
    repl = ChatREPL(_CLI(), args)
    assert repl.system_prompt == PERSONA
