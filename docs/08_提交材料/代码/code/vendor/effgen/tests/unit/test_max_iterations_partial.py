"""A run truncated at the iteration cap is not reported as a completed success.

When the ReAct loop reaches ``max_iterations`` before the model produces a final
answer, the outcome is ``success=False`` with ``partial=True`` metadata so a
caller does not treat a truncated multi-step run as finished. The CLI summary
renders this distinctly from both a success and an outright failure. What the
run had reached is kept in ``metadata["partial_output"]``; the shape of that
outcome is pinned by ``test_iteration_cap_outcome.py``.

Uses an in-process scripted model (no network) to drive the loop
deterministically — not a mock of live API behavior, which the project forbids.
"""

from __future__ import annotations

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_generation import AgentGenerationMixin
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.builtin.calculator import Calculator


class _NoFinalAnswerModel(BaseModel):
    """Emits a distinct calculator action every turn and never a final answer."""

    def __init__(self):
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)
        self.calls = 0

    def load(self):  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        self.calls += 1
        n = self.calls
        text = (
            "Thought: keep computing.\n"
            "Action: calculator\n"
            f'Action Input: {{"operation": "calculate", "expression": "{n} + {n}"}}'
        )
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self.generate(prompt).text

    def count_tokens(self, text):  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self):  # pragma: no cover
        return 4096

    def unload(self):  # pragma: no cover
        pass

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config) for p in prompts]

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):  # pragma: no cover
        return self.generate(prompt, config=config)

    def supports_function_calling(self):
        return False

    def supports_tool_calling(self):
        return False


def _make_agent(max_iterations: int = 3) -> Agent:
    cfg = AgentConfig(
        raise_on_error=False,  # these assert the failure response, not the raise
        name="maxiter-test",
        model=_NoFinalAnswerModel(),
        tools=[Calculator()],
        max_iterations=max_iterations,
        tool_calling_mode="react",
    )
    return Agent(config=cfg)


def test_max_iterations_partial_is_not_success():
    resp = _make_agent(max_iterations=3).run("Compute a running series step by step.")
    assert resp.success is False, "a run cut off at the iteration cap is not a success"
    assert resp.metadata.get("reason") == "max_iterations_partial"
    assert resp.metadata.get("partial") is True
    # The run says what stopped it, and the recovered progress is kept.
    assert resp.output and resp.output.strip()
    assert resp.metadata.get("partial_output")


def test_partial_run_summary_line_is_distinct():
    from effgen.cli.progress import summary_line

    resp = _make_agent(max_iterations=3).run("Compute a running series step by step.")
    plain, markup = summary_line(resp)
    assert "partial result" in plain.lower()
    assert "Failed" not in plain and "Done" not in plain


def test_reconstructed_error_points_at_the_iteration_cap():
    err = AgentGenerationMixin._reconstruct_error(
        {"reason": "max_iterations_partial", "partial": True}
    )
    assert isinstance(err, RuntimeError)
    assert "max_iterations" in str(err)
