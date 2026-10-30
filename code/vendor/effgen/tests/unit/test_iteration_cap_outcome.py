"""A run stopped at its iteration cap reports the stop, not a retrieved passage.

The tool loop can run out of iterations while the model is still calling tools.
What the scratchpad holds at that point is tool output — for a retrieval tool, a
block of source passages — which the model never wrote up as an answer. The run
reports the stop with what to do about it, and carries the recovered text beside
it as ``metadata["partial_output"]`` so nothing is discarded.

Uses an in-process scripted model (no network) to drive the loop
deterministically — not a mock of live API behavior, which the project forbids.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import ToolCategory
from effgen.tools.function_tool import Tool

PASSAGE = (
    "Cloudkite Support Policy. Customers who purchase a paid plan may request a "
    "full refund within 30 days of the charge. The support desk is staffed "
    "Monday through Friday, 9am to 6pm Pacific time."
)


def _lookup(query: str) -> str:
    """Return the passages matching a query."""
    # Each query returns a distinct block, so the loop's repeat guards do not
    # fire and the run reaches the iteration cap the way a real multi-step
    # retrieval run does.
    return f"Result for {query}. {PASSAGE}"


class _AlwaysRetrievesModel(BaseModel):
    """Calls the retrieval tool every turn and never writes a final answer."""

    def __init__(self):
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)
        self.calls = 0

    def load(self):  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        self.calls += 1
        text = (
            "Thought: I should look this up.\n"
            "Action: lookup\n"
            f'Action Input: {{"query": "refund policy {self.calls}"}}'
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


def _agent(max_iterations: int = 3):
    lookup = Tool.from_function(
        _lookup, name="lookup", category=ToolCategory.INFORMATION_RETRIEVAL,
    )
    return Agent(config=AgentConfig(
        raise_on_error=False,  # these assert the failure response, not the raise
        name="cap-test",
        model=_AlwaysRetrievesModel(),
        tools=[lookup],
        max_iterations=max_iterations,
        tool_calling_mode="react",
    ))


def _run(max_iterations: int = 3):
    return _agent(max_iterations).run(
        "What is the refund window and what are the support hours?"
    )


@pytest.fixture(scope="module")
def capped():
    return _run(max_iterations=3)


def test_retrieved_passage_is_not_the_answer(capped):
    assert capped.success is False
    assert "30 days of the charge" not in capped.output, (
        "the retrieved passage was handed back as the answer"
    )


def test_output_states_what_stopped_the_run_and_what_to_do(capped):
    out = capped.output
    assert "3 iterations" in out, out
    assert "without a final answer" in out, out
    assert "max_iterations" in out, out


def test_recovered_progress_is_kept_beside_the_outcome(capped):
    progress = capped.metadata.get("partial_output")
    assert progress, "the recovered text was discarded"
    assert "30 days of the charge" in progress
    assert capped.metadata.get("partial") is True
    assert capped.metadata.get("reason") == "max_iterations_partial"


def test_the_stop_is_a_typed_outcome(capped):
    error = capped.metadata.get("error")
    assert isinstance(error, dict), "the iteration cap reports no typed outcome"
    assert error["type"] == "MaxIterationsReached"
    assert error["category"] == "max_iterations"
    assert error["max_iterations"] == 3
    assert error["retryable"] is False
    assert error["message"] == capped.output


def test_a_cap_of_one_reads_in_the_singular():
    resp = _run(max_iterations=1)
    assert "1 iteration without" in resp.output, resp.output


def test_reconstructed_error_carries_the_message(capped):
    """``raise_on_error`` raises the typed stop, naming it and what to do."""
    from effgen.core.agent_generation import AgentGenerationMixin

    err = AgentGenerationMixin._reconstruct_error(capped.metadata)
    assert isinstance(err, RuntimeError)
    assert str(err).startswith("MaxIterationsReached: "), str(err)
    assert "max_iterations" in str(err)
    assert "30 days of the charge" not in str(err)


def test_cli_frames_the_stop_and_the_progress_separately(capped):
    import io

    # The assertions read what a Rich console wrote into a buffer; without Rich
    # the same text goes to stdout and tests/cli covers that path.
    pytest.importorskip("rich")

    from effgen.cli.commands.run import PARTIAL_PROGRESS_TITLE, present_response
    from effgen.ui.theme import get_console

    class _Cli:
        def __init__(self, console):
            self.console = console
            self.panels: list[str] = []

        def print_error_panel(self, text, title=""):  # pragma: no cover - not hit
            self.panels.append(f"ERROR:{title}")

    buf = io.StringIO()
    cli = _Cli(get_console(file=buf, force_terminal=False, width=100))
    present_response(cli, capped)
    out = buf.getvalue()
    assert not cli.panels, "a run stopped at the cap is not an error panel"
    assert "Stopped" in out
    assert PARTIAL_PROGRESS_TITLE.split(" (")[0] in out
    assert "30 days of the charge" in out


def test_run_card_reports_the_stop_and_the_progress_separately(capped):
    from effgen.ui.report_html import build_html_report

    document = {
        "task": "What is the refund window?",
        "output": capped.output,
        "success": capped.success,
        "model": "scripted-model",
        "iterations": capped.iterations,
        "tool_calls": capped.tool_calls,
        "metadata": {
            k: v for k, v in capped.metadata.items() if k != "debug_trace"
        },
    }
    html = build_html_report(document, kind="run")
    assert "stopped (max_iterations_partial)" in html
    assert "Partial progress" in html
    assert "30 days of the charge" in html
    assert ">failed<" not in html


def test_chat_turn_shows_the_stop_then_the_progress(capsys):
    """A piped chat turn writes the stop, then the progress under its label.

    The stop message says the progress is reported separately, so the surface
    that shows the message shows the progress too.
    """
    from types import SimpleNamespace

    from effgen.cli.chat import ChatREPL
    from effgen.cli.commands.run import PARTIAL_PROGRESS_TITLE

    class _Cli:
        console = None

        def __init__(self):
            self.warnings: list[str] = []

        def _animate(self, args):
            return False

        def print_warning(self, text):
            self.warnings.append(text)

    args = SimpleNamespace(
        model="scripted-model", provider=None, preset=None, tools=None,
        system_prompt=None, guardrails=None, temperature=None, max_tokens=None,
        no_sub_agents=True, verbose=False, quiet=True, session_id=None,
        no_animation=True,
    )
    repl = ChatREPL(_Cli(), args)
    repl.interactive = False
    repl.animate = False
    repl.agent = _agent(max_iterations=2)
    capsys.readouterr()

    answer = repl._run_with_tools("What is the refund window?")
    out = capsys.readouterr().out

    assert "without a final answer" in answer
    assert "30 days of the charge" not in answer, (
        "the retrieved passage was returned as the chat reply"
    )
    assert f"{PARTIAL_PROGRESS_TITLE}:" in out
    assert out.index("without a final answer") < out.index(PARTIAL_PROGRESS_TITLE)
    assert "30 days of the charge" in out


def test_run_history_keeps_the_stop_message_and_no_passage(tmp_path, monkeypatch):
    """The history file records the stop, not the passages it recovered."""
    import json

    from effgen.observability import run_log

    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(run_log, "_pruned", True)
    resp = _run(max_iterations=2)

    files = sorted((tmp_path / "runs").glob("*.jsonl"))
    assert files, "the run was not recorded in the history store"
    records = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    record = records[-1]
    # A run that was carried out but never reached an answer reads as its own
    # status, distinct from a run that could not be carried out at all.
    assert record["status"] == "stopped"
    assert record["stop_reason"] == "max_iterations_partial"
    assert record["output"] is None
    assert "30 days of the charge" not in (record["error"] or "")
    # The whole message is kept, not a mid-sentence slice of it.
    assert record["error"] == resp.output
