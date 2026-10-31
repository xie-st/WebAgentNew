"""Tests for effGen Jupyter magics.

Validates that:
1. The ``effgen.jupyter`` extension loads in an IPython shell.
2. ``%effgen_chat`` sends a message and displays a response (mocked model).
3. ``%%effgen_agent`` runs an agent on a cell body (mocked model).
4. ``%effgen_metrics`` renders a metrics HTML table.
5. Argument parsing works correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _configured_default_model(monkeypatch):
    """The magics no longer ship a hardcoded default model — they require one to
    be configured (so they never silently call a paid cloud model). Provide a
    default for the invocation tests below; the tests that specifically assert
    the no-model behaviour delete these vars again.
    """
    monkeypatch.setenv("EFFGEN_DEFAULT_MODEL", "gpt-5-nano")
    monkeypatch.delenv("EFFGEN_JUPYTER_MODEL", raising=False)

# ---------------------------------------------------------------------------
# Helpers — minimal IPython-shaped kernel mock
# ---------------------------------------------------------------------------


class FakeIPython:
    """Minimal stand-in for IPython.core.interactiveshell.InteractiveShell."""

    def __init__(self):
        self._magics: dict[str, Any] = {}

    def register_magics(self, magics_class: Any) -> None:
        self._magics[magics_class.__name__] = magics_class


# ---------------------------------------------------------------------------
# load_ipython_extension
# ---------------------------------------------------------------------------


class TestLoadExtension:
    def test_load_registers_magics(self):
        """load_ipython_extension registers EffgenMagics with the shell."""
        from effgen.jupyter import load_ipython_extension

        shell = FakeIPython()
        load_ipython_extension(shell)
        assert "EffgenMagics" in shell._magics

    def test_load_twice_is_safe(self):
        """Calling load_ipython_extension twice should not raise."""
        from effgen.jupyter import load_ipython_extension

        shell = FakeIPython()
        load_ipython_extension(shell)
        load_ipython_extension(shell)
        assert "EffgenMagics" in shell._magics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_default_model_from_env(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_JUPYTER_MODEL", "gpt-5-nano")
        from effgen.jupyter import magics  # reimport to hit monkeypatched env

        assert magics._default_model() == "gpt-5-nano"

    def test_default_model_falls_back_to_framework_default(self, monkeypatch):
        """With no Jupyter-specific override, EFFGEN_DEFAULT_MODEL is used."""
        monkeypatch.delenv("EFFGEN_JUPYTER_MODEL", raising=False)
        monkeypatch.setenv("EFFGEN_DEFAULT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        from effgen.jupyter import magics

        assert magics._default_model() == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_default_model_none_when_unset(self, monkeypatch):
        """No hardcoded paid default — returns None so callers can guide the user."""
        monkeypatch.delenv("EFFGEN_JUPYTER_MODEL", raising=False)
        monkeypatch.delenv("EFFGEN_DEFAULT_MODEL", raising=False)
        from effgen.jupyter import magics

        assert magics._default_model() is None

    def test_server_url_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("EFFGEN_JUPYTER_SERVER_URL", raising=False)
        from effgen.jupyter import magics

        assert magics._server_url() is None

    def test_server_url_from_env(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_JUPYTER_SERVER_URL", "http://localhost:9090")
        from effgen.jupyter import magics

        assert magics._server_url() == "http://localhost:9090"

    def test_format_tool_trace_empty(self):
        from effgen.jupyter.magics import _format_tool_trace

        assert _format_tool_trace([]) == ""

    def test_format_tool_trace_nonempty(self):
        from effgen.jupyter.magics import _format_tool_trace

        trace = [{"tool": "calculator", "input": {"expr": "2+2"}, "output": "4"}]
        result = _format_tool_trace(trace)
        assert "calculator" in result
        assert "2+2" in result

    def test_trace_from_response_pairs_tool_events(self):
        """_trace_from_response should pair start+complete tool events."""
        from effgen.jupyter.magics import _trace_from_response

        class _Resp:
            execution_trace = [
                {"type": "task_start", "message": "go", "data": {}},
                {"type": "tool_call_start", "data": {"tool_name": "calculator", "tool_input": {"expr": "17*23"}}},
                {"type": "tool_call_complete", "data": {"tool_name": "calculator", "result": "391"}},
            ]

        trace = _trace_from_response(_Resp())
        assert len(trace) == 1
        assert trace[0]["tool"] == "calculator"
        assert trace[0]["input"] == {"expr": "17*23"}
        assert trace[0]["output"] == "391"

    def test_trace_from_response_empty(self):
        from effgen.jupyter.magics import _trace_from_response

        class _Resp:
            execution_trace = []

        assert _trace_from_response(_Resp()) == []

    def test_context_overflow_hint_detects_overflow(self):
        from effgen.jupyter.magics import _context_overflow_hint

        raw = (
            "Cerebras generation failed: Error code: 400 - "
            "{'message': 'Please reduce the length of the messages', "
            "'code': 'context_length_exceeded'}"
        )
        hint = _context_overflow_hint(raw, "general")
        assert hint is not None
        assert "context window" in hint
        assert "general" in hint

    def test_context_overflow_hint_passes_normal_output(self):
        from effgen.jupyter.magics import _context_overflow_hint

        assert _context_overflow_hint("391", "math") is None
        assert _context_overflow_hint("The answer is 42.", None) is None

    def test_resolve_tools_without_asyncio_run(self):
        """Tool resolution should work inside IPython's already-running loop."""
        from effgen.jupyter.magics import _resolve_tools

        tools = _resolve_tools(["calculator"])
        assert tools
        assert tools[0].name == "calculator"

    def test_prometheus_snapshot_returns_dict(self):
        from effgen.jupyter.magics import _prometheus_snapshot

        snap = _prometheus_snapshot()
        assert isinstance(snap, dict)

    def test_format_metrics_html_empty(self):
        from effgen.jupyter.magics import _format_metrics_html

        html = _format_metrics_html({})
        assert "No Prometheus metrics" in html

    def test_format_metrics_html_nonempty(self):
        from effgen.jupyter.magics import _format_metrics_html

        html = _format_metrics_html({"effgen_model_calls_total": 42.0})
        assert "effgen_model_calls_total" in html
        assert "42" in html


# ---------------------------------------------------------------------------
# %effgen_chat magic
# ---------------------------------------------------------------------------


class _IPythonShell:
    """Minimal IPython shell for running magics without a live kernel."""

    def __init__(self):
        self.user_ns: dict[str, Any] = {}
        self.display_calls: list[Any] = []

    def register_magics(self, magics_class: Any) -> None:
        pass


class TestEffgenChat:
    """Test %effgen_chat with a mocked in-process agent."""

    def _make_magics(self) -> Any:
        from effgen.jupyter.magics import EffgenMagics

        shell = _IPythonShell()
        return EffgenMagics(shell=shell)

    def test_empty_message_shows_usage(self):
        """Empty line should display usage hint without calling the model."""
        magic = self._make_magics()
        displayed = []
        with patch("effgen.jupyter.magics.display", side_effect=displayed.append):
            magic.effgen_chat("")
        assert displayed  # something was displayed
        # Should not have attempted a model call

    def test_chat_calls_in_process(self, monkeypatch):
        """Non-empty message calls _chat_in_process when no server URL is set."""
        monkeypatch.delenv("EFFGEN_JUPYTER_SERVER_URL", raising=False)

        mock_result = "Hello from mock!"
        displayed = []

        with (
            patch("effgen.jupyter.magics._chat_in_process", return_value=mock_result) as mock_fn,
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            magic = self._make_magics()
            magic.effgen_chat("hi there")

        mock_fn.assert_called_once()
        call_args = mock_fn.call_args
        assert call_args[0][0] == "hi there"
        assert displayed

    def test_chat_uses_server_when_url_set(self, monkeypatch):
        """When EFFGEN_JUPYTER_SERVER_URL is set, _chat_via_server is called."""
        monkeypatch.setenv("EFFGEN_JUPYTER_SERVER_URL", "http://localhost:9999")

        displayed = []

        with (
            patch("effgen.jupyter.magics._chat_via_server", return_value="Server reply") as mock_fn,
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            magic = self._make_magics()
            magic.effgen_chat("ping")

        mock_fn.assert_called_once()
        assert displayed

    def test_chat_model_override(self, monkeypatch):
        monkeypatch.delenv("EFFGEN_JUPYTER_SERVER_URL", raising=False)

        captured_model: list[str] = []

        def fake_chat(msg, model):
            captured_model.append(model)
            return f"response from {model}"

        displayed = []
        with (
            patch("effgen.jupyter.magics._chat_in_process", side_effect=fake_chat),
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            magic = self._make_magics()
            magic.effgen_chat("--model gpt-5-nano hello")

        assert captured_model == ["gpt-5-nano"]

    def test_chat_displays_markdown(self, monkeypatch):
        monkeypatch.delenv("EFFGEN_JUPYTER_SERVER_URL", raising=False)

        from IPython.display import Markdown

        displayed = []
        with (
            patch(
                "effgen.jupyter.magics._chat_in_process",
                return_value="Test answer",
            ),
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            magic = self._make_magics()
            magic.effgen_chat("--model gpt-5-nano test question")

        assert any(isinstance(d, Markdown) for d in displayed)
        assert any("Test answer" in d.data for d in displayed if isinstance(d, Markdown))

    def test_chat_no_model_shows_friendly_message(self, monkeypatch):
        """With no model and no env default, show an actionable message — never
        a stale hardcoded id, and never an actual model call."""
        monkeypatch.delenv("EFFGEN_JUPYTER_SERVER_URL", raising=False)
        monkeypatch.delenv("EFFGEN_JUPYTER_MODEL", raising=False)
        monkeypatch.delenv("EFFGEN_DEFAULT_MODEL", raising=False)

        from IPython.display import Markdown

        displayed = []
        with (
            patch("effgen.jupyter.magics._chat_in_process") as mock_fn,
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            magic = self._make_magics()
            magic.effgen_chat("hello with no model")

        mock_fn.assert_not_called()
        assert any(
            isinstance(d, Markdown) and "No model configured" in d.data for d in displayed
        )


# ---------------------------------------------------------------------------
# %%effgen_agent magic
# ---------------------------------------------------------------------------


class TestEffgenAgent:
    def _make_magics(self) -> Any:
        from effgen.jupyter.magics import EffgenMagics

        return EffgenMagics(shell=_IPythonShell())

    def test_empty_cell_shows_hint(self):
        magic = self._make_magics()
        displayed = []
        with patch("effgen.jupyter.magics.display", side_effect=displayed.append):
            magic.effgen_agent("", "")
        assert displayed

    def test_agent_calls_agent_run(self, monkeypatch):
        """%%effgen_agent should invoke Agent.run with the cell body."""
        from effgen.jupyter.magics import EffgenMagics

        mock_response = MagicMock()
        mock_response.output = "The answer is 42."
        mock_response.execution_trace = []

        displayed = []

        with (
            patch("effgen.core.agent.Agent") as MockAgent,
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            mock_agent_instance = MagicMock()
            mock_agent_instance.run.return_value = mock_response
            MockAgent.return_value = mock_agent_instance

            magic = EffgenMagics(shell=_IPythonShell())
            magic.effgen_agent("", "What is 6 * 7?")

        mock_agent_instance.run.assert_called_once_with("What is 6 * 7?")
        assert displayed

    def test_agent_with_preset_uses_create_agent(self, monkeypatch):
        """Providing a known preset should build the agent via create_agent
        (so it is wired with the preset's tools) and run the cell body."""
        mock_response = MagicMock()
        mock_response.output = "done"
        mock_response.execution_trace = []

        calls: list[str] = []

        def fake_run(task):
            calls.append(task)
            return mock_response

        with (
            patch("effgen.presets.create_agent") as mock_create,
            patch("effgen.jupyter.magics.display"),
        ):
            inst = MagicMock()
            inst.run = fake_run
            mock_create.return_value = inst

            magic = __import__(
                "effgen.jupyter.magics", fromlist=["EffgenMagics"]
            ).EffgenMagics(shell=_IPythonShell())
            magic.effgen_agent("coding", "Write a quicksort.")

        # create_agent was called with the preset name; the cell body was run.
        assert mock_create.call_args[0][0] == "coding"
        assert calls and "quicksort" in calls[0]

    def test_agent_error_is_displayed_not_raised(self, monkeypatch):
        """An exception in Agent.run should produce an error message, not raise."""
        displayed = []

        with (
            patch(
                "effgen.core.agent.Agent",
                side_effect=RuntimeError("model not found"),
            ),
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            from effgen.jupyter.magics import EffgenMagics

            magic = EffgenMagics(shell=_IPythonShell())
            magic.effgen_agent("", "some task")  # should not raise

        assert displayed
        from IPython.display import Markdown

        md_texts = [d.data for d in displayed if isinstance(d, Markdown)]
        assert any("error" in t.lower() for t in md_texts)


# ---------------------------------------------------------------------------
# %effgen_metrics magic
# ---------------------------------------------------------------------------


class TestEffgenMetrics:
    def test_metrics_displays_html(self):
        from IPython.display import HTML

        displayed = []

        with (
            patch(
                "effgen.jupyter.magics._prometheus_snapshot",
                return_value={"effgen_model_calls_total": 7.0},
            ),
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            from effgen.jupyter.magics import EffgenMagics

            magic = EffgenMagics(shell=_IPythonShell())
            magic.effgen_metrics("")

        assert any(isinstance(d, HTML) for d in displayed)
        html_obj = next(d for d in displayed if isinstance(d, HTML))
        assert "effgen_model_calls_total" in html_obj.data

    def test_metrics_handles_empty_registry(self):
        """Empty Prometheus registry should display a 'no metrics' message."""
        from IPython.display import HTML

        displayed = []

        with (
            patch("effgen.jupyter.magics._prometheus_snapshot", return_value={}),
            patch("effgen.jupyter.magics.display", side_effect=displayed.append),
        ):
            from effgen.jupyter.magics import EffgenMagics

            magic = EffgenMagics(shell=_IPythonShell())
            magic.effgen_metrics("")

        html_obj = next((d for d in displayed if isinstance(d, HTML)), None)
        assert html_obj is not None
        assert "No Prometheus" in html_obj.data
