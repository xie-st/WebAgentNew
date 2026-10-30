"""Rich output & result presentation.

Covers ``AgentResponse`` presentation (``__str__``/``__repr__``/``_repr_html_``/
``show()``/``trace()``), the shared Rich theme + console factory (``NO_COLOR``
honored), and the dependency-free markdown→HTML converter for the notebook card.
"""

from __future__ import annotations

import io

import pytest

# Every case below asserts what Rich renders, so an install without it has
# nothing here to check — the plain-text presentation is covered in tests/cli.
pytest.importorskip("rich")

from effgen.core.agent import AgentResponse  # noqa: E402
from effgen.ui import get_console
from effgen.ui.render import _markdown_to_html
from effgen.ui.theme import EFFGEN_THEME, color_enabled


def _sample(success: bool = True, output: str = "Paris") -> AgentResponse:
    return AgentResponse(
        output=output,
        success=success,
        iterations=2,
        tool_calls=1,
        tokens_used=1204,
        execution_time=3.2,
        execution_trace=[
            {"type": "task_start", "message": "q", "data": {}},
            {
                "type": "tool_call_start",
                "message": "calling",
                "data": {"tool_name": "calculator", "arguments": "2+2"},
            },
            {"type": "tool_call_complete", "message": "ok", "data": {"result": "4"}},
        ],
        metadata={"reason": "final_answer", "cost_usd": 0.0006},
    )


# --- __str__ → answer, detailed-but-compact __repr__ ------------------


def test_str_is_the_answer():
    r = _sample(output="The answer is 42.")
    assert str(r) == "The answer is 42."


def test_repr_is_compact_and_detailed():
    r = _sample()
    rep = repr(r)
    # Detailed: carries the key structured fields.
    assert "AgentResponse(" in rep
    assert "success=True" in rep
    assert "tokens_used=1204" in rep
    assert "reason='final_answer'" in rep
    # Compact: summarizes the trace rather than dumping it.
    assert "trace_steps=3" in rep
    assert "tool_call_start" not in rep  # the full trace is NOT inlined


def test_repr_truncates_long_output():
    r = _sample(output="x" * 500)
    assert "…" in repr(r)
    assert len(repr(r)) < 400


# --- Jupyter _repr_html_ card ----------------------------------------


def test_repr_html_has_answer_metrics_and_trace():
    r = _sample(output="The capital is **Paris**.")
    html = r._repr_html_()
    assert "<div" in html and "</div>" in html
    assert "Paris" in html
    assert "<strong>Paris</strong>" in html  # markdown bold rendered
    assert "1,204 tokens" in html
    assert "$0.0006" in html
    assert "calculator" in html  # tool surfaced
    assert "Step trace" in html  # collapsible trace present


def test_repr_html_escapes_model_output():
    r = _sample(output="<script>alert('x')</script> & co")
    html = r._repr_html_()
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_repr_html_failed_run():
    r = _sample(success=False, output="boom")
    html = r._repr_html_()
    assert "failed" in html.lower()


def test_repr_html_is_theme_aware():
    # The card must follow the notebook's own light/dark theme (a
    # `prefers-color-scheme: dark` variant) rather than hardcoding
    # light-theme colors that are illegible in a dark JupyterLab/VS Code
    # theme, and it must not clobber a previous card's palette when several
    # cards render in the same notebook — the CSS is scoped to a class, not
    # inlined as fixed colors.
    r = _sample(output="hi")
    html = r._repr_html_()
    assert "prefers-color-scheme: dark" in html
    assert 'class="effgen-card"' in html
    assert "color:#1a1a1a" not in html
    assert "color:#666" not in html
    assert "color:#888" not in html


# --- show() / trace() -------------------------------------------------


def test_show_prints_answer_and_footer():
    r = _sample(output="Paris")
    buf = io.StringIO()
    console = get_console(file=buf, width=80)
    ret = r.show(console=console)
    out = buf.getvalue()
    assert ret is r  # chainable
    assert "Paris" in out
    assert "1,204 tokens" in out
    assert "calculator" in out


def test_trace_renders_steps():
    r = _sample()
    buf = io.StringIO()
    console = get_console(file=buf, width=100)
    r.trace(console=console)
    out = buf.getvalue()
    assert "Reasoning trace" in out
    assert "tool_call_start" in out
    assert "calculator" in out


def test_trace_with_no_steps_reports_no_steps():
    r = AgentResponse(output="hi", success=True, execution_trace=[])
    buf = io.StringIO()
    console = get_console(file=buf, width=80)
    r.trace(console=console)
    out = buf.getvalue()
    assert "No step-by-step trace" in out
    assert "hi" in out


# --- markdown rendering + theme + NO_COLOR ----------------------


def test_theme_exists_with_semantic_keys():
    assert EFFGEN_THEME is not None
    styles = EFFGEN_THEME.styles
    for key in ("effgen.success", "effgen.error", "effgen.warning", "effgen.tool"):
        assert key in styles


def test_no_color_disables_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert color_enabled() is False
    buf = io.StringIO()
    console = get_console(file=buf)
    assert console.no_color is True
    r = _sample(output="**bold**")
    r.show(console=console)
    # No ANSI escape sequences leak when NO_COLOR is set and output is piped.
    assert "\x1b" not in buf.getvalue()


def test_color_enabled_default(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color_enabled() is True


def test_markdown_to_html_code_block():
    html = _markdown_to_html("text\n\n```python\nprint('hi')\n```")
    assert "<pre" in html and "<code>" in html
    assert "print(&#x27;hi&#x27;)" in html or "print('hi')" in html


def test_markdown_to_html_headings_and_lists():
    html = _markdown_to_html("# Title\n\n- one\n- two\n\n1. first")
    assert "<h3" in html  # h1 mapped into card range
    assert "<ul" in html and "<li>one</li>" in html
    assert "<ol" in html


def test_markdown_to_html_inline_and_escape():
    html = _markdown_to_html("Use `x < y` and **bold** and <b>raw</b>")
    assert "<code>x &lt; y</code>" in html
    assert "<strong>bold</strong>" in html
    assert "&lt;b&gt;raw&lt;/b&gt;" in html  # raw html escaped, not honored


def test_markdown_to_html_empty():
    assert "no output" in _markdown_to_html("").lower()


def test_markdown_to_html_table():
    html = _markdown_to_html("| x | y |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |")
    assert "<table" in html and "<thead>" in html and "<tbody>" in html
    assert "<th" in html and "<td" in html
    # The raw pipe rows and the separator must NOT leak as text.
    assert "| x | y |" not in html
    assert "|---|" not in html


def test_markdown_to_html_table_cells_escaped():
    html = _markdown_to_html("| a | b |\n|---|---|\n| `c` | <x> |")
    assert "<code>c</code>" in html  # inline markdown inside cells
    assert "&lt;x&gt;" in html  # raw html in a cell is escaped, not honored


def test_whole_answer_wrapped_in_markdown_fence_is_unwrapped():
    # A common SLM quirk: the entire answer is one ```markdown … ``` fence.
    answer = "```markdown\n## Hi\n\n- a\n- b\n\n| x | y |\n|---|---|\n| 1 | 2 |\n```"
    html = _markdown_to_html(answer)
    assert "<table" in html and "<h4" in html and "<ul" in html
    assert "<pre" not in html  # not rendered as one literal code box


def test_genuine_code_block_is_not_unwrapped():
    # A real code answer must stay a highlighted code block, not be unwrapped.
    answer = "```python\nprint('hi')\n```"
    html = _markdown_to_html(answer)
    assert "<pre" in html and "<code>" in html


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
