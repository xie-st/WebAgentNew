"""The head-to-head report: several models answering one prompt.

Shows the prompt, the fastest and cheapest contender, the judged pick when a
judge ran, the tally of first-token and end-to-end latency, tokens and cost,
and what every contender answered.
"""

from __future__ import annotations

from typing import Any

from .report_html_components import _badge, _bar_chart, _cards, _table
from .report_html_format import _DASH, ReportError, _esc, _int, _mapping, _secs, _sequence, _usd


def _battle_body(data: dict[str, Any]) -> tuple[str, str, str]:
    """Render a head-to-head: the prompt, the tally, the verdict, the answers."""
    contenders = _sequence(data.get("contenders"))
    if not contenders:
        raise ReportError(
            "The battle result carries no contenders to render. Pass the JSON "
            "`effgen battle --json` emitted, with its contenders intact."
        )
    prompt = str(data.get("prompt") or "")
    verdict = _mapping(data.get("verdict"))
    finishers = [c for c in contenders if _mapping(c).get("answer") and not _mapping(c).get("error")]

    parts: list[str] = []

    for key, label in (("fastest", "Fastest"), ("cheapest", "Cheapest")):
        entry = _mapping(verdict.get(key))
        if entry.get("model"):
            parts.append(
                '<div class="verdict">'
                f'<div class="verdict-label">{label}</div>'
                f'<div class="verdict-value">{_esc(entry.get("model"))}</div>'
                f'<p class="verdict-note">{_esc(entry.get("detail", ""))}</p></div>'
            )

    judge = _mapping(verdict.get("judge"))
    if judge.get("winner"):
        parts.append(
            '<div class="verdict">'
            '<div class="verdict-label">Judged pick</div>'
            f'<div class="verdict-value">{_esc(judge.get("winner"))}</div>'
            f'<p class="verdict-note">{_esc(judge.get("reasoning", ""))} '
            f'Judged by {_esc(judge.get("judge_model"))}, separately from the '
            "measurements above.</p></div>"
        )

    parts.append('<h2>Prompt</h2><div class="answer">' + _esc(prompt) + "</div>")

    total_cost = data.get("total_cost_usd")
    if not finishers:
        # Nothing ran, so there is no price to report — not a set of models
        # that publish no price.
        spent = _DASH
    else:
        spent = _usd(total_cost) if total_cost is not None else "unpriced"
    parts.append(_cards([
        ("Contenders", str(len(contenders)), f"{len(finishers)} answered"),
        ("Wall clock", _secs(data.get("wall_s"), 2), "all models in parallel"),
        ("Total cost", spent, ""),
    ]))

    parts.append("<h2>Tally</h2>")
    rows = []
    estimated_any = False
    for raw in contenders:
        c = _mapping(raw)
        if c.get("error"):
            status = _badge("failed", "err")
            cells = [_DASH, _DASH, _DASH, _DASH]
        else:
            status = _badge("answered", "ok")
            cost = c.get("cost_usd")
            tokens = c.get("total_tokens")
            if c.get("estimated_tokens"):
                estimated_any = True
            cells = [
                _secs(c.get("ttft_s"), 2),
                _secs(c.get("latency_s"), 2),
                (_int(tokens) + ("*" if c.get("estimated_tokens") else "")
                 if tokens is not None else _DASH),
                _usd(cost) if cost is not None else "unpriced",
            ]
        rows.append([
            f'<span class="wrap">{_esc(c.get("model"))}</span>',
            status,
            *[f'<span class="num">{cell}</span>' for cell in cells],
        ])
    parts.append(_table(
        ["Model", "Result", "First token", "Latency", "Tokens", "Cost"],
        rows,
        caption=("* counted locally, not reported by the provider" if estimated_any else ""),
    ))

    timed = [
        (str(_mapping(c).get("model")), _mapping(c).get("latency_s"),
         _secs(_mapping(c).get("latency_s"), 2))
        for c in finishers
        if _mapping(c).get("latency_s") is not None
    ]
    if timed:
        parts.append("<h2>Charts</h2>")
        parts.append(_bar_chart("Latency", timed, role="accent2"))
        priced = [
            (str(_mapping(c).get("model")), _mapping(c).get("cost_usd"),
             _usd(_mapping(c).get("cost_usd"))
             if _mapping(c).get("cost_usd") is not None else "unpriced")
            for c in finishers
        ]
        parts.append(_bar_chart(
            "Cost for this run (unpriced models shown as a hatched bar)",
            priced,
            role="accent",
        ))

    parts.append("<h2>Answers</h2>")
    for raw in contenders:
        c = _mapping(raw)
        parts.append(f'<h3 class="wrap">{_esc(c.get("model"))}</h3>')
        if c.get("error"):
            parts.append(f'<div class="answer err">{_esc(c.get("error"))}</div>')
        else:
            parts.append(f'<div class="answer">{_esc(c.get("answer") or "")}</div>')

    subtitle = (
        f"{len(contenders)} models on one prompt · "
        f"{len(finishers)} answered in {_secs(data.get('wall_s'), 2)}"
    )
    return "Model Battle", subtitle, "".join(parts)
