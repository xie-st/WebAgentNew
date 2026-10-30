"""The ``effgen report`` command plus the shared result-artifact writers.

``_main`` parses arguments and dispatches; it re-exports these names. Holds the
extension-to-format mapping, the ``-o``/``--report`` artifact writers shared by
``eval``/``compare``/``cost``, and the ``report`` handler that renders a saved
result document to a self-contained HTML file.
"""

from __future__ import annotations

from pathlib import Path

from effgen.cli.commands._shared import _invoked_command


def _artifact_format(path: str) -> str:
    """Map an output path's extension to a format: ``html``, ``markdown``, or ``json``."""
    suffix = Path(path).suffix.lower()
    if suffix in (".html", ".htm"):
        return "html"
    if suffix in (".md", ".markdown"):
        return "markdown"
    return "json"


def _write_result_artifact(
    path: str,
    *,
    cli,
    data: dict,
    kind: str,
    json_text: str,
    markdown_text: str | None = None,
) -> None:
    """Write a result to *path* in the format its extension names.

    ``.html``/``.htm`` renders the self-contained report, ``.md``/``.markdown``
    writes Markdown when the result has a Markdown form, and any other
    extension writes JSON.
    """
    fmt = _artifact_format(path)
    if fmt == "html":
        from effgen.ui.report_html import write_html_report
        write_html_report(path, data, kind=kind, command=_invoked_command())
    elif fmt == "markdown" and markdown_text is not None:
        Path(path).write_text(markdown_text, encoding="utf-8")
    else:
        Path(path).write_text(json_text, encoding="utf-8")
    cli.print(f"\nResults written to {path}")


def _write_html_report_arg(args, *, cli, data: dict, kind: str) -> None:
    """Write the ``--report`` HTML file when the flag was given."""
    report_path = getattr(args, "report", None)
    if not report_path:
        return
    from effgen.ui.report_html import ReportError, write_html_report
    try:
        written = write_html_report(
            report_path, data, kind=kind, command=_invoked_command()
        )
    except ReportError as exc:
        cli.print_error(str(exc))
        return
    cli.print(f"\nHTML report written to {written}")


def _handle_report_command(args, cli) -> int:
    """Handle 'effgen report' — render a saved result JSON as an HTML report."""
    from effgen.ui.report_html import (
        ReportError,
        detect_report_kind,
        load_result_document,
        write_html_report,
    )

    try:
        data = load_result_document(args.result)
    except ReportError as exc:
        cli.print_error(str(exc))
        return 2

    kind = getattr(args, "kind", None) or detect_report_kind(data)
    output = getattr(args, "output", None) or str(Path(args.result).with_suffix(".html"))
    try:
        written = write_html_report(
            output, data, kind=kind, command=f"effgen report {args.result}"
        )
    except ReportError as exc:
        cli.print_error(str(exc))
        return 2
    cli.print(f"HTML report written to {written} ({kind})")
    return 0
