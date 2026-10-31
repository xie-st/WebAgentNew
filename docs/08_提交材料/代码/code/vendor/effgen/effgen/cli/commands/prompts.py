"""The ``effgen prompts`` command: browse, render, evaluate and run templates.

:mod:`effgen.cli._main` parses arguments and dispatches; it imports this at
module scope and re-exports it (``effgen.cli._handle_prompts_command`` resolves
through the chain). Holds the library listing in table/JSON/Markdown form, the
per-template detail view with its schema and rendered fixture, the golden and
live eval with its pass-rate exit gate, and the non-interactive ``render`` and
``run`` paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from effgen.ui.render import json_ensure_ascii

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def _handle_prompts_command(args, cli: "CLIInterface") -> int:
    """Handle 'effgen prompts' subcommands."""
    import json as _json

    from effgen.cli import _main

    try:
        from effgen.prompts.library import PromptEval, registry
    except ImportError as exc:
        cli.print_error(f"Prompt library not available: {exc}")
        return 1

    cmd = getattr(args, 'prompts_command', None)

    # ---- list ----
    if cmd == 'list' or cmd is None:
        domain_filter = getattr(args, 'domain', None)
        variant_filter = getattr(args, 'variant', None)
        fmt = getattr(args, 'list_format', 'table')

        prompts = registry.search(domain=domain_filter, variant=variant_filter)

        if fmt == 'json':
            rows = [
                {
                    'name': p.name,
                    'domain': p.domain,
                    'variant': p.variant,
                    'description': p.description,
                    'tags': p.tags,
                }
                for p in prompts
            ]
            print(_json.dumps(rows, indent=2, ensure_ascii=json_ensure_ascii()))
            return 0

        if fmt == 'markdown':
            print("| Name | Domain | Variant | Description |")
            print("|------|--------|---------|-------------|")
            for p in prompts:
                print(f"| {p.name} | {p.domain} | {p.variant} | {p.description} |")
            if not prompts:
                print("| — | — | — | No prompts registered yet |")
            return 0

        # table (default)
        if _main.RICH_AVAILABLE and cli.console:
            from rich.table import Table
            t = Table(title="Prompt Library", show_lines=False)
            # Names must never be clipped — they're the id a user types back
            # into `prompts show`/`run`/`render`. "fold" wraps onto extra
            # lines instead of the default ellipsis truncation.
            t.add_column("Name", style="effgen.model", overflow="fold")
            t.add_column("Domain")
            t.add_column("Variant")
            t.add_column("Description")
            for p in prompts:
                t.add_row(p.name, p.domain, p.variant, p.description)
            if not prompts:
                t.add_row("—", "—", "—", "No prompts registered yet")
            cli.console.print(t)
        else:
            if not prompts:
                print("No prompts registered yet.")
            else:
                print(f"{'Name':<45} {'Domain':<12} {'Variant':<12} Description")
                print("-" * 100)
                for p in prompts:
                    print(f"{p.name:<45} {p.domain:<12} {p.variant:<12} {p.description}")
        cli.print(f"\nTotal: {len(prompts)} prompt(s)")
        return 0

    # ---- show ----
    if cmd == 'show':
        name = args.name
        from effgen.cli.playground import _key_error_message, _resolve_prompt
        try:
            p, _resolved_name = _resolve_prompt(name)
        except KeyError as exc:
            cli.print_error(_key_error_message(exc, name))
            return 1

        cli.print_header(f"Prompt: {p.name}")
        cli.print(f"  Domain:      {p.domain}")
        cli.print(f"  Variant:     {p.variant}")
        cli.print(f"  Description: {p.description}")
        cli.print(f"  Tags:        {', '.join(p.tags) or '—'}")
        cli.print("\n[bold]Input Schema:[/bold]" if _main.RICH_AVAILABLE else "\nInput Schema:")
        cli.print_data("  " + _json.dumps(p.input_schema, indent=2).replace("\n", "\n  "))
        cli.print("\n[bold]Fixture:[/bold]" if _main.RICH_AVAILABLE else "\nFixture:")
        cli.print_data("  " + _json.dumps(p.fixture, indent=2).replace("\n", "\n  "))
        try:
            rendered = p.render_fixture()
            cli.print(
                "\n[bold]Rendered (fixture):[/bold]"
                if _main.RICH_AVAILABLE
                else "\nRendered (fixture):"
            )
            cli.print_data(rendered)
        except Exception as exc:
            cli.print_warning(f"Could not render: {exc}")
        return 0

    # ---- eval ----
    if cmd == 'eval':
        domain_filter = getattr(args, 'domain', None)
        live = getattr(args, 'live', False)
        model = getattr(args, 'model', None)
        delay = getattr(args, 'delay', 35.0)
        output_path = getattr(args, 'output', None)
        fail_under = getattr(args, 'fail_under', None)

        prompts = registry.search(domain=domain_filter)
        evaluator = PromptEval()

        cli.print_header("Running golden eval...")
        golden_report = evaluator.eval_all_golden(prompts)
        table = golden_report.as_table()
        print(table)

        full_table = "=== Golden Eval ===\n" + table

        total = len(golden_report.results)
        passed = len(golden_report.passed())

        if live:
            if not model:
                cli.print_error("--model is required for --live eval")
                return 1
            cli.print_header(f"Running live eval with model '{model}'...")
            live_report = evaluator.eval_all_live(prompts, model, delay=delay)
            live_table = live_report.as_table()
            print(live_table)
            full_table += "\n=== Live Eval ===\n" + live_table
            total += len(live_report.results)
            passed += len(live_report.passed())

        if output_path:
            Path(output_path).write_text(full_table)
            cli.print_success(f"Eval table written to {output_path}")

        # Reflect failures in the exit code so the harness can gate CI. With
        # --fail-under, compare the pass rate to the threshold; otherwise any
        # single failure exits non-zero.
        failed = total - passed
        if fail_under is not None:
            rate = (passed / total) if total else 1.0
            if rate < fail_under:
                cli.print_error(
                    f"Pass rate {rate:.1%} ({passed}/{total}) is below the "
                    f"--fail-under threshold of {fail_under:.1%}."
                )
                return 1
            return 0
        if failed:
            cli.print_error(f"{failed} of {total} eval(s) failed.")
            return 1
        return 0

    # ---- playground ----
    if cmd == 'playground':
        from effgen.cli.playground import PlaygroundREPL
        repl = PlaygroundREPL()
        return repl.run()

    # ---- render (non-interactive) ----
    if cmd == 'render':
        from effgen.cli.playground import cmd_render
        name = getattr(args, 'prompt_name', None)
        input_file = getattr(args, 'input_file', None)
        inputs: dict = {}
        if input_file:
            try:
                inputs = _json.loads(Path(input_file).read_text())
            except Exception as exc:
                cli.print_error(f"Could not read input file: {exc}")
                return 1
        return cmd_render(name, inputs)

    # ---- run (non-interactive) ----
    if cmd == 'run':
        from effgen.cli.playground import cmd_run
        name = getattr(args, 'prompt_name', None)
        input_file = getattr(args, 'input_file', None)
        model = getattr(args, 'model', None)
        max_tokens = getattr(args, 'max_tokens', None)
        temperature = getattr(args, 'temperature', None)
        inputs = {}
        if input_file:
            try:
                inputs = _json.loads(Path(input_file).read_text())
            except Exception as exc:
                cli.print_error(f"Could not read input file: {exc}")
                return 1
        return cmd_run(name, inputs, model, max_tokens=max_tokens, temperature=temperature)

    cli.print("Usage: effgen prompts [list|show|eval|playground|render|run]")
    return 1
