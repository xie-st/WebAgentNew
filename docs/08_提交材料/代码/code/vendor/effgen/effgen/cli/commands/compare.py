"""The ``effgen compare`` command handler and its terminal tables.

``_main`` parses arguments and dispatches; it imports these at module scope and
re-exports them (tests reach ``_main._handle_compare_command`` /
``_render_comparison_tables``). Holds the multi-model bake-off run loop, the
per-metric Rich tables, and the recommendation/failure reporting.
"""

from __future__ import annotations

from effgen.cli.commands._shared import resolve_provider_name
from effgen.cli.commands.eval import _resolve_eval_suite
from effgen.cli.commands.report import _write_html_report_arg, _write_result_artifact
from effgen.ui.palette import glyph
from effgen.ui.render import json_ensure_ascii
from effgen.ui.tables import console_is_interactive, empty_state, render_table


def _render_comparison_tables(cli, matrix) -> None:
    """Render a comparison matrix as one Rich table per metric (terminal view).

    Carries the same accuracy / latency / cost cells as ``matrix.to_markdown``
    — including ``ERROR`` for a failed model, ``unpriced`` for a model with no
    published price, and ``—`` for a missing cell — so the terminal and the
    piped Markdown say the same thing.
    """
    if not matrix.scores:
        empty_state(
            cli.console,
            title="No scores recorded",
            message="No model produced a scored result for this run.",
            hints=[
                "Check the model ids with effgen models list",
                "Confirm provider keys with effgen doctor",
            ],
        )
        return
    suites = sorted({s.suite_name for s in matrix.scores})
    models = sorted({s.model_name for s in matrix.scores})
    lookup = {(s.model_name, s.suite_name): s for s in matrix.scores}

    def _cell(sc, kind):
        if sc is None:
            return "—"
        if sc.error:
            return "ERROR"
        if kind == "accuracy":
            return f"{sc.accuracy:.1%}"
        if kind == "latency":
            return f"{sc.avg_latency:.3f}"
        if sc.avg_cost_usd is not None:
            return f"${sc.avg_cost_usd:.6f}"
        return "unpriced"

    for kind, title in (
        ("accuracy", "Accuracy"),
        ("latency", "Avg Latency (s)"),
        ("cost", "Avg Cost (USD/run)"),
    ):
        rows = [
            [m] + [_cell(lookup.get((m, su)), kind) for su in suites]
            for m in models
        ]
        render_table(
            columns=["Model", *suites],
            rows=rows,
            console=cli.console,
            title=title,
            justify=["left", *(["right"] * len(suites))],
            styles=["effgen.model", *([None] * len(suites))],
        )
    if matrix.self_judged is not None:
        from effgen.eval.comparison import _judge_note
        cli.print("\n" + _judge_note(matrix.judge_model, matrix.self_judged))
    # Say why a row reads ERROR, and flag a partial run, so the terminal
    # explains a failure instead of leaving the reader with a bare label.
    failures = [s for s in matrix.scores if s.error or s.error_count]
    if failures:
        g_err = glyph("error")
        cli.print("\n[effgen.heading]Failures[/effgen.heading]")
        for s in sorted(failures, key=lambda x: (x.model_name, x.suite_name)):
            if s.error:
                cli.print(f"  [effgen.error]{g_err}[/effgen.error] {s.model_name} "
                          f"({s.suite_name}): did not run — {s.error}")
            else:
                cli.print(
                    f"  [effgen.error]{g_err}[/effgen.error] {s.model_name} "
                    f"({s.suite_name}): {s.error_count} case(s) failed to run and scored zero"
                )
    if matrix.recommendations:
        g_ok = glyph("success")
        cli.print(f"\n[effgen.heading]Recommendations (optimized for "
                  f"{matrix.optimize})[/effgen.heading]")
        for su, model in sorted(matrix.recommendations.items()):
            why = matrix.recommendation_rationale.get(su)
            cli.print(f"  [effgen.success]{g_ok}[/effgen.success] {su}: "
                      f"[effgen.model]{model}[/effgen.model]"
                      + (f" — {why}" if why else ""))


def _handle_compare_command(args, cli) -> int:
    """Handle 'effgen compare' subcommand."""
    from effgen.eval import ModelComparison
    from effgen.eval.evaluator import ScoringMode

    provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
    if prov_err:
        cli.print_error(prov_err)
        return 2

    model_names = [m.strip() for m in args.models.split(',')]
    suite_name = args.suite
    scoring = ScoringMode(args.scoring)
    threshold = args.threshold
    temperature = getattr(args, 'temperature', None)
    preset_name = getattr(args, 'preset', None)
    difficulty = getattr(args, 'difficulty', None)
    max_cases = getattr(args, 'max_cases', None)
    optimize = getattr(args, 'optimize', 'accuracy')
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True

    # Unknown suite is a user error, not a crash — report it without a traceback
    # and exit 2 with the list of valid suites. A bad data file is reported the
    # same way.
    try:
        suite = _resolve_eval_suite(suite_name, difficulty=difficulty, max_cases=max_cases)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        from effgen.eval import list_suites
        cli.print(
            f"Could not load suite '{suite_name}' ({exc}). "
            f"Use a built-in suite ({', '.join(list_suites())}) "
            "or a path to a .jsonl/.json file of test cases."
        )
        return 2

    agents: dict = {}
    judge_agent = None
    try:
        # Load all models and create agents
        from effgen.models import load_model

        for model_name in model_names:
            cli.print(f"Loading model {model_name}...")
            # --provider is a fallback for a bare id; a model_name that
            # already carries its own "provider:"/"engine:" prefix keeps it.
            model_provider = provider if (provider and ":" not in model_name) else None
            try:
                model = load_model(model_name, provider=model_provider)
                if preset_name:
                    from effgen.presets import create_agent
                    agent = create_agent(preset_name, model, temperature=temperature)
                else:
                    from effgen.core.agent import Agent, AgentConfig
                    config_kwargs: dict = {
                        "name": f"compare-{model_name}", "model": model, "max_iterations": 10,
                    }
                    if temperature is not None:
                        config_kwargs["temperature"] = temperature
                    config_kwargs.setdefault("raise_on_error", False)
                    config = AgentConfig(**config_kwargs)
                    agent = Agent(config)
                agents[model_name] = agent
            except Exception as e:
                cli.print(f"  Warning: Failed to load {model_name}: {e}")

        if not agents:
            cli.print(
                "Error: No models loaded successfully. Check the model ids "
                "(`effgen models list`) and provider keys (`effgen doctor`)."
            )
            return 1

        # A named judge grades every contender, so no model grades its own
        # answers. It is loaded once and reused across the field.
        judge_model = getattr(args, 'judge', None)
        if judge_model:
            if scoring is not ScoringMode.LLM_JUDGE:
                cli.print(
                    f"Ignoring --judge {judge_model}: it applies to "
                    "--scoring llm_judge, and this run scores with "
                    f"'{scoring.value}'."
                )
            else:
                try:
                    from effgen.core.agent import Agent, AgentConfig
                    judge_agent = Agent(AgentConfig(
                        name=f"judge-{judge_model}",
                        raise_on_error=False,
                        model=load_model(judge_model),
                        max_iterations=1,
                    ))
                    cli.print(f"Grading every model's answers with {judge_model}.")
                except Exception as e:
                    cli.print_error(
                        f"Could not load the judge model '{judge_model}': {e}. "
                        "Check the id with `effgen models list`."
                    )
                    return 2

        cli.print(f"\nComparing {len(agents)} models on {suite_name} ({len(suite)} cases)...")
        comparison = ModelComparison(
            scoring=scoring, pass_threshold=threshold, judge_agent=judge_agent
        )
        matrix = comparison.run(agents, [suite], optimize=optimize)

        # Display: rich per-metric tables on a terminal, copy-pasteable Markdown
        # (the same content) when piped or redirected. Under --json the Markdown
        # goes to stderr (via cli.print) so stdout carries only the JSON below.
        if not json_mode and console_is_interactive(cli.console):
            _render_comparison_tables(cli, matrix)
        else:
            cli.print(matrix.to_markdown())

        # Write output — the extension chooses the format.
        if args.output:
            _write_result_artifact(
                args.output,
                cli=cli,
                data=matrix.to_dict(),
                kind="comparison",
                json_text=matrix.to_json(),
                markdown_text=matrix.to_markdown(),
            )

        _write_html_report_arg(args, cli=cli, data=matrix.to_dict(), kind="comparison")

        # Emit the comparison matrix as JSON to stdout for piping/CI gating.
        if json_mode:
            print(matrix.to_json(ensure_ascii=json_ensure_ascii()))

        return 0

    except Exception as e:
        cli.print(f"Comparison failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Release agent resources so the run leaves no GC-close warnings. The
        # judge is closed alongside the contenders it graded.
        for agent in [*agents.values(), judge_agent]:
            close = getattr(agent, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
