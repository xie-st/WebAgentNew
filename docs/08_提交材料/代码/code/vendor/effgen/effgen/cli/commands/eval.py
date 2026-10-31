"""The ``effgen eval`` command handler and test-suite resolution.

``_main`` parses arguments and dispatches; it imports these at module scope and
re-exports them (``_main._handle_eval_command`` / ``_resolve_eval_suite`` are
reached by tests). Holds suite resolution (built-in name or a ``.jsonl``/``.json``
file of custom cases), the run loop, the results table, and the exit-code gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from effgen.cli import progress as _progress
from effgen.cli.commands._shared import resolve_provider_name
from effgen.cli.commands.report import _write_html_report_arg, _write_result_artifact
from effgen.ui.palette import glyph
from effgen.ui.render import json_ensure_ascii
from effgen.ui.tables import console_is_interactive, render_table


def _resolve_eval_suite(suite_arg: str, difficulty=None, max_cases=None):
    """Resolve a ``--suite`` argument to a ``TestSuite``.

    Accepts a built-in suite name **or** a path to a ``.jsonl`` / ``.json`` file
    of your own test cases (each an object with ``query``/``expected_output`` and
    optional ``difficulty``/``tags``), so a bake-off can run on your own data —
    not just the bundled suites. Optionally filters by ``difficulty`` and trims to
    the first ``max_cases``. Raises ``KeyError`` (with the list of valid names)
    for an unknown name, or ``FileNotFoundError`` / ``ValueError`` for a bad file.
    """
    from effgen.eval import get_suite
    from effgen.eval.suites import TestSuite

    p = Path(suite_arg)
    if p.suffix.lower() in (".jsonl", ".json") or p.exists():
        if not p.exists():
            raise FileNotFoundError(f"Test-case file not found: {suite_arg}")
        from effgen.eval.evaluator import TestCase
        raw = p.read_text(encoding="utf-8")
        records = []
        if p.suffix.lower() == ".json":
            data = json.loads(raw)
            records = data if isinstance(data, list) else [data]
        else:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        cases = [TestCase.from_dict(r) for r in records]
        if not cases:
            raise ValueError(f"No test cases found in {suite_arg}")
        suite = TestSuite(test_cases=cases)
        suite.name = p.stem
    else:
        suite = get_suite(suite_arg)

    if difficulty:
        from effgen.eval.evaluator import Difficulty
        suite.test_cases = suite.filter(difficulty=Difficulty(difficulty))
    if max_cases is not None and max_cases > 0:
        suite.test_cases = suite.test_cases[:max_cases]
    return suite


def _handle_eval_command(args, cli) -> int:
    """Handle 'effgen eval' subcommand."""
    from effgen.eval import AgentEvaluator, RegressionTracker, list_suites
    from effgen.eval.evaluator import ScoringMode

    # --json: route human chatter to stderr so stdout carries only the JSON
    # results document (CI gates parse it).
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True

    provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
    if prov_err:
        cli.print_error(prov_err)
        return 2

    suite_name = args.suite
    model_name = getattr(args, 'model', None) or 'Qwen/Qwen2.5-1.5B-Instruct'
    preset_name = getattr(args, 'preset', None)
    scoring = ScoringMode(args.scoring)
    threshold = args.threshold
    fail_under = getattr(args, 'fail_under', 0.5)
    baseline_dir = getattr(args, 'baseline_dir', None)
    temperature = getattr(args, 'temperature', None)
    difficulty = getattr(args, 'difficulty', None)
    max_cases = getattr(args, 'max_cases', None)

    agent = None
    try:
        # List suites if requested
        if suite_name == 'list':
            suites = list_suites()
            if json_mode:
                print(json.dumps(
                    [{"name": n, "description": d} for n, d in suites.items()],
                    indent=2, ensure_ascii=json_ensure_ascii(),
                ))
                return 0
            cli.print_header("Available Evaluation Suites")
            for name, desc in suites.items():
                cli.print(f"  {name:16s} — {desc}")
            return 0

        # A bad data file (unknown field, empty, missing) is a user error, not a
        # crash — report it and exit 2 (matching `compare`) instead of a
        # traceback.
        try:
            suite = _resolve_eval_suite(suite_name, difficulty=difficulty, max_cases=max_cases)
        except (ValueError, FileNotFoundError) as exc:
            cli.print(
                f"Could not load suite '{suite_name}' ({exc}). "
                f"Use a built-in suite ({', '.join(list_suites())}) "
                "or a path to a .jsonl/.json file of test cases."
            )
            return 2

        # Report any narrowing applied
        if difficulty:
            cli.print(f"Filtered to {len(suite.test_cases)} {difficulty} test cases")
        if max_cases:
            cli.print(f"Limited to first {len(suite.test_cases)} cases")

        cli.print(f"Loading model {model_name}...")

        # Create agent
        if preset_name:
            from effgen.models import load_model
            from effgen.presets import create_agent
            model = load_model(model_name, provider=provider)
            agent = create_agent(preset_name, model, temperature=temperature)
        else:
            from effgen.core.agent import Agent, AgentConfig
            from effgen.models import load_model
            model = load_model(model_name, provider=provider)
            config_kwargs: dict = {"name": "eval-agent", "model": model, "max_iterations": 10}
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            # A case that fails is a scored result, not an aborted suite.
            config_kwargs.setdefault("raise_on_error", False)
            config = AgentConfig(**config_kwargs)
            agent = Agent(config)

        cli.print(f"Running {suite_name} suite ({len(suite)} cases, scoring={args.scoring})...")
        evaluator = AgentEvaluator(agent, scoring=scoring, pass_threshold=threshold)
        # --json emits a single JSON document to stdout: no live progress bar,
        # which would otherwise render there on an interactive terminal.
        animate = not json_mode and _progress.animation_enabled(
            quiet=getattr(args, 'quiet', False),
            no_animation=getattr(args, 'no_animation', False),
        )
        with _progress.StepProgress(
            cli.console, total=len(suite), description="Eval", animate=animate,
        ) as _bar:
            results = evaluator.run_suite(
                suite, progress_callback=lambda done, total: _bar.update(done, total),
            )

        # Display results
        summary = results.summary()
        cli.print_header(f"Evaluation Results: {suite_name}")
        # Under --json the summary is human chatter: route it to stderr so
        # stdout carries only the JSON document below.
        render_table(
            columns=["Metric", "Value"],
            rows=[
                ["Accuracy", f"{summary['accuracy']:.1%} ({summary['passed']}/{summary['total']})"],
                ["Avg Latency", f"{summary['avg_latency']:.4f}s"],
                ["Total Tokens", f"{summary['total_tokens']}"],
                ["Tool Accuracy", f"{summary['avg_tool_accuracy']:.1%}"],
            ],
            console=None if json_mode else cli.console,
            styles=["effgen.metric", None],
            file=sys.stderr if json_mode else None,
        )

        if summary.get('by_difficulty'):
            cli.print("\n  By Difficulty:")
            for d, info in sorted(summary['by_difficulty'].items()):
                cli.print(f"    {d:8s}: {info['accuracy']:.1%} ({info['passed']}/{info['total']})")

        # Show per-case details for failures
        failures = [r for r in results.results if not r.passed]
        if failures:
            cli.print(f"\n  Failed cases ({len(failures)}):")
            for r in failures[:10]:
                cli.print(f"    - {r.test_case.query[:60]}...")
                cli.print(f"      Expected: {r.test_case.expected_output[:40]}")
                cli.print(f"      Got:      {r.agent_output[:40]}")
                if r.stop_reason and r.stop_reason != "final_answer":
                    cli.print(f"      Ended:    {r.stop_reason}")

        # Save baseline — keyed on the resolved suite name (a custom dataset's
        # name is the file stem, never the full path) so a baseline file for a
        # nested or absolute suite path does not fail to write.
        if args.save_baseline:
            from effgen import __version__
            tracker = RegressionTracker(baselines_dir=baseline_dir)
            path = tracker.save_baseline(suite.name, results, version=__version__)
            cli.print(f"\n  Baseline saved to {path}")

        # Compare baseline
        report = None
        if args.compare_baseline:
            from effgen import __version__
            tracker = RegressionTracker(baselines_dir=baseline_dir)
            report = tracker.compare(suite.name, results, version=__version__)
            cli.print(f"\n{report.to_markdown()}")

        # Stamp the run context the report header and exit-gate verdict read.
        results.metadata.setdefault("model", model_name)
        results.metadata.setdefault("scoring", args.scoring)
        results.metadata["fail_under"] = fail_under

        # Write output — the extension chooses the format.
        if args.output:
            _write_result_artifact(
                args.output,
                cli=cli,
                data=results.summary(),
                kind="eval",
                json_text=results.to_json(),
                markdown_text=results.to_markdown(),
            )

        _write_html_report_arg(args, cli=cli, data=results.summary(), kind="eval")

        # Emit the same results document to stdout for piping/CI gating.
        if json_mode:
            print(results.to_json(ensure_ascii=json_ensure_ascii()))

        # Exit-code gate. A detected blocking regression against a saved
        # baseline always fails the build; otherwise the gate is the suite
        # accuracy against --fail-under (--threshold is a separate per-case
        # setting and does not drive the exit code).
        # A status glyph leads the verdict on an interactive terminal; the piped
        # line (which CI logs parse) stays exactly as before.
        def _gate_prefix(passed: bool) -> str:
            if not console_is_interactive(cli.console):
                return ""
            role = "effgen.success" if passed else "effgen.error"
            return f"[{role}]{glyph('success' if passed else 'error')}[/{role}] "

        if report is not None and report.has_regressions:
            cli.print(
                f"\n  {_gate_prefix(False)}Exit gate: FAIL — blocking regression "
                "against baseline (--compare-baseline)."
            )
            return 1
        gate_passed = results.accuracy >= fail_under
        cli.print(
            f"\n  {_gate_prefix(gate_passed)}Exit gate: {'PASS' if gate_passed else 'FAIL'} "
            f"— accuracy {results.accuracy:.1%} {'>=' if gate_passed else '<'} "
            f"--fail-under {fail_under:.0%}"
        )
        return 0 if gate_passed else 1

    except KeyError as e:
        cli.print(f"Error: {e}")
        cli.print("Available suites:")
        for name, desc in list_suites().items():
            cli.print(f"  {name:16s} — {desc}")
        return 1
    except Exception as e:
        cli.print(f"Evaluation failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Release agent resources so eval stops emitting the GC-without-close
        # warning on every run (matches `compare`'s cleanup).
        if agent is not None:
            close = getattr(agent, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
