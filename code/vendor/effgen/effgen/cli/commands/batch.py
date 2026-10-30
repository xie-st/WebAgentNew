"""The ``effgen batch`` command.

Runs a file of tasks through one agent and streams the results out as JSONL,
CSV or JSON. :mod:`effgen.cli._main` parses the arguments and re-exports the
names defined here.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Live status / progress presentation (TTY-aware; degrades to plain text).
from effgen.cli import progress as _progress
from effgen.ui.render import json_ensure_ascii


def _batch_structured_kwargs(args) -> dict:
    """Build ``output_schema`` / ``output_model`` run-kwargs from the CLI flags.

    ``--schema PATH`` loads a JSON Schema file; ``--output-model module:Class``
    imports a Pydantic model. Each validates every row and writes the parsed
    object; a row that cannot be coerced to the schema is reported as a failed
    row with a reason rather than a silently off-schema string. Raises
    ``ValueError`` with an actionable message on a bad path or spec.
    """
    schema_path = getattr(args, 'schema_path', None)
    model_spec = getattr(args, 'output_model', None)
    if schema_path and model_spec:
        raise ValueError("Use only one of --schema / --output-model, not both.")
    if schema_path:
        p = Path(schema_path)
        if not p.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        try:
            schema = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{schema_path}: not valid JSON: {e}") from e
        if not isinstance(schema, dict):
            raise ValueError(f"{schema_path}: a JSON Schema must be a JSON object.")
        return {"output_schema": schema}
    if model_spec:
        if ":" not in model_spec:
            raise ValueError(
                "--output-model must be 'module:ClassName' "
                "(e.g. myproject.schemas:Ticket)."
            )
        mod_name, _, cls_name = model_spec.partition(":")
        import importlib
        # Make a project-local module importable from a headless run.
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            raise ValueError(f"Could not import module '{mod_name}': {e}") from e
        cls = getattr(mod, cls_name, None)
        if cls is None:
            raise ValueError(f"Module '{mod_name}' has no attribute '{cls_name}'.")
        from effgen.core.structured_output import is_pydantic_model_class
        if not is_pydantic_model_class(cls):
            raise ValueError(
                f"{model_spec} is not a Pydantic model class "
                "(it must subclass pydantic.BaseModel)."
            )
        return {"output_model": cls}
    return {}


def _read_done_indices(output_path: Path) -> dict:
    """Read an existing JSONL output file into ``{index: row}`` for --resume."""
    done: dict[int, dict] = {}
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = row.get("index") if isinstance(row, dict) else None
            if isinstance(idx, int):
                done[idx] = row
    return done


def _handle_batch_command(args, cli) -> int:
    """Handle the 'batch' CLI subcommand."""
    from effgen.core.batch import _QUERY_ALIASES, BatchConfig, BatchRunner
    from effgen.core.batch import SUPPORTED_OUTPUT_FORMATS as _BATCH_OUTPUT_FORMATS

    # Accept the input file as a positional argument or via -i/--input; the
    # explicit flag wins if both are given.
    input_path = getattr(args, 'input', None) or getattr(args, 'input_file', None)
    output_path = getattr(args, 'output', None)
    model_name = getattr(args, 'model', None) or 'Qwen/Qwen2.5-1.5B-Instruct'
    preset_name = getattr(args, 'preset', None)
    guardrails = getattr(args, 'guardrails', None)
    query_field = getattr(args, 'query_field', 'query')
    max_tokens = getattr(args, 'max_tokens', None)
    temperature = getattr(args, 'temperature', None)
    system_prompt = getattr(args, 'system_prompt', None)
    strict = getattr(args, 'strict', False)
    resume = getattr(args, 'resume', False)

    # Headless JSON contract, same as `run --json`: keep stdout pure (only the
    # JSON result document) by routing human chatter to stderr, and emit a
    # typed error object to stdout on EVERY failure path — early argument/read
    # failures included — so a `| jq` consumer never gets empty input.
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True

    def _json_error(exc: Exception) -> int:
        if json_mode:
            print(json.dumps({
                "success": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }, indent=2, ensure_ascii=json_ensure_ascii()))
        return 1

    if not input_path:
        msg = ("No input file given. Pass one as `effgen batch FILE` "
               "or with -i/--input (JSONL, CSV, JSON, or plain text).")
        cli.print_error(msg)
        return _json_error(ValueError(msg))

    # Extra kwargs forwarded to each agent.run() call.
    run_kwargs: dict = {}
    if max_tokens is not None:
        run_kwargs['max_tokens'] = max_tokens
    if temperature is not None:
        run_kwargs['temperature'] = temperature
    try:
        run_kwargs.update(_batch_structured_kwargs(args))
    except ValueError as e:
        cli.print_error(str(e))
        return _json_error(e)

    out_suffix = Path(output_path).suffix.lower() if output_path else None
    # Reject an unsupported --output extension before loading a model or making
    # a single billed call, naming the formats that do work — rather than
    # running the whole batch and failing at write time.
    if output_path and out_suffix not in _BATCH_OUTPUT_FORMATS:
        shown = out_suffix or "(none)"
        msg = (
            f"Unsupported --output format: {shown}. "
            f"Use one of: {', '.join(sorted(_BATCH_OUTPUT_FORMATS))}."
        )
        cli.print_error(msg)
        return _json_error(ValueError(msg))
    # A .jsonl output streams each finished row as it completes, so a crash
    # mid-job keeps the rows already done; --resume then skips those on rerun.
    stream_jsonl = bool(output_path) and out_suffix == ".jsonl"
    if resume and not stream_jsonl:
        msg = "--resume requires a .jsonl --output file."
        cli.print_error(msg)
        return _json_error(ValueError(msg))

    if not output_path and not json_mode:
        cli.print(
            "Warning: no -o/--output and no --json — each row's answer, cost, "
            "tokens, and any error detail will be discarded; only the summary "
            "line at the end of this run is kept."
        )

    agent = None
    out_fh = None
    try:
        # Create agent
        if preset_name:
            from effgen.models import load_model
            from effgen.presets import create_agent
            model = load_model(model_name)
            agent = create_agent(
                preset_name, model, system_prompt=system_prompt, guardrails=guardrails,
            )
        else:
            from effgen.core.agent import Agent, AgentConfig
            from effgen.models import load_model
            model = load_model(model_name)
            config_kwargs: dict = {}
            if system_prompt is not None:
                config_kwargs['system_prompt'] = system_prompt
            config = AgentConfig(
                name="batch-agent", model=model, max_iterations=5,
                guardrails=guardrails, raise_on_error=False, **config_kwargs,
            )
            agent = Agent(config)

        runner = BatchRunner(agent)
        cli.print(f"Loading queries from {input_path}...")

        # Read queries once. A malformed input line is skipped with a message
        # naming the file and line number (not the parser's byte offset);
        # --strict turns the first bad line into a hard failure instead.
        skipped: list[int] = []
        empty_rows: list[tuple[int, list[str]]] = []

        def _on_skip(lineno: int, msg: str) -> None:
            skipped.append(lineno)
            cli.print(f"Skipping malformed input at {input_path}:{lineno}: {msg}")

        def _on_empty(lineno: int, keys: list[str]) -> None:
            empty_rows.append((lineno, keys))

        def _empty_rows_message() -> str:
            """Name the fields the first unusable row carried and how to point at one."""
            lineno, keys = empty_rows[0]
            fields = ", ".join(keys) if keys else "none"
            more = f" (and {len(empty_rows) - 1} more)" if len(empty_rows) > 1 else ""
            return (
                f"Row {lineno}{more} has no query text. Fields present: {fields}. "
                f"Set the query column with --query-field NAME, or key rows on one "
                f"of: {', '.join(_QUERY_ALIASES)}."
            )

        try:
            queries = runner._read_queries(
                Path(input_path), query_field, strict=strict,
                on_skip=_on_skip, on_empty=_on_empty,
            )
        except Exception as e:  # noqa: BLE001 - one clear message, no traceback
            # When every row lacked a recognized query field the read raises
            # before any query is collected, so report the fields those rows
            # carried instead of a generic "no queries found".
            if empty_rows:
                msg = _empty_rows_message()
                cli.print_error(msg)
                return _json_error(ValueError(msg))
            cli.print_error(f"Could not read {input_path}: {e}")
            return _json_error(e)
        if skipped:
            cli.print(
                f"Skipped {len(skipped)} malformed line(s); "
                f"{len(queries)} queries loaded."
            )

        # A row with no recognized query text (neither --query-field nor the
        # aliases query/input/prompt/question/text) can't run. Name the fields it
        # did carry and how to point at the right one, rather than letting each
        # empty row fail with a generic empty-task message.
        if empty_rows:
            msg = _empty_rows_message()
            cli.print_error(msg)
            return _json_error(ValueError(msg))

        # --resume: skip input rows already present in the JSONL output.
        done_rows: dict[int, dict] = {}
        if resume:
            done_rows = {
                i: row for i, row in _read_done_indices(Path(output_path)).items()
                if 0 <= i < len(queries)
            }
        run_positions = [i for i in range(len(queries)) if i not in done_rows]
        run_queries = [queries[i] for i in run_positions]
        if done_rows:
            cli.print(
                f"Resuming: {len(done_rows)} row(s) already present, "
                f"running the remaining {len(run_queries)}."
            )

        # Open the streaming output before the run so completed rows persist
        # immediately. Resume appends to the existing file; a fresh run truncates.
        if stream_jsonl:
            mode = "a" if (resume and Path(output_path).exists()) else "w"
            out_fh = open(output_path, mode, encoding="utf-8")

        def _on_result(pos: int, query: str, resp) -> None:
            if out_fh is None:
                return
            orig_idx = run_positions[pos]
            row = BatchRunner._result_row(orig_idx, resp, query)
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()

        # --json emits a single JSON document to stdout: no live progress bar,
        # which would otherwise render there on an interactive terminal.
        animate = not json_mode and _progress.animation_enabled(
            quiet=getattr(args, 'quiet', False),
            no_animation=getattr(args, 'no_animation', False),
        )
        with _progress.StepProgress(
            cli.console, total=len(run_queries), description="Batch", animate=animate,
        ) as _bar:
            batch_config = BatchConfig(
                max_concurrency=args.concurrency,
                batch_size=args.batch_size,
                retry_failed=args.retries,
                timeout_per_item=args.timeout,
                progress_callback=lambda done, total: _bar.update(done, total),
                on_result=_on_result if out_fh is not None else None,
            )
            result = runner.run(run_queries, config=batch_config, **run_kwargs)

        if out_fh is not None:
            out_fh.close()
            out_fh = None

        # Combine this run with any rows carried over by --resume so the
        # headline counts and totals reflect the whole job, not just the rerun.
        done_success = sum(1 for r in done_rows.values() if r.get("success"))
        total = len(queries)
        succeeded = done_success + result.succeeded
        failed = total - succeeded

        done_cost = sum(
            r["cost_usd"] for r in done_rows.values()
            if isinstance(r.get("cost_usd"), int | float)
        )
        done_tokens = sum(
            r.get("total_tokens", 0) for r in done_rows.values()
            if isinstance(r.get("total_tokens"), int)
        )
        done_prompt_tokens = sum(
            r.get("prompt_tokens", 0) for r in done_rows.values()
            if isinstance(r.get("prompt_tokens"), int)
        )
        done_completion_tokens = sum(
            r.get("completion_tokens", 0) for r in done_rows.values()
            if isinstance(r.get("completion_tokens"), int)
        )
        cost_present = result.total_cost_usd is not None or done_cost > 0
        total_cost = (result.total_cost_usd or 0.0) + done_cost if cost_present else None
        total_tokens = result.total_tokens + done_tokens
        total_prompt_tokens = result.total_prompt_tokens + done_prompt_tokens
        total_completion_tokens = result.total_completion_tokens + done_completion_tokens

        body = (
            f"Batch complete: {succeeded}/{total} succeeded "
            f"in {result.total_time:.2f}s"
        )
        if total_tokens:
            body += f" · {total_tokens:,} tokens"
        from effgen.ui.render import format_cost
        cost_str = format_cost(total_cost)
        if cost_str is not None:
            body += f" · {cost_str}"
        # A status glyph leads the summary on an interactive terminal; the piped
        # line stays exactly as before so a script reading it is unaffected.
        from effgen.ui.palette import glyph
        from effgen.ui.tables import console_is_interactive
        if console_is_interactive(cli.console):
            role = "effgen.success" if failed == 0 else "effgen.warning"
            g = glyph("success" if failed == 0 else "warning")
            cli.print(f"\n[{role}]{g}[/{role}] {body}")
        else:
            cli.print("\n" + body)

        # Non-streaming formats (.csv/.json) get one batched write at the end.
        if output_path and not stream_jsonl:
            runner.write_results(
                result, output_path, query_list=run_queries,
                excel_bom=getattr(args, 'excel_bom', False),
            )
            cli.print(f"Results written to {output_path}")
        elif output_path:
            cli.print(f"Results written to {output_path}")

        if json_mode:
            rows = list(done_rows.values())
            for pos, resp in enumerate(result.results):
                rows.append(BatchRunner._result_row(run_positions[pos], resp, run_queries[pos]))
            rows.sort(key=lambda r: r.get("index", 0))
            print(json.dumps({
                "total": total,
                "succeeded": succeeded,
                "failed": failed,
                "success_rate": round(succeeded / total, 4) if total else 0.0,
                "total_time": round(result.total_time, 2),
                "total_cost_usd": round(total_cost, 8) if total_cost is not None else None,
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "rows": rows,
            }, indent=2, ensure_ascii=json_ensure_ascii()))

        return 0 if failed == 0 else 1

    except Exception as e:
        cli.print(f"Batch execution failed: {e}")
        return _json_error(e)
    finally:
        if out_fh is not None:
            try:
                out_fh.close()
            except Exception:  # noqa: BLE001
                pass
        # Release the agent's resources so the CLI never trips its own
        # "garbage-collected without close()" warning.
        if agent is not None:
            try:
                agent.close()
            except Exception:
                logging.getLogger(__name__).debug("Batch agent close() failed", exc_info=True)
