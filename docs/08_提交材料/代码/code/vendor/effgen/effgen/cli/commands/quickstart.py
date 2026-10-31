"""The ``effgen quickstart`` / ``effgen tutorial`` command: a guided first run.

:mod:`effgen.cli._main` parses arguments and dispatches; it imports this at
module scope and re-exports these names, so ``effgen.cli._main`` stays the
import path callers use. Walks a new user from nothing to a finished agent run
— pick a model, run a sample task, read the trace and the cost — and then, on
request, has the coding agent write and run a small program in a directory
named before the step starts.

``_quickstart_code_step`` and the coding-readiness report are reached through
:mod:`effgen.cli._main`, so replacing either one there still changes what the
guided run does.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from effgen.cli import onboarding as _onboarding
from effgen.cli import progress as _progress
from effgen.cli.commands._shared import _quickstart_suggest_model, resolve_provider_name

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface

logger = logging.getLogger(__name__)

#: The sample task the quickstart's coding step runs. Small enough for a local
#: model, and it only completes if the agent actually executes what it wrote.
_QUICKSTART_CODE_TASK = (
    "Create fib.py with a function fib(n) returning the nth Fibonacci number, "
    "make the file print fib(10) when it runs, then run it and report the number "
    "it printed."
)


def _quickstart_init_step(args, cli: "CLIInterface") -> int:
    """Handle ``effgen quickstart --init [DIR]`` — scaffold a project.

    Writes the configuration, the ``.env`` template, the example script and the
    ignore file into the named directory, puts a daily spend cap in force when
    none is configured, and prints the next three commands. Prompts for nothing,
    so the same call works on a terminal, in a pipe and in CI.

    Returns ``0`` when every file is in place, ``1`` when one could not be
    written.
    """
    from effgen.cli import scaffold as _scaffold

    directory = getattr(args, "init", None) or "."
    budget = getattr(args, "budget", None)
    if budget is None:
        budget = _scaffold.DEFAULT_DAILY_BUDGET_USD

    result = _scaffold.scaffold_project(
        directory,
        force=getattr(args, "force", False),
        budget=budget,
        model=getattr(args, "model", None),
    )

    cli.print_header("effGen project")
    cli.print(f"Directory: {result.directory}\n")

    widest = max(len(name) for name, _ in _scaffold.SCAFFOLD_FILES)
    for name, description in _scaffold.SCAFFOLD_FILES:
        if name in result.created:
            cli.print(f"  wrote    {name:<{widest}}  {description}")
        elif name in result.skipped:
            cli.print(f"  kept     {name:<{widest}}  already there")
    if result.skipped:
        cli.print("\nExisting files were left as they are; --force replaces them.")

    cli.print(f"\nModel:      {result.model}")
    if result.model_reason:
        cli.print(f"            {result.model_reason}")
    cli.print(f"            change it on the 'model:' line of "
              f"{_scaffold.PROJECT_CONFIG_NAME}")

    if result.daily_budget_usd is not None:
        note = "set now" if result.budget_was_set else "already configured"
        cli.print(f"\nSpend cap:  ${result.daily_budget_usd:.2f} a day across all "
                  f"runs ({note})")
        cli.print("            'effgen cost set-budget N' changes it")
        cli.print("            'effgen cost clear-budget' removes it")
    else:
        cli.print("\nSpend cap:  none in force")
        cli.print("            set one with 'effgen cost set-budget 1.00'")

    if result.errors:
        for problem in result.errors:
            cli.print_error(_onboarding.teach(
                f"Scaffolding {result.directory}: {problem}",
                fix="Pick a directory you can write to, e.g. "
                    "'effgen quickstart --init ~/my-agent'.",
            ))
        return 1

    cli.print_header("Next three commands")
    for index, command in enumerate(_scaffold.next_commands(), 1):
        cli.print(f"  {index}. {command}")
    return 0


def _quickstart_code_wanted(args, *, interactive: bool) -> bool:
    """Whether the guided run should include the coding step.

    ``--code`` runs it, ``--no-code`` skips it, and on a terminal the user is
    asked. A scripted run (``--yes``, piped, CI) skips it unless ``--code`` said
    otherwise, so ``effgen quickstart --yes`` stays a single model call.
    """
    if getattr(args, 'no_code', False):
        return False
    if getattr(args, 'code', False):
        return True
    if not interactive:
        return False
    print("Now the coding agent: it writes a small program, runs it, and reports "
          "the real output.")
    try:
        answer = input("Try it? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        return False
    return answer in ("", "y", "yes")


def _quickstart_code_step(
    args, cli: "CLIInterface", model_id: str, provider: str | None, *, interactive: bool
) -> int:
    """Run a small `effgen code` task end to end and show what it did.

    Returns the exit code for the guided run: ``0`` when the step was skipped or
    completed, ``1`` when it ran and failed. Everything is written inside a
    dedicated directory under the user's effGen state directory, named before the
    step starts, so a first run never touches the directory it was started in.
    """
    from effgen.cli import _main

    if not _quickstart_code_wanted(args, interactive=interactive):
        cli.print("\nSkipping the coding step. Run 'effgen code' whenever you want it, "
                  "or 'effgen quickstart --code' to see it here.")
        return 0

    from effgen.cli.code.engine import CodeEngine, workspace_execution_note
    from effgen.cli.code.permissions import MODE_DESCRIPTIONS, PermissionMode
    from effgen.cli.code.render import print_action, print_diff, print_summary

    workspace = _onboarding.state_dir() / "quickstart-code"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        cli.print_error(_onboarding.teach(
            f"Could not create the coding workspace {workspace}: {e}",
            fix="Set EFFGEN_HOME to a directory you can write to, then re-run "
                "'effgen quickstart --code'.",
        ))
        return 1

    # The step is a self-contained demonstration in a directory named above, so
    # it allows writes, sandboxed runs and shell commands without stopping to
    # ask — a guided first run that dead-ends on an unconfirmable command shows
    # the newcomer nothing. The mode is printed, and confinement still applies.
    mode = PermissionMode.YES
    cli.print_header("The coding agent")
    cli.print(f"Workspace: {workspace}  (the files it changes stay there)")
    cli.print(f"Permissions: {mode.value} — {MODE_DESCRIPTIONS[mode]}")
    # The same sandbox line `effgen doctor` reports, so a newcomer knows how the
    # code they are about to run is confined before it runs.
    sandbox = next(
        (c for c in _main._doctor_code_report(str(workspace)).get("checks", [])
         if c.get("name") == "sandbox"),
        None,
    )
    if sandbox:
        cli.print(f"Sandbox: {sandbox.get('detail', sandbox.get('status', ''))}")
    cli.print(f"Task: {_QUICKSTART_CODE_TASK}\n")

    # The generic note points at -w/--workspace, which the guided run does not
    # take; here the directory follows EFFGEN_HOME, so name that instead.
    if workspace_execution_note(workspace) is not None:
        cli.print_warning(
            f"Code run in the sandbox cannot read {workspace}: this sandbox "
            "cannot bind the workspace into the code's private temp directory, "
            "so the file will be written but running it will fail. Set "
            "EFFGEN_HOME to a directory outside the system temp directory."
        )

    engine = CodeEngine(
        model=model_id,
        provider=provider,
        workspace=workspace,
        mode=mode,
        mode_explicit=True,
        interactive=interactive,
        # No iteration override: the step is a real `effgen code` run on the
        # coding preset's own budget, so what a newcomer sees here is what the
        # command does.
        on_event=lambda record: print_action(cli, record),
        on_diff=lambda edit: print_diff(cli, edit),
    )

    agent = None
    # The action ticks and the summary below report what happened; the agent's
    # internal retry/fallback logging would only add noise to a guided run.
    _effgen_logger = logging.getLogger("effgen")
    _prev_level = _effgen_logger.level
    try:
        agent = engine.build_agent()
        _effgen_logger.setLevel(logging.CRITICAL)
        result = engine.run(_QUICKSTART_CODE_TASK)
    except KeyboardInterrupt:
        cli.print_warning("Interrupted; the coding step stopped where it was.")
        return 0
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        cli.print_error(_onboarding.teach(
            f"The coding step could not run: {e}",
            fix="Try it directly with 'effgen code \"write fib.py and run it\" "
                "--auto-edit' to see the full error.",
            doc="docs/cli/code.md",
        ))
        return 1
    finally:
        _effgen_logger.setLevel(_prev_level)
        if agent is not None:
            try:
                agent.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"quickstart: code agent close failed: {e}")

    cli.print("")
    if cli.console:
        cli.console.print(_main.Panel(
            _main.Markdown(result.answer or "(no output)"),
            border_style="green" if result.success else "red",
        ))
    else:
        print(result.answer)

    print_summary(cli, result)
    if result.files_written:
        cli.print(f"Files written in {workspace}: {', '.join(result.files_written)}")
        cli.print(f"Undo them with: effgen code --undo -w {workspace}")
    for record in result.refused:
        cli.print_warning(record.reason)
    if result.withheld:
        kinds = ", ".join(sorted({a.kind for a in result.withheld}))
        cli.print_warning(
            f"{len(result.withheld)} action(s) ({kinds}) were not carried out, so "
            "the report above may not cover everything the task asked for."
        )

    # The sample task is only demonstrated when code actually ran. A model that
    # stops after writing the file has not computed anything, so say that rather
    # than letting the answer above stand in for a result.
    executed = any(
        a.kind in ("run", "shell") and a.decision == "allowed" and a.outcome != "error"
        for a in result.actions
    )
    if not executed:
        did = (
            "wrote the file but never ran it"
            if result.files_written
            else "did not write or run anything"
        )
        cli.print_warning(
            f"The model {did}, so nothing above was computed by executing code. "
            "Smaller models often stop early on a multi-step task; a larger one "
            "finishes this in two or three steps."
        )

    if result.hit_iteration_cap:
        cli.print_warning(
            f"The model used all {result.iterations} steps of the sample task "
            "without finishing. A larger model usually needs two or three; "
            "'effgen code --max-iterations N' raises the cap for a real task."
        )
    if not result.success:
        cli.print_error(_onboarding.teach(
            "The coding step did not complete.",
            fix="Re-run with a larger model, e.g. 'effgen quickstart --code -m gpt-5-mini'.",
            doc="docs/cli/code.md",
        ))
        return 1
    return 0


def _handle_quickstart_command(args, cli: "CLIInterface") -> int:
    """Handle 'effgen quickstart' / 'effgen tutorial' — a guided first run.

    Walks a brand-new user from nothing to a successful agent run in well under
    two minutes: pick a model → run a sample task → see the trace → see the cost
    → write and run a small program with the coding agent. Fully scriptable
    (``--model``/``--task``/``--yes``/``--code``/``--no-code``) for CI and docs.
    """
    from effgen.cli import _main

    # --init scaffolds a project and stops: no model call, no key needed, and
    # nothing to answer, so it is the one step that works before setup is done.
    if getattr(args, 'init', None) is not None:
        return _main._quickstart_init_step(args, cli)

    interactive = _onboarding.is_interactive() and not getattr(args, 'yes', False)

    if getattr(args, 'code', False) and getattr(args, 'no_code', False):
        cli.print_error("--code and --no-code cannot be combined — they ask for "
                        "opposite things. Pass one.")
        return 1

    cli.print_header("effGen quickstart")
    cli.print("Let's run your first agent. This takes about a minute.\n")

    # --- 1. Choose a model -------------------------------------------------
    provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
    if prov_err:
        cli.print_error(prov_err)
        return 1

    model_id = getattr(args, 'model', None)
    if not model_id:
        model_id, suggested_provider, reason = _quickstart_suggest_model()
        provider = provider or suggested_provider
        cli.print(f"Suggested model: [bold]{model_id}[/bold]"
                  f"{f' (provider: {provider})' if provider else ''} — {reason}"
                  if cli.console else
                  f"Suggested model: {model_id}"
                  f"{f' (provider: {provider})' if provider else ''} — {reason}")
        if interactive:
            cli.print("Press Enter to accept, or type a model id (e.g. gpt-5-nano):")
            try:
                typed = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                typed = ""
                cli.print("")
            if typed:
                model_id = typed
                # A bare cloud id without a provider gets routed by the registry;
                # leave provider as-is so the loader can infer it.
        cli.print("")

    # --- 2. Run a sample task ---------------------------------------------
    _default_task = "What is 25 * 17?"
    task = getattr(args, 'task', None) or _default_task
    is_default_task = task == _default_task
    cli.print(f"Task: [italic]{task}[/italic]\n" if cli.console else f"Task: {task}\n")

    agent = None
    try:
        # Only equip the calculator for the math sample task. A user-supplied
        # task gets a clean direct-answer run so an unrelated question never
        # makes a small local model loop on a tool it doesn't need.
        tools = []
        if is_default_task:
            cli.tool_registry.discover_builtin_tools()
            if "calculator" in cli.tool_registry.list_tools():
                try:
                    tools.append(cli.tool_registry.get_tool_sync("calculator"))
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"quickstart: calculator load failed: {e}")

        agent_config = _main.AgentConfig(
            name="quickstart-agent",
            raise_on_error=False,
            model=model_id,
            provider=provider,
            tools=tools,
            system_prompt="You are a helpful AI assistant.",
            max_iterations=5,
        )
        agent = _main.Agent(agent_config)

        animate = cli._animate(args)
        # Keep the first impression clean: the agent transparently retries and
        # recovers from transient provider hiccups, so suppress that internal
        # log churn during the guided run (restored immediately after). A real
        # failure still surfaces via response.success below.
        _effgen_logger = logging.getLogger("effgen")
        _prev_level = _effgen_logger.level
        _effgen_logger.setLevel(logging.CRITICAL)
        try:
            if animate:
                with _progress.LiveStatus(
                    cli.console,
                    model_label=_progress.short_model_label(model_id),
                    reasoning=_progress.is_reasoning_agent(agent),
                    tracker=agent.execution_tracker,
                ):
                    response = agent.run(task)
            else:
                cli.print("Thinking...")
                response = agent.run(task)
        except KeyboardInterrupt:
            cli._handle_interrupt(agent)
            return 130
        finally:
            _effgen_logger.setLevel(_prev_level)

        # --- 3. Show the answer ------------------------------------------
        cli.print_header("Answer")
        if cli.console:
            cli.console.print(_main.Panel(
                _main.Markdown(response.output or "(no output)"),
                border_style="green" if response.success else "red",
            ))
        else:
            print(response.output)

        if not response.success:
            err = (response.metadata or {}).get("error", {})
            cli.print_error(_onboarding.teach(
                f"The run did not succeed: {err.get('message', 'unknown error')}",
                fix="Run 'effgen doctor --live --cheap' to confirm the model is reachable, "
                    "or try a different model with -m.",
                doc="docs/tutorials/getting-started.md",
            ))
            return 1

        # --- 4. Show the trace -------------------------------------------
        # The agent's execution_trace is a list of event dicts (type/message/
        # data) — not a ReAct action/observation shape — so render it with the
        # shared formatter that understands those events. Hand-reading
        # ``step["action"]`` here used to miss every tool call and wrongly print
        # "answered directly" even when a tool ran (and tool_calls reported it).
        cli.print_header("What the agent did")
        _trace_lines = _progress.execution_trace_lines(response.execution_trace)
        if _trace_lines:
            for _style, _text in _trace_lines:
                if cli.console:
                    cli.console.print(f"  [{_style}]{_text}[/{_style}]", highlight=False)
                else:
                    cli.print(f"  {_text}")
        else:
            cli.print("  (answered directly, no tools needed)")

        # --- 5. Show the cost --------------------------------------------
        _progress.print_summary(cli, response)

        # --- 6. Write and run real code ----------------------------------
        code_rc = _main._quickstart_code_step(args, cli, model_id, provider, interactive=interactive)

        # --- Next steps --------------------------------------------------
        # The sample run succeeded to get here; only claim the whole guided run
        # worked when the coding step did too.
        cli.print_header("You're set! Next steps" if code_rc == 0 else "Next steps")
        cli.print("  • effgen run \"your question\" -m " + model_id + "   # run any task")
        cli.print("  • effgen chat -m " + model_id + "                  # interactive chat")
        cli.print("  • effgen code                                    # coding agent session")
        cli.print("  • effgen tools list                              # see available tools")
        cli.print("  • effgen doctor --live --cheap                   # check all providers")
        return code_rc

    except Exception as e:  # noqa: BLE001
        cli.print_error(_onboarding.teach(
            f"Quickstart could not complete: {e}",
            fix="Run 'effgen doctor' to check your setup, then 'effgen quickstart -m <model>'.",
            doc="docs/tutorials/getting-started.md",
        ))
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"quickstart: agent close failed: {e}")
