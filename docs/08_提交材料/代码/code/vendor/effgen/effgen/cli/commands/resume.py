"""The ``effgen resume`` command: continue a run from a saved checkpoint.

:mod:`effgen.cli._main` parses arguments and dispatches; it imports this at
module scope and re-exports it, so ``effgen.cli._main._handle_resume_command``
keeps resolving. Resolves the checkpoint id, path or directory, picks the model
(an explicit ``-m`` wins, otherwise the one the checkpoint recorded), and
releases the agent when the run ends.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _handle_resume_command(args, cli) -> int:
    """Handle 'effgen resume' command."""
    from effgen import Agent, AgentConfig
    from effgen.core.checkpoint import CheckpointManager
    from effgen.errors import CorruptStateError

    cp_arg = args.checkpoint
    # Determine directory + id
    import os as _os
    if _os.path.isdir(cp_arg):
        ckpt_dir = cp_arg
        cp_id = None
    elif cp_arg.endswith(".json") and _os.path.exists(cp_arg):
        ckpt_dir = _os.path.dirname(_os.path.abspath(cp_arg)) or "."
        cp_id = cp_arg
    else:
        ckpt_dir = "./checkpoints"
        cp_id = cp_arg

    mgr = CheckpointManager(ckpt_dir)
    try:
        cp = mgr.load(cp_id) if cp_id else mgr.load_latest()
    except FileNotFoundError as e:
        cli.print(f"Error: {e}")
        cli.print("List available checkpoints by pointing --checkpoint at their directory.")
        cli.print(
            "Looking for a saved conversation instead? Those are listed by "
            "`effgen sessions list` and continued with `effgen chat --session-id <id>`."
        )
        return 2
    except CorruptStateError as e:
        cli.print(f"Error: {e}")
        return 2
    cli.print(f"Resuming '{cp.task[:80]}' from iteration {cp.iteration}")

    # Choose the model: an explicit --model wins; otherwise reuse the model the
    # checkpoint was created with so the run continues on the same model. Warn
    # loudly if the two disagree (a different model may not complete the task
    # coherently). Fall back to a small local model only if nothing is known.
    saved_model = getattr(cp, "model", "") or ""
    if args.model:
        chosen_model = args.model
        if saved_model and saved_model != args.model:
            cli.print(
                f"Warning: checkpoint was created with '{saved_model}' but resuming "
                f"with '{args.model}'. Results may differ."
            )
    elif saved_model:
        chosen_model = saved_model
        cli.print(f"Using checkpoint's model: {saved_model}")
    else:
        chosen_model = "Qwen/Qwen2.5-1.5B-Instruct"
        cli.print(
            "This checkpoint did not record a model; resuming on a small local "
            f"model ({chosen_model}). Pass -m/--model to choose another."
        )

    try:
        if getattr(args, 'preset', None):
            from effgen.presets import create_agent as _create_preset_agent
            agent = _create_preset_agent(args.preset, chosen_model)
        else:
            cfg = AgentConfig(
                name=cp.agent_name, model=chosen_model, tools=[],
                raise_on_error=False,
            )
            agent = Agent(cfg)
    except Exception as e:  # noqa: BLE001 - surface a clean error, no stack trace
        cli.print(f"Error: could not load model '{chosen_model}' to resume: {e}")
        return 1

    try:
        response = agent.resume(checkpoint_id=cp_id, checkpoint_dir=ckpt_dir)
        cli.print_data(response.output if hasattr(response, 'output') else str(response))
        return 0 if getattr(response, 'success', True) else 1
    finally:
        # Release the agent so resume never emits the "garbage-collected
        # without calling close()" warning (matches the run path).
        try:
            agent.close()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Agent close after resume failed: {e}")
