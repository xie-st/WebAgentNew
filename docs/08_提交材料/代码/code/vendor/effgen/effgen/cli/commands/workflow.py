"""The ``effgen workflow`` command group: validate and run workflow DAGs.

``_main`` parses arguments and dispatches; it imports
:func:`_handle_workflow_command` at module scope and re-exports it. Holds the
YAML-to-DAG wiring, the per-node agent factory, and the run/validate output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from effgen.cli.commands._shared import _print_group_help
from effgen.ui.render import json_ensure_ascii

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def _handle_workflow_command(args, cli: "CLIInterface") -> int:
    """Handle the 'workflow' CLI subcommand."""
    from effgen.core.workflow import WorkflowDAG

    wf_cmd = getattr(args, 'workflow_command', None)
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True
    show_diagram = getattr(args, 'diagram', False)

    def _print_diagram(dag, node_results=None):
        from effgen.ui.workflow_viz import workflow_diagram_lines
        order = dag.topological_order()
        levels = dag._compute_levels(order)
        lines = workflow_diagram_lines(
            dag.name,
            [n.id for n in dag.nodes],
            [e.to_dict() for e in dag.edges],
            levels,
            node_results=node_results,
        )
        for style, text in lines:
            if cli.console and style:
                cli.console.print(f"[{style}]{text}[/{style}]", highlight=False)
            else:
                cli.print(text)

    if wf_cmd == 'validate':
        try:
            dag = WorkflowDAG.from_yaml(args.file)
            order = dag.topological_order()
            if json_mode:
                print(json.dumps({
                    "valid": True,
                    "name": dag.name,
                    "nodes": len(dag.nodes),
                    "edges": len(dag.edges),
                    "execution_order": order,
                }, indent=2, ensure_ascii=json_ensure_ascii()))
                return 0
            cli.print(f"Workflow '{dag.name}' is valid.")
            cli.print(f"  Nodes: {len(dag.nodes)}")
            cli.print(f"  Edges: {len(dag.edges)}")
            cli.print(f"  Execution order: {' -> '.join(order)}")
            if show_diagram:
                cli.print("")
                _print_diagram(dag)
            return 0
        except Exception as e:
            if json_mode:
                print(json.dumps({"valid": False, "error": str(e)}, indent=2, ensure_ascii=json_ensure_ascii()))
                return 1
            cli.print(f"Validation failed: {e}")
            return 1

    elif wf_cmd == 'run':
        try:
            model_name = getattr(args, 'model', None)

            def _agent_factory(nd):
                from effgen.core.agent import Agent, AgentConfig
                from effgen.models import load_model
                agent_field = nd.get('agent')
                explicit = model_name or nd.get('model')
                if explicit:
                    model = load_model(explicit)
                elif agent_field:
                    # No top-level -m/--model and no per-node 'model:' key: the
                    # node's 'agent:' value is the natural place a user sets a
                    # model id (e.g. `agent: gpt-5-nano`). Try it as one before
                    # falling back to the local default; a value that does not
                    # resolve to a real model fails loudly instead of silently
                    # running a different (free, local) model with no warning.
                    try:
                        model = load_model(agent_field)
                    except Exception as exc:
                        raise ValueError(
                            f"Workflow node '{nd['id']}' has agent: {agent_field!r}, "
                            f"which does not resolve to a model ({exc}). Set a "
                            "'model:' key on the node, or pass -m/--model, to "
                            "choose its model explicitly."
                        ) from exc
                else:
                    model = load_model('Qwen/Qwen2.5-1.5B-Instruct')
                # A node may name a preset (research/coding/general/...) to get a
                # ready-made tool-equipped agent; otherwise build a plain agent.
                preset = nd.get('preset')
                if preset:
                    from effgen.presets import create_agent
                    return create_agent(preset, model=model)
                config = AgentConfig(
                    name=agent_field or nd['id'],
                    raise_on_error=False,
                    model=model,
                    max_iterations=nd.get('max_iterations', 5),
                )
                return Agent(config)

            quiet = getattr(args, 'quiet', False)
            dag = WorkflowDAG.from_yaml(args.file, agent_factory=_agent_factory)
            if not quiet:
                cli.print(f"Running workflow '{dag.name}' ({len(dag.nodes)} nodes)...")

            # Per-node ``task:`` strings declared in the YAML become each node's
            # default input (so `effgen workflow run workflow.yaml` works with no
            # flags). --input / --task then override or supplement them.
            yaml_inputs: dict = {}
            for node in dag.nodes:
                node_task = node.metadata.get('task')
                if node_task:
                    yaml_inputs[node.id] = node_task

            bare_task = getattr(args, 'task', None)
            initial_inputs: dict | str = dict(yaml_inputs)
            if getattr(args, 'input', None):
                for node_id, task_str in args.input:
                    initial_inputs[node_id] = task_str
            if bare_task:
                if dag.entry_nodes():
                    for nid in dag.entry_nodes():
                        initial_inputs[nid] = bare_task
                elif not initial_inputs:
                    initial_inputs = bare_task

            try:
                result = dag.run(initial_inputs=initial_inputs)
            finally:
                # Release each node's agent so we don't leak handles / emit
                # garbage-collected-without-close warnings.
                for node in dag.nodes:
                    agent = getattr(node, "agent", None)
                    if agent is not None and hasattr(agent, "close"):
                        try:
                            agent.close()
                        except Exception:
                            pass

            if json_mode:
                print(json.dumps(result.to_dict(), indent=2, default=str, ensure_ascii=json_ensure_ascii()))
                return 0 if result.success else 1

            if not quiet:
                cli.print(f"\nWorkflow {'succeeded' if result.success else 'FAILED'} "
                          f"in {result.execution_time:.2f}s")

                if show_diagram:
                    cli.print("")
                    _print_diagram(dag, node_results=result.node_results)
                else:
                    for nr in result.node_results:
                        status = nr['status']
                        cli.print(f"  [{status:>9s}] {nr['id']} ({nr['execution_time']:.2f}s)")

                if result.success:
                    # Show final outputs
                    cli.print("\nOutputs:")
                    for key, val in result.outputs.items():
                        cli.print(f"  {key}: {str(val)[:200]}")

            return 0 if result.success else 1

        except Exception as e:
            if json_mode:
                print(json.dumps({"success": False, "error": str(e)}, indent=2, ensure_ascii=json_ensure_ascii()))
                return 1
            cli.print(f"Workflow execution failed: {e}")
            return 1

    else:
        return _print_group_help(args)
