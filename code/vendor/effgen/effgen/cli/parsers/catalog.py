"""Argument declarations for the commands that browse what effGen ships.

Configuration, tools, models, examples, presets and the prompt library.
``effgen.cli._main.create_parser`` calls these in the order the top-level
``--help`` lists them; each attaches one command to the given subparsers
action and returns nothing.
"""

from __future__ import annotations

import argparse


def add_config_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen config`` and its show/validate/init/set subcommands."""
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.set_defaults(_group_parser=config_parser)
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config command')

    config_show = config_subparsers.add_parser('show', help='Show configuration')
    config_show.add_argument('-f', '--file', help='Configuration file')

    config_validate = config_subparsers.add_parser('validate', help='Validate configuration')
    config_validate.add_argument('-f', '--file', required=True, help='Configuration file')

    config_init = config_subparsers.add_parser('init', help='Initialize new configuration')
    config_init.add_argument('-o', '--output', help='Output file')
    config_init.add_argument('--force', action='store_true', help='Overwrite existing file')

    config_set = config_subparsers.add_parser('set', help='Set a configuration value (e.g. budget.daily 1.0)')
    config_set.add_argument('key', help='Config key (e.g. budget.daily, budget.monthly)')
    config_set.add_argument('value', help='Config value')

def add_tools_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen tools`` and its list/info/test subcommands."""
    tools_parser = subparsers.add_parser('tools', help='Tool management')
    tools_parser.set_defaults(_group_parser=tools_parser)
    tools_subparsers = tools_parser.add_subparsers(dest='tool_command', help='Tools command')

    tools_list = tools_subparsers.add_parser('list', help='List tools')
    tools_list.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    tools_list.add_argument('--category', help='Filter by category')

    tools_info = tools_subparsers.add_parser('info', help='Show tool information')
    tools_info.add_argument('name', help='Tool name')

    tools_test = tools_subparsers.add_parser('test', help='Test a tool')
    tools_test.add_argument('name', help='Tool name')
    tools_test.add_argument('-i', '--input', help='Tool input (JSON or string)')

def add_models_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen models`` and its list/browse/info/load/unload/status/refresh."""
    models_parser = subparsers.add_parser('models', help='Model management')
    models_parser.set_defaults(_group_parser=models_parser)
    models_subparsers = models_parser.add_subparsers(dest='model_command', help='Models command')

    models_list = models_subparsers.add_parser('list', help='List models')
    models_list.add_argument('--provider', help='Show only this provider\'s models (full detail)')
    models_list.add_argument('--free', action='store_true', help='Show only free-tier models')
    models_list.add_argument('-t', '--tools', action='store_true',
                             help='Show only tool-capable models')
    models_list.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')

    models_browse = models_subparsers.add_parser(
        'browse',
        help='Browse every provider in one table — search, filter, sort, page')
    models_browse.add_argument(
        '--search', metavar='TEXT',
        help='Case-insensitive substring match on model id, family, or provider')
    models_browse.add_argument('--provider', help='Limit to one provider')
    models_browse.add_argument(
        '--free', action='store_true', help='Only free-tier models')
    models_browse.add_argument(
        '-t', '--tools', action='store_true', help='Only tool-calling models')
    models_browse.add_argument(
        '--vision', action='store_true', help='Only vision-capable models')
    models_browse.add_argument(
        '--audio', action='store_true', help='Only audio-capable models')
    models_browse.add_argument(
        '--min-context', type=int, metavar='N', dest='min_context',
        help='Only models with a context window of at least N tokens')
    models_browse.add_argument(
        '--max-price-in', type=float, metavar='USD', dest='max_price_in',
        help='Only models whose input price ($/1M) is at most USD')
    models_browse.add_argument(
        '--max-price-out', type=float, metavar='USD', dest='max_price_out',
        help='Only models whose output price ($/1M) is at most USD')
    models_browse.add_argument(
        '--sort', choices=['provider', 'id', 'context', 'max-out',
                           'price-in', 'price-out'],
        default='provider',
        help='Sort order (default: provider then id)')
    models_browse.add_argument(
        '--desc', action='store_true', help='Sort in descending order')
    models_browse.add_argument(
        '--limit', type=int, metavar='N', help='Show at most N rows')
    models_browse.add_argument(
        '--offset', type=int, default=0, metavar='N',
        help='Skip the first N rows (paging)')
    models_browse.add_argument(
        '--include-local', action='store_true', dest='include_local',
        help='Also list models downloaded in the local HuggingFace cache')
    models_browse.add_argument(
        '--json', dest='output_json', action='store_true', help='Output as JSON')

    models_info = models_subparsers.add_parser('info', help='Show model information')
    models_info.add_argument('name', help='Model name (e.g. gpt-5-nano or openai:gpt-5-nano)')
    models_info.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')

    models_load = models_subparsers.add_parser('load', help='Pre-load a model into memory')
    models_load.add_argument('name', help='Model name (e.g. Qwen/Qwen2.5-1.5B-Instruct)')
    models_load.add_argument('-e', '--engine', help='Engine (vllm, transformers)', default=None)

    models_unload = models_subparsers.add_parser('unload', help='Unload a model from memory')
    models_unload.add_argument('name', help='Model name')

    models_status = models_subparsers.add_parser('status', help='Show loaded models and GPU memory status')
    models_status.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')

    models_refresh = models_subparsers.add_parser(
        'refresh', help="Refresh the model catalog from each provider's live API")
    models_refresh.add_argument(
        '--provider', default=None,
        help='Only refresh this provider (default: all providers with a key)')
    models_refresh.add_argument(
        '--dry-run', action='store_true',
        help='Show what would change without writing the snapshot')

def add_examples_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen examples`` and its list/run subcommands."""
    examples_parser = subparsers.add_parser('examples', help='Run example scripts')
    examples_parser.set_defaults(_group_parser=examples_parser)
    examples_subparsers = examples_parser.add_subparsers(dest='example_command', help='Examples command')

    examples_subparsers.add_parser('list', help='List examples')

    examples_run = examples_subparsers.add_parser('run', help='Run an example')
    examples_run.add_argument('name', help='Example name')

def add_presets_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen presets`` — list the agent presets."""
    presets_parser = subparsers.add_parser('presets', help='List available agent presets')
    presets_parser.add_argument('--json', dest='output_json', action='store_true',
                                help='Output the preset list as JSON')

def add_prompts_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen prompts`` and its list/show/eval/playground/render/run."""
    prompts_parser = subparsers.add_parser('prompts', help='Prompt library management')
    prompts_subparsers = prompts_parser.add_subparsers(dest='prompts_command', help='Prompts command')

    prompts_list = prompts_subparsers.add_parser('list', help='List prompt templates')
    prompts_list.add_argument('--domain', help='Filter by domain')
    prompts_list.add_argument('--variant', help='Filter by variant')
    prompts_list.add_argument('--format', choices=['table', 'json', 'markdown'], default='table',
                              dest='list_format', help='Output format')
    prompts_list.add_argument('--json', action='store_const', const='json', dest='list_format',
                              help='Shorthand for --format json (consistent with `models list`, `cost`).')

    prompts_show = prompts_subparsers.add_parser('show', help='Show prompt details')
    prompts_show.add_argument('name', help='Prompt name')

    prompts_eval = prompts_subparsers.add_parser('eval', help='Evaluate prompts')
    prompts_eval.add_argument('--domain', help='Evaluate only this domain')
    prompts_eval.add_argument('--live', action='store_true', help='Run live model evaluation')
    prompts_eval.add_argument('-m', '--model', help='Model to use for live evaluation')
    prompts_eval.add_argument('--delay', type=float, default=35.0,
                              help='Seconds to wait between live model calls (default: 35)')
    prompts_eval.add_argument('-o', '--output', help='Write eval table to this file')
    prompts_eval.add_argument('--fail-under', type=float, default=None, metavar='FRACTION',
                              help='Exit non-zero if the pass rate is below this fraction '
                                   '(0.0-1.0). Without it, any failing eval exits non-zero.')

    # Playground subcommands
    prompts_subparsers.add_parser('playground', help='Launch interactive prompt playground REPL')

    prompts_render = prompts_subparsers.add_parser('render', help='Non-interactive: render a prompt to stdout')
    prompts_render.add_argument('prompt_name', metavar='name', help='Prompt name (e.g. research.literature_review.v1)')
    prompts_render.add_argument('-i', '--input', dest='input_file', metavar='FILE',
                                help="JSON file with input variables, validated against the prompt's "
                                     "input_schema (see 'prompts show <name>'); omit to render the fixture")

    prompts_run = prompts_subparsers.add_parser('run', help='Non-interactive: render + run through a model')
    prompts_run.add_argument('prompt_name', metavar='name', help='Prompt name')
    prompts_run.add_argument('-i', '--input', dest='input_file', metavar='FILE',
                             help="JSON file with input variables, validated against the prompt's "
                                  "input_schema (see 'prompts show <name>'); omit to render the fixture")
    prompts_run.add_argument('-m', '--model', required=True, help='Model identifier to run against')
    prompts_run.add_argument('--max-tokens', type=int, default=None,
                             help='Completion token cap for this run (raise it when a reasoning '
                                  'or structured prompt returns empty/truncated output)')
    prompts_run.add_argument('--temperature', type=float, default=None,
                             help='Sampling temperature for this run')
