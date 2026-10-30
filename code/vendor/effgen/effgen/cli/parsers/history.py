"""Argument declarations for the saved-conversation and run-history commands.

``effgen.cli._main.create_parser`` calls these in the order the top-level
``--help`` lists them; each attaches one command to the given subparsers
action and returns nothing.
"""

from __future__ import annotations

import argparse


def add_sessions_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen sessions`` and its list/show/browse/delete/export/cleanup."""
    sessions_parser = subparsers.add_parser(
        'sessions', help='Browse and manage saved conversations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen sessions list --search refund --since 2026-07-01\n"
            "  effgen sessions show support-4471 --last 4\n"
            "  effgen sessions browse\n"
            "\n"
            "Continue a saved conversation with `effgen chat --session-id <id>` or\n"
            "`effgen run \"...\" --session-id <id>`. (`effgen resume` replays a\n"
            "workflow checkpoint, which is a different store.)\n"
        ),
    )
    sessions_parser.set_defaults(_group_parser=sessions_parser)
    sessions_subparsers = sessions_parser.add_subparsers(dest='session_command', help='Sessions command')
    _sessions_list = sessions_subparsers.add_parser('list', help='List sessions')
    _sessions_list.add_argument('--json', dest='output_json', action='store_true',
                                help='Output the session list as JSON')
    _sessions_list.add_argument('--search', help='Only sessions whose id, agent or messages match this text')
    _sessions_list.add_argument('--since', help='Only sessions updated on/after this date (YYYY-MM-DD)')
    _sessions_list.add_argument('--until', help='Only sessions updated on/before this date (YYYY-MM-DD)')
    _sessions_list.add_argument('-m', '--model',
                                help='Only sessions answered by a model matching this text')
    _sessions_list.add_argument('--limit', type=int, default=50, help='Maximum sessions to show (default: 50)')
    ss = sessions_subparsers.add_parser(
        'show', help='Read a conversation turn by turn',
    )
    ss.add_argument('session_id', help='Session id')
    ss.add_argument('--last', type=int, help='Show only the last N messages')
    ss.add_argument('--json', dest='output_json', action='store_true',
                    help='Output the conversation as JSON')
    sb = sessions_subparsers.add_parser(
        'browse', help='Pick a session from the list and read it',
    )
    sb.add_argument('--json', dest='output_json', action='store_true',
                    help='Output the session list as JSON instead of prompting')
    sb.add_argument('--limit', type=int, default=20, help='Sessions to offer (default: 20)')
    sd = sessions_subparsers.add_parser('delete', help='Delete a session')
    sd.add_argument('session_id', help='Session id')
    se = sessions_subparsers.add_parser('export', help='Export a session')
    se.add_argument('session_id', help='Session id')
    se.add_argument('--format', choices=['json', 'text'], default='json',
                    help='Export format (default: json)')
    sc = sessions_subparsers.add_parser('cleanup', help='Delete sessions older than N days')
    sc.add_argument('--days', type=int, default=30,
                    help='Delete sessions older than this many days (default: 30)')

def add_runs_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen runs`` and its list/show/cleanup subcommands."""
    runs_parser = subparsers.add_parser(
        'runs', help='Browse agent run history',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen runs list --status failed --since 2026-07-17\n"
            "  effgen runs list --model gpt-5-nano --json\n"
            "  effgen runs show 3f9a1c2b\n"
            "\n"
            "Runs are appended to $EFFGEN_HOME/runs (default ~/.effgen/runs) as\n"
            "one JSONL file per day. Set EFFGEN_RUN_HISTORY=0 to keep history in\n"
            "memory only; EFFGEN_RUN_HISTORY_MAX_DAYS sets retention (default 30).\n"
        ),
    )
    runs_parser.set_defaults(_group_parser=runs_parser)
    runs_subparsers = runs_parser.add_subparsers(dest='runs_command', help='Runs command')
    rl = runs_subparsers.add_parser('list', help='List recent runs')
    rl.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    rl.add_argument('--status', choices=['ok', 'stopped', 'error', 'failed'],
                    help="Only runs with this status ('failed' is an alias for "
                         "'error'; 'stopped' is a run the loop ended before the "
                         "model wrote an answer)")
    rl.add_argument('-m', '--model', help='Only runs on a model matching this text')
    rl.add_argument('--search', help='Only runs whose task, answer, id or error match this text')
    rl.add_argument('--session-id', dest='session_filter', help='Only runs from this session')
    rl.add_argument('--since', help='Only runs on/after this date (YYYY-MM-DD)')
    rl.add_argument('--until', help='Only runs on/before this date (YYYY-MM-DD)')
    rl.add_argument('--limit', type=int, default=20, help='Maximum runs to show (default: 20)')
    rs = runs_subparsers.add_parser('show', help='Show one run in full')
    rs.add_argument('run_id', help='Run id (as shown by `effgen runs list`)')
    rs.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    rs.add_argument('--card', metavar='PATH.html',
                    help='Write a summary HTML card for this stored run to PATH. '
                         'History keeps a truncated answer and no step trace, so '
                         'the card states that; use `effgen run --card` at run time '
                         'for the full answer, trace and sources.')
    rc = runs_subparsers.add_parser('cleanup', help='Delete run history older than N days')
    rc.add_argument('--days', type=int, default=30,
                    help='Delete runs older than this many days (default: 30)')
