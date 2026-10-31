"""The ``effgen sessions`` and ``effgen runs`` command groups.

``_main`` parses arguments and dispatches; it imports the two handlers at module
scope and re-exports them (tests reach them as ``_main._handle_sessions_command``
/ ``_main._handle_runs_command`` / ``_main._fmt_cost``). Holds the session and
run-history store reads, the table rendering, and the shared formatting helpers.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from effgen.cli.commands._shared import _invoked_command, _print_group_help
from effgen.ui.render import json_ensure_ascii
from effgen.ui.tables import render_table


def _short_ts(value: Any) -> str:
    """Render a stored ISO timestamp as 'YYYY-MM-DD HH:MM' for a table cell."""
    if not value:
        return "—"
    text = str(value)
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text[:16]


def _fmt_cost(value: Any) -> str:
    """Format a cost in USD, or '—' when the run was not priced.

    Sub-cent amounts keep enough precision to stay distinguishable from a free
    run, so a priced turn never reads as ``$0.0000``.
    """
    if not isinstance(value, int | float):
        return "—"
    if value == 0:
        return "$0.00"
    if value < 0.0001:
        return f"${value:.6f}"
    return f"${value:.4f}" if value < 1 else f"${value:.2f}"


def _one_line(text: Any, width: int = 60) -> str:
    """Collapse text to a single line no wider than *width*."""
    if text is None:
        return "—"
    flat = " ".join(str(text).split())
    if not flat:
        return "—"
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _session_matches(entry: dict, mgr, *, search: str | None, model: str | None,
                     since: str | None, until: str | None) -> bool:
    """Whether a session summary passes the list filters."""
    updated = str(entry.get("updated_at") or "")
    if since and updated[:10] < since:
        return False
    if until and updated[:10] > until:
        return False
    if model and model.lower() not in str(entry.get("model") or "").lower():
        return False
    if search:
        needle = search.lower()
        if needle in str(entry.get("session_id", "")).lower() or \
           needle in str(entry.get("agent_name", "")).lower():
            return True
        try:
            session = mgr.get(entry["session_id"])
        except Exception:  # noqa: BLE001 - an unreadable file simply cannot match
            return False
        return any(
            needle in str(m.get("content", "")).lower() for m in session.messages
        )
    return True


def _render_session(session, cli, *, last: int | None = None) -> None:
    """Print a conversation with speaker labels and per-turn model/cost."""
    messages = session.messages
    shown = messages[-last:] if last and last > 0 else messages
    cli.print_line(f"Session: {session.session_id}")
    if session.agent_name:
        cli.print_line(f"Agent:   {session.agent_name}")
    if session.metadata.get("model"):
        cli.print_line(f"Model:   {session.metadata['model']}")
    cli.print_line(
        f"Turns:   {len(messages)} message(s)"
        + (f" (showing last {len(shown)})" if len(shown) != len(messages) else "")
    )
    cli.print("")
    for m in shown:
        meta = m.get("metadata") or {}
        role = str(m.get("role", "?"))
        stamp = _short_ts(m.get("timestamp"))
        detail = []
        if meta.get("model"):
            detail.append(str(meta["model"]))
        tokens = meta.get("tokens_used")
        if tokens:
            detail.append(f"{tokens} tok")
        if isinstance(meta.get("cost_usd"), int | float):
            detail.append(_fmt_cost(meta["cost_usd"]))
        if isinstance(meta.get("latency_ms"), int | float):
            detail.append(f"{meta['latency_ms']:.0f} ms")
        suffix = ("  ·  " + "  ".join(detail)) if detail else ""
        cli.print_line(f"--- {role} · {stamp}{suffix}", style="dim")
        cli.print_data(str(m.get("content", "")))
        cli.print("")


def _browse_sessions(sessions: list[dict], mgr, cli) -> int:
    """Offer the listed sessions for selection and read the chosen one.

    On a non-interactive stdin the list is all that is printed, so the command
    stays usable when piped or run from a script.
    """
    if not sessions:
        return 0
    if not sys.stdin.isatty():
        cli.print("Read one with: effgen sessions show <id>")
        return 0
    cli.print("")
    try:
        choice = input("Open which session? (number, id, or Enter to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        cli.print("")
        return 0
    if not choice:
        return 0
    session_id = None
    if choice.isdigit() and 1 <= int(choice) <= len(sessions):
        session_id = sessions[int(choice) - 1]["session_id"]
    else:
        session_id = choice
    try:
        session = mgr.get(session_id)
    except FileNotFoundError:
        cli.print_line(f"Session not found: {session_id}")
        return 1
    cli.print("")
    _render_session(session, cli)
    cli.print_line(f"Continue this conversation: effgen chat --session-id {session.session_id}")
    return 0


def _handle_runs_command(args, cli) -> int:
    """Handle 'effgen runs' subcommands over the run history store."""
    import json as _json

    from effgen.observability import run_log

    cmd = getattr(args, 'runs_command', None)
    if cmd == 'list':
        status = getattr(args, 'status', None)
        if status == 'failed':
            status = 'error'
        runs = run_log.read_runs(
            limit=max(int(getattr(args, 'limit', 20) or 20), 1),
            status=status,
            model=getattr(args, 'model', None),
            search=getattr(args, 'search', None),
            since=getattr(args, 'since', None),
            until=getattr(args, 'until', None),
            session_id=getattr(args, 'session_filter', None),
        )
        if getattr(args, 'output_json', False):
            print(_json.dumps({
                "runs": runs,
                "runs_dir": str(run_log.history_dir()),
                "persisted": run_log.history_enabled(),
            }, indent=2, default=str, ensure_ascii=json_ensure_ascii()))
            return 0
        if not runs:
            filtered = any((
                status, getattr(args, 'model', None), getattr(args, 'search', None),
                getattr(args, 'since', None), getattr(args, 'until', None),
                getattr(args, 'session_filter', None),
            ))
            if filtered:
                cli.print(
                    "No runs match those filters. Widen them, or list everything "
                    "with: effgen runs list"
                )
            else:
                cli.print("No runs recorded yet. Run an agent (effgen run \"...\") and try again.")
            if not run_log.history_enabled():
                cli.print(
                    "Run history is disabled (EFFGEN_RUN_HISTORY=0), so only runs "
                    "from this process are visible."
                )
            return 0
        render_table(
            columns=["Run", "When", "Model", "Task", "Cost", "Time", "Status"],
            rows=[
                [
                    r.get("run_id") or "—",
                    _short_ts(r.get("ts")),
                    _one_line(r.get("model"), 24),
                    _one_line(r.get("task"), 40),
                    _fmt_cost(r.get("cost_usd")),
                    f"{r['duration_s']:.1f}s" if isinstance(r.get("duration_s"), int | float) else "—",
                    r.get("status") or ("error" if r.get("error") else "ok"),
                ]
                for r in runs
            ],
            console=cli.console,
            justify=["left", "left", "left", "left", "right", "right", "left"],
            styles=["cyan", None, None, None, None, None, "yellow"],
            caption=f"Stored in: {run_log.history_dir()}",
        )
        cli.print("Open one with: effgen runs show <run-id>")
        return 0
    if cmd == 'show':
        record = run_log.get_run(args.run_id)
        if record is None:
            cli.print_line(f"Run not found: {args.run_id}")
            cli.print("List the stored runs with: effgen runs list")
            return 1
        card_path = getattr(args, 'card', None)
        if card_path:
            from effgen.ui.report_html import ReportError, write_html_report
            try:
                written = write_html_report(
                    card_path, record, kind="run", command=_invoked_command(),
                )
            except (ReportError, OSError) as exc:
                cli.print_error(f"--card: could not write {card_path}: {exc}")
                return 1
            if not getattr(args, 'output_json', False):
                cli.print(f"Summary card written to {written}")
        if getattr(args, 'output_json', False):
            print(_json.dumps(record, indent=2, default=str, ensure_ascii=json_ensure_ascii()))
            return 0
        cli.print_line(f"Run:      {record.get('run_id') or '—'}")
        cli.print_line(f"When:     {record.get('ts') or '—'}")
        cli.print_line(f"Status:   {record.get('status') or '—'}")
        if record.get("stop_reason"):
            cli.print_line(f"Ended:    {record['stop_reason']}")
        cli.print_line(f"Model:    {record.get('model') or '—'}"
                       + (f" ({record['provider']})" if record.get("provider") else ""))
        if record.get("agent"):
            cli.print_line(f"Agent:    {record['agent']}")
        if record.get("session_id"):
            cli.print_line(f"Session:  {record['session_id']}"
                           f"   (read it with: effgen sessions show {record['session_id']})")
        tokens = f"{record.get('input_tokens') or 0} in / {record.get('output_tokens') or 0} out"
        cli.print_line(f"Tokens:   {tokens}")
        cli.print_line(f"Cost:     {_fmt_cost(record.get('cost_usd'))}")
        if isinstance(record.get("duration_s"), int | float):
            cli.print_line(f"Duration: {record['duration_s']:.2f}s")
        if record.get("task"):
            cli.print("\nTask:")
            cli.print_data(str(record["task"]))
        if record.get("output"):
            cli.print("\nAnswer:")
            cli.print_data(str(record["output"]))
        if record.get("error"):
            cli.print("\nError:")
            cli.print_data(str(record["error"]))
        return 0
    if cmd == 'cleanup':
        removed = run_log.cleanup_runs(older_than_days=args.days)
        cli.print(f"Removed {removed} run history file(s).")
        return 0
    return _print_group_help(args)


def _handle_sessions_command(args, cli) -> int:
    """Handle 'effgen sessions' subcommands."""
    import json as _json

    from effgen.core.session import SessionManager
    from effgen.errors import CorruptStateError
    mgr = SessionManager()
    cmd = getattr(args, 'session_command', None)
    if cmd in ('list', 'browse'):
        sessions, unreadable = mgr.scan()
        limit = getattr(args, 'limit', None)
        if cmd == 'list':
            sessions = [
                s for s in sessions
                if _session_matches(
                    s, mgr,
                    search=getattr(args, 'search', None),
                    model=getattr(args, 'model', None),
                    since=getattr(args, 'since', None),
                    until=getattr(args, 'until', None),
                )
            ]
        if limit and limit > 0:
            sessions = sessions[:limit]
        if getattr(args, 'output_json', False):
            print(_json.dumps({
                "sessions": sessions,
                "unreadable": unreadable,
                "sessions_dir": str(mgr.sessions_dir),
            }, indent=2, default=str, ensure_ascii=json_ensure_ascii()))
            return 0
        if not sessions and not unreadable:
            cli.print(
                "No sessions yet. Start one with: effgen chat  (or effgen run \"...\" "
                "creates a session you can resume)."
            )
            return 0
        if sessions:
            render_table(
                columns=["#", "Session", "Messages", "Model", "Cost", "Updated"],
                rows=[
                    [
                        i,
                        s['session_id'],
                        s['messages'],
                        _one_line(s.get('model'), 28),
                        _fmt_cost(s.get('cost_usd')),
                        _short_ts(s.get('updated_at')),
                    ]
                    for i, s in enumerate(sessions, start=1)
                ],
                console=cli.console,
                justify=["right", "left", "right", "left", "right", "left"],
                styles=[None, "cyan", "yellow", None, None, None],
                caption=f"Stored in: {mgr.sessions_dir}",
            )
        if unreadable:
            names = ", ".join(u["file"] for u in unreadable)
            cli.print_warning(
                f"{len(unreadable)} session file(s) could not be read and are not "
                f"listed: {names}"
            )
        if cmd == 'browse':
            return _browse_sessions(sessions, mgr, cli)
        cli.print("Read one:  effgen sessions show <id>")
        cli.print("Continue:  effgen chat --session-id <id>")
        return 0
    if cmd == 'show':
        try:
            session = mgr.get(args.session_id)
        except FileNotFoundError:
            cli.print_line(f"Session not found: {args.session_id}")
            cli.print("List the stored sessions with: effgen sessions list")
            return 1
        except CorruptStateError as e:
            cli.print(f"Error: {e}")
            return 2
        if getattr(args, 'output_json', False):
            data = session.to_dict()
            last = getattr(args, 'last', None)
            if last and last > 0:
                data["messages"] = data["messages"][-last:]
            print(_json.dumps(data, indent=2, default=str, ensure_ascii=json_ensure_ascii()))
            return 0
        _render_session(session, cli, last=getattr(args, 'last', None))
        cli.print_line(f"Continue this conversation: effgen chat --session-id {session.session_id}")
        return 0
    if cmd == 'delete':
        ok = mgr.delete(args.session_id)
        cli.print("Deleted." if ok else f"Session not found: {args.session_id}")
        return 0 if ok else 1
    if cmd == 'export':
        try:
            cli.print_data(mgr.export(args.session_id, format=args.format))
        except FileNotFoundError:
            cli.print_line(f"Session not found: {args.session_id}")
            return 1
        except CorruptStateError as e:
            cli.print(f"Error: {e}")
            return 2
        return 0
    if cmd == 'cleanup':
        n = mgr.cleanup(older_than_days=args.days)
        cli.print(f"Removed {n} old session(s).")
        return 0
    return _print_group_help(args)
