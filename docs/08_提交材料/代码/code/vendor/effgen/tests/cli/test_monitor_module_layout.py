"""Layout of the ``effgen.cli.monitor`` module split.

The monitor module keeps the command entry point and the interval bounds it
validates against, and composes four siblings: ``monitor_collect`` reads the
sources into one snapshot document, ``monitor_format`` holds the scope labels and
the value formatters, ``monitor_panels`` prints the static labelled tables a pipe
gets, and ``monitor_live`` builds the full-screen layout and drives the refresh
loop. Both of the last two render "panels" from the same snapshot, which is the
likeliest thing to misread, so the split names them apart and the checks here
keep the names apart too.

Three invariants make that split safe to read and to extend:

* ``effgen.cli.monitor`` stays the import path for every name it has ever
  exposed, whichever module now defines it;
* each tunable and each scope label reaches the facade as the *same object* its
  owner defines, so nothing can drift into a second copy;
* the siblings never import the monitor module, and the edges between them run
  one way, so the package's import graph stays acyclic.

Two seams are asserted through the thing that uses them rather than through an
import, because they fail quietly: ``effgen top``'s advertised argparse defaults
are read back off a parser built by ``create_parser()``, and the dispatch arm is
read out of ``effgen.cli._main``'s source.

``effgen.cli.battle`` defines a private ``_run_live`` of its own. It is unrelated
to :func:`effgen.cli.monitor.run_live` and must not be conflated with it when
grepping for either.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import effgen.cli.monitor as monitor_mod
import effgen.cli.monitor_collect as monitor_collect
import effgen.cli.monitor_format as monitor_format
import effgen.cli.monitor_live as monitor_live
import effgen.cli.monitor_panels as monitor_panels

# Names that moved out of the monitor module, and the module that owns each.
OWNERS = {
    # where the numbers come from
    "resolve_server_url": "effgen.cli.monitor_collect",
    "resolve_api_key": "effgen.cli.monitor_collect",
    "_fetch_json": "effgen.cli.monitor_collect",
    "_collect_activity": "effgen.cli.monitor_collect",
    "_collect_server": "effgen.cli.monitor_collect",
    "_collect_spend": "effgen.cli.monitor_collect",
    "_collect_gpu": "effgen.cli.monitor_collect",
    "collect_snapshot": "effgen.cli.monitor_collect",
    "_effgen_version": "effgen.cli.monitor_collect",
    # how a value is rendered, whichever path renders it
    "_preview": "effgen.cli.monitor_format",
    "_fmt_usd": "effgen.cli.monitor_format",
    "_fmt_seconds": "effgen.cli.monitor_format",
    "_fmt_pct": "effgen.cli.monitor_format",
    "_fmt_clock": "effgen.cli.monitor_format",
    "_status_style": "effgen.cli.monitor_format",
    "_health_style": "effgen.cli.monitor_format",
    # the static tables a pipe or --once gets
    "render_snapshot": "effgen.cli.monitor_panels",
    "_emit": "effgen.cli.monitor_panels",
    "_unavailable": "effgen.cli.monitor_panels",
    "_render_activity": "effgen.cli.monitor_panels",
    "_render_activity_table": "effgen.cli.monitor_panels",
    "_render_traffic_table": "effgen.cli.monitor_panels",
    "_render_by_model_table": "effgen.cli.monitor_panels",
    "_render_spend_table": "effgen.cli.monitor_panels",
    "_render_gpu_table": "effgen.cli.monitor_panels",
    # the full-screen view and the loop that refreshes it
    "_build_live_view": "effgen.cli.monitor_live",
    "_QuitKeyReader": "effgen.cli.monitor_live",
    "run_live": "effgen.cli.monitor_live",
}

# What the monitor module itself must keep: the command body and the bounds it
# validates an --interval against, which nothing else reads.
KEPT = ("run_monitor_command",)
KEPT_CONSTANTS = ("DEFAULT_INTERVAL", "MIN_INTERVAL", "MAX_INTERVAL")

#: Constants that moved, and the module that owns each. Every one is checked for
#: object identity against the facade, not equality: two equal copies of
#: ``DEFAULT_ACTIVITY_LIMIT`` would keep every assertion passing while the CLI
#: advertised one value and the collector used the other.
MOVED_CONSTANTS = {
    "DEFAULT_SERVER_URL": monitor_collect,
    "DEFAULT_ACTIVITY_LIMIT": monitor_collect,
    "BURN_WINDOW_S": monitor_collect,
    "_HTTP_TIMEOUT_S": monitor_collect,
    "GPU_SAMPLE_INTERVAL_S": monitor_live,
    "SCOPE_ACTIVITY": monitor_format,
    "SCOPE_TRAFFIC": monitor_format,
    "SCOPE_HTTP_STATUS": monitor_format,
    "SCOPE_SPEND": monitor_format,
    "SCOPE_GPU": monitor_format,
    "_STATUS_STYLES": monitor_format,
}

#: Every attribute ``effgen.cli.monitor`` exposed before the split. The set may
#: only grow: a caller reading any of these must keep working.
BASELINE_MODULE_NAMES = frozenset(
    (
        "Any", "BURN_WINDOW_S", "DEFAULT_ACTIVITY_LIMIT", "DEFAULT_INTERVAL",
        "DEFAULT_SERVER_URL", "GPU_SAMPLE_INTERVAL_S", "MAX_INTERVAL", "MIN_INTERVAL",
        "SCOPE_ACTIVITY", "SCOPE_GPU", "SCOPE_HTTP_STATUS", "SCOPE_SPEND",
        "SCOPE_TRAFFIC", "_HTTP_TIMEOUT_S", "_QuitKeyReader", "_STATUS_STYLES",
        "__builtins__", "__cached__", "__doc__", "__file__", "__loader__",
        "__name__", "__package__", "__spec__", "_build_live_view",
        "_collect_activity", "_collect_gpu", "_collect_server", "_collect_spend",
        "_effgen_version", "_emit", "_fetch_json", "_fmt_clock", "_fmt_pct",
        "_fmt_seconds", "_fmt_usd", "_health_style", "_preview", "_render_activity",
        "_render_activity_table", "_render_by_model_table", "_render_gpu_table",
        "_render_spend_table", "_render_traffic_table", "_status_style",
        "_unavailable", "annotations", "collect_snapshot", "color_enabled",
        "console_is_interactive", "datetime", "get_console", "http", "json",
        "json_ensure_ascii", "os", "render_snapshot", "render_table",
        "resolve_api_key", "resolve_server_url", "run_live", "run_monitor_command",
        "select", "socket", "sys", "time", "timezone", "urllib",
    )
)

NEW_MODULES = (monitor_collect, monitor_format, monitor_live, monitor_panels)
SPLIT_MODULES = (monitor_mod, *NEW_MODULES)

#: The one-way edges between the siblings. ``monitor_format`` is the leaf;
#: ``monitor_panels`` reads only formatters; ``monitor_live`` also needs the
#: collectors, because it re-samples the GPU on its own slower cadence.
ALLOWED_SIBLING_EDGES = {
    "effgen.cli.monitor_format": frozenset(),
    "effgen.cli.monitor_collect": frozenset({"effgen.cli.monitor_format"}),
    "effgen.cli.monitor_panels": frozenset({"effgen.cli.monitor_format"}),
    "effgen.cli.monitor_live": frozenset(
        {"effgen.cli.monitor_format", "effgen.cli.monitor_collect"}
    ),
}

#: No module in the split may grow past this, or the split stops paying for
#: itself.
MAX_MODULE_LINES = 500


@pytest.mark.parametrize(("name", "module"), sorted(OWNERS.items()))
def test_moved_names_are_owned_by_one_module(name: str, module: str) -> None:
    owner = getattr(monitor_mod, name)
    assert owner.__module__ == module
    assert getattr(__import__(module, fromlist=["x"]), name) is owner


@pytest.mark.parametrize("name", KEPT)
def test_the_command_entry_point_stays_in_the_monitor_module(name: str) -> None:
    assert getattr(monitor_mod, name).__module__ == "effgen.cli.monitor"


@pytest.mark.parametrize("name", KEPT_CONSTANTS)
def test_the_interval_bounds_are_defined_in_the_monitor_module(name: str) -> None:
    """Only ``run_monitor_command`` reads these, so they stay beside it."""
    src = inspect.getsource(monitor_mod)
    assert f"\n{name} = " in src, f"{name} is no longer defined in monitor.py"
    for module in NEW_MODULES:
        assert not hasattr(module, name), f"{module.__name__} defines {name} too"


@pytest.mark.parametrize("name", sorted(MOVED_CONSTANTS))
def test_moved_constants_are_shared_not_copied(name: str) -> None:
    assert getattr(monitor_mod, name) is getattr(MOVED_CONSTANTS[name], name)


def test_the_two_server_scopes_have_not_collapsed_into_one() -> None:
    """The generation counters and the all-routes histogram count different things.

    They are labelled separately so nobody tries to reconcile them; a split that
    folded the two strings together would make the Traffic panel claim they do.
    """
    assert monitor_mod.SCOPE_HTTP_STATUS != monitor_mod.SCOPE_TRAFFIC


def test_the_module_name_set_only_grows() -> None:
    """No name this module ever exposed may stop being an attribute of it."""
    dropped = sorted(n for n in BASELINE_MODULE_NAMES if not hasattr(monitor_mod, n))
    assert not dropped, f"no longer attributes of effgen.cli.monitor: {dropped}"


def _module_scope_imports(module) -> list[str]:
    """The dotted names *module* imports at module scope, however it spells them.

    Only top-level statements are read, so an import inside a function — which
    costs nothing at import time and cannot close a cycle — is not reported.
    Every spelling is reduced to the module it names: ``import a.b``,
    ``import a.b as c``, ``from a.b import c`` and ``from a import b`` all report
    ``a.b``.
    """
    import ast

    package = module.__name__.rpartition(".")[0]
    tree = ast.parse(inspect.getsource(module))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = f"{package}.{base}" if base else package
            imported.append(base)
            imported.extend(f"{base}.{alias.name}" for alias in node.names)
    return imported


@pytest.mark.parametrize("module", NEW_MODULES, ids=lambda m: m.__name__)
def test_new_modules_do_not_import_the_monitor_module(module) -> None:
    """Keeps the dependency graph acyclic: monitor imports the four, never the reverse.

    The check reads the import statements rather than matching text, so
    ``from . import monitor``, ``from effgen.cli import monitor`` and
    ``import effgen.cli.monitor`` are caught alongside the ``from ... import``
    form. Any of them binds a half-built module during package import and
    resolves only by the order the package happens to load in.
    """
    offenders = [
        name
        for name in _module_scope_imports(module)
        if name == "effgen.cli.monitor" or name.startswith("effgen.cli.monitor.")
    ]
    assert not offenders, (
        f"{module.__name__} imports the monitor module at module scope: {offenders}"
    )


@pytest.mark.parametrize("module", NEW_MODULES, ids=lambda m: m.__name__)
def test_the_sibling_import_edges_run_one_way(module) -> None:
    siblings = {m.__name__ for m in NEW_MODULES} - {module.__name__}
    reached = {
        name if name in siblings else name.rsplit(".", 1)[0]
        for name in _module_scope_imports(module)
        if name in siblings or name.rsplit(".", 1)[0] in siblings
    }
    offenders = sorted(reached - ALLOWED_SIBLING_EDGES[module.__name__])
    assert not offenders, f"{module.__name__} reaches a sibling it should not: {offenders}"


@pytest.mark.parametrize("module", NEW_MODULES, ids=lambda m: m.__name__)
def test_new_modules_keep_the_heavy_imports_function_local(module) -> None:
    """Module scope stays free of the imports that would cost or cycle.

    ``effgen.cli._main`` at module scope would close a cycle back through the
    dispatch arm that imports this package; ``rich`` at module scope would break
    an install without it, and only the live view needs it at all.
    """
    forbidden = ("effgen.cli._main", "rich")
    offenders = [
        name
        for name in _module_scope_imports(module)
        if any(name == f or name.startswith(f + ".") for f in forbidden)
    ]
    assert not offenders, f"{offenders} must stay inside the function that uses them"


@pytest.mark.parametrize("command", ["top", "monitor"])
def test_effgen_top_still_advertises_the_module_defaults(command: str) -> None:
    """The argparse defaults, read off a real parser rather than off the import.

    ``effgen.cli.parsers.ops`` reads ``DEFAULT_INTERVAL`` and
    ``DEFAULT_ACTIVITY_LIMIT`` from the facade to build these flags, so one of the
    pair failing to re-export would change what the CLI advertises, not merely
    what a test can import. The activity limit is compared against the module
    that *owns* it, so a value shadowed on the facade — which would leave the CLI
    advertising one number and the collector using another — reads as a failure
    rather than as two sides agreeing with each other.
    """
    from effgen.cli._main import create_parser

    args = create_parser().parse_args([command])
    assert args.interval == monitor_mod.DEFAULT_INTERVAL
    assert args.limit == monitor_collect.DEFAULT_ACTIVITY_LIMIT


def test_the_dispatch_arm_reaches_the_facade() -> None:
    """``effgen top`` and ``effgen monitor`` both import from ``effgen.cli.monitor``."""
    from effgen.cli import _main

    src = inspect.getsource(_main)
    assert "from effgen.cli.monitor import run_monitor_command" in src
    assert "elif args.command in ('top', 'monitor'):" in src


@pytest.mark.parametrize("module", SPLIT_MODULES, ids=lambda m: m.__name__)
def test_no_module_in_the_split_is_oversized(module) -> None:
    lines = len(Path(inspect.getsourcefile(module)).read_text().splitlines())
    assert lines <= MAX_MODULE_LINES, f"{module.__name__} is {lines} lines"


def test_the_monitor_module_is_a_composition_layer() -> None:
    """The monitor module no longer carries the functions it delegates."""
    src = inspect.getsource(monitor_mod)
    for name in ("def collect_snapshot(", "def render_snapshot(", "def run_live(",
                 "def _build_live_view(", "def _fetch_json(", "def _fmt_usd("):
        assert name not in src, f"{name!r} is still defined in monitor.py"
