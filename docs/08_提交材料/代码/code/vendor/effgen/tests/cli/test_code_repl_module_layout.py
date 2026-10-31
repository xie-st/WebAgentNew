"""Layout of the ``effgen.cli.code.repl`` module split.

The repl module keeps the session — construction, the engine, the gate
callbacks, readline and the loop — while the command table and dispatcher, the
turn, the session-control commands and the presentation layer are contributed by
mixins in their own sibling modules. Three invariants make that split safe to
read and to extend:

* ``effgen.cli.code.repl`` stays the import path for every name it has ever
  exposed and for the attributes tests and callers patch, whichever module now
  defines them;
* every attribute reaches :class:`~effgen.cli.code.repl.CodeREPL` from exactly
  one class in its MRO, so adding a mixin can never silently shadow a method;
* the sibling modules never import the repl module, so the package's import
  graph stays acyclic.

The seams here fail quietly rather than loudly: a patch aimed at
``effgen.cli.code.repl.CodeREPL`` that stops intercepting leaves the test using
it passing while the guard it installed is gone. Each one is asserted directly.
"""

from __future__ import annotations

import inspect
import io
from types import SimpleNamespace

import pytest

import effgen.cli.code as code_pkg
import effgen.cli.code.repl as repl_mod
import effgen.cli.code.repl_commands as repl_commands
import effgen.cli.code.repl_session as repl_session
import effgen.cli.code.repl_turn as repl_turn
import effgen.cli.code.repl_view as repl_view
from effgen.cli import _main
from effgen.cli.code.repl import CodeREPL


class _EmptyClassBody:
    """A class body that declares nothing, used to read what the interpreter adds."""


#: Attributes the interpreter itself puts on every class body, which therefore
#: appear on every class in a MRO and are not definitions anyone wrote. Read off
#: an empty class rather than listed, so an interpreter that stamps a new one
#: does not read as a name defined twice.
INTERPRETER_CLASS_ATTRIBUTES = frozenset(vars(_EmptyClassBody)) | {
    "__module__",
    "__doc__",
    "__dict__",
    "__weakref__",
    "__qualname__",
}

# Members that moved out of the repl module, and the module that owns each.
OWNERS = {
    # the command table's dispatcher and the commands that change the working set
    "_dispatch": "effgen.cli.code.repl_commands",
    "_cmd_plan": "effgen.cli.code.repl_commands",
    "_cmd_review": "effgen.cli.code.repl_commands",
    "_cmd_diff": "effgen.cli.code.repl_commands",
    "_staged_selection": "effgen.cli.code.repl_commands",
    "_cmd_apply": "effgen.cli.code.repl_commands",
    "_cmd_reject": "effgen.cli.code.repl_commands",
    "_cmd_undo": "effgen.cli.code.repl_commands",
    "_cmd_run": "effgen.cli.code.repl_commands",
    "_cmd_test": "effgen.cli.code.repl_commands",
    "_execute_shell": "effgen.cli.code.repl_commands",
    "_shell_output_text": "effgen.cli.code.repl_commands",
    "_feed_back": "effgen.cli.code.repl_commands",
    "_cmd_context": "effgen.cli.code.repl_commands",
    "_cmd_add": "effgen.cli.code.repl_commands",
    "_cmd_drop": "effgen.cli.code.repl_commands",
    "_cmd_tools": "effgen.cli.code.repl_commands",
    "_cmd_help": "effgen.cli.code.repl_commands",
    # the commands that retune, inspect or persist the session
    "_cmd_mode": "effgen.cli.code.repl_session",
    "_resolve_swap_target": "effgen.cli.code.repl_session",
    "_cmd_model": "effgen.cli.code.repl_session",
    "_cmd_cost": "effgen.cli.code.repl_session",
    "_cmd_trace": "effgen.cli.code.repl_session",
    "_cmd_git": "effgen.cli.code.repl_session",
    "_git_staged": "effgen.cli.code.repl_session",
    "_git_commit": "effgen.cli.code.repl_session",
    "_cmd_reset": "effgen.cli.code.repl_session",
    "_cmd_clear": "effgen.cli.code.repl_session",
    "_cmd_compact": "effgen.cli.code.repl_session",
    "_dump_history": "effgen.cli.code.repl_session",
    "_cmd_save": "effgen.cli.code.repl_session",
    "_cmd_session": "effgen.cli.code.repl_session",
    "_report_coding_suitability": "effgen.cli.code.repl_session",
    "_adopt_session_state": "effgen.cli.code.repl_session",
    "_existing_context_files": "effgen.cli.code.repl_session",
    "_save_coding_state": "effgen.cli.code.repl_session",
    "_record_turn": "effgen.cli.code.repl_session",
    "_begin_session": "effgen.cli.code.repl_session",
    "_resume_stored_session": "effgen.cli.code.repl_session",
    "_attach_session": "effgen.cli.code.repl_session",
    "_cmd_load": "effgen.cli.code.repl_session",
    "_cmd_doctor": "effgen.cli.code.repl_session",
    # one turn, from the composed task to the notes printed after it
    "_compose_task": "effgen.cli.code.repl_turn",
    "_with_context_files": "effgen.cli.code.repl_turn",
    "_read_context_file": "effgen.cli.code.repl_turn",
    "_snapshot": "effgen.cli.code.repl_turn",
    "_do_turn": "effgen.cli.code.repl_turn",
    "_note_written": "effgen.cli.code.repl_turn",
    "_run_agent": "effgen.cli.code.repl_turn",
    "_post_turn_notes": "effgen.cli.code.repl_turn",
    # what the session prints around a turn
    "_say": "effgen.cli.code.repl_view",
    "_status": "effgen.cli.code.repl_view",
    "_banner_line": "effgen.cli.code.repl_view",
    "_banner": "effgen.cli.code.repl_view",
    "_prompt_str": "effgen.cli.code.repl_view",
    "_render_answer": "effgen.cli.code.repl_view",
    "_print_footer": "effgen.cli.code.repl_view",
}

# Members the repl module itself must keep: construction, the engine and its
# agent, the gate callbacks the engine is handed, input plumbing and the loop.
KEPT = (
    "__init__",
    "_initial_mode",
    "_resolve_model",
    "_build_engine",
    "_carry_memory",
    "agent",
    "_tool",
    "_on_event",
    "_on_diff",
    "_setup_readline",
    "_save_readline",
    "_read_input",
    "run",
)

#: Every attribute ``effgen.cli.code.repl`` exposed before the split. The set may
#: only grow: a caller reading any of these must keep working.
BASELINE_MODULE_NAMES = frozenset(
    (
        "ActionRecord", "Any", "CodeEngine", "CodeREPL", "MODE_DESCRIPTIONS", "Path",
        "PermissionMode", "_CODING_TOOLS", "_SLASH_COMMANDS", "__annotations__",
        "__builtins__", "__cached__", "__doc__", "__file__", "__loader__", "__name__",
        "__package__", "__spec__", "_estimate_tokens", "_history_dir", "annotations",
        "asyncio", "build_project_context", "color_enabled", "datetime", "default_mode",
        "json", "logger", "logging", "print_action", "print_diff", "print_plain",
        "print_status", "resolve_workspace", "staged_diff", "sys", "time",
        "undo_workspace", "workspace_env",
    )
)

NEW_MODULES = (repl_commands, repl_session, repl_turn, repl_view)


def _defining_class(name: str) -> type:
    for cls in CodeREPL.__mro__:
        if name in vars(cls):
            return cls
    raise AssertionError(f"{name!r} is not defined anywhere in CodeREPL.__mro__")


@pytest.mark.parametrize(("name", "module"), sorted(OWNERS.items()))
def test_moved_members_are_owned_by_one_module(name: str, module: str) -> None:
    assert _defining_class(name).__module__ == module


@pytest.mark.parametrize("name", KEPT)
def test_session_construction_and_loop_stay_in_the_repl_module(name: str) -> None:
    assert _defining_class(name).__module__ == "effgen.cli.code.repl"


def test_repl_class_is_defined_in_the_repl_module() -> None:
    # ``monkeypatch.setattr("effgen.cli.code.repl.CodeREPL", ...)`` resolves
    # against this module, and so does a future ``patch.object`` on the class.
    assert CodeREPL.__module__ == "effgen.cli.code.repl"
    assert repl_mod.CodeREPL is CodeREPL
    assert code_pkg.CodeREPL is CodeREPL


def test_a_stub_on_the_repl_module_is_what_a_terminal_run_launches(monkeypatch, tmp_path):
    """The launch seam, as behaviour rather than as an import.

    ``run_code_command`` reads ``CodeREPL`` off this module at call time. The
    piped case is covered elsewhere by proving the REPL is *not* launched; this
    proves the positive, so a move that quietly broke the patch target would be
    caught rather than silently disarming that guard.
    """
    from effgen.cli.commands import code as code_cmd

    launched = {"repl": False}

    class _Stub(CodeREPL):
        def run(self):
            launched["repl"] = True
            return 0

    monkeypatch.setattr("effgen.cli.code.repl.CodeREPL", _Stub)

    ws = tmp_path / "ws"
    ws.mkdir()
    args = SimpleNamespace(
        task=None, print_task=None, model="fake:model", provider=None,
        workspace=str(ws), plan_only=False, auto_edit=False, assume_yes=False,
        undo=False, max_iterations=None, temperature=None, max_tokens=None,
        output_json=False, quiet=True,
    )
    cli = _main.CLIInterface()
    cli.console = None

    class _TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", _TTY(""))
    monkeypatch.setattr("sys.stdout", _TTY())
    monkeypatch.setattr("sys.stderr", _TTY())

    rc = code_cmd.run_code_command(cli, args)

    assert launched["repl"] is True
    assert rc == 0


def test_module_level_names_still_resolve_from_the_repl_module() -> None:
    # ``from effgen.cli.code.repl import _SLASH_COMMANDS, _estimate_tokens`` is
    # how the test lane reaches these; they must stay the same objects.
    assert repl_mod._SLASH_COMMANDS is repl_commands._SLASH_COMMANDS
    assert repl_mod._CODING_TOOLS is repl_commands._CODING_TOOLS
    assert repl_mod._estimate_tokens is repl_commands._estimate_tokens


def test_the_module_name_set_only_grows() -> None:
    """No name this module ever exposed may stop being an attribute of it."""
    dropped = sorted(BASELINE_MODULE_NAMES - set(dir(repl_mod)))
    assert not dropped, f"no longer attributes of effgen.cli.code.repl: {dropped}"


def _module_scope_imports(module) -> list[str]:
    """The dotted names *module* imports at module scope, however it spells them.

    Only top-level statements are read, so an import inside a function — which
    costs nothing at import time and cannot close a cycle — is not reported.
    Every spelling is reduced to the module it names: ``import a.b``,
    ``import a.b as c``, ``from a.b import c`` and ``from a import b`` all
    report ``a.b``.
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
def test_new_modules_do_not_import_the_repl_module(module) -> None:
    """Keeps the dependency graph acyclic: repl imports the four, never the reverse.

    The check reads the import statements rather than matching text, so
    ``from . import repl``, ``from effgen.cli.code import repl`` and
    ``import effgen.cli.code.repl`` are caught alongside the ``from ... import``
    form. Any of them binds a half-built module during package import and
    resolves only by the order the package happens to load in.
    """
    offenders = [
        name
        for name in _module_scope_imports(module)
        if name == "effgen.cli.code.repl" or name.startswith("effgen.cli.code.repl.")
    ]
    assert not offenders, f"{module.__name__} imports the repl module at module scope: {offenders}"


@pytest.mark.parametrize("module", NEW_MODULES, ids=lambda m: m.__name__)
def test_new_modules_keep_the_heavy_imports_function_local(module) -> None:
    """Module scope stays free of the imports that would cost or cycle.

    ``effgen.cli._main`` at module scope would close a cycle back through
    ``effgen.cli.commands.code``; ``rich`` and ``readline`` at module scope
    would break an install that has neither.
    """
    forbidden = ("effgen.cli._main", "rich", "readline")
    offenders = [
        name
        for name in _module_scope_imports(module)
        if any(name == f or name.startswith(f + ".") for f in forbidden)
    ]
    assert not offenders, f"{offenders} must stay inside the function that uses them"


def test_every_repl_attribute_is_defined_once_across_the_mro() -> None:
    """No mixin may shadow a name another class in the MRO already defines.

    Two definitions of the same name would make the answer depend on the base
    order, so a split that duplicates one is rejected here rather than in a
    behaviour test that happens to notice.
    """
    seen: dict[str, list[str]] = {}
    for cls in CodeREPL.__mro__:
        if cls is object:
            continue
        for name in vars(cls):
            if name in INTERPRETER_CLASS_ATTRIBUTES:
                continue
            seen.setdefault(name, []).append(f"{cls.__module__}.{cls.__name__}")
    duplicates = {n: where for n, where in seen.items() if len(where) > 1}
    assert not duplicates, f"defined more than once across CodeREPL.__mro__: {duplicates}"


def test_the_command_table_and_the_dispatcher_stay_in_step() -> None:
    """Every documented command is routed, and every route is documented.

    The table and the handler map live in one module so they cannot drift; this
    pins that they describe the same command set.
    """
    handled = set(_dispatch_keys())
    documented = set(repl_commands._SLASH_COMMANDS)
    assert documented - handled == set(), f"documented but not routed: {documented - handled}"
    # ``/quit`` is the one alias the dispatcher accepts without documenting it
    # separately; ``/help`` documents it on the ``/exit`` row.
    assert handled - documented == {"/quit"}
    for cmd, description in repl_commands._SLASH_COMMANDS.items():
        assert description.strip(), f"{cmd} has no description"


def _dispatch_keys() -> list[str]:
    """The command names ``_dispatch`` builds its handler table from."""
    import ast
    import textwrap

    src = inspect.getsource(repl_commands.CodeCommandsMixin._dispatch)
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value.startswith("/")
            for k in node.keys
        ):
            return [k.value for k in node.keys]
    raise AssertionError("no handler table found in _dispatch")


def test_class_shape_survives_the_move() -> None:
    """A ``staticmethod`` and a ``property`` must not flatten into plain functions."""
    assert isinstance(
        inspect.getattr_static(CodeREPL, "_shell_output_text"), staticmethod
    )
    assert isinstance(inspect.getattr_static(CodeREPL, "agent"), property)
    # Callable on the class, without an instance.
    result = SimpleNamespace(output={"stdout": "a\n", "stderr": ""})
    assert CodeREPL._shell_output_text(result) == "a"


def test_the_repl_module_is_a_composition_layer() -> None:
    """The repl module no longer carries the members it delegates."""
    src = inspect.getsource(repl_mod)
    for name in ("def _dispatch(", "def _cmd_help(", "def _do_turn(",
                 "def _banner(", "def _cmd_doctor(", "def _say("):
        assert name not in src, f"{name!r} is still defined in repl.py"
