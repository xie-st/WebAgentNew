"""Layout of the ``effgen.cli.chat`` module split.

The chat module keeps the session — construction, the agent it runs, readline and
the loop — while the command table and dispatcher, one turn, the render layer and
the session-persistence commands are contributed by mixins in their own sibling
modules. Three invariants make that split safe to read and to extend:

* ``effgen.cli.chat`` stays the import path for every name it has ever exposed
  and for the attributes tests and callers patch, whichever module now defines
  them;
* every attribute reaches :class:`~effgen.cli.chat.ChatREPL` from exactly one
  class in its MRO, so adding a mixin can never silently shadow a method;
* the sibling modules never import the chat module, so the package's import
  graph stays acyclic.

The launch seam fails quietly rather than loudly: ``chat_mode`` reads
``ChatREPL`` off ``effgen.cli.chat`` at call time, so a class that moved out
would leave a test patching ``effgen.cli.chat.ChatREPL`` green while the stub it
installed intercepts nothing. That one is asserted as behaviour — a stub that
actually runs — rather than as an import.
"""

from __future__ import annotations

import inspect
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import effgen.cli.chat as chat_mod
import effgen.cli.chat_commands as chat_commands
import effgen.cli.chat_session as chat_session
import effgen.cli.chat_turn as chat_turn
import effgen.cli.chat_view as chat_view
from effgen.cli import _main
from effgen.cli.chat import ChatREPL


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

# Members that moved out of the chat module, and the module that owns each.
OWNERS = {
    # the command table's dispatcher and the commands that report or reset state
    "_dispatch": "effgen.cli.chat_commands",
    "_cmd_help": "effgen.cli.chat_commands",
    "_cmd_tools": "effgen.cli.chat_commands",
    "_cmd_status": "effgen.cli.chat_commands",
    "_cmd_cost": "effgen.cli.chat_commands",
    "_cmd_reset": "effgen.cli.chat_commands",
    "_cmd_clear_screen": "effgen.cli.chat_commands",
    "_cmd_trace": "effgen.cli.chat_commands",
    "_cmd_doctor": "effgen.cli.chat_commands",
    "_teach_model_error": "effgen.cli.chat_commands",
    # the commands that change what the session is, and the history behind them
    "_resolve_swap_target": "effgen.cli.chat_session",
    "_unknown_model_message": "effgen.cli.chat_session",
    "_cmd_model": "effgen.cli.chat_session",
    "_cmd_session": "effgen.cli.chat_session",
    "_cmd_save": "effgen.cli.chat_session",
    "_cmd_load": "effgen.cli.chat_session",
    "_dump_history": "effgen.cli.chat_session",
    "_list_saved": "effgen.cli.chat_session",
    # one turn, from the counters it opens with to the answer it produces
    "_snapshot": "effgen.cli.chat_turn",
    "_do_turn": "effgen.cli.chat_turn",
    "_persist_session_turn": "effgen.cli.chat_turn",
    "_stream_plain": "effgen.cli.chat_turn",
    "_run_with_tools": "effgen.cli.chat_turn",
    "_count_tokens": "effgen.cli.chat_turn",
    # what the session prints around a turn
    "_chrome_console": "effgen.cli.chat_view",
    "_human_stream": "effgen.cli.chat_view",
    "_banner_line": "effgen.cli.chat_view",
    "_prompt_str": "effgen.cli.chat_view",
    "_banner": "effgen.cli.chat_view",
    "_show_answer": "effgen.cli.chat_view",
    "_show_progress": "effgen.cli.chat_view",
    "_footer_text": "effgen.cli.chat_view",
    "_print_footer": "effgen.cli.chat_view",
}

# Members the chat module itself must keep: construction, the agent it runs,
# input plumbing and the loop that ties them together.
KEPT = (
    "__init__",
    "_load_tools",
    "_build_agent",
    "_rebuild",
    "tool_count",
    "_setup_readline",
    "_save_readline",
    "_read_input",
    "run",
)

#: Every attribute ``effgen.cli.chat`` exposed before the split. The set may only
#: grow: a caller reading any of these must keep working. Membership is checked
#: with ``hasattr`` rather than ``dir()`` because a module's ``__annotations__``
#: is materialised on first access, so it is absent from ``dir()`` until read.
BASELINE_MODULE_NAMES = frozenset(
    (
        "Any", "ChatREPL", "Path", "_SLASH_COMMANDS", "__annotations__",
        "__builtins__", "__cached__", "__doc__", "__file__", "__loader__",
        "__name__", "__package__", "__spec__", "_history_dir", "_progress",
        "annotations", "color_enabled", "datetime", "json", "logging", "os",
        "sys", "time",
    )
)

NEW_MODULES = (chat_commands, chat_session, chat_turn, chat_view)

#: The order the facade lists its bases in, and therefore the order the MRO
#: reads. Pinned because a reordering silently changes which class wins when two
#: mixins ever define the same name.
EXPECTED_MRO = (
    "effgen.cli.chat.ChatREPL",
    "effgen.cli.chat_commands.ChatCommandsMixin",
    "effgen.cli.chat_session.ChatSessionMixin",
    "effgen.cli.chat_turn.ChatTurnMixin",
    "effgen.cli.chat_view.ChatViewMixin",
    "builtins.object",
)

#: No module in the split may grow past this, or the split stops paying for
#: itself.
MAX_MODULE_LINES = 500

SPLIT_MODULES = (chat_mod, *NEW_MODULES)


def _defining_class(name: str) -> type:
    for cls in ChatREPL.__mro__:
        if name in vars(cls):
            return cls
    raise AssertionError(f"{name!r} is not defined anywhere in ChatREPL.__mro__")


@pytest.mark.parametrize(("name", "module"), sorted(OWNERS.items()))
def test_moved_members_are_owned_by_one_module(name: str, module: str) -> None:
    assert _defining_class(name).__module__ == module


@pytest.mark.parametrize("name", KEPT)
def test_session_construction_and_loop_stay_in_the_chat_module(name: str) -> None:
    assert _defining_class(name).__module__ == "effgen.cli.chat"


def test_repl_class_is_defined_in_the_chat_module() -> None:
    # ``monkeypatch.setattr("effgen.cli.chat.ChatREPL", ...)`` resolves against
    # this module, and so does a future ``patch.object`` on the class.
    assert ChatREPL.__module__ == "effgen.cli.chat"
    assert chat_mod.ChatREPL is ChatREPL


def test_a_stub_on_the_chat_module_is_what_effgen_chat_launches(monkeypatch) -> None:
    """The launch seam, as behaviour rather than as an import.

    ``chat_mode`` reads ``ChatREPL`` off this module at call time. A class that
    moved into a sibling and was re-exported here would keep this import working
    while a patch on ``effgen.cli.chat.ChatREPL`` stopped intercepting, so the
    guard is asserted by running a stub rather than by comparing attributes.
    """
    from effgen.cli.commands import chat as chat_cmd

    launched = {"built": 0, "ran": 0}

    class _Stub(ChatREPL):
        def __init__(self, cli, args):  # no agent, no model
            launched["built"] += 1

        def run(self):
            launched["ran"] += 1
            return 0

    monkeypatch.setattr("effgen.cli.chat.ChatREPL", _Stub)

    cli = _main.CLIInterface()
    cli.console = None

    class _TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", _TTY(""))
    monkeypatch.setattr("sys.stdout", _TTY())

    rc = chat_cmd.chat_mode(cli, SimpleNamespace(provider=None, verbose=False))

    assert (launched["built"], launched["ran"], rc) == (1, 1, 0)


def test_the_module_name_set_only_grows() -> None:
    """No name this module ever exposed may stop being an attribute of it."""
    dropped = sorted(n for n in BASELINE_MODULE_NAMES if not hasattr(chat_mod, n))
    assert not dropped, f"no longer attributes of effgen.cli.chat: {dropped}"


def test_the_shared_objects_are_shared_not_copied() -> None:
    """Identity, not equality.

    ``_history_dir`` decides where a session's readline history and ``/save``
    snapshots land. Two equal-but-separate definitions would send ``effgen chat``
    and ``effgen code`` to different directories while every equality check kept
    passing, so every binder is checked against the same object -- including
    ``chat_view``, which resolves the directory again when it lists saved
    sessions.
    """
    import effgen.cli.chat_view as chat_view
    import effgen.cli.code.repl as code_repl
    import effgen.cli.code.repl_session as code_repl_session

    assert chat_mod._history_dir is chat_session._history_dir
    assert chat_mod._SLASH_COMMANDS is chat_commands._SLASH_COMMANDS
    assert chat_view._history_dir is chat_session._history_dir
    assert code_repl._history_dir is chat_session._history_dir
    assert code_repl_session._history_dir is chat_session._history_dir


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
def test_new_modules_do_not_import_the_chat_module(module) -> None:
    """Keeps the dependency graph acyclic: chat imports the four, never the reverse.

    The check reads the import statements rather than matching text, so
    ``from . import chat``, ``from effgen.cli import chat`` and
    ``import effgen.cli.chat`` are caught alongside the ``from ... import`` form.
    Any of them binds a half-built module during package import and resolves only
    by the order the package happens to load in.
    """
    offenders = [
        name
        for name in _module_scope_imports(module)
        if name == "effgen.cli.chat" or name.startswith("effgen.cli.chat.")
    ]
    assert not offenders, f"{module.__name__} imports the chat module at module scope: {offenders}"


#: The one-way edges between the siblings. ``chat_session`` is the leaf, and the
#: single edge runs ``chat_view`` to ``chat_session`` — the banner's resume hint
#: reads the directory ``/save`` writes to.
ALLOWED_SIBLING_EDGES = {
    "effgen.cli.chat_commands": frozenset(),
    "effgen.cli.chat_session": frozenset(),
    "effgen.cli.chat_turn": frozenset(),
    "effgen.cli.chat_view": frozenset({"effgen.cli.chat_session"}),
}


@pytest.mark.parametrize("module", NEW_MODULES, ids=lambda m: m.__name__)
def test_the_sibling_import_edges_run_one_way(module) -> None:
    """Only one edge exists between the siblings, and it runs view to session.

    ``_history_dir`` is session state, and the banner's resume hint is a view;
    the edge runs view to session and never back, so neither module can be
    imported without the other resolving first.
    """
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

    ``effgen.cli._main`` at module scope would close a cycle back through
    ``effgen.cli.commands.chat``; ``rich`` and ``readline`` at module scope would
    break an install that has neither.
    """
    forbidden = ("effgen.cli._main", "rich", "readline")
    offenders = [
        name
        for name in _module_scope_imports(module)
        if any(name == f or name.startswith(f + ".") for f in forbidden)
    ]
    assert not offenders, f"{offenders} must stay inside the function that uses them"


def test_the_mro_is_the_one_the_facade_declares() -> None:
    assert tuple(f"{c.__module__}.{c.__name__}" for c in ChatREPL.__mro__) == EXPECTED_MRO


def test_every_repl_attribute_is_defined_once_across_the_mro() -> None:
    """No mixin may shadow a name another class in the MRO already defines.

    Two definitions of the same name would make the answer depend on the base
    order, so a split that duplicates one is rejected here rather than in a
    behaviour test that happens to notice.
    """
    seen: dict[str, list[str]] = {}
    for cls in ChatREPL.__mro__:
        if cls is object:
            continue
        for name in vars(cls):
            if name in INTERPRETER_CLASS_ATTRIBUTES:
                continue
            seen.setdefault(name, []).append(f"{cls.__module__}.{cls.__name__}")
    duplicates = {n: where for n, where in seen.items() if len(where) > 1}
    assert not duplicates, f"defined more than once across ChatREPL.__mro__: {duplicates}"


def test_class_shape_survives_the_move() -> None:
    """A ``property`` must not flatten into a plain method, and the default stays."""
    assert isinstance(inspect.getattr_static(ChatREPL, "tool_count"), property)
    assert "DEFAULT_MODEL" in ChatREPL.__dict__
    assert ChatREPL.DEFAULT_MODEL == "Qwen/Qwen2.5-3B-Instruct"


def _dispatch_keys() -> list[str]:
    """The command names ``_dispatch`` builds its handler table from."""
    import ast
    import textwrap

    src = inspect.getsource(chat_commands.ChatCommandsMixin._dispatch)
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value.startswith("/")
            for k in node.keys
        ):
            return [k.value for k in node.keys]
    raise AssertionError("no handler table found in _dispatch")


def test_the_command_table_and_the_dispatcher_stay_in_step() -> None:
    """Every documented command is routed, and every route is documented.

    The table and the handler map live in one module so they cannot drift; this
    pins that they describe the same command set.
    """
    handled = set(_dispatch_keys())
    documented = set(chat_commands._SLASH_COMMANDS)
    assert documented - handled == set(), f"documented but not routed: {documented - handled}"
    # ``/quit`` is the one alias the dispatcher accepts without documenting it
    # separately; ``/help`` documents it on the ``/exit`` row.
    assert handled - documented == {"/quit"}
    for cmd, description in chat_commands._SLASH_COMMANDS.items():
        assert description.strip(), f"{cmd} has no description"


@pytest.mark.parametrize("module", SPLIT_MODULES, ids=lambda m: m.__name__)
def test_no_module_in_the_split_is_oversized(module) -> None:
    lines = len(Path(inspect.getsourcefile(module)).read_text().splitlines())
    assert lines <= MAX_MODULE_LINES, f"{module.__name__} is {lines} lines"


def test_the_chat_module_is_a_composition_layer() -> None:
    """The chat module no longer carries the members it delegates."""
    src = inspect.getsource(chat_mod)
    for name in ("def _dispatch(", "def _cmd_help(", "def _do_turn(",
                 "def _banner(", "def _cmd_model(", "def _print_footer("):
        assert name not in src, f"{name!r} is still defined in chat.py"
