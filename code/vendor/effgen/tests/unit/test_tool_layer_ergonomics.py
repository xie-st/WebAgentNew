"""Tool-layer ergonomics: lazy discovery, sync accessor, accurate envelopes,
operation aliases, friendly errors, and a round-tripping plugin scaffold.

These guard the fixes for the empty-registry trap, the async-only ``get_tool``
gap, double-wrapped tool results, inconsistent selector params, and the broken
plugin scaffold. No live model calls — pure local behavior.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from effgen.tools.base_tool import _inner_status
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.datetime_tool import DateTimeTool
from effgen.tools.builtin.file_ops import FileOperations
from effgen.tools.builtin.json_tool import JSONTool
from effgen.tools.builtin.python_repl import PythonREPL
from effgen.tools.registry import ToolRegistry


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Lazy discovery + sync accessor
# --------------------------------------------------------------------------

def test_fresh_registry_auto_discovers_builtins():
    reg = ToolRegistry()
    # A brand-new registry must not be an empty trap for programmatic users.
    tools = reg.list_tools()
    assert len(tools) > 10
    assert "calculator" in tools
    assert len(reg) == len(tools)
    assert "calculator" in reg  # __contains__ also discovers


def test_get_tool_sync_returns_instance_not_coroutine():
    reg = ToolRegistry()
    tool = reg.get_tool_sync("calculator")
    # The obvious sync call must yield a real tool, not a coroutine object.
    assert not asyncio.iscoroutine(tool)
    assert tool.name == "calculator"


def test_get_metadata_discovers():
    reg = ToolRegistry()
    md = reg.get_metadata("datetime")
    assert md.name == "datetime"


# --------------------------------------------------------------------------
# Result envelopes — no double-wrapped success/failure
# --------------------------------------------------------------------------

def test_inner_status_helper():
    assert _inner_status({"success": False, "error": "boom"}) == (False, "boom")
    assert _inner_status({"success": False, "message": "nope"}) == (False, "nope")
    assert _inner_status({"success": True}) == (None, None)
    assert _inner_status({"valid": False, "error": "x"}) == (None, None)  # not a failure
    assert _inner_status("plain string") == (None, None)


def test_file_ops_failure_surfaces_on_outer(tmp_path):
    fo = FileOperations(allowed_directories=[str(tmp_path)])
    res = _run(fo.execute(operation="read", path=str(tmp_path / "missing.txt")))
    assert res.success is False
    assert res.error and "not found" in res.error.lower()


def test_file_ops_success_unchanged(tmp_path):
    fo = FileOperations(allowed_directories=[str(tmp_path)])
    res = _run(fo.execute(operation="list", path=str(tmp_path)))
    assert res.success is True


def test_python_repl_error_surfaces_on_outer():
    repl = PythonREPL()
    res = _run(repl.execute(code="1/0"))
    assert res.success is False
    assert res.output["success"] is False
    res_ok = _run(repl.execute(code="2 + 2"))
    assert res_ok.success is True
    assert res_ok.output["result"] == 4


def test_json_validate_invalid_is_not_a_tool_failure():
    # JSON validation returning valid=False is a *successful* check, not a failure.
    js = JSONTool()
    res = _run(js.execute(data="{not json", operation="validate"))
    assert res.success is True
    assert res.output["valid"] is False


# --------------------------------------------------------------------------
# Operation aliases
# --------------------------------------------------------------------------

def test_datetime_value_alias():
    dt = DateTimeTool()
    res = _run(dt.execute(operation="current_time"))
    assert res.success is True


def test_json_value_alias_parse():
    js = JSONTool()
    res = _run(js.execute(data='{"a": 1, "b": 2}', operation="parse"))
    assert res.success is True
    assert res.output["result"] == {"a": 1, "b": 2}


def test_param_name_alias_action_to_operation():
    js = JSONTool()
    res = _run(js.execute(data='{"a": 1}', action="keys"))
    assert res.success is True
    assert res.output["result"] == ["a"]


def test_calculator_alias_and_real_op():
    calc = Calculator()
    res = _run(calc.execute(expression="3 * 4", operation="eval"))
    assert res.success is True


def test_enum_error_lists_allowed_and_aliases():
    calc = Calculator()
    res = _run(calc.execute(expression="1", operation="totally_bogus"))
    assert res.success is False
    assert "calculate" in res.error
    assert "Aliases" in res.error


def test_unknown_op_with_no_alias_still_clear():
    fo = FileOperations(allowed_directories=["/tmp"])
    res = _run(fo.execute(operation="frobnicate", path="/tmp"))
    assert res.success is False
    assert "Allowed" in res.error


def test_selector_alias_never_consumes_a_real_distinct_param():
    # DataFrameTool declares BOTH ``operation`` (selector) and ``op`` (a filter
    # operator value). ``op`` must never be hijacked as a misnamed selector when
    # ``operation`` is absent — it is a real, distinct parameter.
    from effgen.tools.builtin.data_analysis import DataFrameTool

    t = DataFrameTool()
    assert t._selector_param() == "operation"
    norm = t._normalize_selector({"op": ">", "column": "x", "value": 5})
    assert norm.get("op") == ">"
    assert norm.get("operation") is None


# --------------------------------------------------------------------------
# Plugin scaffold renders, imports, and runs
# --------------------------------------------------------------------------

def test_plugin_scaffold_round_trip(tmp_path):
    from effgen.cli._main import _create_plugin_scaffold

    rc = _create_plugin_scaffold("acme_demo", output_dir=str(tmp_path))
    assert rc == 0

    base = tmp_path / "effgen-plugin-acme_demo"
    pkg = base / "acme_demo"
    assert (pkg / "tools.py").exists()
    assert (pkg / "plugin.py").exists()

    # Version must be the plugin's own 0.1.0, not the framework version.
    pyproject = (base / "pyproject.toml").read_text()
    assert 'version = "0.1.0"' in pyproject

    # Import + instantiate + execute the generated tool.
    sys.path.insert(0, str(base))
    try:
        # Drop any cached modules from a previous run.
        for mod in [m for m in sys.modules if m.startswith("acme_demo")]:
            del sys.modules[mod]
        from acme_demo.plugin import AcmeDemoPlugin  # type: ignore
        from acme_demo.tools import ExampleTool  # type: ignore

        tool = ExampleTool()  # must not raise (the original scaffold did)
        assert tool.name == "example_tool"

        reg = ToolRegistry()
        names = AcmeDemoPlugin().register(reg)
        assert names == ["example_tool"]
        assert AcmeDemoPlugin.version == "0.1.0"

        res = _run(reg.get_tool_sync("example_tool").execute(text="hi"))
        assert res.success is True
        assert res.output == {"echo": "hi"}
    finally:
        sys.path.remove(str(base))
        for mod in [m for m in sys.modules if m.startswith("acme_demo")]:
            del sys.modules[mod]


def test_plugin_scaffold_rejects_bad_name(tmp_path):
    from effgen.cli._main import _create_plugin_scaffold

    rc = _create_plugin_scaffold("1bad name", output_dir=str(tmp_path))
    assert rc == 1


def test_plugin_scaffold_normalizes_hyphen(tmp_path):
    from effgen.cli._main import _create_plugin_scaffold

    rc = _create_plugin_scaffold("my-plugin", output_dir=str(tmp_path))
    assert rc == 0
    assert (tmp_path / "effgen-plugin-my_plugin" / "my_plugin" / "plugin.py").exists()


# --------------------------------------------------------------------------
# CLI exit codes for tools info/test
# --------------------------------------------------------------------------

def _cli():
    from effgen.cli._main import CLIInterface

    return CLIInterface()


def _args(**kw):
    import types

    return types.SimpleNamespace(**kw)


def test_cli_tools_info_exit_codes():
    cli = _cli()
    assert cli.tools_commands(_args(tool_command="info", name="calculator")) == 0
    # Not-found is a user-visible error: nonzero, even without a prior 'list'.
    assert cli.tools_commands(_args(tool_command="info", name="no_such_tool")) == 1


def test_cli_tools_test_exit_codes():
    cli = _cli()
    ok = cli.tools_commands(
        _args(tool_command="test", name="calculator", input='{"expression": "2+2"}', verbose=False)
    )
    assert ok == 0
    # Missing input -> show schema/example, nonzero.
    assert cli.tools_commands(
        _args(tool_command="test", name="calculator", input=None, verbose=False)
    ) == 1
    # Unknown tool -> nonzero.
    assert cli.tools_commands(
        _args(tool_command="test", name="no_such_tool", input="{}", verbose=False)
    ) == 1
    # A tool that runs but reports failure -> nonzero (accurate envelope).
    bad = cli.tools_commands(
        _args(tool_command="test", name="calculator", input='{"expression": "1/0"}', verbose=False)
    )
    assert bad == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
