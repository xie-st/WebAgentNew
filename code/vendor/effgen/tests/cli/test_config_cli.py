"""Exit codes for the ``effgen config`` command group.

The exit code is what a CI step and a shell script gate on, so a config file
that is refused has to make the command fail. Printing a refusal and exiting 0
reports a broken file as a good one.
"""

from __future__ import annotations

import argparse

import pytest

from effgen.cli.commands.config import config_commands


def _cli():
    from effgen.cli import CLIInterface

    cli = CLIInterface()
    cli.console = None  # keep the test output quiet
    return cli


def _args(command: str, **kwargs) -> argparse.Namespace:
    ns = argparse.Namespace(config_command=command, json=False)
    for key, value in kwargs.items():
        setattr(ns, key, value)
    return ns


@pytest.mark.parametrize(
    "content",
    ["[]", "- 1\n- 2\n", "a bare string\n", "12\n"],
)
def test_validate_fails_on_a_file_that_is_not_a_config(tmp_path, content) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    assert config_commands(_cli(), _args("validate", file=str(path))) == 1


def test_validate_succeeds_on_a_config(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("models:\n  a:\n    type: cloud\n", encoding="utf-8")
    assert config_commands(_cli(), _args("validate", file=str(path))) == 0


def test_validate_without_a_file_fails(tmp_path) -> None:
    assert config_commands(_cli(), _args("validate", file=None)) == 1


def test_show_fails_on_an_unreadable_file(tmp_path) -> None:
    assert config_commands(
        _cli(), _args("show", file=str(tmp_path / "absent.yaml"))
    ) == 1


def test_show_succeeds_on_a_config(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("models:\n  a:\n    type: cloud\n", encoding="utf-8")
    assert config_commands(_cli(), _args("show", file=str(path))) == 0


def test_init_refuses_to_overwrite_without_force(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("models: {}\n", encoding="utf-8")
    assert config_commands(
        _cli(), _args("init", output=str(path), force=False)
    ) == 1
    assert config_commands(
        _cli(), _args("init", output=str(path), force=True)
    ) == 0


def test_set_fails_on_an_unsupported_key() -> None:
    assert config_commands(
        _cli(), _args("set", key="not.a.key", value="1")
    ) == 1
    assert config_commands(
        _cli(), _args("set", key="budget.hourly", value="1")
    ) == 1


def test_set_fails_when_the_value_is_not_a_number() -> None:
    assert config_commands(
        _cli(), _args("set", key="budget.daily", value="lots")
    ) == 1


def test_set_succeeds_on_a_supported_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(tmp_path / "budget.json"))
    assert config_commands(
        _cli(), _args("set", key="budget.daily", value="2.5")
    ) == 0


def test_an_unknown_subcommand_fails() -> None:
    assert config_commands(_cli(), _args("nonexistent")) == 1
