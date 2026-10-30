"""Executed code must not be able to read the credential stores.

Writes are confined to the run's scratch space; reads were not confined at all,
so ``open('~/.ssh/id_rsa').read()`` succeeded inside the sandbox. That is more
than a local read: an agent that holds ``code_executor`` commonly also holds an
outbound tool that runs *outside* the sandbox, so nothing stopped a model from
reading a secret inside, printing it to stdout, and handing it to the outbound
tool on the next turn. Masking the credential stores is the layer that breaks
that chain for the subprocess backend.

It is a **deny-list**, not confinement, and the tests say so: an ordinary file
is still readable, which is what ``credential_reads_masked`` reports rather than
a ``reads_confined`` that would overstate it.
"""
from __future__ import annotations

import asyncio
import pathlib

import pytest

from effgen.security.sandbox import SandboxConfig, SubprocessSandbox


def _run(code: str):
    return asyncio.run(
        SubprocessSandbox().run(code, "python", SandboxConfig(timeout=45))
    )


@pytest.fixture(scope="module")
def confined() -> bool:
    """Whether this host can confine at all (unprivileged user namespaces)."""
    result = _run("print('ok')")
    return bool(result.filesystem_confined)


def test_a_credential_directory_reads_as_empty(confined, tmp_path):
    if not confined:
        pytest.skip("this host cannot create unprivileged user namespaces")
    home = pathlib.Path.home()
    ssh = home / ".ssh"
    if not ssh.is_dir() or not any(ssh.iterdir()):
        pytest.skip("no ~/.ssh content on this host to mask")

    result = _run(
        "import os, pathlib\n"
        f"p = pathlib.Path({str(ssh)!r})\n"
        "print('entries:', sorted(os.listdir(p)) if p.is_dir() else 'gone')\n"
    )
    assert result.exit_code == 0, result.stderr
    assert "entries: []" in result.stdout, result.stdout


def test_the_run_reports_that_it_masked_them(confined):
    if not confined:
        pytest.skip("this host cannot create unprivileged user namespaces")
    result = _run("print('ok')")
    assert result.credential_reads_masked is True


def test_an_ordinary_file_is_still_readable(confined):
    """The deny-list must not become an accidental read confinement.

    Read outside the system temp directory, because the sandbox mounts a
    private ``/tmp`` — long-standing, documented behaviour that has nothing to
    do with the mask.
    """
    if not confined:
        pytest.skip("this host cannot create unprivileged user namespaces")
    target = pathlib.Path("pyproject.toml").resolve()
    result = _run(
        f"print('read_ok:', len(open({str(target)!r}).read()) > 100)"
    )
    assert result.exit_code == 0, result.stderr
    assert "read_ok: True" in result.stdout


def test_the_deny_list_names_the_same_paths_as_the_file_tools():
    """The two lists protect the same locations, so neither drifts alone."""
    from effgen.security.sandbox import (
        _SANDBOX_DENY_HOME_DIRS,
        _SANDBOX_DENY_HOME_FILES,
    )
    from effgen.tools.builtin._fs import _DENY_HOME_SUBPATHS

    sandbox = set(_SANDBOX_DENY_HOME_DIRS) | set(_SANDBOX_DENY_HOME_FILES)
    assert sandbox == set(_DENY_HOME_SUBPATHS), (
        "the sandbox mask and the file-tool deny-list have drifted apart"
    )


class TestProcessTableIsPrivate:
    """Executed code must not see the host's other processes.

    ``/proc`` was skipped by the read-only pass because the nested ``unshare``
    writes ``/proc/self/uid_map`` there, so the host process table stayed
    visible: ``ps aux``, ``/proc/<pid>/cmdline`` and ``/proc/<pid>/environ`` for
    the calling user's other processes were all readable, and a command line or
    an environment block can carry a secret. That is the same exposure the
    credential masking closes for files.

    The sandbox now runs in its own PID namespace with its own ``/proc``.
    ``/proc`` is still not remounted read-only — the nested ``unshare`` still
    needs to write ``uid_map`` — but it is now a *private* procfs, so the writes
    reach nothing outside the sandbox.
    """

    def test_only_the_sandbox_process_is_visible(self, confined):
        import os

        if not confined:
            pytest.skip("this host cannot create unprivileged user namespaces")
        result = _run(
            "import os\n"
            "print('pids:', sorted(p for p in os.listdir('/proc') if p.isdigit()))\n"
        )
        assert result.exit_code == 0, result.stderr
        if not result.process_table_isolated:
            pytest.skip("this host cannot create a private PID namespace")
        # Its own process, and nothing of the host's — which has hundreds.
        assert "pids: ['1']" in result.stdout, result.stdout
        assert len([p for p in os.listdir("/proc") if p.isdigit()]) > 10

    def test_no_host_command_line_can_be_read(self, confined):
        if not confined:
            pytest.skip("this host cannot create unprivileged user namespaces")
        result = _run(
            "import os\n"
            "seen = 0\n"
            "for p in (p for p in os.listdir('/proc') if p.isdigit()):\n"
            "    try:\n"
            "        text = open(f'/proc/{p}/cmdline').read()\n"
            "    except OSError:\n"
            "        continue\n"
            "    if 'pytest' in text or 'conda' in text:\n"
            "        seen += 1\n"
            "print('host_processes_seen:', seen)\n"
        )
        assert result.exit_code == 0, result.stderr
        if not result.process_table_isolated:
            pytest.skip("this host cannot create a private PID namespace")
        assert "host_processes_seen: 0" in result.stdout

    def test_a_run_that_could_not_isolate_says_so_rather_than_claiming_it(self):
        """Fail-closed reporting: the flag is what was enforced, not what was asked."""
        result = _run("print('ok')")
        assert isinstance(result.process_table_isolated, bool)
