"""The stdin contract of ``effgen code``.

``effgen code`` reads a non-terminal stdin to EOF before the run starts, so a
producer that writes slowly is folded in whole. A stream that never closes
therefore holds the run — these tests pin that the wait is announced on stderr
instead of being silent, that the note names the right thing for a run with a
task and for one without, and that a stream which closes normally still folds
its text in front of the task with nothing extra printed.

The stdin-reading cases run the real module in a child process with a real pipe:
an open file descriptor that never reaches EOF is the whole point, and it cannot
be simulated by handing the reader a string buffer.
"""

from __future__ import annotations

import os
import selectors
import subprocess
import sys
import textwrap
import time

import pytest

from effgen.cli.commands.code import (
    STDIN_WAIT_NOTE_SECONDS,
    _read_piped_stdin,
    _resolve_task,
    _stdin_wait_note,
    _WaitNotice,
)

#: Wait the child uses before announcing, kept short so the tests stay quick.
CHILD_DELAY = 0.3

#: Longest a child is given to print its note before the test calls it silent.
NOTE_DEADLINE = 30.0

_CHILD = textwrap.dedent(
    """
    import argparse, json, sys
    from effgen.cli.commands import code

    code.STDIN_WAIT_NOTE_SECONDS = {delay!r}
    args = argparse.Namespace(print_task={task!r}, task=None)
    task, error = code._resolve_task(args, interactive=False)
    sys.stdout.write(json.dumps({{"task": task, "error": error}}))
    sys.stdout.flush()
    """
)


def _spawn(task: str | None) -> subprocess.Popen:
    """Run ``_resolve_task`` in a child whose stdin is an open pipe."""
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(delay=CHILD_DELAY, task=task)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )


def _await_stderr(proc: subprocess.Popen, deadline: float = NOTE_DEADLINE) -> str:
    """Return the first stderr text the child writes, or ``""`` if it stays quiet."""
    sel = selectors.DefaultSelector()
    sel.register(proc.stderr, selectors.EVENT_READ)
    end = time.monotonic() + deadline
    chunks: list[str] = []
    while time.monotonic() < end and not chunks:
        for key, _ in sel.select(timeout=0.2):
            data = os.read(key.fileobj.fileno(), 65536)
            if data:
                chunks.append(data.decode("utf-8", "replace"))
        if proc.poll() is not None and not chunks:
            break
    sel.close()
    return "".join(chunks)


def _finish(proc: subprocess.Popen, feed: bytes = b"") -> tuple[str, str]:
    """Close the child's stdin (optionally feeding *feed*) and collect its output."""
    try:
        out, err = proc.communicate(input=feed, timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - child wedged
        proc.kill()
        out, err = proc.communicate()
        pytest.fail("the child did not exit after stdin closed")
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


class TestOpenStdinIsAnnounced:
    """A pipe that never closes is reported, not waited on silently."""

    def test_task_in_hand_names_the_context_read_and_how_to_skip_it(self):
        proc = _spawn("write fib.py")
        try:
            note = _await_stderr(proc)
        finally:
            if proc.poll() is None:
                proc.stdin.close()
                proc.wait(timeout=30)
        assert note, "an open stdin with a task in hand printed nothing"
        assert "stdin" in note
        assert "has not closed" in note
        assert "/dev/null" in note

    def test_no_task_names_stdin_as_the_task_source(self):
        proc = _spawn(None)
        try:
            note = _await_stderr(proc)
        finally:
            if proc.poll() is None:
                proc.stdin.write(b"explain the failure\n")
                proc.stdin.close()
                proc.wait(timeout=30)
        assert note, "an open stdin with no task printed nothing"
        assert "task from piped stdin" in note
        assert "-p" in note

    def test_the_note_goes_to_stderr_and_stdout_carries_only_the_result(self):
        proc = _spawn("write fib.py")
        note = _await_stderr(proc)
        out, _ = _finish(proc, feed=b"traceback line\n")
        assert "has not closed" in note
        assert "has not closed" not in out
        assert "traceback line" in out


class TestStreamThatCloses:
    """The documented filter behavior is unchanged when stdin reaches EOF."""

    def test_piped_context_precedes_the_task_with_no_note(self):
        proc = _spawn("explain this")
        out, err = _finish(proc, feed=b"AssertionError: boom\n")
        assert "AssertionError: boom" in out
        assert "explain this" in out
        assert out.index("AssertionError") < out.index("explain this")
        assert "has not closed" not in err

    def test_stdin_alone_becomes_the_task_with_no_note(self):
        proc = _spawn(None)
        out, err = _finish(proc, feed=b"add a --retries flag\n")
        assert "add a --retries flag" in out
        assert "has not closed" not in err

    def test_empty_stdin_leaves_the_given_task_alone(self):
        proc = _spawn("write fib.py")
        out, err = _finish(proc)
        assert "write fib.py" in out
        assert "stdin" not in out
        assert "has not closed" not in err


class TestReaderContract:
    """``_read_piped_stdin`` and the notice it arms, without a child process."""

    def test_a_terminal_is_not_read_and_arms_no_notice(self, monkeypatch):
        calls: list[int] = []

        class _Tty:
            def isatty(self):
                return True

            def read(self):  # pragma: no cover - must not be reached
                raise AssertionError("a terminal stdin was read")

        monkeypatch.setattr(sys, "stdin", _Tty())
        assert _read_piped_stdin(on_wait=lambda: calls.append(1)) == ""
        assert calls == []

    def test_a_read_that_returns_first_never_fires_the_notice(self, monkeypatch):
        calls: list[int] = []

        class _Fast:
            def isatty(self):
                return False

            def read(self):
                return "context"

        monkeypatch.setattr(sys, "stdin", _Fast())
        assert _read_piped_stdin(on_wait=lambda: calls.append(1)) == "context"
        assert calls == []

    def test_a_slow_read_fires_the_notice_exactly_once(self, monkeypatch):
        calls: list[int] = []

        class _Slow:
            def isatty(self):
                return False

            def read(self):
                time.sleep(0.25)
                return "context"

        monkeypatch.setattr("effgen.cli.commands.code.STDIN_WAIT_NOTE_SECONDS", 0.05)
        monkeypatch.setattr(sys, "stdin", _Slow())
        assert _read_piped_stdin(on_wait=lambda: calls.append(1)) == "context"
        assert calls == [1]

    def test_a_closed_stdin_reads_as_empty(self, monkeypatch):
        class _Closed:
            def isatty(self):
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr(sys, "stdin", _Closed())
        assert _read_piped_stdin(on_wait=lambda: None) == ""

    def test_the_default_wait_is_short_enough_to_be_noticed(self):
        assert 0 < STDIN_WAIT_NOTE_SECONDS <= 5.0

    def test_the_notice_fires_once_and_cancel_stops_it(self):
        fired: list[int] = []
        notice = _WaitNotice(lambda: fired.append(1), 0.05)
        with notice:
            time.sleep(0.2)
        assert fired == [1]

        stopped: list[int] = []
        with _WaitNotice(lambda: stopped.append(1), 5.0):
            pass
        assert stopped == []

    def test_leaving_the_notice_leaves_no_live_timer_thread(self):
        with _WaitNotice(lambda: None, 30.0) as notice:
            pass
        assert not notice._timer.is_alive()

    def test_both_notes_name_a_way_out(self):
        assert "/dev/null" in _stdin_wait_note(True)
        assert "-p" in _stdin_wait_note(False)
        for note in (_stdin_wait_note(True), _stdin_wait_note(False)):
            assert note == note.strip()
            assert "\n" not in note


class TestResolveTaskWithoutStdin:
    """Task resolution when stdin contributes nothing."""

    def test_missing_task_reports_every_way_to_supply_one(self, monkeypatch):
        monkeypatch.setattr("effgen.cli.commands.code._read_piped_stdin", lambda **_: "")
        import argparse

        args = argparse.Namespace(print_task=None, task=None)
        task, error = _resolve_task(args, interactive=False)
        assert task is None
        assert error is not None
        assert "-p" in error and "stdin" in error
