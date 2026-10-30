"""How a library warning reaches the terminal.

Python's default rendering prints the raising file and line plus the source
line that happened to be executing, so a one-line heads-up about a model's
price arrived mid-answer pointing at internals the reader did not write.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import warnings

import pytest

from effgen.cli import _warnings as warning_render

EFFGEN_FRAME = "/somewhere/site-packages/effgen/core/agent_streaming.py"
OTHER_FRAME = "/somewhere/site-packages/thirdparty/thing.py"
MESSAGE = "effGen budget: no published price for 'groq:allam-2-7b'"


@pytest.fixture
def rendered():
    """Emit one warning through the hook and return what reached stderr."""
    original = warnings.showwarning
    warning_render.install(force=True)

    def emit(frame: str, message: str = MESSAGE, category=UserWarning) -> str:
        warnings.resetwarnings()
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            warnings.warn_explicit(message, category, frame, 124)
        return stream.getvalue()

    yield emit
    warnings.showwarning = original


class TestOurOwnWarnings:
    def test_the_frame_is_gone(self, rendered):
        out = rendered(EFFGEN_FRAME)
        assert "agent_streaming.py" not in out
        assert "124" not in out

    def test_the_message_survives_whole(self, rendered):
        assert MESSAGE in rendered(EFFGEN_FRAME)

    def test_it_is_one_line(self, rendered):
        assert rendered(EFFGEN_FRAME).strip().count("\n") == 0

    def test_it_is_labelled_as_a_warning(self, rendered):
        assert rendered(EFFGEN_FRAME).lower().startswith("warning:")


class TestEverythingElseIsUntouched:
    """Anything not ours is handed to whatever was rendering warnings before.

    Asserting on stderr will not do here: under pytest the previous renderer is
    pytest's own recorder, so a delegated warning is captured rather than
    printed. What matters is that it was delegated at all.
    """

    @staticmethod
    def _delegated(frame: str, message: str, category) -> list[tuple]:
        seen: list[tuple] = []
        original = warnings.showwarning
        warnings.showwarning = lambda *a, **k: seen.append(a)
        try:
            warning_render.install(force=True)
            warnings.resetwarnings()
            with contextlib.redirect_stderr(io.StringIO()):
                warnings.warn_explicit(message, category, frame, 124)
        finally:
            warnings.showwarning = original
        return seen

    def test_another_packages_warning_is_passed_through(self):
        """Their file and line are what makes their warning actionable."""
        seen = self._delegated(OTHER_FRAME, "third-party thing", UserWarning)
        assert len(seen) == 1
        assert seen[0][2] == OTHER_FRAME and seen[0][3] == 124

    def test_a_deprecation_from_effgen_is_passed_through(self):
        seen = self._delegated(EFFGEN_FRAME, "old api", DeprecationWarning)
        assert len(seen) == 1
        assert seen[0][2] == EFFGEN_FRAME

    def test_our_own_user_warning_is_not_passed_through(self):
        """The counterpart: ours is rendered here, not handed on."""
        assert self._delegated(EFFGEN_FRAME, MESSAGE, UserWarning) == []

    def test_nothing_reaches_stdout(self):
        original = warnings.showwarning
        warning_render.install(force=True)
        try:
            warnings.resetwarnings()
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                warnings.warn_explicit(MESSAGE, UserWarning, EFFGEN_FRAME, 1)
            assert out.getvalue() == ""
            assert MESSAGE in err.getvalue()
        finally:
            warnings.showwarning = original


class TestTheUserStaysInControl:
    def test_an_explicit_warning_filter_is_left_alone(self, monkeypatch):
        """Someone who set PYTHONWARNINGS asked for Python's rendering."""
        monkeypatch.setenv("PYTHONWARNINGS", "default")
        assert warning_render.install() is False

    def test_it_installs_when_nothing_was_asked_for(self, monkeypatch):
        monkeypatch.delenv("PYTHONWARNINGS", raising=False)
        monkeypatch.setattr(sys, "warnoptions", [], raising=False)
        original = warnings.showwarning
        try:
            assert warning_render.install() is True
        finally:
            warnings.showwarning = original


class TestColour:
    """A redirected stream reports ``isatty() == False``, so the plain branch
    is what a pipe gets; the coloured branch needs a real terminal."""

    SNIPPET = (
        "import warnings;"
        "from effgen.cli import _warnings as w;"
        "w.install(force=True);"
        f"warnings.warn_explicit({MESSAGE!r}, UserWarning, {EFFGEN_FRAME!r}, 1)"
    )

    def test_a_pipe_gets_no_escape_codes(self):
        done = subprocess.run(
            [sys.executable, "-c", self.SNIPPET],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONWARNINGS": ""},
        )
        assert MESSAGE in done.stderr
        assert "\x1b[" not in done.stderr

    def test_no_color_is_honoured_on_a_terminal(self):
        pty = pytest.importorskip("pty")
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - the child execs and never returns
            os.environ["NO_COLOR"] = "1"
            os.environ["PYTHONWARNINGS"] = ""
            os.execv(sys.executable, [sys.executable, "-c", self.SNIPPET])
        chunks = []
        with contextlib.suppress(OSError):
            while True:
                data = os.read(fd, 4096)
                if not data:
                    break
                chunks.append(data)
        os.waitpid(pid, 0)
        output = b"".join(chunks).decode(errors="replace")
        assert MESSAGE.split(":")[0] in output
        assert "\x1b[" not in output
