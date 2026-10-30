"""A console that cannot encode the CLI's typography still runs every command.

``PYTHONIOENCODING=ascii``, a ``LANG=C`` login, a CI log collector and a Windows
code page all give the CLI a stream that cannot encode an em-dash, an ellipsis
or a table rule. Writing one raises ``UnicodeEncodeError``, which turned a
listing command that had already done its work into a non-zero exit.

The contract pinned here:

* stdout and stderr are routed through an ASCII fold for the duration of a
  command, so what a stream cannot encode is substituted at the single point
  where text becomes bytes rather than raised;
* a stream that *can* encode those characters is handed back unchanged -- the
  same object, so a normal terminal receives the same bytes as before;
* ``--json`` stays lossless: a payload that cannot be written raw is emitted
  with ``\\uXXXX`` escapes, which decodes to the same value instead of a
  transliterated one.

The whole-surface exit-code gate lives in ``tests/packaging/test_reduced_install``,
which drives every command path enumerated from the parser.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

import pytest

from effgen.cli._main import main as cli_main
from effgen.ui.render import (
    _AsciiFoldingStream,
    ascii_folding_stream,
    fold_for_encoding,
    json_ensure_ascii,
)


def _stream(encoding: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict", newline="")


def _text(stream: io.TextIOWrapper, encoding: str) -> str:
    stream.flush()
    return stream.buffer.getvalue().decode(encoding, "replace")


def _run_cli(argv: list[str], encoding: str) -> tuple[int, str]:
    """Run one command with both streams bound to a *encoding* stream."""
    stream = _stream(encoding)
    rc = 0
    argv_before = sys.argv
    sys.argv = ["effgen", *argv]
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            cli_main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = argv_before
    return rc, _text(stream, encoding)


# --------------------------------------------------------------------------- #
# fold_for_encoding
# --------------------------------------------------------------------------- #
class TestFoldForEncoding:
    def test_ascii_text_is_returned_unchanged(self):
        text = "plain ascii - nothing to do"
        assert fold_for_encoding(text, "ascii") is text

    def test_typography_becomes_its_ascii_stand_in(self):
        assert fold_for_encoding("Cost Summary — Last 24 hours", "ascii") == (
            "Cost Summary - Last 24 hours"
        )
        assert fold_for_encoding("≤ is left alone, … is not", "ascii").endswith("... is not")

    def test_glyphs_use_the_same_stand_in_as_the_glyph_table(self):
        from effgen.ui.palette import GLYPHS, GLYPHS_ASCII

        assert fold_for_encoding(GLYPHS["success"], "ascii") == GLYPHS_ASCII["success"]
        assert fold_for_encoding(GLYPHS["arrow"], "ascii") == GLYPHS_ASCII["arrow"]

    def test_an_unmapped_character_is_replaced_not_raised(self):
        out = fold_for_encoding("temperature 20℃ 你好", "ascii")
        out.encode("ascii")  # must not raise
        assert "?" in out

    def test_a_latin1_stream_keeps_what_latin1_can_carry(self):
        assert fold_for_encoding("café — now", "latin-1") == "café - now"

    def test_an_unknown_codec_falls_back_to_ascii(self):
        assert fold_for_encoding("a — b", "not-a-codec") == "a - b"


# --------------------------------------------------------------------------- #
# ascii_folding_stream
# --------------------------------------------------------------------------- #
class TestAsciiFoldingStream:
    def test_a_utf8_stream_is_handed_back_unchanged(self):
        stream = _stream("utf-8")
        assert ascii_folding_stream(stream) is stream

    def test_none_is_handed_back(self):
        assert ascii_folding_stream(None) is None

    def test_wrapping_is_not_repeated(self):
        wrapped = ascii_folding_stream(_stream("ascii"))
        assert isinstance(wrapped, _AsciiFoldingStream)
        assert ascii_folding_stream(wrapped) is wrapped

    def test_an_ascii_stream_receives_the_folded_bytes(self):
        raw = _stream("ascii")
        wrapped = ascii_folding_stream(raw)
        wrapped.write("effGen Cost Summary — Last 24 hours\n")
        assert _text(raw, "ascii") == "effGen Cost Summary - Last 24 hours\n"

    def test_print_through_the_wrapper_does_not_raise(self):
        raw = _stream("ascii")
        print("math — 77 cases …", file=ascii_folding_stream(raw))
        assert _text(raw, "ascii") == "math - 77 cases ...\n"

    def test_stream_attributes_come_from_the_wrapped_stream(self):
        raw = _stream("ascii")
        wrapped = ascii_folding_stream(raw)
        assert wrapped.encoding == "ascii"
        assert wrapped.writable() is True
        assert wrapped.wrapped is raw

    def test_the_glyph_table_still_reads_the_real_encoding_through_it(self):
        """The wrapper is a safety net, not a claim that the stream is UTF-8."""
        from effgen.ui.palette import glyph, supports_unicode

        wrapped = ascii_folding_stream(_stream("ascii"))
        assert supports_unicode(wrapped) is False
        assert glyph("success", wrapped) == "+"


# --------------------------------------------------------------------------- #
# The CLI entry point installs and removes the fold
# --------------------------------------------------------------------------- #
class TestEntryPointInstallsTheFold:
    def test_a_hard_ascii_stream_is_wrapped_for_the_command(self, monkeypatch):
        seen: dict[str, object] = {}

        def _record(args, cli, parser):
            seen["stdout"] = sys.stdout
            seen["stderr"] = sys.stderr
            return 0

        monkeypatch.setattr("effgen.cli._main._dispatch", _record)
        _run_cli(["presets"], "ascii")
        assert isinstance(seen["stdout"], _AsciiFoldingStream)
        assert isinstance(seen["stderr"], _AsciiFoldingStream)

    def test_a_utf8_stream_is_the_identical_object_the_caller_passed(self, monkeypatch):
        seen: dict[str, object] = {}

        def _record(args, cli, parser):
            seen["stdout"] = sys.stdout
            return 0

        monkeypatch.setattr("effgen.cli._main._dispatch", _record)
        stream = _stream("utf-8")
        sys.argv, argv_before = ["effgen", "presets"], sys.argv
        try:
            with contextlib.redirect_stdout(stream), contextlib.suppress(SystemExit):
                cli_main()
        finally:
            sys.argv = argv_before
        assert seen["stdout"] is stream

    def test_the_original_streams_are_restored_afterwards(self, monkeypatch):
        monkeypatch.setattr("effgen.cli._main._dispatch", lambda *a: 0)
        before_out, before_err = sys.stdout, sys.stderr
        _run_cli(["presets"], "ascii")
        assert sys.stdout is before_out
        assert sys.stderr is before_err


# --------------------------------------------------------------------------- #
# Commands that used to exit 1 on a hard-ascii console
# --------------------------------------------------------------------------- #
# Each of these wrote a character the stream could not encode -- an em-dash in a
# header or placeholder, an ellipsis truncation marker, a table rule -- and the
# failed write became the command's exit code.
ASCII_COMMANDS = [
    ["presets"],
    ["tools", "list"],
    ["models", "list"],
    ["models", "list", "--provider", "openai"],
    ["models", "browse", "--limit", "5"],
    ["models", "status"],
    ["cost"],
    ["cost", "today"],
    ["cost", "week"],
    ["cost", "by-provider"],
    ["prompts", "list"],
    ["sessions", "list"],
    ["runs", "list"],
    ["eval", "--suite", "list"],
    ["doctor"],
    ["top", "--once"],
    ["monitor", "--once"],
]


@pytest.mark.parametrize("argv", ASCII_COMMANDS, ids=lambda a: " ".join(a))
def test_a_hard_ascii_console_gets_the_same_exit_code_as_a_utf8_one(argv):
    rc_ascii, text_ascii = _run_cli(argv, "ascii")
    rc_utf8, _ = _run_cli(argv, "utf-8")
    assert rc_ascii == rc_utf8, f"ascii: {text_ascii[-600:]}"
    assert "codec can't encode" not in text_ascii
    assert "PYTHONIOENCODING" not in text_ascii
    assert "Traceback (most recent call last)" not in text_ascii


# --------------------------------------------------------------------------- #
# --json stays lossless
# --------------------------------------------------------------------------- #
JSON_COMMANDS = [
    ["presets", "--json"],
    ["prompts", "list", "--json"],
    ["models", "list", "--json"],
    ["tools", "list", "--json"],
    ["sessions", "list", "--json"],
    ["runs", "list", "--json"],
    ["top", "--json"],
]

# ``top --json`` reports live GPU and process state, so two runs a second apart
# carry different numbers; every other payload above is stable.
STABLE_JSON_COMMANDS = [argv for argv in JSON_COMMANDS if argv != ["top", "--json"]]


class TestJsonStaysLossless:
    def test_escapes_are_requested_only_when_the_stream_needs_them(self):
        assert json_ensure_ascii(_stream("utf-8")) is False
        assert json_ensure_ascii(_stream("ascii")) is True

    @pytest.mark.parametrize("argv", JSON_COMMANDS, ids=lambda a: " ".join(a))
    def test_the_payload_a_hard_ascii_console_receives_is_ascii_json(self, argv):
        rc_ascii, out_ascii = _run_cli(argv, "ascii")
        assert rc_ascii == 0, out_ascii[-600:]
        out_ascii.encode("ascii")  # the hard-ascii payload really is ASCII
        assert isinstance(json.loads(out_ascii), dict | list)

    @pytest.mark.parametrize("argv", STABLE_JSON_COMMANDS, ids=lambda a: " ".join(a))
    def test_the_payload_decodes_to_the_same_value_on_either_console(self, argv):
        rc_ascii, out_ascii = _run_cli(argv, "ascii")
        rc_utf8, out_utf8 = _run_cli(argv, "utf-8")
        assert rc_ascii == rc_utf8 == 0, out_ascii[-600:]
        assert json.loads(out_ascii) == json.loads(out_utf8)

    def test_a_non_ascii_payload_is_escaped_rather_than_transliterated(self):
        """An em-dash in a description must survive as data, not become '-'."""
        _, out_ascii = _run_cli(["presets", "--json"], "ascii")
        assert "\\u2014" in out_ascii
        assert any("—" in preset.get("description", "") for preset in json.loads(out_ascii))

    @pytest.mark.parametrize("factory", ["eval", "comparison"])
    def test_a_result_document_can_be_rendered_as_escaped_json(self, factory):
        """The result types the eval and compare commands print can escape on demand.

        Their default rendering is raw UTF-8, for a file. Written to a console
        that cannot carry it, the escaped rendering decodes to the same value.
        """
        if factory == "eval":
            from effgen.eval.evaluator import SuiteResults

            doc = SuiteResults(suite_name="capitales — français")
        else:
            from effgen.eval.comparison import ComparisonMatrix

            doc = ComparisonMatrix(recommendations={"accuracy": "模型 — one"})
        escaped = doc.to_json(ensure_ascii=True)
        escaped.encode("ascii")  # must not raise
        assert json.loads(escaped) == json.loads(doc.to_json())


# --------------------------------------------------------------------------- #
# No command may print a JSON document the console can only transliterate
# --------------------------------------------------------------------------- #
def _json_print_offenders(path) -> list[str]:
    """Report ``print`` calls in *path* that emit a JSON document unconditionally raw.

    A ``print`` goes to the console. A payload serialised with a hardcoded
    ``ensure_ascii=False``, or by a ``to_json()`` that defaults to it, is folded
    to look-alike characters on a console that cannot encode it, so a consumer
    reading the pipe gets a different value than on a UTF-8 console. The
    decision has to be the stream's -- ``json_ensure_ascii()``.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"):
            continue
        for arg in node.args[:1]:
            if not isinstance(arg, ast.Call):
                continue
            name = getattr(arg.func, "attr", None)
            keywords = {kw.arg for kw in arg.keywords}
            if name == "dumps" and any(
                kw.arg == "ensure_ascii" and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in arg.keywords
            ):
                offenders.append(f"{path.name}:{node.lineno} json.dumps(ensure_ascii=False)")
            elif name == "to_json" and "ensure_ascii" not in keywords:
                offenders.append(f"{path.name}:{node.lineno} to_json() with no ensure_ascii")
    return offenders


def test_no_command_prints_a_json_document_the_console_can_only_transliterate():
    from pathlib import Path

    import effgen.cli

    root = Path(effgen.cli.__file__).parent
    offenders = sorted(
        item for path in root.rglob("*.py") for item in _json_print_offenders(path)
    )
    assert not offenders, "JSON printed without letting the stream pick the escaping:\n" + "\n".join(
        offenders
    )
