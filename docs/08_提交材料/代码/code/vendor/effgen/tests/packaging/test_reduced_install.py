"""The CLI surface on a reduced install and on a stream that is not UTF-8.

effGen declares ``rich`` and ``torch`` as dependencies, but neither is needed to
answer ``effgen models list`` or to talk to a cloud provider — and both go
missing in the real world (a ``--no-deps`` install, a CPU-only box where torch
was removed, a wheel built for another CUDA line that no longer imports). The
contract this module pins is:

* a package that is absent never reaches the user as ``No module named …`` or as
  a Python traceback — the command either works or names what is missing and how
  to install it;
* the exit code of every command is the same as on a full install, so a script
  that checks ``$?`` behaves the same way;
* a stream that cannot encode a character (``PYTHONIOENCODING=ascii``) never
  produces a traceback either, and every command exits with the code it would
  have on a UTF-8 console: what such a stream cannot carry is substituted, not
  raised. Which ASCII stand-in each character gets is a presentation question
  and is asserted in ``tests/cli/test_ascii_terminal``.

Each cell drives the **whole command surface enumerated from the parser** — every
command path plus a set of offline-safe invocations — inside one interpreter, so
a command added later is covered without touching this file.
"""

from __future__ import annotations

import argparse as _argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Makes the named top-level modules look uninstalled to this interpreter, the way
# an absent package behaves (``ModuleNotFoundError`` raised by the import system).
# argv: 1 = comma-separated modules to hide, 2 = stream encoding, 3 = result path.
_ABSENT_PREAMBLE = '''
import contextlib, io, json, os, sys, traceback

BLOCK = [n for n in sys.argv[1].split(",") if n]
ENCODING = sys.argv[2]
RESULT = sys.argv[3]

if BLOCK:
    class _Absent:
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if root in BLOCK:
                raise ModuleNotFoundError("No module named %r" % root, name=root)
            return None

    for _name in BLOCK:
        for _loaded in [m for m in list(sys.modules) if m == _name or m.startswith(_name + ".")]:
            del sys.modules[_loaded]
    sys.meta_path.insert(0, _Absent())
'''

# Drives every command through ``effgen.cli.main`` with stdout/stderr bound to a
# stream carrying the requested encoding, and records one row per command.
_SURFACE_DRIVER = '''
REAL = [
    ["--version"], ["doctor"], ["models", "list"], ["models", "list", "--json"],
    ["models", "list", "--provider", "openai"], ["models", "list", "--free"],
    ["models", "status"], ["models", "info", "gpt-5-nano"], ["models", "browse", "--limit", "5"],
    ["tools", "list"], ["tools", "list", "--json"], ["tools", "info", "calculator"],
    ["tools", "test", "calculator", "--input", '{"expression": "2+2"}'],
    ["presets"], ["presets", "--json"], ["config", "show"], ["examples", "list"],
    ["cost"], ["cost", "today"], ["cost", "week"], ["cost", "by-provider"],
    ["prompts", "list"], ["prompts", "list", "--json"],
    ["sessions", "list"], ["runs", "list"], ["eval", "--suite", "list"],
    ["top", "--once"], ["top", "--json"], ["monitor", "--once"],
]

import argparse

from effgen.cli._main import create_parser
from effgen.cli._main import main as cli_main


def walk(parser, prefix=()):
    out = [list(prefix)]
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                out += walk(sub, tuple(prefix) + (name,))
    return out


rows = []
for inv in [p + ["--help"] for p in walk(create_parser())] + REAL:
    stream = io.TextIOWrapper(io.BytesIO(), encoding=ENCODING, errors="strict", newline="")
    rc = 0
    sys.argv = ["effgen"] + inv
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            cli_main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc(file=stream)
        rc = -1
    stream.flush()
    text = stream.buffer.getvalue().decode(ENCODING, "replace")
    rows.append({
        "cmd": " ".join(inv),
        "rc": rc,
        "traceback": "Traceback (most recent call last)" in text,
        # A command may legitimately *report* an import failure it diagnosed
        # (``effgen doctor`` does); what it may not do is fail because of one.
        "missing": rc != 0 and (("No module named" in text) or ("ModuleNotFoundError" in text)),
        # A character the stream cannot encode must be substituted at the write
        # boundary, never reported to the user as a failed command.
        "encode_error": ("codec can't encode" in text)
        or ("You may need to add PYTHONIOENCODING" in text),
        "tail": text.strip().splitlines()[-2:],
    })

with open(RESULT, "w", encoding="utf-8") as fh:
    json.dump(rows, fh)
'''

CELL_RUNNER = _ABSENT_PREAMBLE + _SURFACE_DRIVER

# Cells: (id, modules to hide, stream encoding, extra environment).
CELLS = [
    ("full", "", "utf-8", {}),
    ("no-rich", "rich", "utf-8", {}),
    ("no-torch", "torch", "utf-8", {}),
    ("no-rich-no-torch", "rich,torch", "utf-8", {}),
    ("no-rich-no-color", "rich", "utf-8", {"NO_COLOR": "1"}),
    ("ascii", "", "ascii", {}),
    ("no-rich-ascii", "rich", "ascii", {}),
]
REDUCED_CELLS = [cell[0] for cell in CELLS if cell[1]]
ALL_CELLS = [cell[0] for cell in CELLS]


def _run(tmp_path: Path, program: str, blocked: str, encoding: str = "utf-8") -> subprocess.CompletedProcess:
    path = tmp_path / "cell.py"
    path.write_text(program, encoding="utf-8")
    env = dict(os.environ)
    env["EFFGEN_NO_DOTENV"] = "1"
    env["COLUMNS"] = "120"
    env.pop("NO_COLOR", None)
    # The cell binds the command's streams to one carrying *encoding*, but the
    # interpreter's own stdout has to carry it too: a write that goes around
    # ``sys.stdout`` reaches the real file, and on a hard-ascii console that is
    # exactly the write that must not take the command down with it.
    env["PYTHONIOENCODING"] = encoding
    return subprocess.run(
        [sys.executable, str(path), blocked, encoding, str(tmp_path / "result.json")],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=600,
    )


# The fixture below drives seven interpreters through the whole command surface
# before the first test runs, and that does not fit the repository's 120s
# default on a busy machine: every test in this file errored in setup during a
# parallel suite run while the identical tests passed in a lane that raised the
# cap. The work belongs to the module, so the allowance does too — a caller
# should not have to pass a flag to run this file correctly.
pytestmark = pytest.mark.timeout(900)


@pytest.fixture(scope="module")
def cells(tmp_path_factory) -> dict[str, list[dict]]:
    """Drive every cell once; each test reads the rows it needs."""
    out: dict[str, list[dict]] = {}
    for cell_id, blocked, encoding, extra in CELLS:
        tmp_path = tmp_path_factory.mktemp(cell_id.replace("-", "_"))
        env_before = {k: os.environ.get(k) for k in extra}
        os.environ.update(extra)
        try:
            proc = _run(tmp_path, CELL_RUNNER, blocked, encoding)
        finally:
            for key, value in env_before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        result = tmp_path / "result.json"
        if not result.exists():
            pytest.fail(
                f"cell {cell_id!r} did not finish:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        out[cell_id] = json.loads(result.read_text(encoding="utf-8"))
    return out


def _offenders(rows: list[dict], key: str) -> str:
    return "\n".join(f"  effgen {r['cmd']} (rc={r['rc']}): {r['tail']}" for r in rows if r[key])


def test_the_command_surface_is_enumerated_from_the_parser(cells):
    """Every command argparse knows about is driven, not a hand-written subset."""
    from effgen.cli._main import create_parser

    def count(parser) -> int:
        total = 1
        for action in parser._actions:
            if isinstance(action, _argparse._SubParsersAction):
                total += sum(count(sub) for sub in action.choices.values())
        return total

    driven = {row["cmd"] for row in cells["full"]}
    help_paths = {cmd for cmd in driven if cmd.endswith("--help")}
    assert len(help_paths) == count(create_parser())
    assert len(driven) > len(help_paths)


@pytest.mark.parametrize("cell_id", REDUCED_CELLS)
def test_an_absent_package_never_surfaces_as_an_import_error(cells, cell_id):
    offenders = _offenders(cells[cell_id], "missing")
    assert not offenders, f"{cell_id}: commands failed on a missing module:\n{offenders}"


@pytest.mark.parametrize("cell_id", ALL_CELLS)
def test_no_command_prints_a_traceback(cells, cell_id):
    offenders = _offenders(cells[cell_id], "traceback")
    assert not offenders, f"{cell_id}: commands printed a traceback:\n{offenders}"


@pytest.mark.parametrize("cell_id", ["ascii", "no-rich-ascii"])
def test_no_command_fails_on_a_character_the_stream_cannot_encode(cells, cell_id):
    """A hard-ascii console prints a stand-in; it never turns a command into an error."""
    offenders = _offenders(cells[cell_id], "encode_error")
    assert not offenders, f"{cell_id}: commands reported an encoding failure:\n{offenders}"


@pytest.mark.parametrize("cell_id", ["ascii", "no-rich-ascii"])
def test_no_command_is_turned_into_a_failure_by_the_stream(cells, cell_id):
    """The stronger assertion: not merely no traceback, but no new failure.

    Sixteen commands used to exit 1 on a hard-ascii console — a literal em dash
    in a header, table title or placeholder, a ``…`` truncation marker in
    ``runs list`` and inside ``top --json``'s payload, a ``·`` in ``presets``
    and a ``≤`` in a prompt template's constraint text. The listing and results
    headers now fold at the print boundary the way the ``run``/``chat``
    presentation strings always did, so an ASCII console gets a stand-in rather
    than an error.

    A command that fails on a UTF-8 stream too is not this — ``config show``
    with no file is a usage error either way — so the comparison is against the
    full cell, and what is asserted is that the *stream* costs nothing.
    """
    reference = {row["cmd"]: row["rc"] for row in cells["full"]}
    caused_by_the_stream = [
        f"  effgen {row['cmd']}: rc {reference.get(row['cmd'])} on utf-8 -> "
        f"{row['rc']} on ascii: {row['tail']}"
        for row in cells[cell_id]
        if row["rc"] != 0 and reference.get(row["cmd"], row["rc"]) == 0
    ]
    assert not caused_by_the_stream, (
        f"{cell_id}: an ASCII-only stream turned a working command into a "
        "failure:\n" + "\n".join(caused_by_the_stream)
    )


@pytest.mark.parametrize("cell_id", [cell for cell in ALL_CELLS if cell != "full"])
def test_exit_codes_match_the_full_install(cells, cell_id):
    reference = {row["cmd"]: row["rc"] for row in cells["full"]}
    drifted = [
        f"  effgen {row['cmd']}: {reference[row['cmd']]} -> {row['rc']} {row['tail']}"
        for row in cells[cell_id]
        if reference.get(row["cmd"], row["rc"]) != row["rc"]
    ]
    assert not drifted, (
        f"{cell_id}: exit code changed against a full install:\n" + "\n".join(drifted)
    )


def _probe(tmp_path: Path, blocked: str, body: str) -> str:
    """Run *body* in a subprocess where the *blocked* modules are absent."""
    proc = _run(tmp_path, _ABSENT_PREAMBLE + body, blocked)
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_a_local_engine_without_torch_names_the_package_and_the_fix(tmp_path):
    """Loading a local model without torch says what to install."""
    out = _probe(tmp_path, "torch", '''
from effgen import load_model
try:
    load_model("transformers:Qwen/Qwen2.5-0.5B-Instruct")
    print("LOADED")
except Exception as exc:
    print("TYPE:", type(exc).__name__)
    print("MESSAGE:", str(exc))
''')
    assert "LOADED" not in out
    message = out.split("MESSAGE:", 1)[1].strip()
    assert message != "No module named 'torch'"
    assert "PyTorch" in message
    assert "pip install" in message


def test_a_local_engine_names_the_package_a_lazy_import_layer_hides(tmp_path):
    """``transformers`` reaches rich through typer; its lazy loader hides which."""
    out = _probe(tmp_path, "rich", '''
from effgen import load_model
try:
    load_model("transformers:Qwen/Qwen2.5-0.5B-Instruct")
    print("LOADED")
except Exception as exc:
    print("MESSAGE:", str(exc))
''')
    if "LOADED" in out:  # a build whose lazy layer does not reach rich
        return
    message = out.split("MESSAGE:", 1)[1].strip()
    assert "'rich'" in message
    assert "pip install rich" in message
    assert "AutoModelForCausalLM" not in message


@pytest.mark.parametrize(
    ("module", "attribute", "engine"),
    [
        ("effgen.models.transformers_engine", "TransformersEngine", "transformers"),
        ("effgen.models.vllm_engine", "VLLMEngine", "vllm"),
    ],
)
def test_importing_an_engine_module_without_torch_names_the_fix(
    tmp_path, module, attribute, engine
):
    """Every route to a local engine reports the same missing package.

    ``effgen.TransformersEngine`` and ``effgen.models.TransformersEngine`` read
    their text from the engine module's own import failure, so the module is
    what has to carry it.
    """
    out = _probe(tmp_path, "torch", f'''
import effgen
for label, get in (
    ("direct", lambda: __import__({module!r}, fromlist=["x"])),
    ("package", lambda: getattr(__import__("effgen.models", fromlist=["x"]), {attribute!r})),
    ("toplevel", lambda: getattr(effgen, {attribute!r})),
):
    try:
        get()
        print(label, "|", "OK")
    except Exception as exc:
        print(label, "|", type(exc).__name__, "|", str(exc).replace("\\n", " "))
''')
    rows = {line.split(" | ")[0]: line for line in out.splitlines() if " | " in line}
    direct = rows["direct"]
    assert "ImportError" in direct
    assert "No module named" not in direct
    assert f"local '{engine}' engine" in direct
    assert "pip install torch" in direct
    # The two attribute routes wrap the same text in a missing-attribute error.
    for label in ("package", "toplevel"):
        assert "AttributeError" in rows[label]
        assert "pip install torch" in rows[label]


@pytest.mark.parametrize(
    "module",
    [
        "effgen.models.transformers_engine_loading",
        "effgen.models.transformers_engine_placement",
        "effgen.models.transformers_engine_sampling",
        "effgen.models.transformers_engine_generation",
    ],
)
def test_importing_an_engine_submodule_without_torch_names_the_fix(tmp_path, module):
    """A submodule imported on its own reports the missing package the same way.

    The engine spans several modules. Someone who opens the one holding the code
    they care about and imports it directly must get the message naming what to
    install, not ``No module named 'torch'`` raised from deeper down.
    """
    out = _probe(tmp_path, "torch", f'''
try:
    __import__({module!r}, fromlist=["x"])
    print("OK")
except Exception as exc:
    print(type(exc).__name__, "|", str(exc).replace("\\n", " "))
''')
    assert "ImportError" in out
    assert "No module named" not in out
    assert "local 'transformers' engine" in out
    assert "pip install torch" in out


@pytest.mark.parametrize(
    "module",
    [
        "effgen.models.transformers_engine_support",
        "effgen.models.transformers_engine_streaming",
    ],
)
def test_engine_submodules_that_need_no_torch_import_without_it(tmp_path, module):
    """Neither module touches torch or transformers at module scope, so both load.

    Guarding them would report a missing package they never reach for.
    """
    out = _probe(tmp_path, "torch", f'''
try:
    __import__({module!r}, fromlist=["x"])
    print("OK")
except Exception as exc:
    print(type(exc).__name__, "|", str(exc).replace("\\n", " "))
''')
    assert out.strip().splitlines()[-1] == "OK", out


@pytest.mark.parametrize(
    "helper",
    ["print", "print_line", "print_header", "print_success", "print_warning", "print_error"],
)
def test_chatter_reaches_a_redirected_stdout_before_the_process_exits(tmp_path, helper):
    """Without rich, output still arrives while the command is still running.

    A rich console writes through on every call. The plain fallback goes to the
    interpreter's block buffer, so a long-running command redirected to a file —
    ``effgen serve > server.log``, which prints the API key it just minted —
    showed nothing at all until it was stopped. Each helper is driven in its own
    process, so one that stopped flushing is not covered up by another that
    still does.
    """
    marker = "API key: abc123"
    program = _ABSENT_PREAMBLE + f'''
import time

from effgen.cli._main import CLIInterface

cli = CLIInterface()
cli.{helper}("  {marker}")
time.sleep(30)
'''
    path = tmp_path / "chatter.py"
    path.write_text(program, encoding="utf-8")
    log = tmp_path / "chatter.log"
    env = dict(os.environ)
    env["EFFGEN_NO_DOTENV"] = "1"
    with log.open("wb") as fh:
        proc = subprocess.Popen(
            [sys.executable, str(path), "rich", "utf-8", str(tmp_path / "unused.json")],
            stdout=fh, stderr=subprocess.DEVNULL, env=env, cwd=str(tmp_path),
        )
        try:
            deadline = time.time() + 25
            seen = ""
            while time.time() < deadline:
                seen = log.read_text(encoding="utf-8", errors="replace")
                if marker in seen:
                    break
                time.sleep(0.5)
        finally:
            proc.kill()
            proc.wait(timeout=30)
    assert marker in seen, (
        f"CLIInterface.{helper} never reached a redirected stdout while the "
        f"process was still running; saw {seen!r}"
    )


def test_the_gpu_report_without_torch_says_torch_is_absent(tmp_path):
    """A machine with GPUs but no torch is not described as a CUDA build mismatch."""
    out = _probe(tmp_path, "torch", '''
from effgen.gpu.cuda_compat import get_cuda_status
status = get_cuda_status()
print("TORCH_INSTALLED:", status.torch_installed)
print("MESSAGE:", status.message or "")
''')
    assert "TORCH_INSTALLED: False" in out
    message = out.split("MESSAGE:", 1)[1].strip()
    if message:
        assert "not installed" in message
        assert "built for" not in message
