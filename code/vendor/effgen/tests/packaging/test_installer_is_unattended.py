"""The installer never waits for an answer nobody is there to give.

`./install.sh` runs from a pipeline, from another script, and a second time on
a machine that already has the environment. In each case stdin is not a
terminal, so a bare `read` returns non-zero at once — and the installer runs
under `set -e`, which turns that into "Installation failed" for a condition
that is not a failure. This has reached a release candidate before.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Every shell file that can prompt during an install.
INSTALLER_SCRIPTS = (
    "install.sh",
    "scripts/install_effgen.sh",
    "scripts/setup_and_verify.sh",
)

#: A prompt is acceptable when a terminal check guards it. `[ -t 0 ]` is the
#: test; it has to appear close enough above the prompt to be guarding it.
TERMINAL_CHECK = re.compile(r"-t\s+0")
GUARD_WINDOW_LINES = 12

#: Commands that need a terminal to describe itself. `clear` writes "TERM
#: environment variable not set." and exits non-zero when TERM is absent, which
#: under `set -e` ends the run on the line it appears. A CI runner, a piped
#: install and `nohup` all arrive with no TERM.
NEEDS_A_TERMINAL = re.compile(r"^[^#]*\b(clear|tput|stty|reset|tabs)\b\s*(\||;|&|$)")

#: The guarded wrapper each installer script defines for exactly this.
TERMINAL_SAFE_CALL = "clear_screen"

#: Reading a prompt. Matches `read -p`, and `read` with the prompt on the line
#: before it, while ignoring commented-out lines and `while read` loops that
#: consume a pipe rather than a person.
PROMPT = re.compile(r"^[^#]*\bread\s+(-[a-zA-Z]+\s+)*-p\b")


def _scripts() -> list[Path]:
    return [REPO / name for name in INSTALLER_SCRIPTS if (REPO / name).exists()]


def test_at_least_one_installer_script_is_present():
    """A vacuous pass here would hide every check below."""
    assert _scripts(), f"none of {INSTALLER_SCRIPTS} exist to check"


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_every_prompt_is_guarded_by_a_terminal_check(script: Path):
    """A prompt with no terminal check fails any unattended install."""
    lines = script.read_text().splitlines()
    unguarded = []
    for number, line in enumerate(lines, start=1):
        if not PROMPT.search(line):
            continue
        window = "\n".join(lines[max(0, number - GUARD_WINDOW_LINES - 1):number])
        if not TERMINAL_CHECK.search(window):
            unguarded.append(f"{script.name}:{number}: {line.strip()}")

    assert not unguarded, (
        "These prompts run whether or not anyone can answer them, so an "
        "install from CI, from another script, or with stdin redirected ends "
        "at `read` under `set -e`:\n  " + "\n  ".join(unguarded) +
        "\nGuard each with `if [ -t 0 ]` and pick a safe default without one."
    )


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_the_script_parses(script: Path):
    """A syntax error here is only ever found by someone installing."""
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"{script.name}: {result.stderr}"


def test_the_existing_environment_branch_keeps_what_is_there():
    """Without a terminal the installer must not delete an existing environment.

    Removing it is the destructive choice, and nobody asked for it — the branch
    exists precisely because a second run is a normal thing to do.
    """
    text = (REPO / "scripts/install_effgen.sh").read_text()
    start = text.find("already exists")
    assert start != -1, "the existing-environment branch has moved or gone"
    branch = text[start:start + 1500]
    assert "-t 0" in branch, "the existing-environment prompt is not guarded"
    assert "conda env remove" in branch, "the branch no longer offers a rebuild"
    remove_at = branch.find("conda env remove")
    guard_at = branch.find("-t 0")
    assert guard_at < remove_at, (
        "the environment is removed before the terminal check decides whether "
        "anyone asked for that"
    )


def test_every_helper_the_installer_runs_exists():
    """A documented flag must not point at a script nobody wrote.

    `--full` and `--download-models` both reach a helper by path. When that
    path is wrong the installer prints a warning and carries on, so the flag
    reads as supported while doing nothing at all — which is how
    `--download-models` shipped hollow.
    """
    text = (REPO / "scripts/install_effgen.sh").read_text()
    referenced = set(re.findall(r'\$SCRIPT_DIR/([A-Za-z0-9_.-]+\.(?:py|sh))', text))
    assert referenced, "no helper references found — has the syntax changed?"

    missing = sorted(name for name in referenced if not (REPO / "scripts" / name).exists())
    assert not missing, (
        "The installer runs helpers that do not exist, so the flags reaching "
        f"them do nothing: {missing}"
    )


def test_the_model_download_helper_runs_without_a_terminal():
    """`--full` must not stall on a prompt, and must not fail the install."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/download_models.py"), "--list",
         "--interactive"],
        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
    assert "GB" in result.stdout, "the listing does not say how large the set is"


def test_the_model_download_helper_never_redirects_the_shared_cache():
    """Passing a download directory would put a second copy of every model on
    disk and hide the one the rest of the toolchain reads."""
    source = (REPO / "scripts/download_models.py").read_text()
    # An argument, not the word: the file explains in a comment why it passes
    # neither, and that explanation must not fail the check it describes.
    redirects = re.findall(r"^[^#]*\b(cache_dir|local_dir|download_dir)\s*=", source,
                           re.MULTILINE)
    assert not redirects, (
        f"download_models.py passes {sorted(set(redirects))}, which overrides "
        "HF_HUB_CACHE and puts a second copy of every model on disk"
    )


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_nothing_needs_a_terminal_to_describe_itself(script: Path):
    """A bare `clear` ends an unattended install on the line it appears.

    The scripts define ``clear_screen``, which checks for a terminal and a TERM
    before wiping anything; calling ``clear`` directly bypasses that.
    """
    lines = script.read_text().splitlines()
    unguarded = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith(("#", "clear_screen()", "clear_screen ")):
            continue
        if TERMINAL_SAFE_CALL in line:
            continue
        if NEEDS_A_TERMINAL.search(line) and "|| true" not in line:
            unguarded.append(f"{script.name}:{number}: {stripped}")

    assert not unguarded, (
        "These need a terminal that may not exist. Without TERM they fail, and "
        "under `set -e` the install ends there:\n  " + "\n  ".join(unguarded) +
        f"\nCall {TERMINAL_SAFE_CALL} instead, which checks first."
    )


@pytest.mark.parametrize(
    "entry", ["install.sh", "scripts/install_effgen.sh", "scripts/verify.sh"],
)
def test_the_entry_point_parses_and_shows_help_without_a_terminal(entry: str):
    """The most basic unattended run: no TERM, no stdin, --help.

    Reaching help proves the script gets past its banner, which is where the
    terminal assumptions live.
    """
    path = REPO / entry
    if not path.exists():
        pytest.skip(f"{entry} is not in this tree")
    env = {k: v for k, v in os.environ.items() if k != "TERM"}
    result = subprocess.run(
        ["bash", str(path), "--help"], capture_output=True, text=True,
        timeout=120, stdin=subprocess.DEVNULL, cwd=str(REPO), env=env,
    )
    assert "TERM environment variable not set" not in (result.stdout + result.stderr), (
        f"{entry} needs a TERM to print its own help"
    )
