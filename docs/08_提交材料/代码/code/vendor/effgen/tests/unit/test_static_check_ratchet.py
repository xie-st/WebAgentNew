"""The recorded static-check result may improve, never regress.

effGen's mypy configuration is permissive, so the package does not type-check
clean and a "must be clean" gate would have to be switched off. A lane that is
off cannot notice anything, so the recorded result is gated instead:
``scripts/mypy_baseline.json`` holds every error's identity with how often it
occurs, and ``scripts/mypy_ratchet.py`` fails when a new identity appears, when
a recorded one occurs more often, or when the total rises. That keeps
annotations which disagree with the code from accumulating; unannotated public
signatures are a different gate, ``test_public_signature_hints.py``, because
``disallow_untyped_defs = false`` means mypy reports nothing for one.

The checks here run without mypy installed: they validate the baseline document
and drive the comparison over synthetic transcripts, including one carrying a
planted error, so the gate itself is proven to catch a regression. Running mypy
for real is the ``static-check ratchet`` CI job.

The linter's target version is checked here too, because it is the same
decision: ``[tool.ruff] target-version`` tracks the ``requires-python`` floor,
and several pyupgrade rules are version-gated behind it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "scripts" / "mypy_baseline.json"
RATCHET = REPO_ROOT / "scripts" / "mypy_ratchet.py"

# The version-gated pyupgrade rules that a higher `target-version` would turn on
# and that the declared Python floor cannot satisfy. Documented beside
# `target-version` in pyproject.toml.
VERSION_GATED_RULES = {
    "UP017": (3, 11),  # datetime.timezone.utc -> datetime.UTC
    "UP041": (3, 11),  # asyncio.TimeoutError -> TimeoutError
    "UP042": (3, 11),  # class C(str, Enum) -> enum.StrEnum
    "UP047": (3, 12),  # module-level TypeVar -> PEP 695 type parameters
}


def _load_ratchet():
    """Import scripts/mypy_ratchet.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_mypy_ratchet", RATCHET)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mypy_ratchet"] = module
    spec.loader.exec_module(module)
    return module


def _read_pyproject() -> dict:
    import tomllib

    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ratchet():
    return _load_ratchet()


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE.exists(), (
        f"{BASELINE} is missing. Record it with: python scripts/mypy_ratchet.py --update"
    )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The baseline document
# ---------------------------------------------------------------------------


def test_baseline_totals_match_its_identities(baseline):
    """The recorded totals are derived from the recorded identities."""
    identities = baseline["identities"]
    assert baseline["totals"]["identities"] == len(identities)
    assert baseline["totals"]["errors"] == sum(identities.values())


def test_baseline_by_code_matches_its_identities(baseline):
    """The per-code counts add up to the same errors."""
    from collections import Counter

    tally: Counter[str] = Counter()
    for identity, count in baseline["identities"].items():
        tally[identity.split("|", 2)[1]] += count
    assert dict(tally) == baseline["by_code"]


def test_baseline_records_the_toolchain_that_produced_it(baseline):
    """A different mypy reports a different set, so the version is recorded."""
    toolchain = baseline["toolchain"]
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", toolchain["mypy"]), toolchain["mypy"]
    assert toolchain["target"] == "effgen"


def test_baseline_identities_carry_no_line_numbers(baseline):
    """Identities survive code moving: file, code and message only."""
    for identity in baseline["identities"]:
        path, _code, _message = identity.split("|", 2)
        assert path.startswith("effgen/"), identity
        assert not re.search(r"\.py:\d+", path), identity


def test_pinned_mypy_matches_the_recorded_baseline(baseline):
    """requirements-dev.txt pins the mypy the baseline was recorded with."""
    text = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    match = re.search(r"^mypy==(\S+)$", text, re.MULTILINE)
    assert match, "requirements-dev.txt must pin mypy exactly for the ratchet"
    assert match.group(1) == baseline["toolchain"]["mypy"], (
        "requirements-dev.txt pins mypy "
        f"{match.group(1)} but scripts/mypy_baseline.json was recorded with "
        f"{baseline['toolchain']['mypy']} — re-record the baseline or fix the pin."
    )


# ---------------------------------------------------------------------------
# The comparison, driven over synthetic transcripts
# ---------------------------------------------------------------------------


def _transcript(rows: list[tuple[str, int, str, str]]) -> str:
    return "\n".join(
        f"{path}:{line}:1: error: {message}  [{code}]" for path, line, message, code in rows
    )


def test_parses_an_error_line_into_an_identity_without_its_position(ratchet):
    first = ratchet.parse_errors(
        _transcript([("effgen/x.py", 12, 'Name "z" is not defined', "name-defined")])
    )
    moved = ratchet.parse_errors(
        _transcript([("effgen/x.py", 900, 'Name "z" is not defined', "name-defined")])
    )
    assert first == moved
    assert list(first) == ['effgen/x.py|name-defined|Name "z" is not defined']


def test_an_unchanged_run_passes(ratchet, baseline):
    """Replaying the recorded set is a pass."""
    from collections import Counter

    status, lines = ratchet.compare(Counter(baseline["identities"]), baseline)
    assert status == 0, "\n".join(lines)
    assert "ratchet holds" in "\n".join(lines)


def test_an_improved_run_passes_and_reports_what_improved(ratchet, baseline):
    from collections import Counter

    counts = Counter(baseline["identities"])
    dropped = sorted(counts)[0]
    del counts[dropped]
    status, lines = ratchet.compare(counts, baseline)
    assert status == 0, "\n".join(lines)
    assert "improved" in "\n".join(lines)


def test_a_planted_new_error_is_caught(ratchet, baseline):
    """The gate must fail on an identity the baseline has never seen."""
    from collections import Counter

    counts = Counter(baseline["identities"])
    counts["effgen/planted_module.py|no-untyped-def|Function is missing a type annotation"] = 1
    status, lines = ratchet.compare(counts, baseline)
    report = "\n".join(lines)
    assert status == 1, report
    assert "RATCHET VIOLATED" in report
    assert "planted_module.py" in report


def test_a_planted_repeat_of_a_known_error_is_caught(ratchet, baseline):
    """An identity that occurs more often than recorded is also a regression."""
    from collections import Counter

    counts = Counter(baseline["identities"])
    known = sorted(counts)[0]
    counts[known] += 1
    status, lines = ratchet.compare(counts, baseline)
    report = "\n".join(lines)
    assert status == 1, report
    assert "+ MORE" in report


def test_a_planted_error_survives_a_simultaneous_improvement(ratchet, baseline):
    """Fixing one error does not buy the right to add another."""
    from collections import Counter

    counts = Counter(baseline["identities"])
    ordered = sorted(counts)
    del counts[ordered[0]]
    counts["effgen/planted_module.py|attr-defined|\"Q\" has no attribute \"z\""] = 1
    status, lines = ratchet.compare(counts, baseline)
    assert status == 1, "\n".join(lines)


def test_gating_a_saved_transcript_end_to_end(ratchet, baseline, tmp_path, capsys):
    """`--from-output` runs the whole gate over a transcript, exit code included."""
    good = tmp_path / "clean.txt"
    good.write_text(
        "\n".join(
            f"{i.split('|')[0]}:1:1: error: {i.split('|', 2)[2]}  [{i.split('|')[1]}]"
            for i, n in baseline["identities"].items()
            for _ in range(n)
        ),
        encoding="utf-8",
    )
    assert ratchet.main(["--from-output", str(good)]) == 0

    bad = tmp_path / "regressed.txt"
    bad.write_text(
        good.read_text(encoding="utf-8")
        + "\neffgen/planted_module.py:1:1: error: Missing type parameters  [type-arg]",
        encoding="utf-8",
    )
    assert ratchet.main(["--from-output", str(bad)]) == 1
    assert "RATCHET VIOLATED" in capsys.readouterr().out


def test_the_recorded_result_does_not_depend_on_the_installed_extras(ratchet, baseline):
    """A developer with `[all]` and CI with a base install must agree.

    `--ignore-missing-imports` is not enough on its own: it silences the import
    error for a package that is absent, but an installed one still contributes
    its real types. `--no-site-packages` makes every third-party import Any
    either way, which is what keeps the recorded set comparable.
    """
    assert "--no-site-packages" in ratchet.MYPY_ARGS, (
        "Without --no-site-packages the recorded set changes with the "
        "environment, and the gate fails in CI for reasons unrelated to the "
        "change under review."
    )
    assert baseline["toolchain"]["mypy_args"] == ratchet.MYPY_ARGS, (
        "scripts/mypy_baseline.json was recorded with "
        f"{baseline['toolchain']['mypy_args']} but the script now runs "
        f"{ratchet.MYPY_ARGS} — re-record the baseline in the same change."
    )


def test_a_baseline_recorded_with_other_flags_is_refused(ratchet, baseline, tmp_path, capsys):
    """Changing the flags without re-recording is an actionable exit 2."""
    if ratchet.mypy_version() == "absent":
        pytest.skip(
            "the flag mismatch is reported after the mypy-presence check, so it "
            "is only reachable where mypy is installed"
        )
    skewed = dict(baseline)
    skewed["toolchain"] = dict(baseline["toolchain"])
    skewed["toolchain"]["mypy_args"] = ["--ignore-missing-imports"]
    path = tmp_path / "skewed.json"
    path.write_text(json.dumps(skewed), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        ratchet.check_toolchain(skewed)
    assert excinfo.value.code == 2
    assert "--no-site-packages" in capsys.readouterr().err


def test_the_ratchet_does_not_stand_in_for_the_signature_gate():
    """Untyped code is the other gate's job, and the config is why."""
    settings = _read_pyproject()["tool"]["mypy"]
    assert settings["disallow_untyped_defs"] is False, (
        "The ratchet is documented as not covering wholly unannotated code "
        "because mypy is configured not to report it. Turning "
        "disallow_untyped_defs on changes that, and the baseline with it."
    )
    gate = REPO_ROOT / "tests" / "unit" / "test_public_signature_hints.py"
    assert gate.exists(), (
        f"{gate.name} is the zero-tolerance gate on unannotated public "
        "signatures; without it nothing catches an untyped new module."
    )


# ---------------------------------------------------------------------------
# The linter target tracks the declared Python floor
# ---------------------------------------------------------------------------


def test_ruff_target_version_equals_the_requires_python_floor():
    """A higher target would demand syntax the declared floor cannot run."""
    data = _read_pyproject()
    requires = data["project"]["requires-python"]
    floor = re.search(r">=\s*(\d+)\.(\d+)", requires)
    assert floor, requires
    expected = f"py{floor.group(1)}{floor.group(2)}"
    actual = data["tool"]["ruff"]["target-version"]
    assert actual == expected, (
        f"requires-python is {requires!r} but [tool.ruff] target-version is "
        f"{actual!r}. They must match: {sorted(VERSION_GATED_RULES)} are gated "
        "behind the target and rewrite code the floor cannot execute. Move the "
        "floor first, then the target, then apply the rules in their own commit."
    )


def test_the_version_gated_rules_are_documented_beside_the_target():
    """A reader at `target-version` finds out why it is not higher."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[tool.ruff]", 1)[1].split("[tool.ruff.lint]", 1)[0]
    for rule in VERSION_GATED_RULES:
        assert rule in section, (
            f"{rule} is gated behind [tool.ruff] target-version but is not "
            "explained next to it in pyproject.toml"
        )


def test_the_gated_rules_are_not_silenced_by_the_ignore_list():
    """They stay off because of the target, not because they were suppressed."""
    ignored = set(_read_pyproject()["tool"]["ruff"]["lint"]["ignore"])
    assert not (ignored & set(VERSION_GATED_RULES)), (
        "A version-gated rule was added to the ignore list; it would then stay "
        "off after the Python floor moves, which is not the intent."
    )
