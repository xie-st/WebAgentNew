"""
Hypothesis-driven fuzz tests for configuration parsing & merging.

Targets the YAML/JSON config pipeline a user feeds with arbitrary files:

  * ``ConfigLoader._substitute_env_vars`` — ``${VAR}`` / ``${VAR:default}``
    expansion over untrusted file text.
  * ``ConfigLoader._load_single_config`` — YAML/JSON parse + extension routing.
  * ``Config.update`` / ``Config._merge_recursive`` — deep merge of arbitrary
    nested dicts.
  * ``Config`` attribute/item access.

Asserts that:
  1. Env-var substitution never crashes on arbitrary content and always returns
     a string; an undefined variable is left as its literal placeholder (never
     a partial/secret leak).
  2. Loading a malformed YAML/JSON file raises a *clear* typed error
     (``ValueError`` / ``yaml.YAMLError`` / ``json.JSONDecodeError``), never a
     bare crash or a silent empty config.
  3. ``Config.update`` deep-merges arbitrary nested dicts without crashing and
     without mutating the *source* dict.
  4. Attribute access on a missing key raises ``AttributeError`` (never returns
     a misleading empty Config).

Exit criterion: ≥200 examples per text/merge test.
"""

from __future__ import annotations

import json

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.config.loader import Config, ConfigLoader
from effgen.config.validator import ConfigValidator

pytestmark = pytest.mark.fuzz

# ---------------------------------------------------------------------------
# Secret leak guard — substitution must never emit a partial fake key.
# ---------------------------------------------------------------------------

_FAKE_SECRET = "sk-FUZZ-not-a-real-key-0000000000000000"


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Text peppered with ${...} placeholders and braces to stress the regex.
_ENV_TEXT = st.text(
    alphabet="${}:_-ABCabc012 \n=",
    min_size=0,
    max_size=200,
)

_json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**6), max_value=10**6)
    | st.text(max_size=15)
)

_config_dicts = st.recursive(
    st.dictionaries(st.text(min_size=1, max_size=8), _json_scalars, max_size=4),
    lambda children: st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=4),
    max_leaves=20,
)


# ---------------------------------------------------------------------------
# Env-var substitution
# ---------------------------------------------------------------------------


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(_ENV_TEXT)
def test_substitute_never_crashes(text: str) -> None:
    """Arbitrary text through the ${VAR} substituter returns a string, never raises."""
    loader = ConfigLoader()
    out = loader._substitute_env_vars(text)
    assert isinstance(out, str)


def test_substitute_undefined_left_literal() -> None:
    """An undefined ${VAR} is left as the literal placeholder (no empty/secret leak)."""
    loader = ConfigLoader()
    out = loader._substitute_env_vars("token=${DEFINITELY_UNDEFINED_VAR_XYZ}")
    assert "${DEFINITELY_UNDEFINED_VAR_XYZ}" in out


def test_substitute_default_used(monkeypatch) -> None:
    """${VAR:default} uses the default when VAR is unset, env value when set."""
    loader = ConfigLoader()
    monkeypatch.delenv("EFFGEN_FUZZ_VAR", raising=False)
    assert loader._substitute_env_vars("${EFFGEN_FUZZ_VAR:fallback}") == "fallback"
    monkeypatch.setenv("EFFGEN_FUZZ_VAR", "live")
    assert loader._substitute_env_vars("${EFFGEN_FUZZ_VAR:fallback}") == "live"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(st.text(alphabet="${}:_ABC ", max_size=80))
def test_substitute_never_leaks_secret(text: str) -> None:
    """Even if a placeholder references a secret env var, only the resolved value
    (not a mangled partial) is emitted; unresolved placeholders stay literal.

    Env is set/cleared directly (hypothesis forbids function-scoped fixtures).
    """
    import os

    prev = os.environ.get("EFFGEN_SECRET_FUZZ")
    os.environ["EFFGEN_SECRET_FUZZ"] = _FAKE_SECRET
    try:
        loader = ConfigLoader()
        out = loader._substitute_env_vars(text)
        # The substituter must never produce a *truncated* version of the secret.
        for n in range(8, len(_FAKE_SECRET)):
            partial = _FAKE_SECRET[:n]
            if partial in out:
                assert _FAKE_SECRET in out  # only the whole value, never a fragment
    finally:
        if prev is None:
            os.environ.pop("EFFGEN_SECRET_FUZZ", None)
        else:
            os.environ["EFFGEN_SECRET_FUZZ"] = prev


# ---------------------------------------------------------------------------
# Malformed-file loading → clear typed error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ext,content",
    [
        (".yaml", "key: [unclosed\n  bad: : :"),
        (".yaml", "\t- not\n   valid: : indentation"),
        (".json", "{not valid json,,,}"),
        (".json", '{"a": '),
    ],
)
def test_malformed_file_raises_clear_error(tmp_path, ext, content) -> None:
    """A malformed config file raises a typed parse error, not a silent crash."""
    p = tmp_path / f"bad{ext}"
    p.write_text(content)
    loader = ConfigLoader()
    with pytest.raises((ValueError, yaml.YAMLError, json.JSONDecodeError)):
        loader.load_config(p, validate=False)


def test_unsupported_extension_raises_value_error(tmp_path) -> None:
    """An unknown extension fails clearly rather than silently."""
    p = tmp_path / "config.toml"
    p.write_text("a = 1")
    loader = ConfigLoader()
    with pytest.raises(ValueError):
        loader.load_config(p, validate=False)


def test_missing_file_raises_filenotfound() -> None:
    loader = ConfigLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_config("/nonexistent/path/to/config.yaml", validate=False)


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(_config_dicts, _config_dicts)
def test_merge_never_crashes_and_is_nondestructive(base: dict, other: dict) -> None:
    """Deep-merging two arbitrary nested dicts never crashes; source is untouched."""
    import copy

    other_snapshot = copy.deepcopy(other)
    cfg = Config(data=copy.deepcopy(base))
    cfg.update(other)
    assert isinstance(cfg.to_dict(), dict)
    # update() must not mutate the *other* dict it merged from.
    assert other == other_snapshot


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_config_dicts)
def test_roundtrip_via_yaml_and_json(data: dict) -> None:
    """Any config dict survives a YAML and JSON serialise→load round-trip."""
    y = yaml.safe_load(yaml.safe_dump(data))
    j = json.loads(json.dumps(data))
    assert Config(data=y).to_dict() == data
    assert Config(data=j).to_dict() == data


def test_missing_attribute_raises() -> None:
    cfg = Config(data={"present": 1})
    assert cfg.present == 1
    with pytest.raises(AttributeError):
        _ = cfg.absent


# ---------------------------------------------------------------------------
# A file that parses but is not a config document
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".yaml", ".json"])
@pytest.mark.parametrize(
    "content",
    ["[]", "[1, 2, 3]", '"a bare string"', "12", "true", '[{"models": {}}]'],
)
def test_a_document_that_is_not_a_mapping_names_the_file(tmp_path, ext, content) -> None:
    """A config is a mapping of settings; anything else is refused at its file.

    Loading one used to succeed and then fail far away — on a merge, a lookup, or
    a validation pass — with no mention of which file caused it.
    """
    path = tmp_path / f"config{ext}"
    path.write_text(content, encoding="utf-8")
    loader = ConfigLoader(config_dir=str(tmp_path))
    with pytest.raises(ValueError, match="must contain a mapping"):
        loader.load_config(path, validate=False)
    with pytest.raises(ValueError) as excinfo:
        loader.load_config(path, validate=False)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize("ext", [".yaml", ".json"])
def test_an_empty_config_file_loads_as_an_empty_config(tmp_path, ext) -> None:
    path = tmp_path / f"config{ext}"
    path.write_text("{}" if ext == ".json" else "", encoding="utf-8")
    loader = ConfigLoader(config_dir=str(tmp_path))
    assert loader.load_config(path, validate=False).to_dict() == {}


@pytest.mark.parametrize(
    "document",
    [
        {"models": None},
        {"models": "text"},
        {"models": [1, 2]},
        {"models": {"a": "not a mapping"}},
        {"models": {"a": {"type": "openai"}}, "fallback_chain": "a"},
        {"tools": 5},
        {"tools": {"a": 1}},
        {"tools": {"a": {"enabled": True}}, "tool_selection": {"recommendations": 5}},
        {"mcp_servers": "text"},
        {"system_prompts": ["x"]},
        {"system_prompts": {"a": 5}},
        {"chains": 5},
        {"few_shot_examples": {"a": "text"}},
        {"slm_optimization": {"max_few_shot_examples": "many"}},
        {"slm_optimization": []},
    ],
)
def test_a_wrong_typed_section_is_a_validation_error_not_a_crash(document) -> None:
    """Validation reports a wrong-typed section; it does not raise out of a check."""
    validator = ConfigValidator()
    for check in (
        validator.validate_models,
        validator.validate_tools,
        validator.validate_prompts,
    ):
        result = check(document)
        assert isinstance(result.valid, bool)
        assert isinstance(result.errors, list)


@pytest.mark.parametrize("document", [None, [], "text", 5, True, [{"models": {}}]])
def test_a_validator_given_a_non_mapping_reports_it(document) -> None:
    validator = ConfigValidator()
    for check in (
        validator.validate_models,
        validator.validate_tools,
        validator.validate_prompts,
    ):
        result = check(document)
        assert not result.valid
        assert any("must be a mapping" in e for e in result.errors)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_config_dicts)
def test_validation_never_raises_on_an_arbitrary_document(data: dict) -> None:
    validator = ConfigValidator()
    for check in (
        validator.validate_models,
        validator.validate_tools,
        validator.validate_prompts,
    ):
        assert isinstance(check(data).valid, bool)
