from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from webagent.config import CONFIG_SCHEMA_VERSION, ConfigError, load_config, parse_config

CONFIGS = Path(__file__).parents[1] / "configs"


def load_raw(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_shipped_configs_parse_with_expected_defaults() -> None:
    official = load_config(CONFIGS / "official.yaml")
    assert official.profile == "official" and official.backend == "official"
    assert official.agent.strategy == "single"
    assert official.agent.tool_calling_mode == "react"
    assert official.agent.max_iterations is None
    assert official.agent.capability_probe is False
    assert official.agent.sub_agents is False
    assert official.agent.fallback is False
    assert official.agent.speculative_execution is False
    assert official.agent.cross_task_memory is False
    assert official.model.base_url_env is None
    assert official.budget_limits().input_tokens == 24_000
    assert official.budget_limits().planned_context_tokens == 32_768

    real = load_config(CONFIGS / "real.yaml")
    assert real.model.base_url_env == "WEBAGENT_MODEL_BASE_URL"
    assert real.model.api_key_env == "WEBAGENT_MODEL_API_KEY"
    assert real.env_names()["model_api_key"] == "WEBAGENT_MODEL_API_KEY"

    fake = load_config(CONFIGS / "fake.yaml")
    assert fake.profile == "fake" and fake.backend == "fake"
    assert fake.fake.script == "tests/fixtures/fake_scripts/basic.json"
    assert fake.token_counter().source == "estimate:chars_div_4"


def test_summary_is_non_sensitive_and_carries_no_values() -> None:
    config = load_config(CONFIGS / "real.yaml")
    summary = config.summary()
    text = str(summary)
    assert "WEBAGENT_MODEL_API_KEY" in text
    assert "sk-" not in text.lower()
    redacted = config.redactor().redact(
        {"password": "hunter2", "model_api_key": "sk-REAL-SECRET-123456"}
    )
    assert redacted == {"password": "[REDACTED]", "model_api_key": "[REDACTED]"}


def test_unknown_keys_and_invalid_profiles() -> None:
    raw = load_raw("official.yaml")
    raw["extra"] = True
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = load_raw("official.yaml")
    raw["backend"] = "playwright"
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = load_raw("real.yaml")
    raw["model"].pop("api_key_env")
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = load_raw("official.yaml")
    raw["agent"]["capability_probe"] = True
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = load_raw("fake.yaml")
    raw["agent"]["max_iterations"] = 10
    assert parse_config(raw).agent.max_iterations == 10


def test_credentials_in_config_are_rejected() -> None:
    raw = load_raw("real.yaml")
    raw["model"]["api_key"] = "value"
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = load_raw("real.yaml")
    raw["model"]["name"] = "sk-REAL-SECRET-123456"
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = load_raw("real.yaml")
    raw["model"]["note"] = "Authorization: Bearer abcdefghijklmnop"
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_empty_and_missing_config_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(empty)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_example_env_contains_only_fake_placeholders() -> None:
    text = (CONFIGS / "example.env").read_text(encoding="utf-8")
    assert "example.invalid" in text
    assert "sk-FAKE" in text
    assert "hunter2" not in text
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        assert "=" in line


def test_schema_version_is_checked() -> None:
    raw = load_raw("official.yaml")
    raw["schema_version"] = "wrong"
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = load_raw("official.yaml")
    raw["schema_version"] = CONFIG_SCHEMA_VERSION
    parse_config(copy.deepcopy(raw))

def test_official_text_mode_and_deadline_margin_are_explicit_configuration() -> None:
    official = load_config(CONFIGS / "official.yaml")
    assert official.official.text_mode == "decompose"
    assert official.budget.deadline_margin_seconds == 0.0
    assert official.budget_limits().deadline_margin_seconds == 0.0
    assert official.summary()["official"]["text_mode"] == "decompose"
    assert official.summary()["budget"]["deadline_margin_seconds"] == 0.0

    raw = load_raw("official.yaml")
    raw["budget"].pop("deadline_margin_seconds")
    assert parse_config(raw).budget_limits().deadline_margin_seconds == 0.0

    raw = load_raw("official.yaml")
    raw["official"]["text_mode"] = "raw"
    assert parse_config(raw).official.text_mode == "raw"

    raw = load_raw("official.yaml")
    raw["official"]["text_mode"] = "escaped"
    with pytest.raises(ConfigError):
        parse_config(raw)

    raw = load_raw("real.yaml")
    raw["official"] = {"text_mode": "raw"}
    with pytest.raises(ConfigError):
        parse_config(raw)

    raw = load_raw("official.yaml")
    raw["budget"]["deadline_margin_seconds"] = -1
    with pytest.raises(ConfigError):
        parse_config(raw)
