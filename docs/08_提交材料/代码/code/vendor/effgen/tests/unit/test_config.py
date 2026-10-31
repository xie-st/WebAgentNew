"""Unit tests for configuration system."""

import os
from pathlib import Path

import pytest

from effgen.config.loader import Config, ConfigLoader


class TestConfig:
    """Tests for Config dataclass."""

    def test_create_empty_config(self):
        config = Config()
        assert config.data == {}

    def test_create_with_data(self):
        config = Config(data={"key": "value"})
        assert config["key"] == "value"

    def test_dict_access(self):
        config = Config(data={"a": 1, "b": {"c": 2}})
        assert config["a"] == 1
        assert config["b"]["c"] == 2

    def test_set_item(self):
        config = Config()
        config["key"] = "value"
        assert config["key"] == "value"

    def test_missing_key_raises(self):
        config = Config()
        with pytest.raises(KeyError):
            _ = config["nonexistent"]


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_yaml(self, tmp_dir):
        yaml_file = tmp_dir / "test.yaml"
        yaml_file.write_text("name: test\nvalue: 42\n")
        loader = ConfigLoader()
        config = loader.load_config(str(yaml_file))
        assert config["name"] == "test"
        assert config["value"] == 42

    def test_load_json(self, tmp_dir):
        import json
        json_file = tmp_dir / "test.json"
        json_file.write_text(json.dumps({"name": "test", "value": 42}))
        loader = ConfigLoader()
        config = loader.load_config(str(json_file))
        assert config["name"] == "test"
        assert config["value"] == 42

    def test_load_nonexistent_file(self):
        loader = ConfigLoader()
        with pytest.raises((OSError, FileNotFoundError)):
            loader.load_config("/nonexistent/path/config.yaml")

    def test_env_variable_substitution(self, tmp_dir):
        os.environ["EFFGEN_TEST_VAR"] = "hello_world"
        yaml_file = tmp_dir / "test.yaml"
        yaml_file.write_text("value: ${EFFGEN_TEST_VAR}\n")
        loader = ConfigLoader()
        config = loader.load_config(str(yaml_file))
        assert config["value"] is not None
        os.environ.pop("EFFGEN_TEST_VAR", None)

    def test_unparseable_yaml_names_the_file(self, tmp_dir):
        """A YAML syntax error reports the path, not an anonymous string."""
        yaml_file = tmp_dir / "broken.yaml"
        yaml_file.write_text("models:\n\tkey: 1\n")
        loader = ConfigLoader()
        with pytest.raises(Exception) as caught:
            loader.load_config(str(yaml_file), validate=False)
        message = str(caught.value)
        assert str(yaml_file) in message
        assert "<unicode string>" not in message

    def test_unparseable_json_names_the_file(self, tmp_dir):
        """An invalid JSON config reports the path and stays a ValueError."""
        json_file = tmp_dir / "broken.json"
        json_file.write_text("{not valid json,,,}")
        loader = ConfigLoader()
        with pytest.raises(ValueError) as caught:
            loader.load_config(str(json_file), validate=False)
        assert str(json_file) in str(caught.value)

    def test_a_merged_set_names_which_file_is_broken(self, tmp_dir):
        """Loading several files names the one that failed to parse."""
        good = tmp_dir / "good.yaml"
        good.write_text("name: test\n")
        broken = tmp_dir / "second.yaml"
        broken.write_text("models: [unclosed\n")
        loader = ConfigLoader()
        with pytest.raises(Exception) as caught:
            loader.load_config([str(good), str(broken)], validate=False)
        assert str(broken) in str(caught.value)


class TestValidationIsAGate:
    """``load_config(validate=True)`` used to accept anything.

    Each validator's result was discarded and the section's *contents* were
    passed where the whole document was expected, so the validator looked for
    ``config["models"]["models"]``, found nothing, and validated an empty set.
    """

    def test_an_invalid_document_is_refused(self, tmp_dir):
        bad = tmp_dir / "bad.yaml"
        bad.write_text("models:\n  broken:\n    no_type_field: 1\n")
        loader = ConfigLoader()
        with pytest.raises(ValueError) as caught:
            loader.load_config(str(bad), validate=True)
        assert "type" in str(caught.value)

    def test_the_refusal_lists_every_problem(self, tmp_dir):
        bad = tmp_dir / "bad.yaml"
        bad.write_text(
            "models:\n  one:\n    no_type: 1\n  two:\n    also_no_type: 2\n"
        )
        loader = ConfigLoader()
        with pytest.raises(ValueError) as caught:
            loader.load_config(str(bad), validate=True)
        assert len(getattr(caught.value, "errors", [])) == 2

    def test_the_refusal_names_the_file(self, tmp_dir):
        bad = tmp_dir / "bad.yaml"
        bad.write_text("models:\n  broken:\n    no_type_field: 1\n")
        loader = ConfigLoader()
        with pytest.raises(ValueError) as caught:
            loader.load_config(str(bad), validate=True)
        assert str(bad) in str(caught.value)

    def test_a_valid_document_is_accepted(self, tmp_dir):
        good = tmp_dir / "good.yaml"
        good.write_text("models:\n  m1:\n    type: openai\n    model_name: gpt-4o-mini\n")
        loader = ConfigLoader()
        loader.load_config(str(good), validate=True)
        assert loader.validate_config() is True

    @pytest.mark.parametrize(
        "name",
        ["config.yaml", "models.yaml", "tools.yaml", "prompts.yaml", "api_keys.yaml"],
    )
    def test_every_shipped_config_passes_its_own_gate(self, name):
        """A gate effGen's own configuration cannot pass is not a usable gate."""
        shipped = Path(__file__).resolve().parents[2] / "configs" / name
        if not shipped.exists():
            pytest.skip(f"{name} is not shipped in this tree")
        ConfigLoader().load_config(str(shipped), validate=True)

    def test_a_merged_load_records_the_files_it_read(self, tmp_dir):
        """Merging used to leave the file list empty, so reload() had nothing."""
        one = tmp_dir / "one.yaml"
        one.write_text("models:\n  m1:\n    type: openai\n    model_name: a\n")
        loader = ConfigLoader()
        loader.load_config(str(one), validate=True)
        assert [p.name for p in loader._config_files] == ["one.yaml"]
