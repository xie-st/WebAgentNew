"""Non-sensitive configuration loading and validation.

Configuration files contain environment *variable names*, never values. Any
credential-looking key or value is rejected so a committed config cannot leak a
secret by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .budget import (
    BudgetLimits,
    HeuristicTokenCounter,
    TokenCounter,
    TiktokenTokenCounter,
)
from .errors import ConfigError
from .redaction import Redactor

CONFIG_SCHEMA_VERSION = "webagent.config.v1"

_SENSITIVE_KEY = re.compile(
    r"(?i)^(password|passwd|secret|api[_-]?key|apikey|api[_-]?secret|client[_-]?secret|"
    r"token|auth[_-]?token|access[_-]?token|refresh[_-]?token|connection[_-]?token|"
    r"authorization|credential|credentials)$"
)
_SECRET_VALUE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{6,}|bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class AgentConfig:
    strategy: str = "single"
    tool_calling_mode: str = "react"
    max_iterations: int | None = None
    capability_probe: bool = False
    sub_agents: bool = False
    fallback: bool = False
    speculative_execution: bool = False
    cross_task_memory: bool = False


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    name: str
    context_length: int
    tokenizer: str = "tiktoken:cl100k_base"
    token_count_source: str = "estimated"
    base_url_env: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class BudgetConfig:
    input_tokens: int = 24_000
    planned_context_tokens: int = 32_768
    max_model_calls: int | None = None


@dataclass(frozen=True)
class LimitsConfig:
    max_messages: int = 256
    max_request_bytes: int = 1024 * 1024


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    redact_extra: tuple[str, ...] = ()


@dataclass(frozen=True)
class FakeSettings:
    script: str | None = None
    site: str | None = None


@dataclass(frozen=True)
class AppConfig:
    profile: str
    backend: str
    runtime_dir: str = "runtime"
    agent: AgentConfig = field(default_factory=AgentConfig)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider="fake", name="fake-scripted", context_length=32768, tokenizer="heuristic",
        token_count_source="estimated",
    ))
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    fake: FakeSettings = field(default_factory=FakeSettings)
    schema_version: str = CONFIG_SCHEMA_VERSION
    source_path: str | None = None

    def budget_limits(self) -> BudgetLimits:
        return BudgetLimits(
            input_tokens=self.budget.input_tokens,
            planned_context_tokens=self.budget.planned_context_tokens,
            max_messages=self.limits.max_messages,
            max_request_bytes=self.limits.max_request_bytes,
            max_model_calls=self.budget.max_model_calls,
        )

    def token_counter(self) -> TokenCounter:
        tokenizer = self.model.tokenizer
        if tokenizer.startswith("tiktoken:"):
            encoding = tokenizer.split(":", 1)[1]
            return TiktokenTokenCounter(encoding)
        if tokenizer in {"heuristic", "estimate"}:
            return HeuristicTokenCounter()
        raise ConfigError(f"unsupported tokenizer {tokenizer!r}")

    def redactor(self, secrets: tuple[str, ...] = ()) -> Redactor:
        return Redactor(secrets=secrets, extra_patterns=self.logging.redact_extra)

    def env_names(self) -> Mapping[str, str | None]:
        """Environment variable names only; values are read by the runtime."""

        return {
            "model_base_url": self.model.base_url_env,
            "model_api_key": self.model.api_key_env,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "backend": self.backend,
            "runtime_dir": self.runtime_dir,
            "agent": {
                "strategy": self.agent.strategy,
                "tool_calling_mode": self.agent.tool_calling_mode,
                "max_iterations": self.agent.max_iterations,
                "capability_probe": self.agent.capability_probe,
                "sub_agents": self.agent.sub_agents,
                "fallback": self.agent.fallback,
                "speculative_execution": self.agent.speculative_execution,
                "cross_task_memory": self.agent.cross_task_memory,
            },
            "model": {
                "provider": self.model.provider,
                "name": self.model.name,
                "context_length": self.model.context_length,
                "tokenizer": self.model.tokenizer,
                "token_count_source": self.model.token_count_source,
                "env_names": dict(self.env_names()),
            },
            "budget": {
                "input_tokens": self.budget.input_tokens,
                "planned_context_tokens": self.budget.planned_context_tokens,
                "max_model_calls": self.budget.max_model_calls,
                "max_messages": self.limits.max_messages,
                "max_request_bytes": self.limits.max_request_bytes,
            },
            "schema_version": self.schema_version,
        }


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"unknown configuration keys in {where}: {', '.join(unknown)}")


def _expect_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _expect_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a boolean")
    return value


def _expect_int(value: Any, where: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}")
    return value


def _expect_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _scan_for_secrets(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _SENSITIVE_KEY.match(key_text) and not key_text.lower().endswith("_env"):
                raise ConfigError(
                    f"credential key {where}.{key_text} is not allowed; use an environment variable name"
                )
            _scan_for_secrets(item, f"{where}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_for_secrets(item, f"{where}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ConfigError(f"credential-looking value in {where} is not allowed")


def parse_config(data: Mapping[str, Any], *, source_path: str | None = None) -> AppConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("configuration root must be a mapping")
    _check_keys(
        data,
        {"schema_version", "profile", "backend", "runtime_dir", "agent", "model", "budget", "limits", "logging", "fake"},
        "root",
    )
    _scan_for_secrets(data, "config")
    schema_version = data.get("schema_version", CONFIG_SCHEMA_VERSION)
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"unsupported config schema_version {schema_version!r}")
    profile = _expect_text(data.get("profile"), "profile")
    backend = _expect_text(data.get("backend"), "backend")
    allowed = {
        "official": {"official"},
        "real": {"playwright", "browser-use"},
        "fake": {"fake"},
    }
    if profile not in allowed:
        raise ConfigError(f"profile must be one of {sorted(allowed)}")
    if backend not in allowed[profile]:
        raise ConfigError(f"backend {backend!r} is not valid for profile {profile!r}")
    runtime_dir = _expect_text(data.get("runtime_dir", "runtime"), "runtime_dir")

    agent_raw = _expect_mapping(data.get("agent", {}), "agent")
    _check_keys(
        agent_raw,
        {
            "strategy",
            "tool_calling_mode",
            "max_iterations",
            "capability_probe",
            "sub_agents",
            "fallback",
            "speculative_execution",
            "cross_task_memory",
        },
        "agent",
    )
    strategy = _expect_text(agent_raw.get("strategy", "single"), "agent.strategy")
    tool_calling_mode = _expect_text(
        agent_raw.get("tool_calling_mode", "react"), "agent.tool_calling_mode"
    )
    if strategy != "single":
        raise ConfigError("agent.strategy must be 'single' for phase 1")
    if tool_calling_mode != "react":
        raise ConfigError("agent.tool_calling_mode must be 'react' for phase 1")
    max_iterations = agent_raw.get("max_iterations")
    if max_iterations is not None:
        max_iterations = _expect_int(max_iterations, "agent.max_iterations")
    agent = AgentConfig(
        strategy=strategy,
        tool_calling_mode=tool_calling_mode,
        max_iterations=max_iterations,
        capability_probe=_expect_bool(agent_raw.get("capability_probe", False), "agent.capability_probe"),
        sub_agents=_expect_bool(agent_raw.get("sub_agents", False), "agent.sub_agents"),
        fallback=_expect_bool(agent_raw.get("fallback", False), "agent.fallback"),
        speculative_execution=_expect_bool(
            agent_raw.get("speculative_execution", False), "agent.speculative_execution"
        ),
        cross_task_memory=_expect_bool(
            agent_raw.get("cross_task_memory", False), "agent.cross_task_memory"
        ),
    )
    for disabled in (
        "capability_probe",
        "sub_agents",
        "fallback",
        "speculative_execution",
        "cross_task_memory",
    ):
        if getattr(agent, disabled):
            raise ConfigError(f"agent.{disabled} must be false for phase 1")

    model_raw = _expect_mapping(data.get("model"), "model")
    _check_keys(
        model_raw,
        {
            "provider",
            "name",
            "context_length",
            "tokenizer",
            "token_count_source",
            "base_url_env",
            "api_key_env",
        },
        "model",
    )
    provider = _expect_text(model_raw.get("provider"), "model.provider")
    if provider not in {"official", "openai_compatible", "fake"}:
        raise ConfigError("model.provider must be official, openai_compatible or fake")
    if profile == "official" and provider != "official":
        raise ConfigError("official profile requires model.provider=official")
    if profile == "real" and provider != "openai_compatible":
        raise ConfigError("real profile requires model.provider=openai_compatible")
    if profile == "fake" and provider != "fake":
        raise ConfigError("fake profile requires model.provider=fake")
    base_url_env = model_raw.get("base_url_env")
    api_key_env = model_raw.get("api_key_env")
    for name in ("base_url_env", "api_key_env"):
        value = model_raw.get(name)
        if value is not None and (not isinstance(value, str) or not _ENV_NAME.match(value)):
            raise ConfigError(f"model.{name} must be an uppercase environment variable name or null")
    if provider == "openai_compatible":
        if not base_url_env or not api_key_env:
            raise ConfigError("real model needs base_url_env and api_key_env names")
    token_count_source = _expect_text(
        model_raw.get("token_count_source", "estimated"), "model.token_count_source"
    )
    if token_count_source not in {"estimated", "exact"}:
        raise ConfigError("model.token_count_source must be estimated or exact")
    model = ModelConfig(
        provider=provider,
        name=_expect_text(model_raw.get("name"), "model.name"),
        context_length=_expect_int(model_raw.get("context_length", 32768), "model.context_length"),
        tokenizer=_expect_text(model_raw.get("tokenizer", "tiktoken:cl100k_base"), "model.tokenizer"),
        token_count_source=token_count_source,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
    )

    budget_raw = _expect_mapping(data.get("budget", {}), "budget")
    _check_keys(budget_raw, {"input_tokens", "planned_context_tokens", "max_model_calls"}, "budget")
    budget = BudgetConfig(
        input_tokens=_expect_int(budget_raw.get("input_tokens", 24_000), "budget.input_tokens"),
        planned_context_tokens=_expect_int(
            budget_raw.get("planned_context_tokens", 32_768), "budget.planned_context_tokens"
        ),
        max_model_calls=(
            None
            if budget_raw.get("max_model_calls") is None
            else _expect_int(budget_raw["max_model_calls"], "budget.max_model_calls")
        ),
    )
    limits_raw = _expect_mapping(data.get("limits", {}), "limits")
    _check_keys(limits_raw, {"max_messages", "max_request_bytes"}, "limits")
    limits = LimitsConfig(
        max_messages=_expect_int(limits_raw.get("max_messages", 256), "limits.max_messages"),
        max_request_bytes=_expect_int(
            limits_raw.get("max_request_bytes", 1024 * 1024), "limits.max_request_bytes"
        ),
    )
    logging_raw = _expect_mapping(data.get("logging", {}), "logging")
    _check_keys(logging_raw, {"level", "redact_extra"}, "logging")
    extra = logging_raw.get("redact_extra", [])
    if not isinstance(extra, (list, tuple)) or any(not isinstance(item, str) for item in extra):
        raise ConfigError("logging.redact_extra must be a list of regex strings")
    logging = LoggingConfig(
        level=_expect_text(logging_raw.get("level", "INFO"), "logging.level"),
        redact_extra=tuple(extra),
    )
    fake_raw = _expect_mapping(data.get("fake", {}), "fake")
    _check_keys(fake_raw, {"script", "site"}, "fake")
    if profile != "fake" and fake_raw:
        raise ConfigError("fake settings are only valid for the fake profile")
    fake = FakeSettings(
        script=fake_raw.get("script"),
        site=fake_raw.get("site"),
    )
    config = AppConfig(
        profile=profile,
        backend=backend,
        runtime_dir=runtime_dir,
        agent=agent,
        model=model,
        budget=budget,
        limits=limits,
        logging=logging,
        fake=fake,
        schema_version=schema_version,
        source_path=source_path,
    )
    # Reuse the same positive/limit validation as the budget packer.
    config.budget_limits()
    try:
        config.token_counter()
    except ConfigError:
        raise
    except Exception as exc:  # invalid tiktoken encoding or missing tokenizer
        raise ConfigError(f"cannot construct model.tokenizer {config.model.tokenizer!r}: {exc}") from exc
    return config


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path.name}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"configuration file is empty: {config_path}")
    return parse_config(raw, source_path=str(config_path))
