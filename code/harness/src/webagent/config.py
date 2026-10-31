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
    deadline_margin_seconds: float = 0.0


@dataclass(frozen=True)
class LimitsConfig:
    max_messages: int = 256
    max_request_bytes: int = 1024 * 1024


@dataclass(frozen=True)
class OfficialBehaviors:
    """Explicit phase-2 behavior switches for the official entry point.

    Each switch removes the whole behavior from the production path when off:
    the tools are not registered, their schemas are not in the prompt and the
    finish gate is not passed to the driver. ``recovery`` enables the R-wave
    takeover/pause/handoff, bounded recovery and restart wiring; it defaults to
    off so the official path stays at its pre-R-wave behavior unless a run
    explicitly asks for it.
    """

    verification: bool = True
    state_tools: bool = True
    recovery: bool = False


@dataclass(frozen=True)
class OfficialSettings:
    """Wire-text policy for the official platform adapter.

    ``text_mode`` is the one switch a real-page verification run changes:

    * ``decompose`` (default) never puts a line break in one ``type`` action; a
      multi-line proposal becomes ``type`` + ``press [Enter]`` sub-actions.
    * ``raw`` sends the text unchanged, including line breaks, so the platform's
      real character semantics can be observed before the default is fixed.
    """

    text_mode: str = "decompose"
    behaviors: OfficialBehaviors = field(default_factory=OfficialBehaviors)


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    redact_extra: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationSettings:
    """Phase-2 verification bounds (E1).

    Semantic checks are allowed by default but remain opt-in per condition; the
    call cap is a local safety net on top of the shared model budget.
    """

    semantic_checks: bool = True
    max_semantic_calls: int = 20


@dataclass(frozen=True)
class RecoverySettings:
    """Phase-2 recovery bounds (E1); all values are explicit and enforceable."""

    stagnation_window: int = 6
    stagnation_repeats: int = 2
    max_recoveries_per_goal: int = 2
    max_recoveries_total: int = 5
    retry_backoff_seconds: tuple[float, ...] = (1.0, 2.0)
    reflection_limit: int = 3


@dataclass(frozen=True)
class ParallelConfig:
    """F-wave parallel scheduling skeleton settings.

    The whole section is optional and defaults to serial execution: only an
    explicit ``parallel`` block with ``enabled: true`` turns the coordinator on.
    When enabled without a capacity the documented default is two active
    branches. ``model_concurrency`` is fixed at one in this release, so taking
    browser waits in parallel can never widen the model-call concurrency,
    ``site_ops_per_second`` is one shared schedule for every branch, and
    ``backoff_seconds`` is the shared cooldown when a branch sees a 429 or a page
    cooling hint without a Retry-After value.
    """

    enabled: bool = False
    capacity: int = 2
    model_concurrency: int = 1
    site_ops_per_second: float = 1.0
    backoff_seconds: tuple[float, ...] = (5.0, 10.0, 20.0)
    max_rate_retries: int = 3

    @property
    def interval_seconds(self) -> float:
        return 1.0 / self.site_ops_per_second


@dataclass(frozen=True)
class ExperienceSettings:
    """Phase-2 approved text-experience loading (G wave, G06).

    ``enabled`` is the single independent switch. It defaults to off, so the
    shipped official path sends exactly its pre-G-wave prompt. ``allowlist`` is
    the local allowlist file that names approved entries; empty means zero load
    even when the switch is on. No page content can add an entry or change its
    approval state.
    """

    enabled: bool = False
    allowlist: str | None = None
    max_entries: int = 8
    max_chars: int = 4000


@dataclass(frozen=True)
class ContextSettings:
    """G-wave necessary-context protection and the final input-budget check.

    ``enabled`` is a single, independent switch: while it is false (the shipped
    default) no protector is built, nothing is appended to the prompt and the
    driver's send path is exactly the pre-G-wave one. When it is true, the
    harness assembles the task, current AX view, ledger short view and
    unconfirmed-effect list, appends its own ledger/effect blocks to the prompt
    and checks the final send once before every model call.

    ``max_input_tokens`` and ``max_request_bytes`` are the project-side limits
    checked on that same final send; they default to the existing input budget
    and the platform request size limit so the native budget/compaction path and
    the project check share one effective token ceiling. ``ledger_max_chars``
    and ``effects_max_chars`` bound the two harness blocks; only the optional
    ledger detail is ever shortened, never the task, AX text or effect records.
    """

    enabled: bool = False
    max_input_tokens: int = 24_000
    max_request_bytes: int = 1024 * 1024
    ledger_max_chars: int = 2_000
    effects_max_chars: int = 1_200


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
    official: OfficialSettings = field(default_factory=OfficialSettings)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    fake: FakeSettings = field(default_factory=FakeSettings)
    verification: VerificationSettings = field(default_factory=VerificationSettings)
    recovery: RecoverySettings = field(default_factory=RecoverySettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    experience: ExperienceSettings = field(default_factory=ExperienceSettings)
    schema_version: str = CONFIG_SCHEMA_VERSION
    source_path: str | None = None

    def budget_limits(self) -> BudgetLimits:
        return BudgetLimits(
            input_tokens=self.budget.input_tokens,
            planned_context_tokens=self.budget.planned_context_tokens,
            max_messages=self.limits.max_messages,
            max_request_bytes=self.limits.max_request_bytes,
            max_model_calls=self.budget.max_model_calls,
            deadline_margin_seconds=self.budget.deadline_margin_seconds,
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
                "deadline_margin_seconds": self.budget.deadline_margin_seconds,
            },
            "official": {
                "text_mode": self.official.text_mode,
                "behaviors": {
                    "verification": self.official.behaviors.verification,
                    "state_tools": self.official.behaviors.state_tools,
                    "recovery": self.official.behaviors.recovery,
                },
            },
            "verification": {
                "semantic_checks": self.verification.semantic_checks,
                "max_semantic_calls": self.verification.max_semantic_calls,
            },
            "recovery": {
                "stagnation_window": self.recovery.stagnation_window,
                "stagnation_repeats": self.recovery.stagnation_repeats,
                "max_recoveries_per_goal": self.recovery.max_recoveries_per_goal,
                "max_recoveries_total": self.recovery.max_recoveries_total,
                "retry_backoff_seconds": list(self.recovery.retry_backoff_seconds),
                "reflection_limit": self.recovery.reflection_limit,
            },
            "context": {
                "enabled": self.context.enabled,
                "max_input_tokens": self.context.max_input_tokens,
                "max_request_bytes": self.context.max_request_bytes,
                "ledger_max_chars": self.context.ledger_max_chars,
                "effects_max_chars": self.context.effects_max_chars,
            },
            "parallel": {
                "enabled": self.parallel.enabled,
                "capacity": self.parallel.capacity,
                "model_concurrency": self.parallel.model_concurrency,
                "site_ops_per_second": self.parallel.site_ops_per_second,
                "max_rate_retries": self.parallel.max_rate_retries,
            },
            "experience": {
                "enabled": self.experience.enabled,
                "allowlist": self.experience.allowlist,
                "max_entries": self.experience.max_entries,
                "max_chars": self.experience.max_chars,
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


def _expect_number(value: Any, where: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{where} must be a number >= {minimum}")
    return float(value)


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
        {
            "schema_version",
            "profile",
            "backend",
            "runtime_dir",
            "agent",
            "model",
            "budget",
            "limits",
            "official",
            "logging",
            "fake",
            "verification",
            "recovery",
            "context",
            "parallel",
            "experience",
        },
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
    _check_keys(
        budget_raw,
        {"input_tokens", "planned_context_tokens", "max_model_calls", "deadline_margin_seconds"},
        "budget",
    )
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
        deadline_margin_seconds=_expect_number(
            budget_raw.get("deadline_margin_seconds", 0.0),
            "budget.deadline_margin_seconds",
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
    official_raw = _expect_mapping(data.get("official", {}), "official")
    _check_keys(official_raw, {"text_mode", "behaviors"}, "official")
    if profile != "official" and official_raw:
        raise ConfigError("official settings are only valid for the official profile")
    text_mode = _expect_text(official_raw.get("text_mode", "decompose"), "official.text_mode")
    if text_mode not in {"raw", "decompose"}:
        raise ConfigError("official.text_mode must be 'raw' or 'decompose'")
    behaviors_raw = _expect_mapping(
        official_raw.get("behaviors", {}), "official.behaviors"
    )
    _check_keys(
        behaviors_raw, {"verification", "state_tools", "recovery"}, "official.behaviors"
    )
    behaviors = OfficialBehaviors(
        verification=_expect_bool(
            behaviors_raw.get("verification", True), "official.behaviors.verification"
        ),
        state_tools=_expect_bool(
            behaviors_raw.get("state_tools", True), "official.behaviors.state_tools"
        ),
        recovery=_expect_bool(
            behaviors_raw.get("recovery", False), "official.behaviors.recovery"
        ),
    )
    official = OfficialSettings(text_mode=text_mode, behaviors=behaviors)
    fake_raw = _expect_mapping(data.get("fake", {}), "fake")
    _check_keys(fake_raw, {"script", "site"}, "fake")
    if profile != "fake" and fake_raw:
        raise ConfigError("fake settings are only valid for the fake profile")
    fake = FakeSettings(
        script=fake_raw.get("script"),
        site=fake_raw.get("site"),
    )
    verification_raw = _expect_mapping(data.get("verification", {}), "verification")
    _check_keys(verification_raw, {"semantic_checks", "max_semantic_calls"}, "verification")
    verification = VerificationSettings(
        semantic_checks=_expect_bool(
            verification_raw.get("semantic_checks", True), "verification.semantic_checks"
        ),
        max_semantic_calls=_expect_int(
            verification_raw.get("max_semantic_calls", 20),
            "verification.max_semantic_calls",
        ),
    )
    recovery_raw = _expect_mapping(data.get("recovery", {}), "recovery")
    _check_keys(
        recovery_raw,
        {
            "stagnation_window",
            "stagnation_repeats",
            "max_recoveries_per_goal",
            "max_recoveries_total",
            "retry_backoff_seconds",
            "reflection_limit",
        },
        "recovery",
    )
    backoff_raw = recovery_raw.get("retry_backoff_seconds", [1.0, 2.0])
    if not isinstance(backoff_raw, (list, tuple)) or not backoff_raw:
        raise ConfigError("recovery.retry_backoff_seconds must be a non-empty list")
    backoff = tuple(
        _expect_number(item, f"recovery.retry_backoff_seconds[{index}]")
        for index, item in enumerate(backoff_raw)
    )
    recovery = RecoverySettings(
        stagnation_window=_expect_int(
            recovery_raw.get("stagnation_window", 6), "recovery.stagnation_window", minimum=2
        ),
        stagnation_repeats=_expect_int(
            recovery_raw.get("stagnation_repeats", 2),
            "recovery.stagnation_repeats",
            minimum=2,
        ),
        max_recoveries_per_goal=_expect_int(
            recovery_raw.get("max_recoveries_per_goal", 2),
            "recovery.max_recoveries_per_goal",
        ),
        max_recoveries_total=_expect_int(
            recovery_raw.get("max_recoveries_total", 5),
            "recovery.max_recoveries_total",
        ),
        retry_backoff_seconds=backoff,
        reflection_limit=_expect_int(
            recovery_raw.get("reflection_limit", 3), "recovery.reflection_limit"
        ),
    )
    context_raw = _expect_mapping(data.get("context", {}), "context")
    _check_keys(
        context_raw,
        {
            "enabled",
            "max_input_tokens",
            "max_request_bytes",
            "ledger_max_chars",
            "effects_max_chars",
        },
        "context",
    )
    context = ContextSettings(
        enabled=_expect_bool(context_raw.get("enabled", False), "context.enabled"),
        max_input_tokens=_expect_int(
            context_raw.get("max_input_tokens", 24_000), "context.max_input_tokens"
        ),
        max_request_bytes=_expect_int(
            context_raw.get("max_request_bytes", 1024 * 1024),
            "context.max_request_bytes",
        ),
        ledger_max_chars=_expect_int(
            context_raw.get("ledger_max_chars", 2_000), "context.ledger_max_chars"
        ),
        effects_max_chars=_expect_int(
            context_raw.get("effects_max_chars", 1_200), "context.effects_max_chars"
        ),
    )
    if context.enabled and context.max_input_tokens > budget.input_tokens:
        raise ConfigError(
            "context.max_input_tokens must not exceed budget.input_tokens; the "
            "native compaction budget and the project final send check share one "
            "effective token ceiling"
        )
    parallel_raw = _expect_mapping(data.get("parallel", {}), "parallel")
    _check_keys(
        parallel_raw,
        {
            "enabled",
            "capacity",
            "model_concurrency",
            "site_ops_per_second",
            "backoff_seconds",
            "max_rate_retries",
        },
        "parallel",
    )
    parallel_enabled = _expect_bool(parallel_raw.get("enabled", False), "parallel.enabled")
    parallel_capacity = _expect_int(
        parallel_raw.get("capacity", 2), "parallel.capacity", minimum=1
    )
    model_concurrency = _expect_int(
        parallel_raw.get("model_concurrency", 1), "parallel.model_concurrency", minimum=1
    )
    if model_concurrency != 1:
        raise ConfigError(
            "parallel.model_concurrency must be 1 in the first release; "
            "parallel browser waits never widen model concurrency"
        )
    site_ops_per_second = _expect_number(
        parallel_raw.get("site_ops_per_second", 1.0), "parallel.site_ops_per_second"
    )
    if site_ops_per_second <= 0:
        raise ConfigError("parallel.site_ops_per_second must be greater than zero")
    parallel_backoff_raw = parallel_raw.get("backoff_seconds", [5.0, 10.0, 20.0])
    if not isinstance(parallel_backoff_raw, (list, tuple)) or not parallel_backoff_raw:
        raise ConfigError("parallel.backoff_seconds must be a non-empty list")
    parallel_backoff = tuple(
        _expect_number(item, f"parallel.backoff_seconds[{index}]")
        for index, item in enumerate(parallel_backoff_raw)
    )
    max_rate_retries = _expect_int(
        parallel_raw.get("max_rate_retries", 3), "parallel.max_rate_retries", minimum=1
    )
    if profile == "official" and parallel_enabled:
        raise ConfigError(
            "official profile does not support parallel execution or Tab tools; "
            "parallel.enabled must stay false"
        )
    parallel = ParallelConfig(
        enabled=parallel_enabled,
        capacity=parallel_capacity,
        model_concurrency=model_concurrency,
        site_ops_per_second=site_ops_per_second,
        backoff_seconds=parallel_backoff,
        max_rate_retries=max_rate_retries,
    )
    experience_raw = _expect_mapping(data.get("experience", {}), "experience")
    _check_keys(
        experience_raw,
        {"enabled", "allowlist", "max_entries", "max_chars"},
        "experience",
    )
    experience_allowlist = experience_raw.get("allowlist")
    if experience_allowlist is not None and not isinstance(experience_allowlist, str):
        raise ConfigError("experience.allowlist must be a path string or null")
    if isinstance(experience_allowlist, str) and not experience_allowlist.strip():
        # The default is "empty"; both null and an empty string mean zero load.
        experience_allowlist = None
    experience = ExperienceSettings(
        enabled=_expect_bool(
            experience_raw.get("enabled", False), "experience.enabled"
        ),
        allowlist=experience_allowlist,
        max_entries=_expect_int(
            experience_raw.get("max_entries", 8), "experience.max_entries"
        ),
        max_chars=_expect_int(
            experience_raw.get("max_chars", 4000), "experience.max_chars"
        ),
    )
    config = AppConfig(
        profile=profile,
        backend=backend,
        runtime_dir=runtime_dir,
        agent=agent,
        model=model,
        budget=budget,
        limits=limits,
        official=official,
        logging=logging,
        fake=fake,
        verification=verification,
        recovery=recovery,
        context=context,
        parallel=parallel,
        experience=experience,
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
