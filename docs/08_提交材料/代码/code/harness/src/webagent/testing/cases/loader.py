"""Case manifest loading, validation and the Agent-visible / hidden split.

Each ``cases/*.yaml`` file declares the task template, generator parameters,
public task parameters, setup, completion conditions, fault plan, reset method,
supported modes/budget and hidden judging parameters.

``CaseDefinition.agent_view()`` is the only payload a future runner may hand to
the model or place in a prompt. Hidden judge configuration, fault plans and
generator internals are separate attributes and are never serialized into it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..site.data import generate_dataset
from ..site.faults import normalize_fault_plan

_REQUIRED = (
    "case_id",
    "case_version",
    "title",
    "task_template",
    "data_generator",
    "public",
    "setup",
    "completion",
    "fault_plan",
    "reset",
    "support",
    "hidden",
)

_CASES_DIR = Path(__file__).resolve().parent
# Additional P-wave cases are kept in their own namespace so the frozen
# L01-L07 base manifest (and its exact-set regression test) is untouched.
_EXTENDED_CASES_DIR = _CASES_DIR / "extended"


class CaseValidationError(ValueError):
    """Manifest is incomplete or internally inconsistent."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaseValidationError(f"{name} must be a mapping")
    return value


def _deep_plain(value: Any) -> Any:
    """Return plain JSON-compatible data (no YAML aliases / tuples)."""

    if isinstance(value, Mapping):
        return {str(key): _deep_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    case_version: str
    title: str
    task_template: str
    data_generator: Mapping[str, Any]
    public: Mapping[str, Any]
    setup: Mapping[str, Any]
    completion: Mapping[str, Any]
    fault_plan: tuple[Mapping[str, Any], ...]
    reset: Mapping[str, Any]
    support: Mapping[str, Any]
    hidden: Mapping[str, Any]
    source_path: Path

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.support.get("seeds", ()))

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(str(mode) for mode in self.support.get("modes", ()))

    @property
    def start_path(self) -> str:
        return str(self.public.get("start_path", "/"))

    @property
    def task_params(self) -> Mapping[str, Any]:
        return _require_mapping(self.public.get("task_params", {}), "public.task_params")

    @property
    def budget_seconds(self) -> int:
        return int(self.support.get("budget_seconds", 600))

    @property
    def judge_config(self) -> Mapping[str, Any]:
        return _require_mapping(self.hidden.get("judge", {}), "hidden.judge")

    @property
    def generator_params(self) -> Mapping[str, Any]:
        params = _require_mapping(
            self.data_generator.get("params", {}), "data_generator.params"
        )
        return params

    def render_task(self) -> str:
        try:
            task = self.task_template.format(**dict(self.task_params))
        except (KeyError, IndexError, ValueError) as exc:
            raise CaseValidationError(
                f"{self.case_id}: task_template cannot be rendered: {exc}"
            ) from exc
        if not task.strip():
            raise CaseValidationError(f"{self.case_id}: rendered task is empty")
        return task

    def agent_view(self) -> dict[str, Any]:
        """The only case payload safe for a model prompt / Agent tool result."""

        return {
            "case_id": self.case_id,
            "case_version": self.case_version,
            "task": self.render_task(),
            "start_path": self.start_path,
            "task_params": _deep_plain(self.task_params),
            "budget_seconds": self.budget_seconds,
            "modes": list(self.modes),
        }

    def hidden_view(self) -> dict[str, Any]:
        """Test-side view: hidden judge config and fault truth. Never prompt it."""

        return {
            "case_id": self.case_id,
            "case_version": self.case_version,
            "hidden": _deep_plain(self.hidden),
            "fault_plan": [_deep_plain(fault) for fault in self.fault_plan],
            "data_generator": _deep_plain(self.data_generator),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_version": self.case_version,
            "title": self.title,
            "task_template": self.task_template,
            "data_generator": _deep_plain(self.data_generator),
            "public": _deep_plain(self.public),
            "setup": _deep_plain(self.setup),
            "completion": _deep_plain(self.completion),
            "fault_plan": [_deep_plain(fault) for fault in self.fault_plan],
            "reset": _deep_plain(self.reset),
            "support": _deep_plain(self.support),
            "hidden": _deep_plain(self.hidden),
        }


def _validate_support(support: Mapping[str, Any], case_id: str) -> None:
    seeds = support.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise CaseValidationError(f"{case_id}: support.seeds must be a non-empty list")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise CaseValidationError(f"{case_id}: seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise CaseValidationError(f"{case_id}: seeds must be unique")
    modes = support.get("modes")
    if not isinstance(modes, list) or not modes:
        raise CaseValidationError(f"{case_id}: support.modes must be a non-empty list")
    if not isinstance(support.get("budget_seconds", 600), int):
        raise CaseValidationError(f"{case_id}: budget_seconds must be an integer")


def _validate_generator(generator: Mapping[str, Any], case_id: str) -> None:
    if generator.get("name") != "catalog":
        raise CaseValidationError(
            f"{case_id}: data_generator.name must be 'catalog' for this batch"
        )
    params = _require_mapping(generator.get("params", {}), "data_generator.params")
    for key in ("products", "posts", "page_size"):
        value = params.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise CaseValidationError(
                f"{case_id}: data_generator.params.{key} must be a positive integer"
            )


def _validate_hidden(hidden: Mapping[str, Any], case_id: str) -> None:
    judge = _require_mapping(hidden.get("judge", {}), "hidden.judge")
    object_type = judge.get("object_type")
    if object_type not in ("product", "post"):
        raise CaseValidationError(f"{case_id}: hidden.judge.object_type is invalid")
    role = judge.get("role")
    if not isinstance(role, str) or not role:
        raise CaseValidationError(f"{case_id}: hidden.judge.role is required")
    expected_count = judge.get("expected_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 1:
        raise CaseValidationError(
            f"{case_id}: hidden.judge.expected_count must be a positive integer"
        )


def _validate_completion(completion: Mapping[str, Any], case_id: str) -> None:
    if completion.get("method") != "local_judge":
        raise CaseValidationError(f"{case_id}: completion.method must be 'local_judge'")
    criteria = completion.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise CaseValidationError(f"{case_id}: completion.criteria must be a non-empty list")


def _validate_data(case: CaseDefinition) -> None:
    judge = case.judge_config
    role = str(judge["role"])
    expected = int(judge["expected_count"])
    for seed in case.seeds:
        dataset = generate_dataset(seed, case.generator_params)
        matching = [record for record in dataset.hidden if record.role == role]
        if len(matching) != expected:
            raise CaseValidationError(
                f"{case.case_id}: seed {seed} generated {len(matching)} hidden {role} "
                f"records, expected {expected}"
            )
        if any(record.item_type != judge["object_type"] for record in matching):
            raise CaseValidationError(
                f"{case.case_id}: hidden role {role} mixes item types"
            )


def manifest_path(case_id: str) -> Path:
    """Resolve a case manifest, falling back to the extended-case namespace."""

    primary = _CASES_DIR / f"{case_id}.yaml"
    if primary.is_file():
        return primary
    return _EXTENDED_CASES_DIR / f"{case_id}.yaml"


def load_case(case_id: str, *, path: str | Path | None = None) -> CaseDefinition:
    source = Path(path) if path is not None else manifest_path(case_id)
    if not source.is_file():
        raise CaseValidationError(f"case manifest not found: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CaseValidationError(f"{source.name}: invalid YAML: {exc}") from exc
    raw = _require_mapping(raw, source.name)
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise CaseValidationError(
            f"{source.name}: missing required fields: {', '.join(missing)}"
        )
    if str(raw["case_id"]) != case_id:
        raise CaseValidationError(
            f"{source.name}: case_id {raw['case_id']!r} does not match requested {case_id!r}"
        )
    data_generator = _require_mapping(raw["data_generator"], "data_generator")
    public = _require_mapping(raw["public"], "public")
    setup = _require_mapping(raw["setup"], "setup")
    completion = _require_mapping(raw["completion"], "completion")
    reset = _require_mapping(raw["reset"], "reset")
    support = _require_mapping(raw["support"], "support")
    hidden = _require_mapping(raw["hidden"], "hidden")
    _validate_generator(data_generator, case_id)
    _validate_support(support, case_id)
    _validate_hidden(hidden, case_id)
    _validate_completion(completion, case_id)
    if not isinstance(raw["fault_plan"], list):
        raise CaseValidationError(f"{case_id}: fault_plan must be a list")
    normalized = normalize_fault_plan(raw["fault_plan"])
    task_params = _require_mapping(public.get("task_params", {}), "public.task_params")
    start_path = public.get("start_path", "/")
    if not isinstance(start_path, str) or not start_path.startswith("/"):
        raise CaseValidationError(f"{case_id}: public.start_path must start with '/'")
    page_size = int(data_generator["params"]["page_size"])
    if int(setup.get("page_size", page_size)) != page_size:
        raise CaseValidationError(
            f"{case_id}: setup.page_size must match data_generator.params.page_size"
        )
    case = CaseDefinition(
        case_id=str(raw["case_id"]),
        case_version=str(raw["case_version"]),
        title=str(raw["title"]),
        task_template=str(raw["task_template"]),
        data_generator=_deep_plain(data_generator),
        public=_deep_plain(public),
        setup=_deep_plain(setup),
        completion=_deep_plain(completion),
        fault_plan=tuple(fault.to_store_row() for fault in normalized),
        reset=_deep_plain(reset),
        support=_deep_plain(support),
        hidden=_deep_plain(hidden),
        source_path=source,
    )
    case.render_task()
    if not task_params:
        raise CaseValidationError(f"{case_id}: public.task_params must not be empty")
    _validate_data(case)
    return case


def list_cases(*, validate_data: bool = True) -> list[CaseDefinition]:
    cases: list[CaseDefinition] = []
    for path in sorted(_CASES_DIR.glob("*.yaml")):
        case_id = path.stem
        if validate_data:
            cases.append(load_case(case_id, path=path))
        else:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(raw, Mapping) and raw.get("case_id") == case_id:
                cases.append(load_case(case_id, path=path))
    return cases


def case_ids() -> tuple[str, ...]:
    return tuple(path.stem for path in sorted(_CASES_DIR.glob("*.yaml")))


def extended_case_ids() -> tuple[str, ...]:
    return tuple(path.stem for path in sorted(_EXTENDED_CASES_DIR.glob("*.yaml")))


def list_extended_cases(*, validate_data: bool = True) -> list[CaseDefinition]:
    cases: list[CaseDefinition] = []
    for path in sorted(_EXTENDED_CASES_DIR.glob("*.yaml")):
        case_id = path.stem
        if validate_data:
            cases.append(load_case(case_id, path=path))
        else:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(raw, Mapping) and raw.get("case_id") == case_id:
                cases.append(load_case(case_id, path=path))
    return cases


def agent_payload(case: CaseDefinition) -> str:
    """Convenience serialization used by tests to prove no hidden leak."""

    return json.dumps(case.agent_view(), ensure_ascii=False, sort_keys=True)


__all__ = [
    "CaseDefinition",
    "CaseValidationError",
    "agent_payload",
    "case_ids",
    "extended_case_ids",
    "list_cases",
    "list_extended_cases",
    "load_case",
    "manifest_path",
]
