"""Test preparation: isolated data copies, setup and fault injection.

The preparation layer is the programmatic entry used by tests and by a future
runner. It starts one site per case/run, owns an isolated SQLite database,
applies the manifest setup and optional fault injections, and offers reset and
judge calls. It never exposes hidden truth through the HTTP site.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..cases import CaseDefinition, load_case
from ...protocol import validate_run_id
from .faults import FaultDefinition
from .server import LocalSite

DEFAULT_RUNTIME_DIR = Path("runtime") / "phase2_d1"


@dataclass
class PreparedCase:
    """One started case/run with its own database and local site."""

    case: CaseDefinition
    seed: int
    site: LocalSite
    run_dir: Path

    @property
    def base_url(self) -> str:
        return self.site.base_url

    @property
    def db_path(self) -> Path:
        return self.site.db_path

    def reset(self, seed: int | None = None) -> "PreparedCase":
        if seed is not None:
            allowed = set(self.case.seeds) | {
                int(item) for item in self.case.support.get("reserved_seeds", ())
            }
            if int(seed) not in allowed:
                raise ValueError(
                    f"{self.case.case_id}: seed {seed} is not one of the declared "
                    f"seeds {sorted(allowed)}"
                )
            self.seed = int(seed)
        self.site.reset(self.seed)
        return self

    def agent_view(self) -> dict[str, Any]:
        view = self.case.agent_view()
        view["start_url"] = self.site.url(self.case.start_path)
        return view

    def hidden_truth(self, *, role: str | None = None, item_type: str | None = None):
        return self.site.hidden_truth(role=role, item_type=item_type)

    def hidden_snapshot(self) -> dict[str, Any]:
        return self.site.hidden_snapshot()

    def submission_count(self, item_id: str | None = None) -> int:
        return self.site.submission_count(item_id)

    def fault_events(self) -> list[dict[str, Any]]:
        return self.site.fault_events()

    def inject_fault(self, fault: Mapping[str, Any]) -> FaultDefinition:
        return self.site.inject_fault(fault)

    def judge(
        self,
        *,
        evidence: Any = None,
        answer: Any = None,
        evidence_root: str | Path | None = None,
    ):
        from ..evaluation import judge as judge_run

        return judge_run(self, evidence=evidence, answer=answer, evidence_root=evidence_root)

    def stop(self) -> None:
        self.site.stop()

    def __enter__(self) -> "PreparedCase":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class TestPreparer:
    """Creates and resets isolated local test runs."""

    def __init__(self, runtime_dir: str | Path = DEFAULT_RUNTIME_DIR) -> None:
        self.runtime_dir = Path(runtime_dir)

    def prepare(
        self,
        case_id: str,
        seed: int = 101,
        *,
        run_id: str | None = None,
        faults: Iterable[Mapping[str, Any]] | None = None,
    ) -> PreparedCase:
        case = load_case(case_id)
        allowed = set(case.seeds) | {
            int(item) for item in case.support.get("reserved_seeds", ())
        }
        if int(seed) not in allowed:
            raise ValueError(
                f"{case_id}: seed {seed} is not one of the declared seeds {sorted(allowed)}"
            )
        run_id = run_id or f"{case_id.lower()}-{int(seed)}-{uuid.uuid4().hex[:8]}"
        validate_run_id(run_id)
        run_dir = (self.runtime_dir / "runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        fault_plan = [*case.fault_plan, *(faults or ())]
        site = LocalSite(
            run_dir / "site.sqlite3",
            case_id=case.case_id,
            case_version=case.case_version,
            seed=int(seed),
            generator_params=case.generator_params,
            hidden_config={"judge": case.judge_config, "setup": case.setup},
            fault_plan=fault_plan,
            setup=case.setup,
        )
        site.start()
        return PreparedCase(case=case, seed=int(seed), site=site, run_dir=run_dir)

    def reset(self, prepared: PreparedCase, seed: int | None = None) -> PreparedCase:
        return prepared.reset(seed)

    def judge(self, prepared: PreparedCase, **kwargs: Any):
        return prepared.judge(**kwargs)


def start(
    case_id: str,
    seed: int = 101,
    *,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    run_id: str | None = None,
    faults: Iterable[Mapping[str, Any]] | None = None,
) -> PreparedCase:
    """Programmatic entry: start one isolated local case run."""

    return TestPreparer(runtime_dir).prepare(
        case_id, seed, run_id=run_id, faults=faults
    )


def reset(prepared: PreparedCase, seed: int | None = None) -> PreparedCase:
    """Programmatic entry: reset a prepared case (same seed keeps the data)."""

    return prepared.reset(seed)


def judge(prepared: PreparedCase, **kwargs: Any):
    """Programmatic entry: judge a prepared case from hidden state."""

    return prepared.judge(**kwargs)


__all__ = [
    "DEFAULT_RUNTIME_DIR",
    "PreparedCase",
    "TestPreparer",
    "judge",
    "reset",
    "start",
]
