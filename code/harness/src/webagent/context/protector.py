"""The G-wave protector: assemble the two harness blocks and check every send.

One protector belongs to one official task and is created only when
``context.enabled`` is on. It has two jobs on the live path and both are plain
program rules:

* :meth:`ContextProtector.turn_note` renders the ledger short view and the
  unconfirmed-effect list. The driver appends that text to the prompt it is
  assembling, so the necessary ledger actually enters the sent prompt instead of
  only existing in a tool the model may never call.
* :meth:`ContextProtector.check_final_send` runs once before each model call on
  the exact prompt about to be handed to the model adapter. It verifies that the
  task text, the current AX view and both harness blocks are present
  byte-for-byte, measures the request against the project budget and refuses the
  send with an explicit error instead of dropping a protected region.

No model is called here and no clock or random value is read; a report is a pure
function of the session/ledger state and the prompt string.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..budget import BudgetLimits, TokenCounter
from ..config import ContextSettings
from ..protocol import Observation
from ..session import Session
from ..state.store import StateStore
from .errors import ContextBudgetExceeded, ContextProtectionError
from .guard import ContextCheckReport, describe, evaluate
from .sections import (
    ContextBlock,
    NecessarySection,
    byte_length,
    effects_block,
    ledger_block,
    build_necessary_sections,
)

#: One JSON line per measured final send, next to the run's other evidence.
RECORDS_FILE_NAME = "context_checks.jsonl"


class ContextProtector:
    """Necessary-context protection for one task's final sends."""

    def __init__(
        self,
        session: Session,
        store: StateStore | None,
        *,
        settings: ContextSettings,
        counter: TokenCounter,
        limits: BudgetLimits | None = None,
        branch: str | None = None,
    ) -> None:
        if not isinstance(session, Session):
            raise TypeError("ContextProtector needs a webagent Session")
        if store is not None and not isinstance(store, StateStore):
            raise TypeError("ContextProtector store must be a StateStore or None")
        if not isinstance(settings, ContextSettings):
            raise TypeError("ContextProtector needs ContextSettings")
        if not isinstance(counter, TokenCounter):
            raise TypeError("ContextProtector needs a TokenCounter")
        if branch is not None and (not isinstance(branch, str) or not branch.strip()):
            raise ValueError("ContextProtector branch must be a non-empty string or None")
        base = limits or BudgetLimits()
        self.session = session
        self.store = store
        self.settings = settings
        self.counter = counter
        self.branch = branch
        # The effective limits: the project token/byte budget from the context
        # section, the official message limit from the shared limits section.
        self.limits = BudgetLimits(
            input_tokens=settings.max_input_tokens,
            planned_context_tokens=max(
                settings.max_input_tokens, base.planned_context_tokens
            ),
            max_messages=base.max_messages,
            max_request_bytes=settings.max_request_bytes,
            max_model_calls=base.max_model_calls,
            deadline_margin_seconds=base.deadline_margin_seconds,
        )
        self._blocks: tuple[ContextBlock, ContextBlock] | None = None
        self._trimmed_optional_chars = 0
        self._optional_bytes = 0
        self._last_report: ContextCheckReport | None = None
        self._checks = 0

    # ------------------------------------------------------------------ views
    def current_blocks(self) -> tuple[ContextBlock, ContextBlock]:
        """The ledger and effect blocks for the state as it is right now."""

        ledger = ledger_block(
            self.store,
            max_chars=self.settings.ledger_max_chars,
            branch=self.branch,
        )
        effects = effects_block(
            self.store,
            max_chars=self.settings.effects_max_chars,
        )
        self._blocks = (ledger, effects)
        self._trimmed_optional_chars = ledger.trimmed_optional_chars
        self._optional_bytes = byte_length(ledger.optional)
        return self._blocks

    def turn_note(self) -> str:
        """The text the driver appends to the prompt it is assembling this turn."""

        ledger, effects = self.current_blocks()
        return ledger.text + "\n\n" + effects.text

    # ------------------------------------------------------------------- check
    def check_final_send(self, prompt: Any) -> ContextCheckReport:
        """Protect and measure the exact prompt about to be sent; raise on refusal."""

        sections = self._sections()
        report = evaluate(
            prompt=prompt,
            sections=sections,
            limits=self.limits,
            counter=self.counter,
            trimmed_optional_chars=self._trimmed_optional_chars,
            optional_bytes=self._optional_bytes,
            state=self._state_info(),
        )
        self._checks += 1
        self._last_report = report
        self._record(report)
        if not report.mandatory_intact:
            raise ContextProtectionError(
                "the final send lost a necessary context region ("
                + ", ".join(report.missing)
                + "); the request is refused instead of sending without it; "
                + describe(report),
                report=report,
            )
        if report.over_limit:
            raise ContextBudgetExceeded(
                "the final send exceeds the project context budget; every necessary "
                "region is intact and no message is trimmed at this exit "
                "(optional ledger detail is shortened only while the block is "
                "built); " + describe(report),
                report=report,
            )
        return report

    @property
    def last_report(self) -> ContextCheckReport | None:
        return self._last_report

    @property
    def checks(self) -> int:
        """How many final sends this protector measured for the task."""

        return self._checks

    # ---------------------------------------------------------------- internal
    def _sections(self) -> tuple[NecessarySection, ...]:
        session = self.session
        task = session.task
        observation = session.observation
        if not isinstance(task, str) or not task.strip():
            raise ContextProtectionError(
                "the final send has no task text; no model request is allowed"
            )
        if not isinstance(observation, Observation):
            raise ContextProtectionError(
                "the final send has no current observation; no model request is allowed"
            )
        ledger, effects = self._blocks or self.current_blocks()
        return build_necessary_sections(
            task=task,
            observation=observation,
            ledger=ledger,
            effects=effects,
        )

    def _state_info(self) -> Mapping[str, Any]:
        store = self.store
        info: dict[str, Any] = {"branch": self.branch}
        if store is not None:
            state = store.state
            info["revision"] = state.revision
            info["plan_version"] = state.plan_version
        return info

    @property
    def records_path(self) -> str:
        """Where the per-send reports are recorded for this run."""

        return str(self.session.evidence.run_dir / RECORDS_FILE_NAME)

    def _record(self, report: ContextCheckReport) -> None:
        """Append this send's measurement to the run's context records.

        The harness keeps its own file instead of a new event type: the event
        schema is shared with the other phase-2 waves, and a refused send must
        still leave a readable record of what was measured.
        """

        try:
            path = self.session.evidence.run_dir / RECORDS_FILE_NAME
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        except Exception:  # noqa: BLE001 - recording cannot change the refusal
            pass


__all__ = ["RECORDS_FILE_NAME", "ContextProtector"]
