"""Scripted character probe for the official platform.

The platform only executes actions that arrive through the frozen SDK's
``reset`` / ``step`` / ``close`` callbacks, so there is no side channel for
asking "what does a bracket or a tab mean on the real page?" This adapter
answers that by becoming a deliberately dumb SDK agent:

* it never plans or executes the assigned task;
* after ``reset`` it takes the current observation, picks the lowest-numbered
  typeable AX element and performs one fixed script;
* every step still goes through the normal ``Session`` proposal/dispatch path,
  so ``proposed``/``dispatched``/``observation``/``feedback`` events and the
  observation/screenshot evidence are written exactly as in a production run;
* it forces ``text_mode=raw`` for its own backend, because the whole point is to
  observe the platform's literal handling of ``[``, ``]``, ``\\`` and tab;
* it ends with ``stop [character probe completed; task not executed]`` and
  records the probe nature in ``result.json`` (never a fake task completion).

Activation is opt-in: pass ``probe=True`` to the constructor, or set
``WEBAGENT_PROBE=1``. With the flag off the subclass delegates every SDK hook to
:class:`OfficialAgent` and behaves exactly like the production adapter. The
module is new; no production file is modified.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from .backends.official import (
    OfficialActionFormatError,
    OfficialBackend,
    TextMode,
    validate_action_text,
)
from .budget import Budget, Deadline
from .clock import Clock
from .config import AppConfig
from .errors import DeadlineExceeded, SessionStateError
from .official_agent import OfficialAgent, _deadline_value
from .protocol import (
    ActionOp,
    FeedbackStatus,
    Finish,
    Observation,
    TerminationReason,
    WebAction,
    new_action_id,
)
from .session import Session

PROBE_ENV_VAR = "WEBAGENT_PROBE"

# Plaintext probe strings. They are kept as module constants so the report and
# the tests quote the same bytes that go on the wire.
#
# A: one line, deliberately containing a bracket pair and a backslash
#    (the Python literal ``"P[1]\\a"`` is ``P`` ``[`` ``1`` ``]`` ``\`` ``a``).
# B: two characters groups separated by one real TAB character
#    (the Python literal ``"L2\tL3"`` is ``L2`` TAB ``L3``).
PROBE_TEXT_A = "P[1]\\a"
PROBE_TEXT_B = "L2\tL3"

PROBE_STOP_MESSAGE = "character probe completed; task not executed"
PROBE_NO_TYPEABLE_MESSAGE = "probe: no typeable element"
PROBE_TASK_NOT_EXECUTED = "task not executed"

# AX roles the probe accepts as "typeable". ``textbox``/``searchbox`` are the
# common official AX roles; ``contenteditable`` is accepted in case the platform
# labels a rich-text field that way. Element numbers are compared numerically.
_TYPEABLE_ROLES = frozenset(
    {
        "textbox",
        "searchbox",
        "contenteditable",
        "text area",
        "textarea",
        "textfield",
        "text_field",
    }
)


def probe_enabled_from_env(env: Mapping[str, str] | None = None) -> bool:
    """Whether ``WEBAGENT_PROBE`` asks for probe behaviour (``1/true/yes/on``)."""

    source = os.environ if env is None else env
    return str(source.get(PROBE_ENV_VAR, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_probe_flag(probe: bool | None) -> bool:
    """Explicit constructor value wins; otherwise fall back to the environment."""

    if probe is None:
        return probe_enabled_from_env()
    if not isinstance(probe, bool):
        raise ValueError("probe must be a bool or None")
    return probe


@dataclass(frozen=True)
class ProbeCommand:
    """One scripted probe action; no model is involved in choosing it."""

    op: ActionOp
    text: str | None = None


def build_probe_script(text_a: str, text_b: str) -> tuple[ProbeCommand, ...]:
    """The fixed sequence: type A, press Enter, type B."""

    return (
        ProbeCommand(ActionOp.TYPE, text_a),
        ProbeCommand(ActionOp.PRESS),
        ProbeCommand(ActionOp.TYPE, text_b),
    )


def _checked_probe_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise OfficialActionFormatError(f"{field} must be a non-empty string")
    # The same wire validation the production path uses, so a probe string that
    # could never be sent is refused at construction instead of mid-page.
    return validate_action_text(value, field)


class OfficialProbeAgent(OfficialAgent):
    """An ``OfficialAgent`` subclass that probes characters instead of tasks."""

    def __init__(
        self,
        *,
        app_config: AppConfig | None = None,
        runtime_dir: str | None = None,
        clock: Clock | None = None,
        model_api: Any | None = None,
        browser_api: Any | None = None,
        download_artifacts: frozenset[str] | set[str] | tuple[str, ...] | None = None,
        probe: bool | None = None,
        probe_text_a: str = PROBE_TEXT_A,
        probe_text_b: str = PROBE_TEXT_B,
    ) -> None:
        super().__init__(
            app_config=app_config,
            runtime_dir=runtime_dir,
            clock=clock,
            model_api=model_api,
            browser_api=browser_api,
            download_artifacts=download_artifacts,
        )
        self.probe_enabled = resolve_probe_flag(probe)
        self.configured_text_mode: TextMode = self.text_mode
        self.probe_text_a = _checked_probe_text(probe_text_a, "probe.text_a")
        self.probe_text_b = _checked_probe_text(probe_text_b, "probe.text_b")
        self._probe_script: deque[ProbeCommand] = deque()
        self._probe_errors: list[str] = []
        self._probe_screenshot: dict[str, Any] | None = None
        self._probe_dispatched = 0
        if self.probe_enabled:
            # The probe owns the wire policy: always raw, whatever the task
            # config says. A screenshot of the final probe page is mandatory.
            self.text_mode = TextMode.RAW
            self.download_artifacts = frozenset(set(self.download_artifacts) | {"screenshot"})

    # ---------------------------------------------------------------- read API
    @property
    def probe_screenshot(self) -> Mapping[str, Any] | None:
        return None if self._probe_screenshot is None else dict(self._probe_screenshot)

    @property
    def probe_errors(self) -> tuple[str, ...]:
        return tuple(self._probe_errors)

    # ---------------------------------------------------------------- SDK hooks
    def reset(self, request: Mapping[str, Any]) -> None:
        super().reset(request)
        self._reset_probe_state()

    def _step(self, request: Mapping[str, Any]) -> dict[str, str]:
        if not self.probe_enabled:
            return super()._step(request)
        if not isinstance(request, Mapping):
            raise ValueError("step request must be a mapping")
        if self._platform_run_id is None:
            raise ValueError("reset is required before step")
        if request.get("run_id") != self._platform_run_id:
            raise ValueError("run_id mismatch; reset is required")
        self._step_index += 1
        if self._finished:
            if self._last_stop_action is not None:
                return dict(self._last_stop_action)
            return {"action": "stop [task already ended]"}
        if self._session is None:
            return self._probe_start(request)
        return self._probe_continue(request)

    # --------------------------------------------------------------- lifecycle
    def _reset_probe_state(self) -> None:
        self._probe_script = deque()
        self._probe_errors = []
        self._probe_screenshot = None
        self._probe_dispatched = 0

    def _probe_start(self, request: Mapping[str, Any]) -> dict[str, str]:
        fallback = self._reset_deadline
        if fallback is None:
            fallback = self.clock.now() + 300
        deadline = _deadline_value(request, fallback)
        if self.clock.now() >= deadline:
            action = self._stop_action("task deadline reached before local work began")
            self._finished = True
            self._last_stop_action = action
            return dict(action)
        task = request.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("first step has no complete task text")
        assert self._local_run_id is not None
        session = Session(
            self._local_run_id,
            runtime_dir=self.runtime_dir,
            clock=self.clock,
            budget=Budget(
                limits=self.app_config.budget_limits(),
                deadline=Deadline(deadline, clock=self.clock),
                clock=self.clock,
            ),
            page_id="official-page",
        )
        self._session = session
        session.start_task(task)
        backend = OfficialBackend(
            self._local_run_id,
            session=session,
            evidence=session.evidence,
            page_id="official-page",
            browser_api=self.browser_api,
            download_artifacts=self.download_artifacts,
            deadline_check=session.check_deadline,
            text_mode=TextMode.RAW,
        )
        self._backend = backend
        initial = backend.from_request(request)
        session.ingest_observation(initial)
        self._pending_observation = initial
        self._probe_script = deque(build_probe_script(self.probe_text_a, self.probe_text_b))
        return self._probe_dispatch_next()

    def _probe_continue(self, request: Mapping[str, Any]) -> dict[str, str]:
        session = self._session
        backend = self._backend
        pending = self._pending_action
        if session is None or backend is None:
            raise RuntimeError("probe task container is incomplete")
        if pending is None:
            raise RuntimeError("no probe action is awaiting platform feedback")
        try:
            session.check_deadline("probe step")
            observation = backend.from_request(
                request,
                previous_action=pending,
                previous_observation=self._pending_observation,
            )
            self._pending_observation = observation
            session.ingest_observation(observation)
        except DeadlineExceeded:
            return self._probe_finish(
                "character probe stopped at the task deadline; task not executed",
                reason=TerminationReason.DEADLINE,
            )
        feedback = observation.feedback
        if feedback is not None and feedback.status is FeedbackStatus.ERROR:
            # A platform rejection is evidence, not a reason to retry: the next
            # scripted action still runs, and the stop message lists the error.
            self._probe_errors.append(
                f"{pending.op.value}: {feedback.error_type}: {feedback.message}"
            )
        return self._probe_dispatch_next()

    # ------------------------------------------------------------ script steps
    def _probe_dispatch_next(self) -> dict[str, str]:
        session = self._session
        backend = self._backend
        if session is None or backend is None:
            raise RuntimeError("probe task container is incomplete")
        observation = session.observation
        if observation is None:
            raise RuntimeError("probe has no current observation")
        while self._probe_script:
            command = self._probe_script[0]
            observation = session.observation
            if observation is None:
                raise RuntimeError("probe has no current observation")
            if command.op is ActionOp.TYPE:
                target = self._typeable_target(observation)
                if target is None:
                    return self._probe_finish(
                        PROBE_NO_TYPEABLE_MESSAGE,
                        local_message=f"{PROBE_NO_TYPEABLE_MESSAGE}; {PROBE_TASK_NOT_EXECUTED}",
                    )
                args: dict[str, Any] = {
                    "target": target,
                    "text": command.text or "",
                    "submit": False,
                }
            else:
                args = {"keys": "Enter"}
            action = WebAction(
                action_id=new_action_id(),
                run_id=observation.run_id,
                page_id=observation.page_id,
                observation_id=observation.observation_id,
                op=command.op,
                args=args,
            )
            try:
                session.propose(action)
                # Serialize before dispatch, as the production path does: a
                # locally unrepresentable command never becomes an in-flight
                # platform action.
                wire = backend.serialize_action(action)
                session.consume(action.action_id)
            except Exception as exc:  # noqa: BLE001 - recorded, then next script step
                self._record_script_error(command, exc)
                try:
                    session.drop_proposal("probe_action_error")
                except Exception:  # noqa: BLE001 - the proposal may already be gone
                    pass
                self._probe_script.popleft()
                continue
            self._probe_script.popleft()
            self._pending_action = action
            self._probe_dispatched += 1
            return {"action": wire}
        return self._probe_finish(PROBE_STOP_MESSAGE)

    def _typeable_target(self, observation: Observation) -> str | None:
        best: str | None = None
        for number, element in observation.elements.items():
            role = (element.role or "").strip().lower()
            name = (element.name or "").lower()
            typeable = (
                role in _TYPEABLE_ROLES
                or "contenteditable" in role
                or "contenteditable" in name
            )
            if not typeable:
                continue
            if best is None or int(number) < int(best):
                best = number
        return best

    def _record_script_error(self, command: ProbeCommand, exc: BaseException) -> None:
        label = command.op.value
        detail = f"{label}: {type(exc).__name__}: {exc}"
        self._probe_errors.append(detail)
        session = self._session
        if session is not None:
            try:
                session.log_exception("probe_action", exc)
            except Exception:  # noqa: BLE001 - logging never replaces the stop
                pass

    # ---------------------------------------------------------------- terminal
    def _probe_finish(
        self,
        wire_message: str,
        *,
        local_message: str | None = None,
        reason: TerminationReason = TerminationReason.INCOMPLETE,
    ) -> dict[str, str]:
        """Download the final screenshot and end with an honest stop."""

        session = self._session
        backend = self._backend
        screenshot = self._download_probe_screenshot(session, backend)
        wire_text = self._message_with_errors(wire_message)
        local_text = self._message_with_errors(local_message or wire_message)
        refs: tuple[str, ...] = ()
        claim = f"character probe: {self._probe_dispatched} scripted actions dispatched"
        if screenshot is None:
            claim += "; screenshot unavailable"
        elif screenshot.get("available"):
            reference = screenshot.get("reference")
            claim += f"; screenshot {reference}"
            if isinstance(reference, str) and reference:
                refs = (reference,)
        else:
            claim += f"; screenshot unavailable ({screenshot.get('reason')})"
        claim += f"; {PROBE_TASK_NOT_EXECUTED}"
        finish = Finish(
            claim=claim,
            message=local_text,
            reason=reason,
            evidence_refs=refs,
        )
        if session is not None:
            try:
                if session.proposed_action is not None or session.in_flight_action is not None:
                    session.finish_incomplete(finish)
                else:
                    session.finish(finish)
            except SessionStateError:
                # The session was already terminal (e.g. a deadline path). Do
                # not rewrite its terminal evidence; the stop still goes out.
                pass
        self._finished = True
        action = self._stop_action(wire_text)
        self._last_stop_action = action
        self._finalize_run()
        return dict(action)

    def _download_probe_screenshot(
        self,
        session: Session | None,
        backend: OfficialBackend | None,
    ) -> dict[str, Any] | None:
        if session is None or backend is None or session.observation is None:
            return None
        try:
            artifact = backend.download_artifact("screenshot")
        except Exception as exc:  # noqa: BLE001 - a missing screenshot is evidence too
            self._probe_errors.append(f"screenshot: {type(exc).__name__}: {exc}")
            if session is not None:
                try:
                    session.log_exception("probe_screenshot", exc)
                except Exception:  # noqa: BLE001
                    pass
            return {"available": False, "reason": type(exc).__name__, "reference": None}
        self._probe_screenshot = {
            "available": artifact.available,
            "reason": artifact.reason,
            "reference": artifact.reference,
        }
        return self._probe_screenshot

    def _message_with_errors(self, base: str) -> str:
        if not self._probe_errors:
            return base
        details: list[str] = []
        for detail in self._probe_errors:
            text = " ".join(str(detail).split())
            # A stop line is one structural command: keep its bracket pair the
            # only one, without escaping the surrounding message.
            text = text.replace("[", "(").replace("]", ")")
            if len(text) > 160:
                text = text[:157] + "..."
            details.append(text)
        suffix = "; action errors: " + " | ".join(details)
        if len(suffix) > 400:
            suffix = suffix[:397] + "..."
        return base + suffix


__all__ = [
    "PROBE_ENV_VAR",
    "PROBE_NO_TYPEABLE_MESSAGE",
    "PROBE_STOP_MESSAGE",
    "PROBE_TEXT_A",
    "PROBE_TEXT_B",
    "OfficialProbeAgent",
    "ProbeCommand",
    "build_probe_script",
    "probe_enabled_from_env",
    "resolve_probe_flag",
]
