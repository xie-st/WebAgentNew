"""The controlled execution loop.

One EffGen generator, one backend, one coordinator. Differences from
``BackendLoopRunner`` are only at the control seams:

* before a dispatch the gate is checked (and the user's control file polled);
* a takeover detected in the action's returned observation pauses *before* the
  generator is advanced, so no model call or internal tool runs while the user
  owns the browser;
* after an explicit resume the page is re-observed, old proposals are dropped
  with an explicit ``not executed`` result, and the same generator continues.

The deferred-feedback path is deliberate. The public driver API has no method
that ingests a feedback observation without also advancing the generator (and
therefore starting a model turn), so the control runner keeps the truthful
result and feeds it back only after the user resume signal. Until then no
generator advance happens at all.

If the phase-2 bridge gains ``ingest_feedback_without_advance`` this module can
use it; that requested integration is listed in the report rather than added to
the shared file here.
"""

from __future__ import annotations

from typing import Any, Callable

from ..backend import Backend
from ..effgen_bridge import DriverStatus, DriverStep, EffGenLoopDriver
from ..errors import DeadlineExceeded
from ..protocol import ExecutionResult, FeedbackStatus, Observation, WebAction
from ..session import Session, SessionState
from ..state.integration import (
    EffectIntent,
    EffectPolicy,
    ObservedObject,
    evidence_ref_from_artifact,
    record_observed_object,
)
from ..state.records import EffectStatus, PendingEffect
from ..state.store import StateStore
from .coordinator import ControlCoordinator
from .errors import ControlCancelled, ControlPaused
from .state import ControlState


class ControlledLoopRunner:
    """Drive one task with pause/takeover/resume control around every action."""

    def __init__(
        self,
        driver: EffGenLoopDriver,
        backend: Backend,
        coordinator: ControlCoordinator,
        *,
        store: StateStore | None = None,
        branch: str = "root",
        goal_id: str | None = None,
        effect_policy: EffectPolicy | None = None,
        observation_recorder: (
            Callable[[Observation, WebAction | None], ObservedObject | None] | None
        ) = None,
        on_step: Callable[[DriverStep], None] | None = None,
    ) -> None:
        if not isinstance(driver, EffGenLoopDriver):
            raise TypeError("ControlledLoopRunner needs an EffGenLoopDriver")
        self.driver = driver
        self.backend = backend
        self.coordinator = coordinator
        self.session: Session = driver.session
        self.store = store
        self.branch = branch
        self.goal_id = goal_id
        self.effect_policy = effect_policy or EffectPolicy()
        self.observation_recorder = observation_recorder
        self.on_step = on_step
        self.effects_prepared: list[PendingEffect] = []
        self.effects_confirmed: list[PendingEffect] = []
        self.observed_objects: list[str] = []
        self.recorded_observations: list[dict[str, Any]] = []
        self.last_fact_id: str | None = None
        self.last_error: BaseException | None = None
        self._prepared: tuple[EffectIntent, PendingEffect] | None = None
        self.coordinator.bind_session(self.session)
        self.session.set_action_gate(self.coordinator.gate)

    # ------------------------------------------------------------------- run
    def run(self) -> DriverStep:
        coordinator = self.coordinator
        if not coordinator.started:
            coordinator.start()
        self._record_observation(self.session.observation, None)

        # A takeover that is already visible before the loop opens still pauses
        # before any model turn or action.
        initial_signal = coordinator.detect(self.session.observation)
        if initial_signal is not None:
            coordinator.request_pause(
                initial_signal.reason,
                source="auto",
                details=initial_signal.to_dict(),
            )
        if not coordinator.action_allowed():
            step = self._handoff_before_start()
        else:
            step = self.driver.start()
        return self._loop(step)

    # ------------------------------------------------------------------ loop
    def _loop(self, step: DriverStep) -> DriverStep:
        coordinator = self.coordinator
        while True:
            if step.status is not DriverStatus.ACTION:
                coordinator.finish(step.status.value)
                return step

            command = coordinator.poll_external_command()
            if command == "cancel":
                return self._cancel("user_cancel")
            if coordinator.state is ControlState.PAUSE_REQUESTED:
                step = self._handoff_before_dispatch()
                if step.status is not DriverStatus.ACTION:
                    coordinator.finish(step.status.value)
                    return step
                continue
            if coordinator.state in (
                ControlState.DRAINING,
                ControlState.USER_CONTROL,
                ControlState.CANCELLED,
                ControlState.BLOCKED,
            ):
                return self._cancel(
                    f"unexpected_control_state_{coordinator.state.value}"
                )

            if self.on_step is not None:
                self.on_step(step)

            try:
                action = self.driver.take_action(before_dispatch=self._before_dispatch)
            except ControlPaused:
                step = self._handoff_before_dispatch()
                if step.status is not DriverStatus.ACTION:
                    coordinator.finish(step.status.value)
                    return step
                continue
            except ControlCancelled:
                return self._cancel("user_cancel")
            except DeadlineExceeded as exc:
                coordinator.block("deadline_before_dispatch")
                return self.driver.cancel(f"deadline_before_dispatch: {exc}")
            except Exception as exc:  # noqa: BLE001 - ledger refused the dispatch
                dropped = self.session.proposed_action
                if self._prepared is not None and dropped is not None:
                    self._confirm_effect(
                        dropped,
                        EffectStatus.FAILED,
                        note="dispatch was aborted after the pending effect was saved",
                    )
                return self._cancel("dispatch_blocked", exc)

            coordinator.note_action_in_flight(action)
            try:
                result = self.backend.execute(action)
            except Exception as exc:  # noqa: BLE001 - keep the real failure
                coordinator.note_action_finished("exception")
                self._confirm_effect(
                    action, EffectStatus.UNKNOWN, note="backend raised before feedback"
                )
                return self._cancel("backend_error", exc)
            coordinator.note_action_finished(result.feedback.status.value)
            self._confirm_effect_from_result(action, result)

            signal = coordinator.detect(result.observation)
            if signal is not None and coordinator.state is ControlState.RUNNING:
                coordinator.request_pause(
                    signal.reason, source="auto", details=signal.to_dict()
                )
            # A user command can also arrive while the browser is executing.
            command = coordinator.poll_external_command()
            if command == "cancel":
                return self._cancel("user_cancel")
            if not coordinator.action_allowed():
                if coordinator.state is ControlState.RUNNING:
                    coordinator.request_pause("control_gate_closed", source="control")
                if coordinator.state not in (
                    ControlState.PAUSE_REQUESTED,
                    ControlState.DRAINING,
                ):
                    return self._cancel(
                        f"unexpected_control_state_{coordinator.state.value}"
                    )
                step = self._handoff_with_deferred_result(action, result)
                if step.status is not DriverStatus.ACTION:
                    coordinator.finish(step.status.value)
                    return step
                continue

            try:
                step = self.driver.accept_observation(result.observation)
            except DeadlineExceeded as exc:
                coordinator.block("deadline_after_feedback")
                return self.driver.cancel(f"deadline_after_feedback: {exc}")
            except Exception as exc:  # noqa: BLE001
                self.last_error = exc
                return self._cancel("feedback_rejected", exc)
            self._record_observation(result.observation, action)

    # ------------------------------------------------------- before dispatch
    def _before_dispatch(self, action: WebAction) -> None:
        coordinator = self.coordinator
        command = coordinator.poll_external_command()
        if command == "cancel":
            raise ControlCancelled("the user cancelled before dispatch")
        coordinator.gate("dispatch")
        if self.store is None:
            return
        observation = self.session.observation
        if observation is None:
            return
        intent = self.effect_policy.classify(action, observation)
        if intent is None:
            return
        effect = self.store.prepare_effect(
            action_id=action.action_id,
            target=intent.target,
            object_id=intent.object_id,
            goal_id=self.goal_id,
            branch_id=self.branch,
            observation_id=observation.observation_id,
            note=intent.note,
        )
        self._prepared = (intent, effect)
        self.effects_prepared.append(effect)

    # ------------------------------------------------------------- handoffs
    def _handoff_before_start(self) -> DriverStep:
        coordinator = self.coordinator
        coordinator.begin_drain(note="takeover visible before the loop opened")
        coordinator.begin_user_control()
        signal = self._wait_for_signal()
        if signal == "cancel":
            return self._cancel("user_cancel")
        try:
            coordinator.check_platform_deadline("resume_after_handoff")
        except DeadlineExceeded as exc:
            coordinator.block("deadline_during_handoff")
            return self._cancel("deadline_during_handoff", exc)
        coordinator.begin_refresh()
        try:
            fresh = self.backend.observe()
            self.session.pause()
            self.session.resume(fresh)
        except DeadlineExceeded as exc:
            coordinator.block("deadline_during_handoff")
            return self._cancel("deadline_during_handoff", exc)
        except Exception as exc:  # noqa: BLE001
            coordinator.block("reobserve_failed")
            return self._cancel("reobserve_failed", exc)
        coordinator.finish_refresh(observation=fresh)
        self._record_observation(fresh, None)
        return self.driver.start()

    def _handoff_before_dispatch(self) -> DriverStep:
        coordinator = self.coordinator
        coordinator.begin_drain(note="pause requested before page dispatch")
        dropped = self.session.proposed_action
        if self.session.state is SessionState.RUNNING:
            try:
                self.session.pause()
            except Exception as exc:  # noqa: BLE001
                coordinator.block("pause_rejected")
                return self._cancel("pause_rejected", exc)
        if dropped is not None:
            coordinator.discard_proposal(
                dropped,
                reason=f"paused_before_dispatch:{coordinator.pause_reason or 'pause'}",
            )
        coordinator.begin_user_control()
        signal = self._wait_for_signal()
        if signal == "cancel":
            return self._cancel("user_cancel")
        try:
            coordinator.check_platform_deadline("resume_after_handoff")
        except DeadlineExceeded as exc:
            coordinator.block("deadline_during_handoff")
            return self._cancel("deadline_during_handoff", exc)
        coordinator.begin_refresh()
        try:
            fresh = self.backend.observe()
            step = self.driver.resume(fresh)
        except DeadlineExceeded as exc:
            coordinator.block("deadline_during_handoff")
            return self._cancel("deadline_during_handoff", exc)
        except Exception as exc:  # noqa: BLE001
            coordinator.block("resume_after_takeover_failed")
            return self._cancel("resume_after_takeover_failed", exc)
        self._record_observation(fresh, None)
        coordinator.finish_refresh(observation=fresh)
        return step

    def _handoff_with_deferred_result(
        self, action: WebAction, result: ExecutionResult
    ) -> DriverStep:
        """Pause after a browser action without advancing the generator.

        The action already executed, so its feedback must be reported truthfully
        rather than as ``not executed``. The result is kept and ingested only
        after the explicit resume; no model/internal tool runs during the wait.
        """

        coordinator = self.coordinator
        coordinator.begin_drain(
            executed_action_id=action.action_id,
            feedback_status=result.feedback.status.value,
            note="browser action finished; feedback deferred until resume",
        )
        coordinator.begin_user_control()
        signal = self._wait_for_signal()
        if signal == "cancel":
            # The action already executed; record its real feedback before the
            # terminal cancellation so the in-flight platform/browser result is
            # not lost. This is local bookkeeping only: no generator advance.
            try:
                self.session.ingest_observation(result.observation)
                self._record_observation(result.observation, action)
            except Exception as exc:  # noqa: BLE001 - cancellation must still happen
                self.session.log_exception("deferred_feedback_on_cancel", exc)
            return self._cancel("user_cancel")
        try:
            coordinator.check_platform_deadline("resume_after_handoff")
        except DeadlineExceeded as exc:
            coordinator.block("deadline_during_handoff")
            return self._cancel("deadline_during_handoff", exc)
        coordinator.begin_refresh()
        try:
            step = self.driver.accept_observation(result.observation)
        except DeadlineExceeded as exc:
            coordinator.block("deadline_on_deferred_feedback")
            return self._cancel("deadline_on_deferred_feedback", exc)
        except Exception as exc:  # noqa: BLE001
            coordinator.block("deferred_feedback_rejected")
            return self._cancel("deferred_feedback_rejected", exc)
        self._record_observation(result.observation, action)

        if step.status is not DriverStatus.ACTION:
            coordinator.finish_refresh(observation=self.session.observation)
            return step

        # The resume model turn used the pre-handoff observation. Its proposal
        # is valid only in that observation, so it is dropped and the driver is
        # resumed again with the freshly observed page.
        dropped = self.session.proposed_action
        if dropped is not None:
            if self.session.state is SessionState.RUNNING:
                try:
                    self.session.pause()
                except Exception as exc:  # noqa: BLE001
                    coordinator.block("pause_rejected_after_handoff")
                    return self._cancel("pause_rejected_after_handoff", exc)
            coordinator.discard_proposal(
                dropped,
                reason="proposal_used_pre_handoff_observation",
            )
        try:
            fresh = self.backend.observe()
            step = self.driver.resume(fresh)
        except DeadlineExceeded as exc:
            coordinator.block("deadline_during_reobserve")
            return self._cancel("deadline_during_reobserve", exc)
        except Exception as exc:  # noqa: BLE001
            coordinator.block("resume_after_deferred_feedback_failed")
            return self._cancel("resume_after_deferred_feedback_failed", exc)
        self._record_observation(fresh, None)
        coordinator.finish_refresh(observation=fresh)
        return step

    def _wait_for_signal(self) -> str:
        coordinator = self.coordinator
        try:
            return coordinator.wait_for_user_signal(
                timeout_seconds=coordinator.wait_timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 - timeout is not a resume
            self.last_error = exc
            self.session.log_exception("control_wait", exc)
            coordinator.block("no_explicit_resume")
            return "cancel"

    # -------------------------------------------------------------- outcomes
    def _cancel(self, reason: str, exc: BaseException | None = None) -> DriverStep:
        if exc is not None:
            self.last_error = exc
            try:
                self.session.log_exception("control_cancel", exc)
            except Exception:  # noqa: BLE001
                pass
        if self.coordinator.state not in (
            ControlState.CANCELLED,
            ControlState.BLOCKED,
        ):
            self.coordinator.cancel(reason)
        return self.driver.cancel(reason)

    def _confirm_effect(
        self, action: WebAction, status: EffectStatus, *, note: str
    ) -> None:
        prepared = self._prepared
        self._prepared = None
        if prepared is None or self.store is None:
            return
        try:
            confirmed = self.store.confirm_effect(
                action.action_id, status=status, note=note
            )
        except Exception as exc:  # noqa: BLE001 - keep the first failure visible
            self.last_error = exc
            return
        self.effects_confirmed.append(confirmed)

    def _confirm_effect_from_result(
        self, action: WebAction, result: ExecutionResult
    ) -> None:
        status = result.feedback.status
        if status is FeedbackStatus.OBSERVED:
            effect_status = EffectStatus.CONFIRMED
        elif status is FeedbackStatus.ERROR:
            effect_status = EffectStatus.FAILED
        else:
            effect_status = EffectStatus.UNKNOWN
        self._confirm_effect(
            action,
            effect_status,
            note=f"feedback={status.value}",
        )

    def _record_observation(
        self,
        observation: Observation | None,
        action: WebAction | None,
    ) -> None:
        if observation is None or self.observation_recorder is None:
            return
        if self.store is None:
            return
        observed = self.observation_recorder(observation, action)
        if observed is None:
            return
        artifact = observation.artifacts.get("screenshot")
        evidence_id = f"ev-{observation.observation_id}-screenshot"
        try:
            ref = evidence_ref_from_artifact(
                artifact,
                run_id=self.session.run_id,
                branch_id=self.branch,
                evidence_id=evidence_id,
                observation_id=observation.observation_id,
                action_id=None if action is None else action.action_id,
            )
            fact = record_observed_object(
                self.store,
                object_id=observed.object_id,
                fields=observed.fields,
                evidence=(ref,),
                source=f"observation:{observation.observation_id}",
                goal_id=self.goal_id,
                branch_id=self.branch,
            )
        except Exception as exc:  # noqa: BLE001 - control must not hide ledger gaps
            self.last_error = exc
            try:
                self.session.log_exception("control_observation_record", exc)
            except Exception:  # noqa: BLE001
                pass
            return
        self.observed_objects.append(observed.object_id)
        self.last_fact_id = fact.fact_id
        self.recorded_observations.append(
            {
                "observation_id": observation.observation_id,
                "url": observation.url,
                "object_id": observed.object_id,
                "fields": dict(observed.fields),
                "fact_id": fact.fact_id,
            }
        )


__all__ = ["ControlledLoopRunner"]
