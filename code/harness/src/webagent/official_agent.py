"""SDK entry point for the official WebArena platform.

The frozen ``third_party_agent`` client imports ``webagent.official_agent:
OfficialAgent`` and drives it with ``reset`` / ``step`` / ``close``. Each task
gets one EffGen Agent and one interactive generator; the platform, not this
process, executes page actions.

Drive order (matching the phase-1 contract):

* ``reset`` creates an empty task container and closes any previous task.
* the first ``step`` reads the complete task, observation and deadline, then
  creates the Session, OfficialBackend, OfficialModel and EffGen driver;
* each later ``step`` turns that request's observation into the next observation,
  binds it to the previous action, sends feedback into the same generator and
  returns at most one action;
* a terminal generator state becomes a ``stop [说明]`` command, never an
  "official passed" claim;
* ``close`` is idempotent and releases only local task resources.

No real connection, credential file or network call is made by this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .backends.official import OfficialBackend, serialize_stop
from .budget import Budget, Deadline
from .clock import Clock, SystemClock
from .config import AppConfig, load_config, parse_config
from .effgen_bridge import (
    DriverStatus,
    DriverStep,
    EffGenLoopDriver,
    create_driver,
)
from .errors import DeadlineExceeded, SessionStateError
from .models.official import OfficialCallRecord, OfficialModel
from .protocol import Observation, WebAction, validate_run_id
from .session import Session, SessionState

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "official.yaml"


def _default_config() -> AppConfig:
    if _DEFAULT_CONFIG_PATH.is_file():
        return load_config(_DEFAULT_CONFIG_PATH)
    return parse_config(
        {
            "schema_version": "webagent.config.v1",
            "profile": "official",
            "backend": "official",
            "runtime_dir": "runtime",
            "agent": {
                "strategy": "single",
                "tool_calling_mode": "react",
                "max_iterations": None,
                "capability_probe": False,
                "sub_agents": False,
                "fallback": False,
                "speculative_execution": False,
                "cross_task_memory": False,
            },
            "model": {
                "provider": "official",
                "name": "platform-unified",
                "context_length": 32768,
            },
        }
    )


def _safe_local_run_id(platform_run_id: str) -> str:
    """A path-safe local run id that still maps deterministically to the task."""

    try:
        return validate_run_id(platform_run_id)
    except Exception:  # noqa: BLE001 - the official id can contain other characters
        digest = hashlib.sha256(platform_run_id.encode("utf-8")).hexdigest()[:40]
        return "official-" + digest


def _deadline_value(request: Mapping[str, Any], fallback: float | None) -> float:
    value = request.get("execution_deadline")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if fallback is not None:
        return float(fallback)
    raise ValueError("step request has no execution_deadline")


class OfficialAgent:
    """One task's official SDK adapter, instantiated by the SDK factory."""

    def __init__(
        self,
        *,
        app_config: AppConfig | None = None,
        runtime_dir: str | Path | None = None,
        clock: Clock | None = None,
        model_api: Any | None = None,
        browser_api: Any | None = None,
        download_artifacts: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.app_config = app_config or _default_config()
        if self.app_config.profile != "official" or self.app_config.backend != "official":
            raise ValueError("OfficialAgent needs the official profile and backend")
        self.runtime_dir = (
            Path(runtime_dir)
            if runtime_dir is not None
            else _PROJECT_ROOT / "runtime"
        )
        self.clock = clock or SystemClock()
        self.model_api = model_api
        self.browser_api = browser_api
        self.download_artifacts = (
            frozenset({"screenshot"})
            if download_artifacts is None
            else frozenset(download_artifacts)
        )
        self._platform_run_id: str | None = None
        self._local_run_id: str | None = None
        self._reset_deadline: float | None = None
        self._session: Session | None = None
        self._backend: OfficialBackend | None = None
        self._model: OfficialModel | None = None
        self._driver: EffGenLoopDriver | None = None
        self._pending_action: WebAction | None = None
        self._pending_observation: Observation | None = None
        self._terminal_step: DriverStep | None = None
        self._last_stop_action: dict[str, str] | None = None
        self._finished = False
        self._step_index = 0
        self._closed = False

    # ---------------------------------------------------------------- properties
    @property
    def platform_run_id(self) -> str | None:
        return self._platform_run_id

    @property
    def local_run_id(self) -> str | None:
        return self._local_run_id

    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def backend(self) -> OfficialBackend | None:
        return self._backend

    @property
    def model(self) -> OfficialModel | None:
        return self._model

    @property
    def driver(self) -> EffGenLoopDriver | None:
        return self._driver

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def terminal_step(self) -> DriverStep | None:
        return self._terminal_step

    # ---------------------------------------------------------------- SDK hooks
    def reset(self, request: Mapping[str, Any]) -> None:
        """Start an empty task container; no task or observation is assumed."""

        if not isinstance(request, Mapping):
            raise ValueError("reset request must be a mapping")
        platform_run_id = request.get("run_id")
        if not isinstance(platform_run_id, str) or not platform_run_id.strip():
            raise ValueError("reset request has no run_id")
        self._shutdown("reset")
        self._platform_run_id = platform_run_id
        self._local_run_id = _safe_local_run_id(platform_run_id)
        reset_deadline = request.get("execution_deadline")
        self._reset_deadline = (
            float(reset_deadline)
            if isinstance(reset_deadline, (int, float)) and not isinstance(reset_deadline, bool)
            else None
        )
        self._session = None
        self._backend = None
        self._model = None
        self._driver = None
        self._pending_action = None
        self._pending_observation = None
        self._terminal_step = None
        self._last_stop_action = None
        self._finished = False
        self._step_index = 0
        self._closed = False

    def step(self, request: Mapping[str, Any]) -> dict[str, str]:
        """Advance one platform step and return exactly one action string."""

        if self._closed:
            raise RuntimeError("OfficialAgent is closed; reset is required")
        try:
            return self._step(request)
        except DeadlineExceeded as exc:
            return self._stop_after_error("deadline", exc)
        except Exception as exc:  # noqa: BLE001 - return a visible stop, log the type
            return self._stop_after_error("step", exc)

    def close(self) -> None:
        """Release this task's local resources; repeated calls are safe."""

        self._shutdown("close")

    # ------------------------------------------------------------------ step flow
    def _step(self, request: Mapping[str, Any]) -> dict[str, str]:
        if not isinstance(request, Mapping):
            raise ValueError("step request must be a mapping")
        if self._platform_run_id is None:
            raise ValueError("reset is required before step")
        run_id = request.get("run_id")
        if run_id != self._platform_run_id:
            raise ValueError("run_id mismatch; reset is required")
        self._step_index += 1
        if self._finished:
            if self._last_stop_action is not None:
                return dict(self._last_stop_action)
            return {"action": "stop [task already ended]"}
        if self._session is None:
            return self._start_task(request)
        return self._continue_task(request)

    def _start_task(self, request: Mapping[str, Any]) -> dict[str, str]:
        # The SDK's own worker also falls back to 300 seconds when a request
        # omits a deadline; mirror that only as a compatibility fallback.
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
        )
        self._backend = backend
        initial = backend.from_request(request)
        session.ingest_observation(initial)
        self._pending_observation = initial
        model = OfficialModel(
            self.app_config.model,
            model_api=self.model_api,
        )
        model.set_call_sink(self._record_model_call)
        self._model = model
        driver = create_driver(
            session,
            model,
            app_config=self.app_config,
            artifact_loader=backend.download_artifact,
        )
        self._driver = driver
        session.check_deadline("before loop")
        return self._consume_step(driver.start())

    def _continue_task(self, request: Mapping[str, Any]) -> dict[str, str]:
        session = self._session
        backend = self._backend
        driver = self._driver
        action = self._pending_action
        if session is None or backend is None or driver is None:
            raise RuntimeError("official task container is incomplete")
        session.check_deadline("official step")
        if action is None:
            raise RuntimeError("no action is awaiting platform feedback")
        observation = backend.from_request(
            request,
            previous_action=action,
            previous_observation=self._pending_observation,
        )
        self._pending_observation = observation
        return self._consume_step(driver.accept_observation(observation))

    def _consume_step(self, step: DriverStep) -> dict[str, str]:
        if step.status is DriverStatus.ACTION:
            backend = self._backend
            driver = self._driver
            if backend is None or driver is None or step.action is None:
                raise RuntimeError("action step has no official backend or action")
            # Serialize before consuming: an unrepresentable command is a local
            # refusal, not a dispatched platform action with no feedback.
            command = backend.serialize_action(step.action)
            action = driver.take_action()
            self._pending_action = action
            return {"action": command}
        return self._terminal_response(step)

    def _terminal_response(self, step: DriverStep) -> dict[str, str]:
        self._terminal_step = step
        self._finished = True
        if step.status is DriverStatus.COMPLETED:
            message = step.message or "task completed"
        elif step.status is DriverStatus.DEADLINE:
            message = "task deadline reached before completion"
        elif step.status is DriverStatus.CANCELLED:
            message = "task cancelled"
        else:
            message = f"task not completed ({step.status.value})"
        action = self._stop_action(message)
        self._last_stop_action = action
        return dict(action)

    # --------------------------------------------------------------- model sink
    def _record_model_call(self, record: OfficialCallRecord) -> None:
        session = self._session
        if session is None:
            return
        session.record_model_call(
            request_id=record.request_id,
            input_tokens=None,
            output_tokens=None,
            token_count_source="platform_usage_unavailable",
            latency_ms=record.latency_ms,
            failure_type=record.failure_type,
            model_name=self.app_config.model.name,
        )

    # ------------------------------------------------------------------ failure
    def _stop_after_error(self, stage: str, exc: BaseException) -> dict[str, str]:
        session = self._session
        if session is not None:
            try:
                session.log_exception(f"official_{stage}", exc)
            except Exception:  # noqa: BLE001 - logging must not replace the stop
                pass
            try:
                if self._driver is not None and self._driver.terminal is None:
                    self._driver.cancel(f"official_{stage}")
                elif session.state in (SessionState.RUNNING, SessionState.PAUSED):
                    session.cancel(f"official_{stage}")
            except Exception:  # noqa: BLE001
                pass
        self._finished = True
        message = (
            "task deadline reached" if stage == "deadline"
            else f"task not completed due to local error ({type(exc).__name__})"
        )
        action = self._stop_action(message)
        self._last_stop_action = action
        return dict(action)

    def _stop_action(self, message: str) -> dict[str, str]:
        try:
            command = serialize_stop(message)
        except Exception:  # noqa: BLE001 - a scalar fallback must always exist
            command = "stop [task stopped]"
        return {"action": command}

    # --------------------------------------------------------------- lifecycle
    def _shutdown(self, reason: str) -> None:
        driver = self._driver
        if driver is not None:
            try:
                if driver.terminal is None:
                    driver.cancel(reason)
                driver.close()
            except Exception:  # noqa: BLE001
                pass
        backend = self._backend
        if backend is not None:
            try:
                backend.close()
            except Exception:  # noqa: BLE001
                pass
        session = self._session
        if session is not None:
            try:
                if session.state not in (
                    SessionState.FINISHED,
                    SessionState.CANCELLED,
                    SessionState.CLOSED,
                ):
                    try:
                        session.cancel(reason)
                    except SessionStateError:
                        pass
                session.close()
            except Exception:  # noqa: BLE001
                pass
        model = self._model
        if model is not None:
            try:
                model.unload()
            except Exception:  # noqa: BLE001
                pass
        self._driver = None
        self._backend = None
        self._session = None
        self._model = None
        self._pending_action = None
        self._pending_observation = None
        self._closed = True


__all__ = ["OfficialAgent"]
