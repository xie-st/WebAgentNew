"""Command-line entry for the real-browser path.

Commands:

* ``doctor`` checks configuration, imports and directories only. It never calls
  a model or launches a browser.
* ``run`` executes an external task file with Playwright or browser-use.

Pause/resume/cancel are file-driven (``control.jsonl`` inside the run directory),
so no background keyboard thread or hidden async loop is introduced. Commands are
read between actions; pause is refused while an action is in flight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from .backends import available_backends, create_backend
from .budget import Budget, Deadline
from .clock import SystemClock
from .config import AppConfig, ConfigError, load_config
from .control import RunControl
from .effgen_bridge import DriverStatus, EffGenLoopDriver, create_driver
from .errors import WebAgentError
from .models.fake import ScriptedModel
from .models.real import ModelCallRecord, RealModel
from .protocol import validate_run_id
from .session import Session
from .tools import render_catalog

_TASK_KEYS = frozenset(
    {
        "task",
        "start_url",
        "run_id",
        "seed",
        "draft_text",
        "max_seconds",
        "headless",
        "description",
    }
)
_PLAYWRIGHT_PROFILE_GLOBS = (
    "chromium-*/chrome-win64/chrome.exe",
    "chromium-*/chrome-win/chrome.exe",
    "chromium-*/chrome-linux/chrome",
)


def _print(message: str = "") -> None:
    print(message, flush=True)


def load_task_file(path: str | Path) -> dict[str, Any]:
    task_path = Path(path)
    if not task_path.is_file():
        raise ConfigError(f"task file not found: {task_path}")
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"task file is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("task file root must be an object")
    unknown = sorted(set(payload) - _TASK_KEYS)
    if unknown:
        raise ConfigError(f"unknown task keys: {', '.join(unknown)}")
    task = payload.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ConfigError("task file needs a non-empty 'task' string")
    start_url = payload.get("start_url")
    if start_url is not None and (not isinstance(start_url, str) or not start_url.strip()):
        raise ConfigError("task.start_url must be a non-empty string or omitted")
    run_id = payload.get("run_id")
    if run_id is not None:
        validate_run_id(run_id)
    headless = payload.get("headless")
    if headless is not None and not isinstance(headless, bool):
        raise ConfigError("task.headless must be a boolean or omitted")
    max_seconds = payload.get("max_seconds")
    if max_seconds is not None and (
        not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool) or max_seconds <= 0
    ):
        raise ConfigError("task.max_seconds must be a positive number or omitted")
    draft_text = payload.get("draft_text")
    if draft_text is not None and not isinstance(draft_text, str):
        raise ConfigError("task.draft_text must be a string or omitted")
    normalized = dict(payload)
    normalized["task"] = _apply_draft_text(task, draft_text)
    return normalized


def _apply_draft_text(task: str, draft_text: str | None) -> str:
    """Pass the task condition, never an action sequence.

    ``draft_text`` is optional and is appended as an exact-value requirement.
    The executor does not use it to choose elements or invent actions.
    """

    if draft_text is None:
        return task
    if "{draft_text}" in task:
        return task.replace("{draft_text}", draft_text)
    return task + "\n\nExact draft text to enter: " + draft_text


def _chromium_hint() -> str | None:
    root = Path(
        os.environ.get(
            "PLAYWRIGHT_BROWSERS_PATH",
            str(Path.home() / "AppData" / "Local" / "ms-playwright"),
        )
    )
    for pattern in _PLAYWRIGHT_PROFILE_GLOBS:
        matches = sorted(root.glob(pattern))
        if matches:
            return str(matches[-1])
    return None


def doctor(config_path: str, *, as_json: bool = False) -> int:
    checks: list[dict[str, Any]] = []
    errors = 0

    def add(name: str, ok: bool, detail: str, *, fatal: bool = True) -> None:
        nonlocal errors
        checks.append({"check": name, "ok": ok, "detail": detail})
        if not ok and fatal:
            errors += 1

    try:
        config = load_config(config_path)
        add("config", True, f"loaded {config_path} (profile={config.profile})")
    except Exception as exc:  # noqa: BLE001 - doctor reports the cause
        add("config", False, f"{type(exc).__name__}: {exc}")
        config = None

    if config is not None:
        available = available_backends()
        if config.backend not in available:
            add("backend_name", False, f"unsupported backend {config.backend!r}")
        else:
            for name, ok in sorted(available.items()):
                add(
                    f"backend_import:{name}",
                    ok,
                    f"{name} import {'ok' if ok else 'missing'}",
                    fatal=(name == config.backend),
                )
        if config.backend == "playwright":
            executable = _chromium_hint()
            add(
                "chromium_binary",
                executable is not None,
                executable or "no Playwright Chromium binary found under ms-playwright",
                fatal=False,
            )
        try:
            from importlib.metadata import version

            add("browser_use_version", True, version("browser-use"), fatal=False)
        except Exception:  # noqa: BLE001 - optional dependency
            add("browser_use_version", False, "browser-use not installed", fatal=False)
        if config.profile == "real":
            env_names = config.env_names()
            for label, name in env_names.items():
                if name is None:
                    add(f"env:{label}", False, "no environment variable name configured")
                    continue
                present = bool(os.environ.get(name))
                add(f"env:{label}", present, f"{name} is {'set' if present else 'not set'}", fatal=False)
        runtime = Path(config.runtime_dir)
        try:
            runtime.mkdir(parents=True, exist_ok=True)
            probe = runtime / f".doctor-{uuid.uuid4().hex}.tmp"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            add("runtime_dir", True, str(runtime.resolve()))
        except OSError as exc:
            add("runtime_dir", False, f"{runtime}: {exc}")
        if config.profile == "real":
            add(
                "real_profile",
                config.model.provider == "openai_compatible",
                "real profile uses openai_compatible text model",
            )
        if config.profile == "official":
            add("official_profile", False, "B1 CLI does not start the official task path", fatal=False)

    payload = {
        "ok": errors == 0,
        "error_count": errors,
        "config": None if config is None else config.summary(),
        "checks": checks,
    }
    if as_json:
        _print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print(f"config: {config_path}")
        for check in checks:
            mark = "ok" if check["ok"] else "FAIL"
            _print(f"[{mark:4s}] {check['check']}: {check['detail']}")
        _print("doctor ok" if errors == 0 else f"doctor found {errors} error(s)")
    return 0 if errors == 0 else 1


def _new_run_id() -> str:
    return "b1-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def _write_model_calls(session: Session, model: RealModel) -> Path:
    path = session.evidence.run_dir / "model_calls.json"
    payload = [record.to_dict() for record in model.calls]
    path.write_text(
        json.dumps(session.redactor.redact(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _record_sink(session: Session):
    def sink(record: ModelCallRecord) -> None:
        session.record_model_call(
            request_id=record.request_id,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            token_count_source=(
                "provider_usage"
                if record.usage_source == "provider"
                else "unknown"
            ),
            latency_ms=record.latency_ms,
            failure_type=record.failure_type,
            model_name=record.model,
        )

    return sink


def run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.profile != "real":
        raise ConfigError("B1 run needs a real-profile configuration")
    backend_name = args.backend or config.backend
    if backend_name not in {"playwright", "browser-use"}:
        raise ConfigError(f"unsupported backend {backend_name!r}")
    if args.backend is not None and backend_name not in {"playwright", "browser-use"}:
        raise ConfigError(f"unsupported backend {backend_name!r}")
    task_data = load_task_file(args.task_file)
    task_text = str(task_data["task"])
    start_url = args.start_url or task_data.get("start_url")
    if not start_url:
        raise ConfigError("run needs a start URL: --start-url or task.start_url")
    if not str(start_url).startswith(("http://", "https://", "file://")):
        raise ConfigError("start URL must be absolute http(s) or file URL")
    run_id = args.run_id or task_data.get("run_id") or _new_run_id()
    validate_run_id(run_id)
    max_seconds = args.max_seconds or task_data.get("max_seconds")
    headless = (
        args.headless
        if args.headless is not None
        else bool(task_data.get("headless", False))
    )
    runtime_dir = Path(args.runtime_dir or config.runtime_dir)
    clock = SystemClock()
    deadline = (
        Deadline(clock.now() + float(max_seconds), clock=clock) if max_seconds else None
    )
    budget = Budget(config.budget_limits(), deadline=deadline, clock=clock)
    session = Session(run_id, runtime_dir=runtime_dir, clock=clock, budget=budget)
    session.start_task(task_text)
    control_path = session.evidence.run_dir / "control.jsonl"
    control = RunControl(control_path)
    model: ScriptedModel | RealModel
    if args.model_script:
        model = ScriptedModel.from_script(args.model_script)
        model.load()
        _print(f"model: scripted offline model ({args.model_script})")
    else:
        model = RealModel.from_app_config(config)
        model.set_call_sink(_record_sink(session))
        model.set_deadline_check(session.check_deadline)
        model.load()
        _print(f"model: {config.model.name} via {config.model.base_url_env}")
    if not args.offline:
        _print("tools catalog:\n" + render_catalog())
    backend = None
    driver = None
    step = None
    exit_code = 2
    try:
        backend = create_backend(
            backend_name,
            run_id=run_id,
            evidence=session.evidence,
            headless=headless,
            runtime_dir=runtime_dir,
            profile_dir=args.profile_dir,
            action_timeout_ms=args.action_timeout_ms,
            navigation_timeout_ms=args.navigation_timeout_ms,
        )
        _print(f"backend: {backend_name} ({'headless' if headless else 'headed'})")
        _print(f"run: {run_id}")
        _print(f"control file: {control_path}")
        _print("commands: append 'pause', 'resume' or 'cancel' on its own line")
        backend.start(str(start_url))
        initial = backend.observe()
        session.ingest_observation(initial)
        driver = create_driver(session, model, app_config=config)
        step = driver.start()
        while True:
            _print(_step_line(step))
            if step.status is not DriverStatus.ACTION:
                break
            command = control.poll()
            if command == "cancel":
                step = driver.cancel("user_cancel")
                break
            if command == "pause":
                session.pause()
                _print(
                    "PAUSED: no action is in flight; the proposed action was dropped. "
                    "You may edit the page. Append 'resume' to re-observe and continue, "
                    "or 'cancel' to stop."
                )
                command = control.wait_for(
                    {"resume", "cancel"},
                    timeout_seconds=args.pause_timeout,
                )
                if command is None:
                    step = driver.cancel("pause_timeout")
                    _print("pause timed out; run cancelled")
                    break
                if command == "cancel":
                    step = driver.cancel("user_cancel")
                    break
                fresh = backend.observe()
                _print(
                    "RESUMED: re-observed page; previous element numbers are invalid"
                )
                step = driver.resume(fresh)
                continue
            try:
                action = driver.take_action()
                _print(f"action: {action.op.value} {json.dumps(dict(action.args), ensure_ascii=False)}")
                result = backend.execute(action)
            except Exception as exc:  # noqa: BLE001 - record the real failure
                session.log_exception("backend_execute", exc)
                _print(f"ERROR: {type(exc).__name__}: {exc}")
                step = driver.cancel("backend_error")
                break
            _print(
                "feedback: "
                + result.feedback.status.value
                + (" " + result.feedback.message if result.feedback.message else "")
            )
            step = driver.accept_observation(result.observation)
        _print(_step_line(step))
        exit_code = 0 if step.status is DriverStatus.COMPLETED else 130 if step.status is DriverStatus.CANCELLED else 2
    except KeyboardInterrupt:
        _print("interrupted by user; cancelling")
        if driver is not None:
            driver.cancel("keyboard_interrupt")
        exit_code = 130
    except (WebAgentError, OSError, ValueError) as exc:
        session.log_exception("cli_run", exc)
        _print(f"ERROR: {type(exc).__name__}: {exc}")
        if driver is not None:
            driver.cancel("cli_error")
        exit_code = 2
    finally:
        if driver is not None:
            driver.close()
        if backend is not None:
            backend.close()
        if isinstance(model, RealModel):
            try:
                if model._is_loaded:
                    _write_model_calls(session, model)
            except Exception as exc:  # noqa: BLE001 - do not hide the run
                _print(f"warning: could not write model_calls.json: {type(exc).__name__}: {exc}")
            model.unload()
        session.close()
        _print(f"result: {session.evidence.result_path}")
        _print(f"events: {session.evidence.events_path}")
        screenshots = list(session.evidence.screenshots_dir.glob("*.png"))
        _print(f"screenshots: {len(screenshots)} in {session.evidence.screenshots_dir}")
    return exit_code


def _step_line(step: Any) -> str:
    if step is None:
        return "loop: not started"
    suffix = f" ({step.message})" if getattr(step, "message", "") else ""
    return f"loop: {step.status.value}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="webagent", description="WebAgentNew phase-1 harness")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_parser = sub.add_parser("doctor", help="check config/imports/directories only")
    doctor_parser.add_argument("--config", required=True)
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    doctor_parser.set_defaults(func=lambda args: doctor(args.config, as_json=args.as_json))

    run_parser = sub.add_parser("run", help="run one external task file")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--backend", choices=("playwright", "browser-use"), default=None)
    run_parser.add_argument("--task-file", required=True)
    run_parser.add_argument("--model-script", default=None, help="offline scripted model (tests only)")
    run_parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--start-url", default=None)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--runtime-dir", default=None)
    run_parser.add_argument("--profile-dir", default=None)
    run_parser.add_argument("--max-seconds", type=float, default=None)
    run_parser.add_argument("--pause-timeout", type=float, default=600.0)
    run_parser.add_argument("--action-timeout-ms", type=int, default=10_000)
    run_parser.add_argument("--navigation-timeout-ms", type=int, default=20_000)
    run_parser.add_argument("--offline", action="store_true", help="suppress non-essential output")
    run_parser.set_defaults(func=run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        _print(f"config error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _print(f"error: {type(exc).__name__}: {exc}")
        return 2


__all__ = ["build_parser", "doctor", "load_task_file", "main", "run_command"]
