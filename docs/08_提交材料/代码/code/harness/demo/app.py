"""WebAgent 本地演示面板。

一条命令启动（默认 ``python code/harness/demo/app.py --port 8787``），在
127.0.0.1 上提供中文演示页面：输入任务与起始网址后，通过子进程调用现有
``python -m webagent run`` CLI，EffGen + Playwright 在可见浏览器中真实执行；
面板持续展示 run_id、状态、动作日志、最终回答与真实截图，并支持
暂停 / 继续 / 取消（复用 ``control.jsonl`` 行协议）、证据下载与强制结束。

只用 Python 标准库 http.server + 原生 HTML/CSS/JS + 短轮询，不引入前端构建链。
密钥只从现有环境变量读取，不落盘、不进日志与下载。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from webagent.config import ConfigError, load_config
from webagent.redaction import Redactor

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "code" / "harness" / "configs" / "real.yaml"
DEFAULT_OFFLINE_SCRIPT = (
    REPO_ROOT / "code" / "harness" / "tests" / "fixtures" / "fake_scripts" / "local_chain.json"
)
DEFAULT_FIXTURE_DIR = REPO_ROOT / "code" / "harness" / "tests" / "fixtures" / "site"
DEFAULT_RUNTIME_DIR = REPO_ROOT / "runtime"
STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_BODY_BYTES = 1 << 20
MAX_TASK_CHARS = 4000
MAX_SECONDS = 14400
SCREENSHOT_RE = re.compile(r"^[A-Za-z0-9._-]+\.png$")
DOWNLOAD_NAMES = frozenset(
    {"task.json", "result.json", "events.jsonl", "stdout.log", "stderr.log", "meta.json"}
)

PRESET_TASK = (
    "在本地目录站点中搜索 quantum，打开第一条结果，"
    "在草稿输入框中输入 Draft for quantum note，"
    "点击 Preview draft 预览，确认预览显示该文字后保存一张截图。不要提交任何内容。"
)

STATE_LABELS = {
    "idle": "待运行",
    "starting": "启动中",
    "running": "运行中",
    "pause_requested": "暂停请求中",
    "paused": "已暂停",
    "resume_requested": "继续请求中",
    "cancel_requested": "取消请求中",
    "ended": "已结束",
}
OUTCOME_LABELS = {
    "completed": "完成",
    "incomplete": "未完成",
    "failed": "失败",
    "timeout": "超时",
    "cancelled": "已取消",
    "interrupted": "已强制中断",
    "start_failed": "启动失败",
}
TERMINATION_LABELS = {
    "completed": "完成",
    "incomplete": "未完成",
    "deadline": "超时",
    "error": "失败",
    "cancelled": "已取消",
    "budget_exhausted": "未完成（预算耗尽）",
}
ACTIVE_STATES = frozenset(
    {"starting", "running", "pause_requested", "paused", "resume_requested", "cancel_requested"}
)


class DemoError(Exception):
    """面板可理解的业务错误，消息直接展示给用户。"""


def _fmt_args(args: Any) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= 200 else text[:200] + "…"


def _event_entry(event: dict[str, Any]) -> dict[str, Any] | None:
    """把结构化事件映射为展示行；只使用真实存在的字段。"""

    kind = event.get("event_type")
    result = event.get("result")
    elapsed = round(float(event.get("elapsed_ms") or 0) / 1000, 1)
    stamp = str(event.get("time") or "")[11:19]
    entry: dict[str, Any] = {"time": stamp, "elapsed_s": elapsed, "kind": kind}

    if kind == "run_start":
        entry["text"] = "运行开始"
    elif kind == "task_start":
        entry["text"] = "任务开始"
    elif kind == "observation":
        entry["text"] = f"观察页面：{event.get('url') or '暂无'}"
    elif kind == "artifact_saved":
        if isinstance(result, dict) and result.get("kind") == "screenshot":
            ref = str(result.get("reference") or "")
            entry["text"] = f"已保存截图：{Path(ref).name}"
        else:
            return None
    elif kind == "model_call":
        total = event.get("model_calls_total")
        entry["text"] = f"模型调用 #{total if total is not None else '?'}"
    elif kind == "proposed":
        entry["text"] = f"模型提议 {event.get('action_type')} {_fmt_args(event.get('action_arguments'))}"
    elif kind == "dispatched":
        entry["text"] = f"执行动作 {event.get('action_type')} {_fmt_args(event.get('action_arguments'))}"
    elif kind == "feedback":
        entry["text"] = f"页面反馈：{result if isinstance(result, str) else '暂无'}"
    elif kind == "paused":
        entry["text"] = "已暂停：当前无进行中动作，可在浏览器中接管操作"
    elif kind == "resumed":
        entry["text"] = "已恢复：重新观察页面，之前的元素编号已失效"
    elif kind == "cancelled":
        reason = result.get("reason") if isinstance(result, dict) else None
        entry["text"] = f"动作已取消（{reason or 'cancelled'}）"
    elif kind == "agent_claim":
        text = ""
        if isinstance(result, dict):
            text = str(result.get("message") or result.get("claim") or "")
        entry["text"] = f"Agent 返回结论：{text}"
    elif kind == "run_end":
        term = result.get("termination") if isinstance(result, dict) else None
        entry["text"] = f"运行结束：{TERMINATION_LABELS.get(str(term), term or '未知')}"
    else:
        return None
    return entry


class RunRecord:
    """一次面板启动的运行（面板侧状态与元数据）。"""

    def __init__(
        self,
        *,
        run_id: str,
        mode: str,
        task: str,
        start_url: str,
        max_seconds: float,
        demo_dir: Path,
        run_dir: Path,
        argv: list[str],
        process: subprocess.Popen[Any] | None,
        model_name: str,
    ) -> None:
        self.run_id = run_id
        self.mode = mode
        self.task = task
        self.start_url = start_url
        self.max_seconds = max_seconds
        self.demo_dir = demo_dir
        self.run_dir = run_dir
        self.argv = argv
        self.process = process
        self.model_name = model_name

        self.state = "starting"
        self.outcome: str | None = None
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.exit_code: int | None = None
        self.forced = False

        self.log_entries: list[dict[str, Any]] = []
        self.screenshots: list[dict[str, Any]] = []
        self.stdout_tail: list[str] = []
        self.stderr_tail: list[str] = []
        self.current_url: str | None = None
        self.last_action: str | None = None
        self.has_events = False
        self.saw_paused = False
        self.saw_resumed = False
        self.result: dict[str, Any] | None = None
        self._events_line_count = 0

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES


class DemoApp:
    """面板核心：单活动 run 锁、子进程管理、状态机与证据访问。"""

    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        runtime_dir: Path = DEFAULT_RUNTIME_DIR,
        config_path: Path = DEFAULT_CONFIG,
        offline_script: Path = DEFAULT_OFFLINE_SCRIPT,
        fixture_dir: Path = DEFAULT_FIXTURE_DIR,
        port: int = 8787,
        headless: bool | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.runtime_dir = Path(runtime_dir)
        self.config_path = Path(config_path)
        self.offline_script = Path(offline_script)
        self.fixture_dir = Path(fixture_dir)
        self.port = port
        # 默认可见浏览器；测试可用环境变量改为 headless。
        if headless is None:
            headless = os.environ.get("WEBAGENT_DEMO_HEADLESS", "") == "1"
        self.headless = headless

        self.lock = threading.RLock()
        self.current: RunRecord | None = None
        self.history: list[str] = []

        self._config: Any | None = None
        self._config_error: str | None = None
        # 密钥值只用于日志脱敏，不写入任何文件；base_url 非密钥，不做字面替换。
        api_key = os.environ.get("WEBAGENT_MODEL_API_KEY")
        self.redactor = Redactor(secrets=[api_key] if api_key else [])

    # ---------- 配置与模式 ----------

    def _load_config(self) -> None:
        if self._config is not None or self._config_error is not None:
            return
        try:
            config = load_config(str(self.config_path))
            if config.profile != "real":
                self._config_error = f"配置 {self.config_path.name} 不是 real 档案，无法用于真实模型模式"
            else:
                self._config = config
        except (ConfigError, OSError, ValueError) as exc:
            self._config_error = f"{type(exc).__name__}: {exc}"

    def real_mode_status(self) -> dict[str, Any]:
        self._load_config()
        info: dict[str, Any] = {
            "available": False,
            "reason": None,
            "model_name": None,
            "config_path": str(self.config_path),
            "base_url_env": "WEBAGENT_MODEL_BASE_URL",
            "api_key_env": "WEBAGENT_MODEL_API_KEY",
        }
        if self._config_error is not None or self._config is None:
            info["reason"] = self._config_error or "配置不可用"
            return info
        info["model_name"] = self._config.model.name
        missing = [
            name
            for name in self._config.env_names().values()
            if name and not os.environ.get(name)
        ]
        if missing:
            info["reason"] = (
                "缺少模型环境变量：" + "、".join(missing) + "。未设置前只能使用离线脚本演示模式。"
            )
            return info
        info["available"] = True
        return info

    def fixture_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/fixture/index.html?seed=alpha-17"

    # ---------- 启动运行 ----------

    def start(self, mode: str, task: str, start_url: str, max_seconds: Any) -> str:
        with self.lock:
            current = self.current
            if current is not None and current.state != "ended":
                raise DemoError(
                    f"已有任务在运行（run_id={current.run_id}，状态：{STATE_LABELS[current.state]}）。"
                    "请等待其结束后再开始新任务。"
                )
            if mode not in ("real", "offline"):
                raise DemoError("运行模式无效")
            if mode == "offline" and (max_seconds is None or max_seconds == ""):
                max_seconds = 600  # 离线预设默认时限
            seconds = self._validate_seconds(max_seconds)

            if mode == "real":
                status = self.real_mode_status()
                if not status["available"]:
                    raise DemoError(status["reason"] or "真实模型模式当前不可用")
                task_text = self._validate_task(task)
                url = self._validate_url(start_url)
            else:
                task_text = PRESET_TASK
                url = self.fixture_url()

            run_id = "demo-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
            demo_dir = self.runtime_dir / "demo" / run_id
            run_dir = self.runtime_dir / "runs" / run_id
            demo_dir.mkdir(parents=True, exist_ok=True)

            task_payload = {
                "task": task_text,
                "start_url": url,
                "run_id": run_id,
                "max_seconds": seconds,
                "headless": self.headless,
            }
            task_file = demo_dir / "task.json"
            task_file.write_text(
                json.dumps(task_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            argv = self.build_argv(mode, task_file, run_id, seconds)
            meta = {
                "run_id": run_id,
                "mode": mode,
                "task": task_text,
                "start_url": url,
                "max_seconds": seconds,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "argv": argv,
                "run_dir": str(run_dir),
            }
            (demo_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            env = dict(os.environ)
            env["PYTHONUTF8"] = "1"
            stdout_handle = (demo_dir / "stdout.log").open("wb")
            stderr_handle = (demo_dir / "stderr.log").open("wb")
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=str(self.repo_root),
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                    start_new_session=(os.name != "nt"),
                )
            except OSError as exc:
                stdout_handle.close()
                stderr_handle.close()
                raise DemoError(f"启动执行子进程失败：{type(exc).__name__}: {exc}") from exc

            self._load_config()
            record = RunRecord(
                run_id=run_id,
                mode=mode,
                task=task_text,
                start_url=url,
                max_seconds=seconds,
                demo_dir=demo_dir,
                run_dir=run_dir,
                argv=argv,
                process=process,
                model_name=self._config.model.name if self._config else None,
            )
            self.current = record
            self.history.append(run_id)
            return run_id

    def build_argv(
        self, mode: str, task_file: Path, run_id: str, seconds: float
    ) -> list[str]:
        argv = [
            sys.executable,
            "-u",
            "-m",
            "webagent",
            "run",
            "--config",
            str(self.config_path),
            "--backend",
            "playwright",
            "--task-file",
            str(task_file),
            "--run-id",
            run_id,
            "--runtime-dir",
            str(self.runtime_dir),
            "--max-seconds",
            repr(float(seconds)),
            "--offline",
            "--headless" if self.headless else "--no-headless",
        ]
        if mode == "offline":
            argv += ["--model-script", str(self.offline_script)]
        return argv

    def _validate_task(self, task: Any) -> str:
        if not isinstance(task, str) or not task.strip():
            raise DemoError("任务描述不能为空，请输入要完成的任务")
        if len(task) > MAX_TASK_CHARS:
            raise DemoError(f"任务描述过长（{len(task)} 字符，上限 {MAX_TASK_CHARS}）")
        return task

    def _validate_url(self, url: Any) -> str:
        if not isinstance(url, str) or not url.strip():
            raise DemoError("起始网址不能为空，请输入 http(s) 网址")
        text = url.strip()
        parsed = urlparse(text)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise DemoError("起始网址必须是完整的 http:// 或 https:// 网址")
        return text

    def _validate_seconds(self, value: Any) -> float:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            raise DemoError("最长运行秒数必须是正数") from None
        if not seconds > 0 or seconds != seconds:
            raise DemoError("最长运行秒数必须是正数")
        if seconds > MAX_SECONDS:
            raise DemoError(f"最长运行秒数不能超过 {MAX_SECONDS}")
        return seconds

    # ---------- 控制 ----------

    def control(self, command: str) -> None:
        with self.lock:
            record = self.current
            if record is None or record.state == "ended":
                raise DemoError("当前没有可以控制的运行")
            state = record.state
            if command == "pause":
                if state not in ("running", "starting"):
                    raise DemoError(f"当前状态（{STATE_LABELS[state]}）不能暂停")
                record.state = "pause_requested"
            elif command == "resume":
                if state != "paused":
                    raise DemoError(f"当前状态（{STATE_LABELS[state]}）不能继续，只有已暂停后才能继续")
                record.state = "resume_requested"
            elif command == "cancel":
                if state not in ACTIVE_STATES:
                    raise DemoError("当前没有可以取消的运行")
                record.state = "cancel_requested"
            else:
                raise DemoError("未知控制命令")
            control_path = record.run_dir / "control.jsonl"
            control_path.parent.mkdir(parents=True, exist_ok=True)
            with control_path.open("a", encoding="utf-8") as handle:
                handle.write(command + "\n")

    def force_end(self) -> None:
        with self.lock:
            record = self.current
            if record is None or not record.active:
                raise DemoError("当前没有需要强制结束的运行")
            record.forced = True
            record.state = "cancel_requested"
        self._kill_current_tree()

    def _kill_current_tree(self) -> None:
        record = self.current
        if record is None or record.process is None:
            return
        if record.process.poll() is not None:
            return
        self._kill_tree(record.process)
        try:
            record.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def _kill_tree(self, process: subprocess.Popen[Any]) -> None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def shutdown(self) -> None:
        """关闭服务时只清理本面板启动的子进程与浏览器。"""

        with self.lock:
            record = self.current
            if record is None or record.state == "ended":
                return
            record.forced = True
        try:
            self._kill_current_tree()
        finally:
            self.refresh()

    # ---------- 状态采集 ----------

    def refresh(self) -> None:
        """增量读取运行产物并推进状态机；文件不完整时下次重试。"""

        with self.lock:
            record = self.current
            if record is None or record.state == "ended":
                return
            self._ingest(record)

            process = record.process
            if process is not None and process.poll() is not None:
                record.exit_code = process.returncode
                if record.ended_at is None:
                    record.ended_at = time.time()
                record.state = "ended"
                record.outcome = self._map_outcome(record)
                return

            if record.state == "starting":
                if record.has_events:
                    record.state = "running"
            elif record.state == "pause_requested":
                if record.saw_paused:
                    record.state = "paused"
            elif record.state == "resume_requested":
                if record.saw_resumed:
                    record.state = "running"

    def _ingest(self, record: RunRecord) -> None:
        self._ingest_events(record)
        self._ingest_console(record, "stdout.log", "stdout_tail")
        self._ingest_console(record, "stderr.log", "stderr_tail")
        self._ingest_result(record)

    def _ingest_events(self, record: RunRecord) -> None:
        events_path = record.run_dir / "events.jsonl"
        try:
            raw = events_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        lines = raw.splitlines()
        if not raw.endswith("\n"):
            lines = lines[:-1]  # 尾行可能仍在写入，等下次完整后再解析
        if len(lines) < record._events_line_count:
            # 文件被截断重写：回到开头重新解析
            record._events_line_count = 0
            record.log_entries.clear()
        new_lines = lines[record._events_line_count:]
        record._events_line_count = len(lines)
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            record.has_events = True
            kind = event.get("event_type")
            if kind == "observation" and event.get("url"):
                record.current_url = str(event["url"])
            if kind == "proposed" and event.get("action_arguments") is not None:
                record.last_action = (
                    f"{event.get('action_type')} {_fmt_args(event.get('action_arguments'))}"
                )
            elif kind == "dispatched" and event.get("action_arguments") is not None:
                record.last_action = (
                    f"{event.get('action_type')} {_fmt_args(event.get('action_arguments'))}"
                )
            if record.state == "pause_requested" and kind == "paused":
                record.saw_paused = True
            if record.state == "resume_requested" and kind == "resumed":
                record.saw_resumed = True
            if kind == "artifact_saved":
                result = event.get("result")
                if isinstance(result, dict) and result.get("kind") == "screenshot":
                    ref = str(result.get("reference") or "")
                    name = Path(ref).name
                    if name and all(s["file"] != name for s in record.screenshots):
                        record.screenshots.append(
                            {
                                "file": name,
                                "url": event.get("url"),
                                "observation_id": event.get("observation_id"),
                                "time": str(event.get("time") or "")[11:19],
                            }
                        )
            entry = _event_entry(event)
            if entry is not None:
                record.log_entries.append(entry)

    def _ingest_console(self, record: RunRecord, name: str, target: str) -> None:
        path = record.demo_dir / name
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        lines = [self.redactor.redact_text(line) for line in raw.splitlines()]
        setattr(record, target, lines[-200:])

    def _ingest_result(self, record: RunRecord) -> None:
        path = record.run_dir / "result.json"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            record.result = payload

    def _map_outcome(self, record: RunRecord) -> str:
        if record.forced:
            return "interrupted"
        termination = (record.result or {}).get("termination")
        if termination == "completed":
            return "completed"
        if termination == "deadline":
            return "timeout"
        if termination == "cancelled":
            return "cancelled"
        if termination == "error":
            return "failed"
        if termination in ("incomplete", "budget_exhausted"):
            return "incomplete"
        if not record.has_events or record.exit_code == 1:
            return "start_failed"
        return "failed"

    # ---------- 证据访问 ----------

    def resolve_download(self, run_id: str, name: str) -> tuple[Path, str]:
        if run_id not in self.history:
            raise DemoError("该运行不由本面板启动，拒绝访问")
        if name not in DOWNLOAD_NAMES:
            raise DemoError("不允许下载该文件")
        record = self.current
        if record and record.run_id == run_id:
            demo_dir, run_dir = record.demo_dir, record.run_dir
        else:
            demo_dir = self.runtime_dir / "demo" / run_id
            run_dir = self.runtime_dir / "runs" / run_id
        for directory in (demo_dir, run_dir):
            path = (directory / name).resolve()
            if path.is_file():
                return path, name
        raise DemoError("文件尚不存在")

    def resolve_screenshot(self, run_id: str, filename: str) -> Path:
        if run_id not in self.history:
            raise DemoError("该运行不由本面板启动，拒绝访问")
        if not SCREENSHOT_RE.match(filename):
            raise DemoError("非法截图文件名")
        record = self.current
        run_dir = record.run_dir if record and record.run_id == run_id else self.runtime_dir / "runs" / run_id
        screenshots_dir = (run_dir / "screenshots").resolve()
        path = (screenshots_dir / filename).resolve()
        if not str(path).startswith(str(screenshots_dir) + os.sep) or not path.is_file():
            raise DemoError("截图不存在")
        return path

    def _available_files(self, record: RunRecord) -> list[str]:
        names: list[str] = []
        for name in sorted(DOWNLOAD_NAMES):
            if (record.demo_dir / name).is_file() or (record.run_dir / name).is_file():
                names.append(name)
        return names

    # ---------- 快照 ----------

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self.refresh()
            mode_status = self.real_mode_status()
            payload: dict[str, Any] = {
                "server": {
                    "port": self.port,
                    "fixture_url": self.fixture_url(),
                    "runtime_dir": str(self.runtime_dir),
                },
                "mode": mode_status,
                "preset": {"task": PRESET_TASK, "start_url": self.fixture_url()},
                "run": None,
                "history": list(self.history),
            }
            record = self.current
            if record is None:
                return payload

            ended = record.state == "ended"
            elapsed = (record.ended_at or time.time()) - record.started_at
            result = record.result or {}
            finish = result.get("finish") if isinstance(result.get("finish"), dict) else {}
            answer = finish.get("message") or finish.get("claim")
            termination = result.get("termination")
            verification = result.get("verification")

            payload["run"] = {
                "run_id": record.run_id,
                "mode": record.mode,
                "state": record.state,
                "state_label": STATE_LABELS[record.state],
                "outcome": record.outcome,
                "outcome_label": OUTCOME_LABELS.get(record.outcome or "", None),
                "task": record.task,
                "start_url": record.start_url,
                "max_seconds": record.max_seconds,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "elapsed_s": round(elapsed, 1),
                "pid": record.process.pid if record.process else None,
                "exit_code": record.exit_code,
                "forced": record.forced,
                "current_url": record.current_url,
                "last_action": record.last_action,
                "model_name": record.model_name,
                "answer": answer,
                "termination": termination,
                "termination_label": TERMINATION_LABELS.get(str(termination), None)
                if termination
                else None,
                "unfinished": result.get("unfinished"),
                "verification_text": (
                    json.dumps(verification, ensure_ascii=False)
                    if verification
                    else "未提供本地核验结果"
                ),
                "run_dir": str(record.run_dir.resolve()),
                "demo_dir": str(record.demo_dir.resolve()),
                "events": record.log_entries[-300:],
                "screenshots": record.screenshots[-50:],
                "stdout_tail": record.stdout_tail,
                "stderr_tail": record.stderr_tail,
                "files": self._available_files(record),
            }
            return payload


# ---------- HTTP 层 ----------

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


def make_handler(app: DemoApp) -> type[BaseHTTPRequestHandler]:
    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "WebAgentDemo/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            sys.stdout.write("demo: " + (format % args) + "\n")

        # ----- 基础应答 -----

        def _send_bytes(
            self, data: bytes, ctype: str, status: int = 200, download: str | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if download:
                safe = download.replace('"', "")
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{safe}"'
                )
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8", status=status)

        def _send_error_json(self, message: str, status: int = 400) -> None:
            self._send_json({"error": message}, status=status)

        def _send_file(self, path: Path, download: str | None = None) -> None:
            ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
            try:
                data = path.read_bytes()
            except OSError:
                self._send_error_json("文件读取失败", status=500)
                return
            self._send_bytes(data, ctype, download=download)

        def _static_root(self, rel: str, root: Path) -> None:
            base = root.resolve()
            target = (base / rel).resolve()
            if target == base:
                target = base / "index.html"
            if not str(target).startswith(str(base) + os.sep) and target != base:
                self._send_error_json("路径非法", status=403)
                return
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                self._send_error_json("未找到文件", status=404)
                return
            self._send_file(target)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > MAX_BODY_BYTES:
                raise DemoError("请求体过大")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise DemoError(f"请求体不是有效 JSON：{exc}") from exc
            if not isinstance(payload, dict):
                raise DemoError("请求体必须是 JSON 对象")
            return payload

        # ----- 路由 -----

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            try:
                if path in ("/", "/index.html"):
                    self._static_root("index.html", STATIC_DIR)
                elif path.startswith("/static/"):
                    self._static_root(path[len("/static/"):], STATIC_DIR)
                elif path.startswith("/fixture/"):
                    self._static_root(path[len("/fixture/"):], app.fixture_dir)
                elif path == "/api/state":
                    self._send_json(app.snapshot())
                elif path == "/api/screenshot":
                    run_id = (query.get("run") or [""])[0]
                    filename = (query.get("file") or [""])[0]
                    try:
                        image = app.resolve_screenshot(run_id, filename)
                    except DemoError as exc:
                        self._send_error_json(str(exc), status=404)
                        return
                    self._send_file(image)
                elif path == "/api/download":
                    run_id = (query.get("run") or [""])[0]
                    name = (query.get("name") or [""])[0]
                    try:
                        file_path, download_name = app.resolve_download(run_id, name)
                    except DemoError as exc:
                        self._send_error_json(str(exc), status=403)
                        return
                    if name in ("stdout.log", "stderr.log"):
                        text = app.redactor.redact_text(
                            file_path.read_text(encoding="utf-8", errors="replace")
                        )
                        self._send_bytes(
                            text.encode("utf-8"),
                            "text/plain; charset=utf-8",
                            download=download_name,
                        )
                    else:
                        self._send_file(file_path, download=download_name)
                else:
                    self._send_error_json("未知路径", status=404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = self._read_json_body()
                if path == "/api/start":
                    run_id = app.start(
                        mode=str(body.get("mode") or "real"),
                        task=body.get("task"),
                        start_url=body.get("start_url"),
                        max_seconds=body.get("max_seconds"),
                    )
                    self._send_json({"run_id": run_id})
                elif path == "/api/control":
                    command = str(body.get("command") or "")
                    app.control(command)
                    self._send_json({"ok": True, "command": command})
                elif path == "/api/force-end":
                    app.force_end()
                    self._send_json({"ok": True})
                else:
                    self._send_error_json("未知路径", status=404)
            except DemoError as exc:
                status = 409 if "已有任务" in str(exc) else 400
                self._send_error_json(str(exc), status=status)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return DemoHandler


def make_server(app: DemoApp) -> ThreadingHTTPServer:
    handler = make_handler(app)
    server = ThreadingHTTPServer(("127.0.0.1", app.port), handler)
    server.daemon_threads = True
    app.port = server.server_address[1]
    return server


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="demo", description="WebAgent 本地演示面板")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--runtime-dir", default=None, help="覆盖运行数据根目录")
    args = parser.parse_args(argv)

    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else DEFAULT_RUNTIME_DIR
    app = DemoApp(port=args.port, runtime_dir=runtime_dir)

    try:
        server = make_server(app)
    except OSError as exc:
        print(
            f"端口 {args.port} 无法使用（{exc}）。可能有其他程序占用该端口，"
            f"可用 --port 更换端口后重试，面板地址与预设网址会随端口变化。",
            flush=True,
        )
        return 2

    url = f"http://127.0.0.1:{app.port}"
    print(f"WebAgent 演示面板已启动：{url}", flush=True)
    print("离线脚本模式无需密钥；真实模型模式需设置 WEBAGENT_MODEL_BASE_URL / WEBAGENT_MODEL_API_KEY。", flush=True)
    print("按 Ctrl+C 停止面板（会一并清理本面板启动的执行进程与浏览器）。", flush=True)

    if not args.no_browser:
        threading.Timer(0.5, lambda: _open_browser(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止面板并清理本面板启动的进程…", flush=True)
    finally:
        app.shutdown()
        server.server_close()
    print("面板已停止。", flush=True)
    return 0


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - 自动打开失败不影响使用
        print(f"自动打开浏览器失败，请手动访问：{url}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
