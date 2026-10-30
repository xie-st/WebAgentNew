"""
effGen Alerting — Alertmanager-compatible alert rules + webhook dispatcher.

AlertWebhook
------------
Posts alert payloads to a Slack or Discord webhook URL.  The URL is never
written to logs in full — the path (which encodes the secret token) is
replaced with ``***`` before any logging occurs.

    from effgen.observability.alerting import AlertWebhook, Alert, AlertSeverity

    hook = AlertWebhook("https://hooks.slack.com/services/T.../B.../TOKEN")
    alert = Alert(
        name="HighErrorRate",
        severity=AlertSeverity.CRITICAL,
        summary="Error rate exceeded 5% for 10 minutes",
        value=0.08,
    )
    result = hook.fire(alert)   # returns dict with ok/error

AlertRuleValidator
------------------
Validates the bundled ``alert_rules.yaml`` against the Alertmanager rule
schema (promtool or built-in syntactic check).

    from effgen.observability.alerting import validate_alert_rules_yaml
    ok, errors = validate_alert_rules_yaml(path)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .redact import get_redactor
from .slo import SLOTracker

log = logging.getLogger(__name__)

_TIMEOUT = 15  # seconds — explicit, never None


# ---------------------------------------------------------------------------
# Alert model
# ---------------------------------------------------------------------------

class AlertSeverity(StrEnum):
    """Severity level of an alert: info, warning, or critical."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A single alert payload."""

    name: str
    """Short alert name, e.g. ``HighErrorRate``."""

    severity: AlertSeverity = AlertSeverity.WARNING
    """Severity level."""

    summary: str = ""
    """Human-readable summary of the alert condition."""

    description: str = ""
    """Longer description / runbook pointer."""

    value: float | None = None
    """Current metric value that triggered the alert (optional)."""

    threshold: float | None = None
    """Threshold that was crossed (optional)."""

    labels: dict[str, str] = field(default_factory=dict)
    """Additional key/value labels (provider, model, etc.)."""

    fired_at: float = field(default_factory=time.time)
    """UNIX timestamp when the alert was created."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    def to_slack_blocks(self) -> list[dict[str, Any]]:
        """Render as Slack Block Kit blocks."""
        sev_emoji = {"info": ":information_source:", "warning": ":warning:", "critical": ":red_circle:"}
        emoji = sev_emoji.get(self.severity.value, ":bell:")
        header_text = f"{emoji} *effGen Alert: {self.name}*  [{self.severity.value.upper()}]"
        body = self.summary or self.description
        fields: list[dict] = []
        if self.value is not None:
            fields.append({"type": "mrkdwn", "text": f"*Value:* {self.value}"})
        if self.threshold is not None:
            fields.append({"type": "mrkdwn", "text": f"*Threshold:* {self.threshold}"})
        for k, v in self.labels.items():
            fields.append({"type": "mrkdwn", "text": f"*{k}:* {v}"})

        blocks: list[dict] = [
            {"type": "section", "text": {"type": "mrkdwn", "text": header_text}},
        ]
        if body:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": body}})
        if fields:
            blocks.append({"type": "section", "fields": fields})
        blocks.append({"type": "divider"})
        return blocks

    def to_discord_embeds(self) -> list[dict[str, Any]]:
        """Render as Discord embed objects."""
        color_map = {"info": 3447003, "warning": 16776960, "critical": 15158332}
        color = color_map.get(self.severity.value, 10070709)
        fields = []
        if self.value is not None:
            fields.append({"name": "Value", "value": str(self.value), "inline": True})
        if self.threshold is not None:
            fields.append({"name": "Threshold", "value": str(self.threshold), "inline": True})
        for k, v in self.labels.items():
            fields.append({"name": k, "value": v, "inline": True})
        return [{
            "title": f"effGen Alert: {self.name}",
            "description": self.summary or self.description,
            "color": color,
            "fields": fields,
            "footer": {"text": f"severity: {self.severity.value}"},
        }]


# ---------------------------------------------------------------------------
# URL redaction
# ---------------------------------------------------------------------------

def _redact_webhook_url(url: str) -> str:
    """Keep host, replace path+token with ***."""
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return "***"
        return f"{parsed.scheme}://{parsed.netloc}/***"
    except Exception:
        return "***"


def _is_slack_url(url: str) -> bool:
    return "hooks.slack.com" in url


def _is_discord_url(url: str) -> bool:
    return "discord.com" in url or "discordapp.com" in url


# ---------------------------------------------------------------------------
# Bridge to the built-in Slack/Discord webhook tools
# ---------------------------------------------------------------------------

def _slack_tool() -> Any:
    """Construct a ``SlackWebhookTool`` (imported lazily to avoid cycles)."""
    from ..tools.builtin.slack_webhook import SlackWebhookTool  # noqa: PLC0415
    return SlackWebhookTool()


def _discord_tool() -> Any:
    """Construct a ``DiscordWebhookTool`` (imported lazily to avoid cycles)."""
    from ..tools.builtin.discord_webhook import DiscordWebhookTool  # noqa: PLC0415
    return DiscordWebhookTool()


def _run_tool_sync(tool: Any, kwargs: dict[str, Any]) -> Any:
    """Run an async ``BaseTool.execute`` from sync code and return its result.

    Uses a private event loop in a worker thread when called from within a
    running loop, so :meth:`AlertWebhook.fire` stays synchronous and safe to
    call from any context.
    """
    import asyncio  # noqa: PLC0415

    coro_factory = lambda: tool.execute(**kwargs)  # noqa: E731

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to drive one directly.
        return asyncio.run(coro_factory())

    # A loop is already running in this thread; offload to a fresh one.
    import concurrent.futures  # noqa: PLC0415

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_factory())).result()


# ---------------------------------------------------------------------------
# AlertWebhook
# ---------------------------------------------------------------------------

class AlertWebhook:
    """
    Dispatches alert payloads to a Slack or Discord webhook URL.

    The URL is detected automatically (Slack vs Discord).  Custom webhook
    types can be handled via the ``fire`` method by subclassing.

    Parameters
    ----------
    url:
        Slack Incoming Webhook URL **or** Discord webhook URL.
        **Never logged in full** — the path/token is replaced with ``***``.
    """

    def __init__(self, url: str) -> None:
        if not url:
            raise ValueError("AlertWebhook: url must be a non-empty string")
        self._url = url
        self._redacted = _redact_webhook_url(url)
        self._redactor = get_redactor()

    @property
    def redacted_url(self) -> str:
        """Public read-only view of the redacted URL (safe to log/display)."""
        return self._redacted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fire(self, alert: Alert) -> dict[str, Any]:
        """
        Post *alert* to the configured webhook.

        Returns a dict:
        - ``{"ok": True, "webhook": "<redacted>"}`` on success.
        - ``{"ok": False, "error": "...", "webhook": "<redacted>"}`` on error.

        Delivery for Slack and Discord URLs is handed to the corresponding
        built-in webhook tools (``SlackWebhookTool`` / ``DiscordWebhookTool``);
        any other URL falls back to a generic Alertmanager-style JSON POST.

        This method is **non-raising** — delivery failures are logged but
        never propagated so that alerting never blocks the agent.
        """
        try:
            if _is_slack_url(self._url):
                result = self._fire_via_tool(_slack_tool(), self._slack_kwargs(alert))
            elif _is_discord_url(self._url):
                result = self._fire_via_tool(_discord_tool(), self._discord_kwargs(alert))
            else:
                # Generic JSON POST (Alertmanager-style) — no v0.2.6 tool covers
                # arbitrary receivers, so post directly.
                result = self._post(json.dumps(self._generic_payload(alert)).encode())

            log.info(
                "alert.fired  name=%s  severity=%s  webhook=%s  status=%s",
                alert.name,
                alert.severity.value,
                self._redacted,
                result.get("status"),
            )
            return {"ok": True, "webhook": self._redacted, **result}
        except Exception as exc:
            log.error(
                "alert.fire_failed  name=%s  webhook=%s  error=%r",
                alert.name,
                self._redacted,
                str(exc)[:200],
            )
            return {"ok": False, "error": str(exc)[:200], "webhook": self._redacted}

    def _fire_via_tool(self, tool: Any, tool_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Run a built-in webhook tool synchronously and normalise its result.

        Raises ``RuntimeError`` on delivery failure so the caller's non-raising
        wrapper records it uniformly with the direct-POST path.
        """
        result = _run_tool_sync(tool, dict(tool_kwargs, webhook_url=self._url))
        # BaseTool.execute() returns success=True whenever _execute() did not
        # raise; the tool's own delivery flag lives inside output["success"].
        payload = result.output if isinstance(result.output, dict) else {}
        delivered = bool(result.success) and bool(payload.get("success", result.success))
        if not delivered:
            err = payload.get("error") or result.error or "webhook delivery failed"
            raise RuntimeError(str(err))
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        return {"status": (data or {}).get("response_code", 200)}

    def fire_many(self, alerts: list[Alert]) -> list[dict[str, Any]]:
        """Fire multiple alerts; returns a list of per-alert results."""
        return [self.fire(a) for a in alerts]

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _slack_kwargs(self, alert: Alert) -> dict[str, Any]:
        """Kwargs for SlackWebhookTool.execute()."""
        return {
            "text": f"effGen Alert: {alert.name} [{alert.severity.value.upper()}] — {alert.summary}",
            "blocks": alert.to_slack_blocks(),
        }

    def _discord_kwargs(self, alert: Alert) -> dict[str, Any]:
        """Kwargs for DiscordWebhookTool.execute()."""
        return {
            "content": f"**effGen Alert: {alert.name}** [{alert.severity.value.upper()}]",
            "embeds": alert.to_discord_embeds(),
        }

    def _slack_payload(self, alert: Alert) -> bytes:
        """Raw Slack JSON body (kept for direct inspection / tests)."""
        return json.dumps(self._slack_kwargs(alert)).encode()

    def _discord_payload(self, alert: Alert) -> bytes:
        """Raw Discord JSON body (kept for direct inspection / tests)."""
        return json.dumps(self._discord_kwargs(alert)).encode()

    def _generic_payload(self, alert: Alert) -> dict[str, Any]:
        return {
            "version": "4",
            "groupKey": alert.name,
            "status": "firing",
            "receiver": "effgen-webhook",
            "alerts": [alert.to_dict()],
        }

    # ------------------------------------------------------------------
    # HTTP POST
    # ------------------------------------------------------------------

    def _post(self, payload: bytes) -> dict[str, Any]:
        req = Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
                body = resp.read().decode("utf-8", errors="replace")
                return {"status": resp.status, "body": body[:200]}
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# SLO -> alert bridge
# ---------------------------------------------------------------------------

def check_slo_and_alert(
    tracker: SLOTracker,
    name: str,
    webhook: AlertWebhook,
    *,
    burn_rate_threshold: float = 1.0,
    severity: AlertSeverity = AlertSeverity.CRITICAL,
) -> dict[str, Any] | None:
    """Check one SLO's burn rate and fire a webhook alert when it crosses the threshold.

    Evaluates the SLO's current burn rate against *tracker* and fires an alert
    via *webhook* when it exceeds *burn_rate_threshold*.

    This is the reusable form of the "read the live meter, decide, fire" loop
    an operator would otherwise hand-write around
    :meth:`~effgen.observability.slo.SLOTracker.status`. Call it on a schedule
    (a cron job, a background task, or after each batch of requests) to turn
    a registered :class:`~effgen.observability.slo.SLO` into an actual paging
    alert without bespoke glue.

    Args:
        tracker: An :class:`~effgen.observability.slo.SLOTracker` (e.g.
            ``effgen.observability.slo.get_tracker()``).
        name: The registered SLO name to evaluate.
        webhook: The :class:`AlertWebhook` to fire through when over budget.
        burn_rate_threshold: Fire when ``burn_rate > burn_rate_threshold``
            (default ``1.0`` — exactly on-budget or better never fires).
        severity: Severity to attach to the fired alert.

    Returns:
        The :meth:`AlertWebhook.fire` result dict when an alert was fired, or
        ``None`` when the SLO is within budget (nothing sent).
    """
    status = tracker.status(name)
    burn_rate = status["burn_rate"]
    if burn_rate <= burn_rate_threshold:
        return None
    alert = Alert(
        name=f"{name}_burn_rate",
        severity=severity,
        summary=(
            f"{name} burn rate {burn_rate:.2f}x over threshold "
            f"{burn_rate_threshold:.2f}x (target {status['target_pct']}%, "
            f"{status['bad_events']}/{status['total_events']} bad events)"
        ),
        value=burn_rate,
        threshold=burn_rate_threshold,
        labels={"slo": name},
    )
    return webhook.fire(alert)


# ---------------------------------------------------------------------------
# Alert rules YAML validator
# ---------------------------------------------------------------------------

_REQUIRED_RULE_FIELDS = {"alert", "expr", "for", "labels", "annotations"}


def validate_alert_rules_yaml(path: str | Path) -> tuple[bool, list[str]]:
    """
    Validate *path* against the Alertmanager rule file schema.

    *path* is a filesystem path to a YAML file, given as a ``str`` or a
    ``pathlib.Path``. Returns ``(ok, errors)`` where *errors* is a list of
    human-readable problem descriptions (empty on success).

    If ``promtool`` is on PATH, it is invoked first and its output used.
    Otherwise a pure-Python structural check is performed.
    """
    import subprocess  # noqa: PLC0415

    errors: list[str] = []

    if isinstance(path, str):
        if "\n" in path:
            # The single most likely reason a *string* argument contains a
            # newline is that the caller passed YAML text directly instead
            # of a path to it -- name that mistake instead of stat()'ing a
            # multi-line string and reporting a confusing "file not found".
            return False, [
                ("validate_alert_rules_yaml expects a path to a YAML file, "
                "not YAML text. Write the document to a file and pass its "
                "path, e.g. Path('/tmp/rules.yaml').")
            ]
        path = Path(path)
    elif not isinstance(path, Path):
        raise TypeError(
            "validate_alert_rules_yaml expects a str or pathlib.Path "
            f"file path, got {type(path).__name__}"
        )

    if not path.exists():
        return False, [f"File not found: {path}"]

    # Try promtool first
    try:
        result = subprocess.run(  # noqa: S603
            ["promtool", "check", "rules", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, []
        errors.append(f"promtool: {result.stderr.strip() or result.stdout.strip()}")
        return False, errors
    except FileNotFoundError:
        pass  # promtool not installed — fall through to built-in check
    except Exception as exc:
        errors.append(f"promtool error: {exc}")
        return False, errors

    # Built-in structural validation
    try:
        import yaml  # noqa: PLC0415
        with open(path) as fh:
            doc = yaml.safe_load(fh)
    except Exception as exc:
        return False, [f"YAML parse error: {exc}"]

    if not isinstance(doc, dict):
        return False, ["Top-level document must be a YAML mapping"]

    if "groups" not in doc:
        errors.append("Missing top-level 'groups' key")
        return False, errors

    for gi, group in enumerate(doc["groups"]):
        if not isinstance(group, dict):
            errors.append(f"groups[{gi}] is not a mapping")
            continue
        if "name" not in group:
            errors.append(f"groups[{gi}] missing 'name'")
        rules = group.get("rules", [])
        if not isinstance(rules, list):
            errors.append(f"groups[{gi}].rules must be a list")
            continue
        for ri, rule in enumerate(rules):
            if not isinstance(rule, dict):
                errors.append(f"groups[{gi}].rules[{ri}] is not a mapping")
                continue
            # Alerting rules (vs recording rules) have an 'alert' key
            if "alert" in rule:
                for field_name in ("expr", "labels", "annotations"):
                    if field_name not in rule:
                        errors.append(
                            f"groups[{gi}].rules[{ri}] ({rule.get('alert')!r}) "
                            f"missing required field '{field_name}'"
                        )

    return len(errors) == 0, errors
