"""
Tests for effgen.observability.alerting.

Covers:
- Alert dataclass construction and serialisation
- Slack/Discord payload rendering
- AlertWebhook.fire() non-raising contract
- URL redaction at the AlertWebhook layer
- validate_alert_rules_yaml() structural checks
- fire_many() batch dispatch
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from effgen.observability.alerting import (
    Alert,
    AlertSeverity,
    AlertWebhook,
    _redact_webhook_url,
    check_slo_and_alert,
    validate_alert_rules_yaml,
)
from effgen.observability.slo import SLO, SLOTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALERT_RULES_YAML = (
    Path(__file__).parents[2] / "docs" / "observability" / "alert_rules.yaml"
)


@pytest.fixture()
def basic_alert() -> Alert:
    return Alert(
        name="TestAlert",
        severity=AlertSeverity.WARNING,
        summary="Test summary",
        description="Test description",
        value=0.07,
        threshold=0.05,
        labels={"provider": "cerebras", "model": "llama3.1-8b"},
    )


@pytest.fixture()
def slack_hook() -> AlertWebhook:
    return AlertWebhook("https://hooks.slack.com/services/T123/B456/TOKEN")


@pytest.fixture()
def discord_hook() -> AlertWebhook:
    return AlertWebhook("https://discord.com/api/webhooks/123456/TOKEN")


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

class TestAlert:
    def test_defaults(self):
        a = Alert(name="X")
        assert a.severity == AlertSeverity.WARNING
        assert a.summary == ""
        assert a.labels == {}
        assert isinstance(a.fired_at, float)

    def test_to_dict_keys(self, basic_alert):
        d = basic_alert.to_dict()
        assert d["name"] == "TestAlert"
        assert d["severity"] == "warning"
        assert d["value"] == 0.07
        assert d["threshold"] == 0.05
        assert d["labels"] == {"provider": "cerebras", "model": "llama3.1-8b"}

    def test_fired_at_is_recent(self):
        before = time.time()
        a = Alert(name="X")
        after = time.time()
        assert before <= a.fired_at <= after

    def test_severity_enum_values(self):
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.WARNING == "warning"
        assert AlertSeverity.CRITICAL == "critical"


# ---------------------------------------------------------------------------
# Slack / Discord payload rendering
# ---------------------------------------------------------------------------

class TestPayloadRendering:
    def test_slack_blocks_structure(self, basic_alert):
        blocks = basic_alert.to_slack_blocks()
        assert isinstance(blocks, list)
        assert len(blocks) >= 2
        # First block contains the alert name
        first = json.dumps(blocks[0])
        assert "TestAlert" in first

    def test_slack_blocks_severity_emoji(self):
        a = Alert(name="X", severity=AlertSeverity.CRITICAL, summary="boom")
        blocks = a.to_slack_blocks()
        first_text = blocks[0]["text"]["text"]
        assert "CRITICAL" in first_text

    def test_discord_embeds_structure(self, basic_alert):
        embeds = basic_alert.to_discord_embeds()
        assert isinstance(embeds, list)
        assert len(embeds) == 1
        embed = embeds[0]
        assert embed["title"] == "effGen Alert: TestAlert"
        assert "color" in embed

    def test_discord_color_critical(self):
        a = Alert(name="X", severity=AlertSeverity.CRITICAL)
        embeds = a.to_discord_embeds()
        assert embeds[0]["color"] == 15158332  # red

    def test_discord_color_warning(self):
        a = Alert(name="X", severity=AlertSeverity.WARNING)
        embeds = a.to_discord_embeds()
        assert embeds[0]["color"] == 16776960  # yellow

    def test_discord_fields_include_value(self, basic_alert):
        embeds = basic_alert.to_discord_embeds()
        field_names = [f["name"] for f in embeds[0]["fields"]]
        assert "Value" in field_names
        assert "Threshold" in field_names

    def test_labels_appear_in_discord_fields(self, basic_alert):
        embeds = basic_alert.to_discord_embeds()
        field_names = [f["name"] for f in embeds[0]["fields"]]
        assert "provider" in field_names
        assert "model" in field_names

    def test_labels_appear_in_slack_fields(self, basic_alert):
        blocks = basic_alert.to_slack_blocks()
        content = json.dumps(blocks)
        assert "cerebras" in content
        assert "llama3.1-8b" in content


# ---------------------------------------------------------------------------
# URL redaction
# ---------------------------------------------------------------------------

class TestUrlRedaction:
    def test_slack_redaction(self):
        url = "https://hooks.slack.com/services/T123/B456/TOKEN"
        redacted = _redact_webhook_url(url)
        assert redacted == "https://hooks.slack.com/***"
        assert "TOKEN" not in redacted
        assert "T123" not in redacted

    def test_discord_redaction(self):
        url = "https://discord.com/api/webhooks/123456/TOKEN"
        redacted = _redact_webhook_url(url)
        assert redacted == "https://discord.com/***"
        assert "TOKEN" not in redacted

    def test_empty_url_returns_stars(self):
        assert _redact_webhook_url("") == "***"

    def test_hook_exposes_redacted_url(self, slack_hook):
        assert "***" in slack_hook.redacted_url
        assert "TOKEN" not in slack_hook.redacted_url

    def test_hook_redacted_url_keeps_host(self, slack_hook):
        assert "hooks.slack.com" in slack_hook.redacted_url

    def test_fire_result_contains_only_redacted_url(self, slack_hook, basic_alert):
        # Patch the tool-delivery boundary to avoid a real network call
        slack_hook._fire_via_tool = MagicMock(return_value={"status": 200})
        result = slack_hook.fire(basic_alert)
        assert "TOKEN" not in result.get("webhook", "")
        assert "***" in result.get("webhook", "")


# ---------------------------------------------------------------------------
# AlertWebhook.fire() — non-raising contract
# ---------------------------------------------------------------------------

class TestAlertWebhookFire:
    def test_fire_success_result(self, slack_hook, basic_alert):
        slack_hook._fire_via_tool = MagicMock(return_value={"status": 200})
        result = slack_hook.fire(basic_alert)
        assert result["ok"] is True
        assert result["webhook"] == slack_hook.redacted_url

    def test_fire_returns_false_on_network_error(self, slack_hook, basic_alert):
        slack_hook._fire_via_tool = MagicMock(side_effect=RuntimeError("Network error"))
        result = slack_hook.fire(basic_alert)
        assert result["ok"] is False
        assert "error" in result
        assert "Network error" in result["error"]

    def test_fire_never_raises(self, slack_hook, basic_alert):
        """fire() must not propagate any exception."""
        slack_hook._fire_via_tool = MagicMock(side_effect=Exception("Boom"))
        # Should not raise
        result = slack_hook.fire(basic_alert)
        assert result["ok"] is False

    def test_fire_many_returns_list(self, slack_hook):
        slack_hook._fire_via_tool = MagicMock(return_value={"status": 200})
        alerts = [Alert(name=f"A{i}") for i in range(3)]
        results = slack_hook.fire_many(alerts)
        assert len(results) == 3
        assert all(r["ok"] for r in results)

    def test_fire_many_partial_failure(self, slack_hook):
        """Some fires fail; others succeed. No exception escapes."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("even-failure")
            return {"status": 200}

        slack_hook._fire_via_tool = MagicMock(side_effect=side_effect)
        alerts = [Alert(name=f"A{i}") for i in range(4)]
        results = slack_hook.fire_many(alerts)
        assert len(results) == 4
        oks = [r["ok"] for r in results]
        assert True in oks
        assert False in oks

    def test_discord_hook_fire_success(self, discord_hook, basic_alert):
        discord_hook._fire_via_tool = MagicMock(return_value={"status": 204})
        result = discord_hook.fire(basic_alert)
        assert result["ok"] is True

    def test_generic_hook_fire(self, basic_alert):
        """A non-Slack/non-Discord URL posts a generic Alertmanager payload."""
        hook = AlertWebhook("https://my-alertmanager.internal/webhook")
        hook._post = MagicMock(return_value={"status": 200, "body": "ok"})
        result = hook.fire(basic_alert)
        assert result["ok"] is True

    def test_slack_fire_delegates_to_slack_tool(self, slack_hook, basic_alert):
        """Slack delivery must go through SlackWebhookTool with the URL set."""
        from effgen.tools.base_tool import ToolResult

        captured = {}

        def fake_run(tool, kwargs):
            captured["tool_type"] = type(tool).__name__
            captured["kwargs"] = kwargs
            return ToolResult(
                success=True,
                output={"success": True, "data": {"ok": True, "webhook": "x"}},
            )

        with patch("effgen.observability.alerting._run_tool_sync", side_effect=fake_run):
            result = slack_hook.fire(basic_alert)

        assert result["ok"] is True
        assert captured["tool_type"] == "SlackWebhookTool"
        # The secret URL is forwarded to the tool (which redacts it for logs).
        assert captured["kwargs"]["webhook_url"] == slack_hook._url
        assert "blocks" in captured["kwargs"]

    def test_discord_fire_delegates_to_discord_tool(self, discord_hook, basic_alert):
        from effgen.tools.base_tool import ToolResult

        captured = {}

        def fake_run(tool, kwargs):
            captured["tool_type"] = type(tool).__name__
            captured["kwargs"] = kwargs
            return ToolResult(
                success=True,
                output={"success": True, "data": {"ok": True, "response_code": 204}},
            )

        with patch("effgen.observability.alerting._run_tool_sync", side_effect=fake_run):
            result = discord_hook.fire(basic_alert)

        assert result["ok"] is True
        assert captured["tool_type"] == "DiscordWebhookTool"
        assert "embeds" in captured["kwargs"]

    def test_tool_delivery_failure_marks_not_ok(self, slack_hook, basic_alert):
        """If the tool reports delivery failure, fire() returns ok=False."""
        from effgen.tools.base_tool import ToolResult

        def fake_run(tool, kwargs):
            return ToolResult(
                success=True,
                output={"success": False, "data": None, "error": "HTTP 404"},
            )

        with patch("effgen.observability.alerting._run_tool_sync", side_effect=fake_run):
            result = slack_hook.fire(basic_alert)

        assert result["ok"] is False
        assert "404" in result["error"]


# ---------------------------------------------------------------------------
# AlertWebhook construction guards
# ---------------------------------------------------------------------------

class TestAlertWebhookConstruction:
    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="url must be a non-empty string"):
            AlertWebhook("")

    def test_valid_slack_url(self):
        hook = AlertWebhook("https://hooks.slack.com/services/T/B/X")
        assert hook.redacted_url == "https://hooks.slack.com/***"

    def test_valid_discord_url(self):
        hook = AlertWebhook("https://discord.com/api/webhooks/1/TOKEN")
        assert hook.redacted_url == "https://discord.com/***"


# ---------------------------------------------------------------------------
# Alert rules YAML validation
# ---------------------------------------------------------------------------

class TestValidateAlertRulesYaml:
    def test_bundled_rules_file_validates(self):
        """The shipped alert_rules.yaml must pass structural validation."""
        if not ALERT_RULES_YAML.exists():
            pytest.skip("alert_rules.yaml not found")
        ok, errors = validate_alert_rules_yaml(ALERT_RULES_YAML)
        assert ok, f"Validation errors: {errors}"

    def test_missing_file_returns_error(self, tmp_path):
        ok, errors = validate_alert_rules_yaml(tmp_path / "nonexistent.yaml")
        assert not ok
        assert any("not found" in e.lower() or "nonexistent" in e for e in errors)

    def test_empty_groups_passes(self, tmp_path):
        p = tmp_path / "rules.yaml"
        p.write_text("groups: []\n")
        ok, errors = validate_alert_rules_yaml(p)
        assert ok

    def test_missing_groups_key_fails(self, tmp_path):
        p = tmp_path / "rules.yaml"
        p.write_text("foo: bar\n")
        ok, errors = validate_alert_rules_yaml(p)
        assert not ok
        assert any("groups" in e for e in errors)

    def test_rule_missing_expr_fails(self, tmp_path):
        doc = {
            "groups": [{
                "name": "test",
                "rules": [{
                    "alert": "MyAlert",
                    # missing "expr"
                    "labels": {"severity": "warning"},
                    "annotations": {"summary": "x"},
                }]
            }]
        }
        p = tmp_path / "rules.yaml"
        p.write_text(yaml.dump(doc))
        ok, errors = validate_alert_rules_yaml(p)
        assert not ok
        assert any("expr" in e for e in errors)

    def test_rule_missing_labels_fails(self, tmp_path):
        doc = {
            "groups": [{
                "name": "test",
                "rules": [{
                    "alert": "MyAlert",
                    "expr": "up == 0",
                    # missing "labels"
                    "annotations": {"summary": "x"},
                }]
            }]
        }
        p = tmp_path / "rules.yaml"
        p.write_text(yaml.dump(doc))
        ok, errors = validate_alert_rules_yaml(p)
        assert not ok

    def test_rule_missing_annotations_fails(self, tmp_path):
        doc = {
            "groups": [{
                "name": "test",
                "rules": [{
                    "alert": "MyAlert",
                    "expr": "up == 0",
                    "labels": {"severity": "warning"},
                    # missing "annotations"
                }]
            }]
        }
        p = tmp_path / "rules.yaml"
        p.write_text(yaml.dump(doc))
        ok, errors = validate_alert_rules_yaml(p)
        assert not ok

    def test_valid_minimal_rule_passes(self, tmp_path):
        doc = {
            "groups": [{
                "name": "test",
                "rules": [{
                    "alert": "TestAlert",
                    "expr": "up == 0",
                    "for": "5m",
                    "labels": {"severity": "warning"},
                    "annotations": {"summary": "up is 0"},
                }]
            }]
        }
        p = tmp_path / "rules.yaml"
        p.write_text(yaml.dump(doc))
        ok, errors = validate_alert_rules_yaml(p)
        assert ok, errors

    def test_invalid_yaml_returns_error(self, tmp_path):
        p = tmp_path / "rules.yaml"
        p.write_text(": : : invalid yaml { [")
        ok, errors = validate_alert_rules_yaml(p)
        assert not ok
        assert any("yaml" in e.lower() or "parse" in e.lower() for e in errors)

    def test_non_dict_top_level_fails(self, tmp_path):
        p = tmp_path / "rules.yaml"
        p.write_text("- item1\n- item2\n")
        ok, errors = validate_alert_rules_yaml(p)
        assert not ok

    def test_recording_rule_no_alert_field_passes(self, tmp_path):
        """Recording rules (no 'alert' key) should not be flagged for missing expr etc."""
        doc = {
            "groups": [{
                "name": "recording",
                "rules": [{
                    "record": "job:up:ratio",
                    "expr": "avg(up) by (job)",
                }]
            }]
        }
        p = tmp_path / "rules.yaml"
        p.write_text(yaml.dump(doc))
        ok, errors = validate_alert_rules_yaml(p)
        assert ok, errors

    def test_str_path_accepted_like_a_path_object(self, tmp_path):
        p = tmp_path / "rules.yaml"
        p.write_text("groups: []\n")
        ok, errors = validate_alert_rules_yaml(str(p))
        assert ok, errors

    def test_yaml_text_passed_directly_names_the_mistake(self):
        # Passing the document's text instead of a path to it is a natural
        # first try; it must report a clear, actionable error instead of a
        # bare AttributeError from stat()'ing a multi-line string.
        yaml_text = "groups:\n- name: g\n  rules: []\n"
        ok, errors = validate_alert_rules_yaml(yaml_text)
        assert not ok
        assert any("YAML text" in e or "yaml text" in e.lower() for e in errors)

    def test_wrong_type_raises_typeerror_naming_expected_type(self):
        with pytest.raises(TypeError, match="str or pathlib.Path"):
            validate_alert_rules_yaml(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Payload construction helpers (private but tested for correctness)
# ---------------------------------------------------------------------------

class TestPayloadBytes:
    def test_slack_payload_is_valid_json(self, slack_hook, basic_alert):
        payload = slack_hook._slack_payload(basic_alert)
        d = json.loads(payload)
        assert "text" in d
        assert "blocks" in d

    def test_discord_payload_is_valid_json(self, discord_hook, basic_alert):
        payload = discord_hook._discord_payload(basic_alert)
        d = json.loads(payload)
        assert "content" in d
        assert "embeds" in d

    def test_generic_payload_has_alerts_key(self, basic_alert):
        hook = AlertWebhook("https://example.com/webhook")
        d = hook._generic_payload(basic_alert)
        assert "alerts" in d
        assert d["alerts"][0]["name"] == "TestAlert"


# ---------------------------------------------------------------------------
# SLO -> alert bridge
# ---------------------------------------------------------------------------

class TestCheckSloAndAlert:
    def _tracker_with(self, good: int, bad: int, target_pct: float = 99.0) -> SLOTracker:
        tracker = SLOTracker()
        tracker.register(SLO("model_call_success", target_pct=target_pct, window_seconds=3600))
        for _ in range(good):
            tracker.record("model_call_success", ok=True)
        for _ in range(bad):
            tracker.record("model_call_success", ok=False)
        return tracker

    def test_fires_when_over_budget(self, slack_hook):
        tracker = self._tracker_with(good=90, bad=10)  # burn_rate = 10x
        slack_hook._fire_via_tool = MagicMock(return_value={"status": 200})
        result = check_slo_and_alert(tracker, "model_call_success", slack_hook)
        assert result is not None
        assert result["ok"] is True
        slack_hook._fire_via_tool.assert_called_once()

    def test_returns_none_when_within_budget(self, slack_hook):
        tracker = self._tracker_with(good=100, bad=0)  # burn_rate = 0.0
        slack_hook._fire_via_tool = MagicMock(return_value={"status": 200})
        result = check_slo_and_alert(tracker, "model_call_success", slack_hook)
        assert result is None
        slack_hook._fire_via_tool.assert_not_called()

    def test_respects_custom_threshold(self, slack_hook):
        # burn_rate = 2x — over the default 1.0x threshold but under a 5x one.
        tracker = self._tracker_with(good=98, bad=2)
        slack_hook._fire_via_tool = MagicMock(return_value={"status": 200})
        assert check_slo_and_alert(
            tracker, "model_call_success", slack_hook, burn_rate_threshold=5.0
        ) is None
        assert check_slo_and_alert(
            tracker, "model_call_success", slack_hook, burn_rate_threshold=1.0
        ) is not None

    def test_alert_summary_names_the_slo_and_burn_rate(self, slack_hook):
        tracker = self._tracker_with(good=90, bad=10)
        captured: dict = {}
        original_fire = slack_hook.fire

        def capture_fire(alert):
            captured["alert"] = alert
            slack_hook._fire_via_tool = MagicMock(return_value={"status": 200})
            return original_fire(alert)

        slack_hook.fire = capture_fire
        check_slo_and_alert(tracker, "model_call_success", slack_hook)
        alert = captured["alert"]
        assert "model_call_success" in alert.summary
        assert alert.value == pytest.approx(10.0)
        assert alert.labels == {"slo": "model_call_success"}

    def test_never_raises_on_webhook_failure(self, slack_hook):
        """The webhook's own non-raising contract carries through unchanged."""
        tracker = self._tracker_with(good=0, bad=10)
        slack_hook._fire_via_tool = MagicMock(side_effect=RuntimeError("network down"))
        result = check_slo_and_alert(tracker, "model_call_success", slack_hook)
        assert result["ok"] is False
