"""Tests for the dashboard History view (stored runs + saved sessions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from effgen.core.session import Session
from effgen.observability import run_log

STATIC = Path(__file__).resolve().parents[2] / "effgen" / "dashboard" / "static"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    run_log.clear()
    yield
    run_log.clear()


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from effgen.server.app import create_app

    return TestClient(create_app(dev_mode=True), raise_server_exceptions=False)


def _seed():
    run_log.record_run(model="gpt-5-nano", run_id="r-ok", task="refund policy",
                       output="an answer", cost_usd=0.0001, duration_s=1.2,
                       session_id="support-1")
    run_log.record_run(model="bogus", run_id="r-bad", task="broken run",
                       error="model not found")
    session = Session(session_id="support-1", agent_name="cli-agent")
    session.add_message("user", "hi", model="gpt-5-nano", run_id="r-ok")
    session.add_message("assistant", "hello", model="gpt-5-nano", run_id="r-ok",
                        cost_usd=0.0001)
    session.metadata["model"] = "gpt-5-nano"
    session.save()


def test_history_endpoint_returns_runs_and_sessions(client):
    _seed()

    data = client.get("/dashboard/history.json").json()

    assert {r["run_id"] for r in data["runs"]} == {"r-ok", "r-bad"}
    assert [s["session_id"] for s in data["sessions"]] == ["support-1"]
    assert data["persisted"] is True


def test_history_records_carry_the_drill_in_fields(client):
    _seed()

    run = next(r for r in client.get("/dashboard/history.json").json()["runs"]
               if r["run_id"] == "r-ok")

    assert run["task"] == "refund policy"
    assert run["output"] == "an answer"
    assert run["session_id"] == "support-1"
    assert run["status"] == "ok"
    assert run["cost_usd"] == pytest.approx(0.0001)


def test_history_filters_by_status_and_search(client):
    _seed()

    failed = client.get("/dashboard/history.json?status=failed").json()["runs"]
    assert [r["run_id"] for r in failed] == ["r-bad"]

    found = client.get("/dashboard/history.json?search=refund").json()["runs"]
    assert [r["run_id"] for r in found] == ["r-ok"]


def test_history_lookup_of_a_single_run(client):
    _seed()

    data = client.get("/dashboard/history.json?run_id=r-bad").json()

    assert data["run"]["error"] == "model not found"


def test_history_is_empty_not_broken_without_data(client):
    data = client.get("/dashboard/history.json").json()

    assert data["runs"] == []
    assert data["sessions"] == []


def test_history_requires_the_dashboard_access_rule(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from effgen.server.app import create_app

    monkeypatch.delenv("EFFGEN_PUBLIC_DASHBOARD", raising=False)
    monkeypatch.setenv("EFFGEN_API_KEY", "test-key-not-supplied")
    authed = TestClient(create_app(), raise_server_exceptions=False)

    assert authed.get("/dashboard/history.json").status_code == 401


def test_history_view_uses_only_local_assets():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()

    assert 'id="panel-history"' in html
    assert 'id="history-tbody"' in html
    assert "/dashboard/history.json" in js
    for source in (html, js):
        assert "http://" not in source
        assert "https://" not in source.replace("https://www.w3.org", "")


def test_history_elements_referenced_by_the_script_exist():
    html = (STATIC / "index.html").read_text()

    for element_id in ("history-sub", "history-tbody", "history-detail",
                       "history-sessions-tbody", "hist-search", "hist-status"):
        assert f'id="{element_id}"' in html


def test_run_detail_offers_a_summary_card_command_and_names_its_limits():
    js = (STATIC / "app.js").read_text()
    css = (STATIC / "style.css").read_text()

    # The command the button copies exports the selected run from history.
    assert "effgen runs show" in js and "--card" in js
    # The button says what the stored record can and cannot back, so nobody
    # expects the answer and trace a run-time card carries.
    assert "truncated answer, no step trace" in js
    assert "effgen run --card" in js
    assert ".run-export" in css
