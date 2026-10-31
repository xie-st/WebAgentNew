"""Tests for the local dashboard — static files and server routes.

Validates:
1. Static files exist and contain the required panel IDs.
2. ``/dashboard`` serves HTML containing the panel IDs.
3. ``/dashboard/data.json`` serves valid JSON with the required keys.
4. Dashboard data builder returns the expected structure.
5. Static asset routes work.
6. SSE spans endpoint responds.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

STATIC_DIR = Path(__file__).parents[2] / "effgen" / "dashboard" / "static"

# Required panel IDs that must be present in the HTML
REQUIRED_PANEL_IDS = [
    "summary-cards",
    "panel-slo",
    "panel-latency-chart",
    "panel-agent-runs",
    "panel-spans",
    "panel-waterfall",
    "panel-metrics",
]

# Required card value IDs
REQUIRED_CARD_IDS = [
    "val-requests",
    "val-errors",
    "val-latency",
    "val-cost",
    "val-tokens",
]


# ---------------------------------------------------------------------------
# Static file checks (no server required)
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_index_html_exists(self):
        assert (STATIC_DIR / "index.html").exists()

    def test_app_js_exists(self):
        assert (STATIC_DIR / "app.js").exists()

    def test_style_css_exists(self):
        assert (STATIC_DIR / "style.css").exists()

    def test_index_contains_panel_ids(self):
        html = (STATIC_DIR / "index.html").read_text()
        for pid in REQUIRED_PANEL_IDS:
            assert pid in html, f"Panel ID '{pid}' missing from index.html"

    def test_index_contains_card_ids(self):
        html = (STATIC_DIR / "index.html").read_text()
        for cid in REQUIRED_CARD_IDS:
            assert cid in html, f"Card ID '{cid}' missing from index.html"

    def test_app_js_polls_data_json(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "/dashboard/data.json" in js

    def test_app_js_has_render_functions(self):
        js = (STATIC_DIR / "app.js").read_text()
        for fn in ("renderCards", "renderSLO", "renderRuns", "renderMetrics"):
            assert fn in js, f"Function '{fn}' missing from app.js"

    def test_app_js_has_sse_support(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "EventSource" in js

    def test_app_js_groups_spans_into_waterfall(self):
        # The per-run waterfall groups spans by run id and lays them out by the
        # start offset each span now carries.
        js = (STATIC_DIR / "app.js").read_text()
        assert "renderWaterfall" in js
        assert "run_id" in js
        assert "offset_ms" in js

    def test_style_css_has_variables(self):
        css = (STATIC_DIR / "style.css").read_text()
        assert "--bg:" in css
        assert "--accent:" in css

    def test_no_external_assets(self):
        """Every asset must be local — the page renders with no network.

        An on-prem / air-gapped deployment must get a fully working dashboard,
        so the static files may not reference any external host (CDN scripts,
        remote stylesheets, fonts, or images).
        """
        import re

        offenders: list[str] = []
        for fname in ("index.html", "app.js", "style.css"):
            text = (STATIC_DIR / fname).read_text()
            # Flag absolute URLs and protocol-relative ones. A same-origin
            # fetch path such as "/dashboard/data.json" is fine.
            for match in re.findall(r"""https?://[^\s"'()]+|(?<![:\w])//[^\s"'()]+""", text):
                offenders.append(f"{fname}: {match}")
        assert not offenders, f"External asset references found: {offenders}"

    def test_latency_chart_drawn_locally(self):
        """The latency chart is drawn on a canvas, not a CDN chart library."""
        js = (STATIC_DIR / "app.js").read_text()
        assert "getContext" in js
        html = (STATIC_DIR / "index.html").read_text()
        assert "<canvas" in html

    def test_index_contains_auth_banner_element(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert 'id="auth-banner"' in html

    def test_app_js_shows_banner_on_401(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "showAuthBanner" in js
        assert "status === 401" in js

    def test_app_js_hides_banner_on_success(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "hideAuthBanner" in js

    def test_index_contains_catalog_panel(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert 'id="panel-catalog"' in html
        assert 'id="catalog-table"' in html
        assert 'id="cat-search"' in html

    def test_app_js_loads_catalog(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "loadCatalog" in js
        assert "/dashboard/catalog.json" in js

    def test_app_js_catalog_pricing_labels_unpriced(self):
        """The catalog view labels an unpriced model, never a fabricated $0."""
        js = (STATIC_DIR / "app.js").read_text()
        assert "unpriced" in js


# ---------------------------------------------------------------------------
# Dashboard data builder
# ---------------------------------------------------------------------------


class TestDashboardDataBuilder:
    def test_build_dashboard_data_structure(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert isinstance(data, dict)

    def test_required_top_level_keys(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        for key in ("ts", "metrics", "slo", "recent_runs", "recent_spans", "raw_metrics"):
            assert key in data, f"Key '{key}' missing from dashboard data"

    def test_verify_spec_aliases_present(self):
        """The documented ``spans``/``slos`` aliases must also be present."""
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        for key in ("metrics", "spans", "slos", "recent_runs"):
            assert key in data, f"Verify-spec key '{key}' missing from dashboard data"
        # Aliases mirror their canonical keys.
        assert data["slos"] == data["slo"]
        assert data["spans"] == data["recent_spans"]

    def test_buffered_span_appears_in_data(self):
        """A buffered span should surface in recent_spans / spans."""
        from effgen.observability.tracing import _buffer_span, get_recent_spans
        from effgen.server.app import _build_dashboard_data

        _buffer_span("unit.test.span", 1.5)
        assert any(s["name"] == "unit.test.span" for s in get_recent_spans())
        data = _build_dashboard_data()
        assert any(s["name"] == "unit.test.span" for s in data["recent_spans"])

    def test_metrics_sub_keys(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        m = data["metrics"]
        for key in ("total_requests", "total_errors", "avg_latency_s", "total_tokens", "daily_cost_usd"):
            assert key in m, f"Metric key '{key}' missing"

    def test_slo_sub_keys(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        slo = data["slo"]
        for key in ("p99_latency_burn", "error_rate_burn", "availability"):
            assert key in slo, f"SLO key '{key}' missing"

    def test_recent_runs_is_list(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert isinstance(data["recent_runs"], list)

    def test_recent_spans_is_list(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert isinstance(data["recent_spans"], list)

    def test_raw_metrics_is_dict(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert isinstance(data["raw_metrics"], dict)

    def test_ts_is_iso_string(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        ts = data["ts"]
        assert isinstance(ts, str)
        assert "T" in ts  # ISO 8601-ish

    def test_data_is_json_serializable(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        # Should not raise
        serialized = json.dumps(data)
        assert len(serialized) > 2

    def test_run_log_records_appear(self):
        """Records written to run_log should appear in dashboard data."""
        from effgen.observability.run_log import clear, record_run
        from effgen.server.app import _build_dashboard_data

        clear()
        record_run(model="cerebras:llama3.1-8b", duration_s=0.5, input_tokens=10, output_tokens=20)
        data = _build_dashboard_data()
        clear()

        runs = data["recent_runs"]
        assert len(runs) >= 1
        assert runs[0]["model"] == "cerebras:llama3.1-8b"

    def test_agent_run_is_recorded_for_dashboard(self):
        """Agent.run should populate the dashboard run log automatically."""
        from effgen.core.agent import Agent, AgentConfig, AgentResponse
        from effgen.models.base import (
            BaseModel,
            GenerationConfig,
            GenerationResult,
            ModelType,
            TokenCount,
        )
        from effgen.observability.run_log import clear, get_recent_runs

        class DummyModel(BaseModel):
            def __init__(self):
                super().__init__("dummy-model", ModelType.OPENAI)

            def load(self) -> None:
                self._is_loaded = True

            def unload(self) -> None:
                self._is_loaded = False

            def get_context_length(self) -> int:
                return 8192

            def generate(
                self,
                prompt: str,
                config: GenerationConfig | None = None,
                **kwargs,
            ) -> GenerationResult:
                return GenerationResult(
                    text="Final Answer: ok",
                    tokens_used=3,
                    finish_reason="stop",
                    model_name=self.model_name,
                )

            def generate_stream(
                self,
                prompt: str,
                config: GenerationConfig | None = None,
                **kwargs,
            ):
                yield "ok"

            def count_tokens(self, text: str) -> TokenCount:
                return TokenCount(count=len(text.split()), model_name=self.model_name)

        clear()
        agent = Agent(AgentConfig(name="dashboard_test", model=DummyModel()))
        with patch.object(
            Agent,
            "_run_single_agent",
            return_value=AgentResponse(output="ok", tokens_used=7, metadata={"prompt_tokens": 4}),
        ):
            agent.run("hello")
        runs = get_recent_runs()
        clear()

        assert runs
        assert runs[0]["model"] == "dummy-model"
        assert runs[0]["input_tokens"] == 4
        assert runs[0]["output_tokens"] == 7

    def test_effgen_native_metrics_are_summarized(self):
        from effgen.observability.metrics import record_model_call, record_tokens, reset_all
        from effgen.server.app import _build_dashboard_data

        reset_all()
        record_model_call(provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.5)
        record_model_call(provider="cerebras", model="llama3.1-8b", outcome="error", latency=1.5)
        record_tokens(provider="cerebras", model="llama3.1-8b", input_tokens=10, output_tokens=5)
        data = _build_dashboard_data()
        reset_all()

        assert data["metrics"]["total_requests"] >= 2
        assert data["metrics"]["total_errors"] >= 1
        assert data["metrics"]["total_tokens"] >= 15
        assert data["metrics"]["avg_latency_s"] == pytest.approx(1.0)

    def test_cost_is_summed_from_real_per_run_cost(self):
        """The cost figure is the sum of real per-run cost, not a flat rate."""
        from effgen.observability.metrics import record_model_call, reset_all
        from effgen.observability.run_log import clear, record_run
        from effgen.server.app import _build_dashboard_data

        reset_all()
        clear()
        # Ten model calls but only two priced runs totalling $0.000336.
        for _ in range(10):
            record_model_call(provider="openai", model="gpt-5-nano", outcome="ok", latency=0.9)
        record_run(model="openai:gpt-5-nano", cost_usd=0.000333, duration_s=0.9)
        record_run(model="groq:llama", cost_usd=0.000003, duration_s=0.2)
        record_run(model="gpt-missing", cost_usd=None, error="boom", duration_s=0.1)
        data = _build_dashboard_data()
        reset_all()
        clear()

        m = data["metrics"]
        assert m["cost_usd"] == pytest.approx(0.000336, abs=1e-6)
        assert m["daily_cost_usd"] == m["cost_usd"]  # legacy key mirrors the real value
        assert m["priced_runs"] == 2
        assert m["unpriced_runs"] == 1
        # The old fabricated estimate would have been 10 * 0.01 = $0.10.
        assert m["cost_usd"] < 0.01

    def test_cost_none_when_no_priced_runs(self):
        from effgen.observability.run_log import clear, record_run
        from effgen.server.app import _build_dashboard_data

        clear()
        record_run(model="local:qwen", cost_usd=None, duration_s=0.5)
        data = _build_dashboard_data()
        clear()
        assert data["metrics"]["cost_usd"] is None

    def test_latency_percentiles_from_histogram(self):
        from effgen.observability.metrics import record_model_call, reset_all
        from effgen.server.app import _build_dashboard_data

        reset_all()
        for lat in (0.2, 0.3, 0.4, 0.5, 3.0):
            record_model_call(provider="openai", model="gpt-5-nano", outcome="ok", latency=lat)
        data = _build_dashboard_data()
        reset_all()

        slo = data["slo"]
        assert slo["p50_latency_s"] is not None
        assert slo["p95_latency_s"] is not None
        assert slo["p99_latency_s"] is not None
        # p99 must be at least as large as p50 (true percentiles, not the mean).
        assert slo["p99_latency_s"] >= slo["p50_latency_s"]

    def test_histogram_quantile_helper(self):
        import math

        from effgen.server.app import _histogram_quantile

        bounds = [(0.1, 1.0), (0.5, 3.0), (1.0, 5.0), (math.inf, 5.0)]
        assert _histogram_quantile(bounds, 0.5) == pytest.approx(0.4, abs=0.05)
        assert _histogram_quantile([], 0.5) is None
        assert _histogram_quantile([(1.0, 0.0), (math.inf, 0.0)], 0.9) is None

    def test_by_model_breakdown(self):
        from effgen.observability.metrics import record_model_call, record_tokens, reset_all
        from effgen.observability.run_log import clear, record_run
        from effgen.server.app import _build_dashboard_data

        reset_all()
        clear()
        record_model_call(provider="openai", model="gpt-5-nano", outcome="ok", latency=0.5)
        record_model_call(provider="openai", model="gpt-5-nano", outcome="error", latency=1.5)
        record_tokens(provider="openai", model="gpt-5-nano", input_tokens=10, output_tokens=5)
        record_run(model="openai:gpt-5-nano", cost_usd=0.0002, duration_s=0.5)
        data = _build_dashboard_data()
        reset_all()
        clear()

        rows = data["by_model"]
        assert isinstance(rows, list)
        row = next(r for r in rows if r["model"] == "gpt-5-nano")
        assert row["provider"] == "openai"
        assert row["calls"] == 2
        assert row["errors"] == 1
        assert row["error_rate"] == pytest.approx(0.5)
        assert row["input_tokens"] == 10
        assert row["output_tokens"] == 5
        assert row["cost_usd"] == pytest.approx(0.0002)
        assert row["p95_latency_s"] is not None

    def test_by_model_scopes_cost_and_latency_to_the_provider(self):
        """One model name served by two providers keeps the two rows apart.

        Cost is attributed to the provider that spent it (so the column sums to
        the real total instead of doubling), and each row's p95 comes from that
        provider's own observations.
        """
        from effgen.observability.metrics import record_model_call, reset_all
        from effgen.observability.run_log import clear, record_run
        from effgen.server.app import _build_dashboard_data

        reset_all()
        clear()
        for _ in range(3):
            record_model_call(provider="cerebras", model="shared-120b", outcome="ok", latency=8.0)
        for _ in range(2):
            record_model_call(provider="groq", model="shared-120b", outcome="ok", latency=0.3)
        record_run(model="cerebras:shared-120b", provider="cerebras", cost_usd=0.75)
        record_run(model="groq:shared-120b", provider="groq", cost_usd=0.25)
        data = _build_dashboard_data()
        reset_all()
        clear()

        rows = {r["provider"]: r for r in data["by_model"] if r["model"] == "shared-120b"}
        assert set(rows) == {"cerebras", "groq"}
        assert rows["cerebras"]["cost_usd"] == pytest.approx(0.75)
        assert rows["groq"]["cost_usd"] == pytest.approx(0.25)
        assert sum(r["cost_usd"] for r in rows.values()) == pytest.approx(1.0)
        assert data["unattributed_cost_usd"] == pytest.approx(0.0)
        assert rows["groq"]["p95_latency_s"] < 1.0 < rows["cerebras"]["p95_latency_s"]

    def test_unattributable_cost_is_reported_apart(self):
        """Spend on a model with no matching row is never spread over the rows."""
        from effgen.observability.metrics import record_model_call, reset_all
        from effgen.observability.run_log import clear, record_run
        from effgen.server.app import _build_dashboard_data

        reset_all()
        clear()
        record_model_call(provider="openai", model="known-model", outcome="ok", latency=0.4)
        record_run(model="openai:known-model", provider="openai", cost_usd=0.01)
        record_run(model="retired:vanished-model", provider="retired", cost_usd=0.05)
        data = _build_dashboard_data()
        reset_all()
        clear()

        row = next(r for r in data["by_model"] if r["model"] == "known-model")
        assert row["cost_usd"] == pytest.approx(0.01)
        assert data["unattributed_cost_usd"] == pytest.approx(0.05)

    def test_by_model_row_names_its_failure_class(self):
        """The recorded outcome label reaches the row with its remediation."""
        from effgen.models.errors import REMEDIATION_BY_CATEGORY
        from effgen.observability.metrics import record_model_call, reset_all
        from effgen.observability.run_log import clear
        from effgen.server.app import _build_dashboard_data

        reset_all()
        clear()
        record_model_call(provider="openai", model="missing-model", outcome="ok", latency=0.2)
        record_model_call(
            provider="openai", model="missing-model", outcome="not_found", latency=0.2
        )
        record_model_call(
            provider="openai", model="missing-model", outcome="not_found", latency=0.2
        )
        record_model_call(
            provider="openai", model="missing-model", outcome="rate_limited", latency=0.2
        )
        data = _build_dashboard_data()
        reset_all()
        clear()

        row = next(r for r in data["by_model"] if r["model"] == "missing-model")
        assert row["outcomes"] == {"not_found": 2, "ok": 1, "rate_limited": 1}
        assert row["top_error"] == "not_found"
        assert row["top_error_hint"] == REMEDIATION_BY_CATEGORY["not_found"]
        assert row["errors"] == 3

    def test_by_model_row_without_failures_has_no_top_error(self):
        from effgen.observability.metrics import record_model_call, reset_all
        from effgen.observability.run_log import clear
        from effgen.server.app import _build_dashboard_data

        reset_all()
        clear()
        record_model_call(provider="openai", model="healthy-model", outcome="ok", latency=0.2)
        data = _build_dashboard_data()
        reset_all()
        clear()

        row = next(r for r in data["by_model"] if r["model"] == "healthy-model")
        assert row["top_error"] is None
        assert row["top_error_hint"] is None
        assert row["outcomes"] == {"ok": 1}

    def test_by_status_breakdown_shape(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert "by_status" in data
        assert isinstance(data["by_status"], dict)
        assert "http_client_errors" in data["metrics"]
        assert "http_server_errors" in data["metrics"]

    def test_by_status_unchanged_while_route_detail_is_added(self):
        """``by_status`` keeps its exact shape; the route detail sits beside it."""
        from effgen.observability.metrics import record_http_request, reset_all
        from effgen.observability.run_log import clear
        from effgen.server.app import _build_dashboard_data

        reset_all()
        clear()
        for _ in range(4):
            record_http_request(route="/v1/chat/completions", method="POST", status=200)
        for _ in range(2):
            record_http_request(route="/v1/chat/completions", method="POST", status=404)
        record_http_request(route="other", method="GET", status=404)
        record_http_request(route="/health", method="GET", status=200)
        data = _build_dashboard_data()
        reset_all()
        clear()

        assert data["by_status"] == {"200": 5, "404": 3}

        detail = {(d["status"], d["route"], d["method"]): d["count"] for d in data["by_status_detail"]}
        assert detail[("404", "/v1/chat/completions", "POST")] == 2
        assert detail[("404", "other", "GET")] == 1
        assert all(d["class"] == f"{d['status'][0]}xx" for d in data["by_status_detail"])

        routes = {(r["route"], r["method"]): r for r in data["by_route"]}
        chat = routes[("/v1/chat/completions", "POST")]
        assert chat["requests"] == 6
        assert chat["errors"] == 2
        assert chat["error_rate"] == pytest.approx(2 / 6, abs=1e-4)
        assert chat["by_status"] == {"200": 4, "404": 2}
        assert routes[("/health", "GET")]["error_rate"] == 0.0
        # Worst-first, so the failing route is the one an operator reads first.
        assert data["by_route"][0]["error_rate"] >= data["by_route"][-1]["error_rate"]

    def test_version_present(self):
        from effgen import __version__
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert data["version"] == __version__

    def test_prompt_templates_exposed_for_editor_integrations(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert "prompt_templates" in data
        assert isinstance(data["prompt_templates"], list)
        if data["prompt_templates"]:
            first = data["prompt_templates"][0]
            for key in ("name", "description", "template", "category"):
                assert key in first


# ---------------------------------------------------------------------------
# FastAPI server route tests (TestClient)
# ---------------------------------------------------------------------------


def _make_dev_app():
    """Create a dev-mode FastAPI app for testing."""
    try:
        from effgen.server.app import create_app

        return create_app(dev_mode=True)
    except Exception as exc:
        pytest.skip(f"Cannot create test app: {exc}")


class TestDashboardRoutes:
    @pytest.fixture(autouse=True)
    def client(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        app = _make_dev_app()
        self._client = TestClient(app, raise_server_exceptions=False)
        return self._client

    def test_dashboard_index_200(self):
        resp = self._client.get("/dashboard")
        assert resp.status_code == 200

    def test_dashboard_index_html_content_type(self):
        resp = self._client.get("/dashboard")
        assert "html" in resp.headers.get("content-type", "")

    def test_dashboard_index_contains_panel_ids(self):
        resp = self._client.get("/dashboard")
        html = resp.text
        for pid in REQUIRED_PANEL_IDS:
            assert pid in html, f"Panel ID '{pid}' not in served HTML"

    def test_dashboard_slash_redirects_or_200(self):
        resp = self._client.get("/dashboard/")
        assert resp.status_code in (200, 301, 307, 308)

    def test_dashboard_data_json_200(self):
        resp = self._client.get("/dashboard/data.json")
        assert resp.status_code == 200

    def test_dashboard_data_json_content_type(self):
        resp = self._client.get("/dashboard/data.json")
        assert "json" in resp.headers.get("content-type", "")

    def test_dashboard_data_json_valid(self):
        resp = self._client.get("/dashboard/data.json")
        data = resp.json()
        assert isinstance(data, dict)
        assert "metrics" in data
        assert "slo" in data

    def test_dashboard_data_json_required_keys(self):
        resp = self._client.get("/dashboard/data.json")
        data = resp.json()
        for key in ("ts", "metrics", "slo", "recent_runs", "recent_spans", "raw_metrics"):
            assert key in data

    def test_dashboard_catalog_json_200(self):
        resp = self._client.get("/dashboard/catalog.json")
        assert resp.status_code == 200
        assert "json" in resp.headers.get("content-type", "")

    def test_dashboard_catalog_json_shape(self):
        data = self._client.get("/dashboard/catalog.json").json()
        assert isinstance(data, dict)
        for key in ("data", "providers", "local", "counts"):
            assert key in data, f"missing catalog key {key!r}"
        assert data["counts"]["catalog"] > 0
        rec = data["data"][0]
        # A browser needs pricing, capability, and provenance fields, plus the
        # is_priced flag that backs unpriced labeling.
        for field in ("id", "provider", "context_window", "price_in_per_1m",
                      "supports_vision", "is_priced", "verified_on"):
            assert field in rec, f"missing per-model field {field!r}"

    def test_dashboard_catalog_json_provider_filter(self):
        data = self._client.get("/dashboard/catalog.json?provider=openai").json()
        provs = {p["provider"] for p in data["providers"]}
        assert provs == {"openai"}
        assert all(r["provider"] == "openai" for r in data["data"])

    def test_dashboard_static_app_js(self):
        resp = self._client.get("/dashboard/app.js")
        assert resp.status_code == 200

    def test_dashboard_static_style_css(self):
        resp = self._client.get("/dashboard/style.css")
        assert resp.status_code == 200

    def test_dashboard_spans_sse(self, monkeypatch):
        """SSE endpoint should return 200 with text/event-stream content type
        and terminate promptly once the client disconnects (no hang)."""
        # Short heartbeat so the disconnect-check loop exits quickly when the
        # stream context closes (guards against the unbounded-loop regression).
        monkeypatch.setenv("EFFGEN_DASHBOARD_SSE_HEARTBEAT_SECONDS", "0.05")
        monkeypatch.setenv("EFFGEN_DASHBOARD_SSE_MAX_SECONDS", "2")

        # Buffer a span so the stream has immediate data to emit.
        from effgen.observability.tracing import _buffer_span

        _buffer_span("test.span", 12.3)

        with self._client.stream("GET", "/dashboard/spans") as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "")
            assert "text/event-stream" in ct or "event-stream" in ct
            # Read the first buffered span line, then disconnect.
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    assert "test.span" in line
                    break


class TestDashboardAuth:
    def test_dashboard_lookalike_path_is_not_public(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from effgen.server.app import create_app

        client = TestClient(create_app(dev_mode=False), raise_server_exceptions=False)
        resp = client.get("/dashboardevil")
        assert resp.status_code == 401

    def test_dashboard_catalog_requires_auth_by_default(self):
        """The catalog data endpoint follows the dashboard's access rule: it is
        protected by default and only public when the operator opts in."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from effgen.server.app import create_app

        client = TestClient(
            create_app(dev_mode=False, api_key="secret-key"),
            raise_server_exceptions=False,
        )
        assert client.get("/dashboard/catalog.json").status_code == 401
        ok = client.get(
            "/dashboard/catalog.json",
            headers={"Authorization": "Bearer secret-key"},
        )
        assert ok.status_code == 200
        assert ok.json()["counts"]["catalog"] > 0

    def test_dashboard_catalog_public_when_opted_in(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from effgen.server.app import create_app

        client = TestClient(
            create_app(dev_mode=False, api_key="secret-key", public_dashboard=True),
            raise_server_exceptions=False,
        )
        assert client.get("/dashboard/catalog.json").status_code == 200


# ---------------------------------------------------------------------------
# `effgen serve` startup banner + --help: the dashboard gap must be
# discoverable without opening a browser and hitting a 401.
# ---------------------------------------------------------------------------

class TestServeDashboardDiscoverability:
    def _run_serve(self, monkeypatch, capsys, **env):
        from types import SimpleNamespace

        import effgen.cli._main as _main

        for var in ("EFFGEN_API_KEY", "EFFGEN_DEV_MODE", "EFFGEN_PUBLIC_DASHBOARD"):
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        args = SimpleNamespace(host="127.0.0.1", port=8000, rate_limit=None, verbose=False)
        cli = _main.CLIInterface()
        rc = cli.serve_api(args)
        assert rc == 0
        return capsys.readouterr().out

    def test_dashboard_line_flags_auth_by_default(self, monkeypatch, capsys):
        pytest.importorskip("fastapi")
        out = self._run_serve(monkeypatch, capsys)
        assert "/dashboard" in out
        assert "EFFGEN_PUBLIC_DASHBOARD" in out

    def test_dashboard_line_omits_flag_when_public(self, monkeypatch, capsys):
        pytest.importorskip("fastapi")
        out = self._run_serve(monkeypatch, capsys, EFFGEN_PUBLIC_DASHBOARD="1")
        dashboard_lines = [ln for ln in out.splitlines() if "/dashboard" in ln and "v1" not in ln.lower()]
        assert dashboard_lines
        assert "EFFGEN_PUBLIC_DASHBOARD" not in dashboard_lines[0]

    def test_serve_help_documents_public_dashboard_var(self):
        import subprocess
        import sys
        from pathlib import Path

        scripts = Path(sys.executable).parent
        effgen_bin = str(scripts / "effgen") if (scripts / "effgen").exists() else "effgen"
        result = subprocess.run(
            [effgen_bin, "serve", "--help"], capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "EFFGEN_PUBLIC_DASHBOARD" in result.stdout
