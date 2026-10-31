"""
Domain tools — real (non-mock) tests.

All tests exercise the actual tool code paths end-to-end:
- Finance / Knowledge: hit real free public APIs (frankfurter.app,
  CoinGecko, Yahoo Finance v8, arXiv Atom feed, Stack Exchange,
  GitHub). Skipped automatically if the network is unreachable.
- Data Science: run against real pandas / matplotlib / numpy / scipy.
- DevOps: run real git / subprocess / psutil calls.
- Communication: exercise real formatting logic (drafts are offline).

No mocking, no monkeypatching. If an optional dependency is missing,
the test is skipped with a clear reason.
"""

from __future__ import annotations

import asyncio
import csv
import os
import socket

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import (
    ArxivTool,
    CryptoTool,
    CurrencyConverterTool,
    DataFrameTool,
    DockerTool,
    EmailDraftTool,
    GitHubTool,
    GitTool,
    HTTPTool,
    NotificationTool,
    PlotTool,
    SlackDraftTool,
    StackOverflowTool,
    StatsTool,
    StockPriceTool,
    SystemInfoTool,
    WolframAlphaTool,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reachable(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    """Bounded TCP reachability check for ``host``.

    Runs the connect on a daemon thread joined with a wall-clock timeout so a
    slow/dead host — or a hanging DNS lookup, which a plain socket timeout does
    *not* bound — can never stall the caller past ``timeout``. An abandoned
    daemon thread won't block interpreter exit.
    """
    import threading

    result = {"ok": False}

    def _connect() -> None:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                result["ok"] = True
        except OSError:
            pass

    t = threading.Thread(target=_connect, daemon=True)
    t.start()
    t.join(timeout + 1.0)
    return result["ok"]


def _has_network(host: str = "api.github.com", port: int = 443, timeout: float = 3.0) -> bool:
    return _reachable(host, port, timeout)


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def _ok(result: ToolResult):
    assert isinstance(result, ToolResult)
    assert result.success, f"tool failed: {result.error}"
    return result.output


def _ok_from_result(result: ToolResult):
    """Like _ok but takes a pre-asserted ToolResult (caller already checked success)."""
    assert isinstance(result, ToolResult)
    assert result.success, f"tool failed: {result.error}"
    return result.output


# ---------------------------------------------------------------------------
# 10.1 Finance
# ---------------------------------------------------------------------------

@needs_net
def test_currency_converter_usd_to_eur():
    result = _run(CurrencyConverterTool().execute(
        amount=100.0, from_currency="USD", to_currency="EUR",
    ))
    if not result.success:
        err = str(result.error or "")
        # frankfurter.app occasionally returns transient upstream errors (Cloudflare
        # 5xx like HTTP 521, 429 rate-limits, timeouts). Treat those as a skip, not a
        # failure — same pattern as the crypto test above.
        if "HTTP" in err or "429" in err or "timed out" in err.lower():
            pytest.skip(f"frankfurter.app transient error: {err[:100]}")
    out = _ok(result)
    assert out["from"] == "USD"
    assert out["to"] == "EUR"
    assert isinstance(out["converted"], float) and out["converted"] > 0
    assert 0.5 < out["rate"] < 2.0  # sanity band
    assert "disclaimer" in out
    assert out["source"] == "frankfurter.app"


def test_currency_converter_identity():
    # Same currency — no network required, runs identity fast-path
    out = _ok(_run(CurrencyConverterTool().execute(
        amount=42.0, from_currency="JPY", to_currency="JPY",
    )))
    assert out["rate"] == 1.0
    assert out["converted"] == 42.0
    assert out["source"] == "identity"


def test_currency_converter_invalid_currency_fails():
    r = _run(CurrencyConverterTool().execute(
        amount=10.0, from_currency="USD", to_currency="ZZZ",
    ))
    assert not r.success
    assert r.error is not None


@needs_net
def test_crypto_bitcoin_usd():
    result = _run(CryptoTool().execute(coin="bitcoin", vs_currency="usd"))
    if not result.success:
        err = str(result.error or "")
        # CoinGecko's free tier rate-limits aggressively. Treat transient HTTP
        # errors as a skip, not a failure — same pattern as the unknown-coin
        # test below.
        if "429" in err or "HTTP" in err or "timed out" in err.lower():
            pytest.skip(f"coingecko transient error: {err[:100]}")
    out = _ok(result)
    assert out["coin"] == "bitcoin"
    assert out["vs_currency"] == "usd"
    assert isinstance(out["price"], float) and out["price"] > 0
    assert out["source"] == "coingecko"


def test_crypto_unknown_coin_fails():
    if not NETWORK:
        pytest.skip("no network")
    r = _run(CryptoTool().execute(coin="not-a-real-coin-xyz", vs_currency="usd"))
    assert not r.success
    err = r.error or ""
    # Accept either the semantic "Unknown coin" error or a transient
    # upstream rate-limit / HTTP failure from CoinGecko's free tier.
    assert ("Unknown coin" in err) or ("HTTP" in err) or ("429" in err)


@needs_net
def test_stock_price_aapl():
    out = _ok(_run(StockPriceTool().execute(symbol="AAPL")))
    assert out["symbol"] == "AAPL"
    assert out["price"] is None or out["price"] > 0
    assert out["source"] in ("yfinance", "yahoo_api")
    assert "disclaimer" in out


def test_stock_price_missing_symbol_parameter():
    r = _run(StockPriceTool().execute())  # missing required
    assert not r.success
    assert "symbol" in (r.error or "")


# ---------------------------------------------------------------------------
# 10.2 Data science
# ---------------------------------------------------------------------------

def test_stats_summary_numpy_required():
    pytest.importorskip("numpy")
    out = _ok(_run(StatsTool().execute(operation="summary", data=[1, 2, 3, 4, 5])))
    assert out["n"] == 5
    assert out["mean"] == 3.0
    assert out["median"] == 3.0
    assert out["min"] == 1.0
    assert out["max"] == 5.0
    assert abs(out["std"] - 1.5811388300841898) < 1e-9


def test_stats_correlation_and_regression():
    pytest.importorskip("numpy")
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # perfectly linear y = 2x
    corr = _ok(_run(StatsTool().execute(operation="correlation", data=x, y=y)))
    assert abs(corr["result"] - 1.0) < 1e-9

    reg = _ok(_run(StatsTool().execute(operation="regression", data=x, y=y)))
    assert abs(reg["slope"] - 2.0) < 1e-9
    assert abs(reg["intercept"]) < 1e-9


def test_stats_basic_ops():
    pytest.importorskip("numpy")
    d = [10, 20, 30]
    assert _ok(_run(StatsTool().execute(operation="mean", data=d)))["result"] == 20.0
    assert _ok(_run(StatsTool().execute(operation="median", data=d)))["result"] == 20.0
    assert _ok(_run(StatsTool().execute(operation="min", data=d)))["result"] == 10.0
    assert _ok(_run(StatsTool().execute(operation="max", data=d)))["result"] == 30.0


def test_stats_empty_data_fails():
    pytest.importorskip("numpy")
    r = _run(StatsTool().execute(operation="mean", data=[]))
    assert not r.success


def test_stats_mismatched_lengths_fails():
    pytest.importorskip("numpy")
    r = _run(StatsTool().execute(operation="correlation", data=[1, 2, 3], y=[1, 2]))
    assert not r.success


def test_dataframe_csv_head_filter_aggregate(tmp_path):
    pytest.importorskip("pandas")
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "city", "age", "salary"])
        w.writerow(["alice", "NYC", 30, 100000])
        w.writerow(["bob", "NYC", 25, 80000])
        w.writerow(["carol", "SF", 35, 120000])
        w.writerow(["dave", "SF", 40, 150000])

    # Exercise the strict allow-list mode: confine reads to the temp dir, as a
    # caller hardening the tool to a single root would.
    tool = DataFrameTool(allowed_directories=[str(tmp_path)])

    head = _ok(_run(tool.execute(file_path=str(csv_path), operation="head", n=2)))
    assert head["rows"] == 4
    assert len(head["data"]) == 2
    assert head["columns"] == ["name", "city", "age", "salary"]

    desc = _ok(_run(tool.execute(file_path=str(csv_path), operation="describe")))
    assert "data" in desc

    info = _ok(_run(tool.execute(file_path=str(csv_path), operation="info")))
    assert info["data"]["non_null"]["age"] == 4

    flt = _ok(_run(tool.execute(
        file_path=str(csv_path), operation="filter",
        column="city", op="==", value="NYC",
    )))
    assert flt["matched_rows"] == 2

    agg = _ok(_run(tool.execute(
        file_path=str(csv_path), operation="aggregate",
        agg="mean", group_by="city",
    )))
    # Grouped-by-city mean salaries
    assert agg["data"]["salary"]["NYC"] == 90000.0
    assert agg["data"]["salary"]["SF"] == 135000.0


def test_dataframe_missing_file_fails(tmp_path):
    pytest.importorskip("pandas")
    r = _run(DataFrameTool(allowed_directories=[str(tmp_path)]).execute(
        file_path=str(tmp_path / "nope.csv"), operation="head",
    ))
    assert not r.success
    assert "not found" in (r.error or "").lower() or "no such" in (r.error or "").lower()


def test_dataframe_filter_requires_column_and_op(tmp_path):
    pytest.importorskip("pandas")
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("a,b\n1,2\n")
    r = _run(DataFrameTool(allowed_directories=[str(tmp_path)]).execute(
        file_path=str(csv_path), operation="filter"))
    assert not r.success


def test_plot_line_creates_png(tmp_path):
    pytest.importorskip("matplotlib")
    out_path = tmp_path / "chart.png"
    out = _ok(_run(PlotTool().execute(
        chart_type="line",
        x=[1, 2, 3, 4],
        y=[10, 20, 15, 25],
        title="test",
        xlabel="x",
        ylabel="y",
        output_path=str(out_path),
    )))
    assert out["file_path"] == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 100  # non-trivial PNG
    # PNG magic bytes
    assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_histogram_to_tempfile():
    pytest.importorskip("matplotlib")
    out = _ok(_run(PlotTool().execute(
        chart_type="hist", y=list(range(100)),
    )))
    assert os.path.exists(out["file_path"])
    assert out["file_path"].endswith(".png")
    os.unlink(out["file_path"])


def test_plot_scatter_requires_x():
    pytest.importorskip("matplotlib")
    r = _run(PlotTool().execute(chart_type="scatter", y=[1, 2, 3]))
    assert not r.success


# ---------------------------------------------------------------------------
# 10.3 DevOps
# ---------------------------------------------------------------------------

def test_git_status_on_repo():
    out = _ok(_run(GitTool().execute(operation="status", cwd=".")))
    assert out["returncode"] == 0
    assert "command" in out
    assert out["command"].startswith("git status")


def test_git_log_returns_commits():
    out = _ok(_run(GitTool().execute(operation="log", n=3, cwd=".")))
    assert out["returncode"] == 0
    # Repo has at least 3 commits (per git history)
    lines = [ln for ln in out["stdout"].splitlines() if ln.strip()]
    assert len(lines) >= 1


def test_git_branch():
    out = _ok(_run(GitTool().execute(operation="branch", cwd=".")))
    assert out["returncode"] == 0


def test_git_rejects_disallowed_operation():
    r = _run(GitTool().execute(operation="push", cwd="."))
    # validate_parameters will reject via enum first
    assert not r.success


def test_docker_version_or_skip():
    import shutil
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    out = _ok(_run(DockerTool().execute(operation="version")))
    # docker may not be running; returncode !=0 is OK, we only verify the
    # tool executed the subprocess without raising.
    assert "returncode" in out


def test_docker_logs_requires_container():
    import shutil
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    r = _run(DockerTool().execute(operation="logs"))
    assert not r.success


def test_system_info_all():
    pytest.importorskip("psutil")
    out = _ok(_run(SystemInfoTool().execute(kind="all")))
    for key in ("cpu", "memory", "disk", "network"):
        assert key in out
    assert out["memory"]["total"] > 0
    assert out["cpu"]["count_logical"] >= 1


def test_system_info_memory_only():
    pytest.importorskip("psutil")
    out = _ok(_run(SystemInfoTool().execute(kind="memory")))
    assert "memory" in out
    assert "cpu" not in out


@needs_net
def test_http_get_json_response():
    # httpbin.org is frequently overloaded/down; a stalled connect or DNS lookup
    # would otherwise hang past the suite timeout. Bounded pre-check + a short
    # request timeout keep a flaky host a clean skip, not a CI failure.
    if not _reachable("httpbin.org"):
        pytest.skip("httpbin.org unreachable")
    result = _run(HTTPTool().execute(
        url="https://httpbin.org/get",
        method="GET",
        headers={"User-Agent": "effGen-tests"},
        timeout=10,
    ))
    if not result.success:
        pytest.skip(f"httpbin.org unreachable: {result.error}")
    out = result.output
    if out.get("status", 0) >= 500:
        pytest.skip(f"httpbin.org returned upstream {out['status']}")
    assert out["status"] == 200
    assert out["json"] is not None
    assert "headers" in out["json"]


def test_http_invalid_scheme_rejected():
    r = _run(HTTPTool().execute(url="ftp://example.com"))
    assert not r.success


# ---------------------------------------------------------------------------
# 10.4 Knowledge
# ---------------------------------------------------------------------------

@needs_net
def test_arxiv_search_returns_papers():
    result = _run(ArxivTool().execute(query="transformer", max_results=3))
    if not result.success:
        err = str(result.error or "")
        # arXiv's export API is flaky under load: 429s, read timeouts,
        # connection resets all happen in healthy networks. Treat any
        # transient network error as a skip.
        if any(
            needle in err.lower()
            for needle in ("429", "timed out", "timeout", "connection", "http")
        ):
            pytest.skip(f"arxiv transient error: {err[:100]}")
    out = _ok_from_result(result)
    assert out["count"] >= 1
    first = out["results"][0]
    assert "title" in first and len(first["title"]) > 0
    assert "url" in first and first["url"].startswith("http")
    assert isinstance(first["authors"], list)


@needs_net
def test_stackoverflow_search():
    # The Stack Exchange API throttles anonymous callers by IP, and its
    # full-text search answers an anonymous request from a shared address with
    # an empty item list rather than an error (measured 2026-08-03: every query
    # shape — q=, intitle=, tagged= — returned `{"items": []}` with quota to
    # spare). Neither condition is something this tree can correct, so the
    # response shape is what is asserted; a throttled or empty answer skips.
    result = _run(StackOverflowTool().execute(query="python list comprehension", max_results=3))
    if not result.success:
        err = (result.error or "").lower()
        if any(word in err for word in ("throttle", "rate limit", "429", "400", "timed out")):
            pytest.skip(f"Stack Exchange API unavailable to this caller: {result.error}")
        pytest.fail(f"StackOverflowTool failed unexpectedly: {result.error}")
    out = result.output
    assert out["query"] == "python list comprehension"
    assert out["count"] == len(out["results"])
    if not out["results"]:
        pytest.skip("Stack Exchange search returned no items for an anonymous caller")
    first = out["results"][0]
    assert "title" in first
    assert first["link"].startswith("https://stackoverflow.com")


@needs_net
def test_github_search_repositories():
    # Unauthenticated GitHub search has a 60 req/hour/IP cap — CI matrix legs
    # share an Actions runner pool and routinely trip it. Use a stable query
    # and assert only on the response shape; skip cleanly on rate-limit or
    # empty result so a transient API condition doesn't fail the suite.
    result = _run(GitHubTool().execute(query="pytest", kind="repositories", max_results=3))
    if not result.success:
        err = (result.error or "").lower()
        if "rate limit" in err or "403" in err or "429" in err or "abuse" in err:
            pytest.skip(f"GitHub API rate-limited: {result.error}")
        pytest.fail(f"GitHubTool failed unexpectedly: {result.error}")
    out = result.output
    assert "results" in out
    assert out["total_count"] >= 0
    if not out["results"]:
        pytest.skip("GitHub search returned no results (likely rate-limited or transient)")
    first = out["results"][0]
    # Each result must carry the basic schema the tool promises.
    assert "full_name" in first
    assert isinstance(first.get("full_name"), str) and "/" in first["full_name"]


def test_wolfram_alpha_without_key_fails():
    # Ensure no key leaked from env
    os.environ.pop("WOLFRAM_ALPHA_APPID", None)
    tool = WolframAlphaTool()
    r = _run(tool.execute(query="2+2"))
    assert not r.success
    assert "api key" in (r.error or "").lower()


# ---------------------------------------------------------------------------
# 10.5 Communication
# ---------------------------------------------------------------------------

def test_email_draft_basic():
    out = _ok(_run(EmailDraftTool().execute(
        to=["alice@example.com", "bob@example.com"],
        subject="Hi",
        body="Hello there",
        cc=["carol@example.com"],
        from_address="me@example.com",
    )))
    assert out["sent"] is False
    assert "DRAFT ONLY" in out["notice"]
    draft = out["draft"]
    assert "From: me@example.com" in draft
    assert "To: alice@example.com, bob@example.com" in draft
    assert "Cc: carol@example.com" in draft
    assert "Subject: Hi" in draft
    assert "Hello there" in draft


def test_email_draft_requires_recipients():
    r = _run(EmailDraftTool().execute(subject="x", body="y"))
    assert not r.success


def test_slack_draft_with_mentions_and_thread():
    out = _ok(_run(SlackDraftTool().execute(
        channel="#eng",
        text="deploy finished",
        mentions=["alice", "@bob"],
        thread_ts="1234.5678",
    )))
    assert out["sent"] is False
    assert "#eng" in out["draft"]
    assert "<@alice>" in out["draft"]
    assert "<@bob>" in out["draft"]
    assert "1234.5678" in out["draft"]


def test_slack_draft_plain():
    out = _ok(_run(SlackDraftTool().execute(channel="#general", text="hi")))
    assert out["draft"] == "[#general]: hi"
    assert out["sent"] is False


def test_notification_succeeds_without_plyer():
    # Whether plyer is installed or not, this tool must succeed and
    # report shown={True|False} without raising.
    out = _ok(_run(NotificationTool().execute(
        title="test", message="from pytest", timeout=1,
    )))
    assert out["title"] == "test"
    assert out["message"] == "from pytest"
    assert "shown" in out


# ---------------------------------------------------------------------------
# registry integration
# ---------------------------------------------------------------------------

def test_registry_auto_discovers_domain_tools():
    from effgen.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.discover_builtin_tools()
    names = set(reg.list_tools())
    expected = {
        "stock_price", "currency_converter", "crypto_price",
        "dataframe", "plot", "stats",
        "git", "docker", "system_info", "http",
        "arxiv", "stackoverflow", "github", "wolfram_alpha",
        "email_draft", "slack_draft", "notification",
    }
    missing = expected - names
    assert not missing, f"registry missing domain tools: {missing}"
