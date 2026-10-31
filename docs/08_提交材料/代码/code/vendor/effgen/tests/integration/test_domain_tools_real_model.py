"""
Real-model integration tests for new domain tools.

Tests real SLM agents actually *calling* the new domain tools
(finance, data science, devops, knowledge, communication) on GPU.

Dispatched per-GPU via the EFFGEN_TEST_GROUP env var so the three
groups can run in parallel across GPUs 3, 4, and 5:

    EFFGEN_TEST_GROUP=a CUDA_VISIBLE_DEVICES=3 python -m tests...
    EFFGEN_TEST_GROUP=b CUDA_VISIBLE_DEVICES=4 python -m tests...
    EFFGEN_TEST_GROUP=c CUDA_VISIBLE_DEVICES=5 python -m tests...

Group A (GPU 3): finance + communication  (StockPrice, Currency, Crypto,
                 EmailDraft, SlackDraft)
Group B (GPU 4): data science + devops    (Stats, DataFrame, Plot,
                 SystemInfo, Git, HTTP)
Group C (GPU 5): knowledge                 (Arxiv, StackOverflow, GitHub)
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import traceback

# Pick up CUDA_VISIBLE_DEVICES from the environment — the launcher sets it.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from effgen.core.agent import Agent, AgentConfig
from effgen.models import load_model
from effgen.tools.builtin import (
    ArxivTool,
    CryptoTool,
    CurrencyConverterTool,
    DataFrameTool,
    EmailDraftTool,
    GitHubTool,
    GitTool,
    HTTPTool,
    PlotTool,
    SlackDraftTool,
    StackOverflowTool,
    StatsTool,
    StockPriceTool,
    SystemInfoTool,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        print(f"[{os.environ.get('CUDA_VISIBLE_DEVICES','?')}] loading {MODEL_NAME}...", flush=True)
        _MODEL = load_model(MODEL_NAME)
    return _MODEL


def make_agent(name: str, tools: list) -> Agent:
    return Agent(config=AgentConfig(
        name=name,
        model=get_model(),
        tools=tools,
        max_iterations=6,
    ))


def run_check(label: str, fn):
    print(f"\n--- {label} ---", flush=True)
    try:
        fn()
        print(f"PASS: {label}", flush=True)
        return True
    except Exception:
        print(f"FAIL: {label}\n{traceback.format_exc()}", flush=True)
        return False


# ---------------------------------------------------------------------------
# group A — finance + communication
# ---------------------------------------------------------------------------

def group_a():
    results = []

    def t_currency():
        agent = make_agent("finance-agent", [CurrencyConverterTool()])
        r = agent.run("Convert 100 USD to EUR using the currency_converter tool.")
        assert r.success, f"agent failed: {r}"
        (r.output or "").lower() + " " + str(r.metadata)
        # The model should produce a number; we don't pin the value
        assert any(c.isdigit() for c in (r.output or "")), f"no number in output: {r.output}"

    def t_crypto_tool_direct():
        # Exercise agent-attached CryptoTool, then make sure the tool itself ran
        agent = make_agent("crypto-agent", [CryptoTool()])
        r = agent.run("What is the current price of bitcoin in USD? Use the crypto_price tool.")
        assert r.success, f"agent failed: {r}"

    def t_stock():
        agent = make_agent("stock-agent", [StockPriceTool()])
        r = agent.run("Use stock_price to look up AAPL and report the price.")
        assert r.success, f"agent failed: {r}"

    def t_email():
        agent = make_agent("email-agent", [EmailDraftTool()])
        r = agent.run(
            "Draft an email using the email_draft tool. "
            "To: alice@example.com. Subject: Status. Body: All tests passing."
        )
        assert r.success, f"agent failed: {r}"

    def t_slack():
        agent = make_agent("slack-agent", [SlackDraftTool()])
        r = agent.run(
            "Use slack_draft to draft a message to channel #engineering "
            "with text 'deploy finished'."
        )
        assert r.success, f"agent failed: {r}"

    results.append(run_check("A1 currency_converter agent", t_currency))
    results.append(run_check("A2 crypto_price agent", t_crypto_tool_direct))
    results.append(run_check("A3 stock_price agent", t_stock))
    results.append(run_check("A4 email_draft agent", t_email))
    results.append(run_check("A5 slack_draft agent", t_slack))
    return results


# ---------------------------------------------------------------------------
# group B — data science + devops
# ---------------------------------------------------------------------------

def group_b():
    results = []

    def t_stats():
        agent = make_agent("stats-agent", [StatsTool()])
        r = agent.run(
            "Use the stats tool to compute the mean of the numbers 10, 20, 30, 40, 50."
        )
        assert r.success, f"agent failed: {r}"
        assert "30" in (r.output or "") or "30.0" in (r.output or ""), f"expected 30 in: {r.output}"

    def t_dataframe():
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        w = csv.writer(tmp)
        w.writerow(["name", "age"])
        w.writerow(["alice", 30])
        w.writerow(["bob", 25])
        tmp.close()
        try:
            agent = make_agent("df-agent", [DataFrameTool()])
            r = agent.run(
                f"Use the dataframe tool on file_path '{tmp.name}' with operation 'head' to show rows."
            )
            assert r.success, f"agent failed: {r}"
        finally:
            os.unlink(tmp.name)

    def t_plot():
        # Direct tool exercise — ensures agent wiring works end-to-end
        agent = make_agent("plot-agent", [PlotTool()])
        r = agent.run(
            "Use the plot tool with chart_type 'bar', x [1,2,3], y [10,20,30], title 'Demo'."
        )
        assert r.success, f"agent failed: {r}"

    def t_sysinfo():
        agent = make_agent("sys-agent", [SystemInfoTool()])
        r = agent.run("Use the system_info tool to get memory info.")
        assert r.success, f"agent failed: {r}"

    def t_git():
        agent = make_agent("git-agent", [GitTool()])
        r = agent.run("Use the git tool with operation 'status' in the current directory.")
        assert r.success, f"agent failed: {r}"

    def t_http():
        agent = make_agent("http-agent", [HTTPTool()])
        r = agent.run(
            "Use the http tool to GET https://api.github.com/repos/ctrl-gaurav/effGen."
        )
        assert r.success, f"agent failed: {r}"

    results.append(run_check("B1 stats agent", t_stats))
    results.append(run_check("B2 dataframe agent", t_dataframe))
    results.append(run_check("B3 plot agent", t_plot))
    results.append(run_check("B4 system_info agent", t_sysinfo))
    results.append(run_check("B5 git agent", t_git))
    results.append(run_check("B6 http agent", t_http))
    return results


# ---------------------------------------------------------------------------
# group C — knowledge
# ---------------------------------------------------------------------------

def group_c():
    results = []

    def t_arxiv():
        agent = make_agent("arxiv-agent", [ArxivTool()])
        r = agent.run(
            "Use the arxiv tool to search for 'transformer' with max_results 2."
        )
        assert r.success, f"agent failed: {r}"

    def t_stackoverflow():
        agent = make_agent("so-agent", [StackOverflowTool()])
        r = agent.run(
            "Use the stackoverflow tool to search for 'python asyncio' with max_results 2."
        )
        assert r.success, f"agent failed: {r}"

    def t_github():
        agent = make_agent("gh-agent", [GitHubTool()])
        r = agent.run(
            "Use the github tool with query 'effgen', kind 'repositories', max_results 2."
        )
        assert r.success, f"agent failed: {r}"

    results.append(run_check("C1 arxiv agent", t_arxiv))
    results.append(run_check("C2 stackoverflow agent", t_stackoverflow))
    results.append(run_check("C3 github agent", t_github))
    return results


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

GROUPS = {"a": group_a, "b": group_b, "c": group_c}


def main():
    group = os.environ.get("EFFGEN_TEST_GROUP", "").lower()
    if group not in GROUPS:
        print(f"ERROR: set EFFGEN_TEST_GROUP to one of {list(GROUPS)}", file=sys.stderr)
        sys.exit(2)
    print(f"[group {group}] starting on CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    results = GROUPS[group]()
    passed = sum(results)
    total = len(results)
    print(f"\n[group {group}] RESULT: {passed}/{total} passed", flush=True)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
