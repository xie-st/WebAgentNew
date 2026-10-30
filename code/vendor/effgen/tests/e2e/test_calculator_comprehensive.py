#!/usr/bin/env python3
"""
Calculator Agent Comprehensive Test Script.
Tests T1 through T7 on a single model (specified via --model / CUDA_VISIBLE_DEVICES).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from effgen import load_model
from effgen.presets import create_agent

# Enable detailed logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)


def run_test(agent, test_id, question, expected_answer=None, expect_tool=None, description=""):
    """Run a single test and return results dict."""
    print(f"\n{'='*60}")
    print(f"TEST {test_id}: {description}")
    print(f"Q: {question}")
    print(f"Expected: {expected_answer}")
    if expect_tool:
        print(f"Expected tool: {expect_tool}")
    print("-" * 60)

    t0 = time.time()
    try:
        resp = agent.run(question)
        dt = time.time() - t0
    except Exception as e:
        dt = time.time() - t0
        print(f"CRASH: {e}")
        return {
            "test_id": test_id,
            "description": description,
            "status": "CRASH",
            "error": str(e),
            "time": dt,
        }

    output = resp.output if resp.output else ""
    print(f"A: {output[:500]}")
    print(f"Success: {resp.success}, Iterations: {resp.iterations}, Tool calls: {resp.tool_calls}")
    print(f"Time: {dt:.1f}s")

    # Check result
    status = "UNKNOWN"
    notes = ""

    if not resp.success:
        status = "FAIL"
        notes = f"Agent reported failure: {output[:200]}"
    elif expected_answer is not None:
        # Check if expected answer appears in output
        if str(expected_answer).lower() in output.lower():
            status = "PASS"
        else:
            # Try numeric comparison
            try:
                # Extract numbers from output
                import re
                numbers = re.findall(r'[\d,]+\.?\d*', output.replace(",", ""))
                if any(abs(float(n) - float(expected_answer)) < 0.01 for n in numbers):
                    status = "PASS"
                else:
                    status = "PARTIAL"
                    notes = f"Expected '{expected_answer}' not found in output"
            except (ValueError, TypeError):
                status = "PARTIAL"
                notes = f"Expected '{expected_answer}' not found in output"

    if expect_tool:
        if resp.tool_calls > 0:
            notes += f" | Tool calls: {resp.tool_calls}"
        else:
            notes += " | WARNING: No tool calls made"
            if status == "PASS":
                status = "PARTIAL"  # Got right answer but didn't use tool

    print(f"STATUS: {status}")
    if notes:
        print(f"NOTES: {notes}")

    return {
        "test_id": test_id,
        "description": description,
        "status": status,
        "output": output[:500],
        "expected": str(expected_answer),
        "tool_calls": resp.tool_calls,
        "iterations": resp.iterations,
        "time": dt,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculator Agent Comprehensive Tests")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tests", default="all", help="Comma-separated test IDs or 'all'")
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print("Calculator Agent Comprehensive Tests")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Create math agent
    agent = create_agent("math", model)
    print(f"Math agent created with tools: {list(agent.tools.keys())}")

    results = []

    # P2-T1: Simple arithmetic
    results.append(run_test(
        agent, "P2-T1",
        "What is 247 * 83?",
        expected_answer="20501",
        expect_tool="calculator",
        description="Simple arithmetic (247 * 83 = 20501)"
    ))

    # P2-T2: Math function
    results.append(run_test(
        agent, "P2-T2",
        "What is the square root of 144?",
        expected_answer="12",
        expect_tool="calculator",
        description="Math function (sqrt(144) = 12)"
    ))

    # P2-T3: Multi-step expression
    results.append(run_test(
        agent, "P2-T3",
        "What is (15 + 27) * 3 - 10?",
        expected_answer="116",
        expect_tool="calculator",
        description="Multi-step expression ((15+27)*3-10 = 116)"
    ))

    # P2-T4: Tool selection (Calculator vs PythonREPL)
    results.append(run_test(
        agent, "P2-T4",
        "Add 5 and 3, then multiply the result by 2.",
        expected_answer="16",
        expect_tool="calculator",
        description="Tool selection — should use Calculator or PythonREPL"
    ))

    # P2-T5: Non-JSON tool input resilience
    results.append(run_test(
        agent, "P2-T5",
        "Use the calculator to compute 2+2.",
        expected_answer="4",
        expect_tool="calculator",
        description="Non-JSON tool input test"
    ))

    # P2-T6: Wrong tool name handling
    results.append(run_test(
        agent, "P2-T6",
        "Use your math_solver tool to compute 10 + 5.",
        expected_answer="15",
        description="Wrong tool name — model may hallucinate 'math_solver'"
    ))

    # P2-T7: Direct answer without tools
    results.append(run_test(
        agent, "P2-T7",
        "What is 2+2?",
        expected_answer="4",
        description="Direct answer — trivial math, may skip tools"
    ))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model}")
    print(f"{'='*60}")
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    partial_count = sum(1 for r in results if r["status"] == "PARTIAL")
    fail_count = sum(1 for r in results if r["status"] in ("FAIL", "CRASH"))
    total = len(results)

    for r in results:
        tool_info = f" (tools={r.get('tool_calls', '?')})" if r.get("tool_calls") else ""
        print(f"  {r['test_id']}: {r['status']:8s} — {r['description']}{tool_info}")
        if r.get("notes"):
            print(f"           {r['notes']}")

    print(f"\nTotal: {pass_count} PASS, {partial_count} PARTIAL, {fail_count} FAIL out of {total}")

    # Save results
    results_file = f"/tmp/p2_results_{args.model.replace('/', '_')}.json"
    with open(results_file, "w") as f:
        json.dump({"model": args.model, "gpu": gpu, "results": results}, f, indent=2)
    print(f"Results saved to {results_file}")

    # Cleanup
    model.unload()
    print("Done.")


if __name__ == "__main__":
    # Only needed when running this file as a standalone script (not under
    # pytest). Done here — never at import time — so test collection cannot
    # leave tests/ on sys.path, which would shadow the repo-root ``deploy``
    # namespace package with tests/deploy/ and break the lambda import tests.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
