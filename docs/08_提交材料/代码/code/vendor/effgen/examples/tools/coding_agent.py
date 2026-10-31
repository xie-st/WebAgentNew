#!/usr/bin/env python3
"""
effGen — Code Execution Agent (Write + Run + Iterate)

A coding agent that writes Python code, executes it, reads output, and iterates
on errors. Demonstrates PythonREPL and CodeExecutor tools, markdown fence stripping,
and multi-iteration reasoning.

Recommended models:
  - Qwen/Qwen2.5-7B-Instruct (7B)      — best quality, fastest
  - Qwen/Qwen2.5-3B-Instruct (3B)      — excellent quality (default)
  - Qwen/Qwen2.5-1.5B-Instruct (1.5B)  — surprisingly strong for code tasks
  - microsoft/Phi-4-mini-instruct       — very accurate

Tools used: CodeExecutor, PythonREPL, FileOperations, BashTool
Preset: coding

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/coding_agent.py
  CUDA_VISIBLE_DEVICES=0 python examples/coding_agent.py --model Qwen/Qwen2.5-7B-Instruct
  CUDA_VISIBLE_DEVICES=0 python examples/coding_agent.py --interactive
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.presets import create_agent

CODING_SYSTEM_PROMPT = """You are an expert Python coding agent. You write, execute, and debug code.

TOOL SELECTION RULES:
- For running Python code: use the 'python_repl' tool with {"code": "your code here"}
- For running code in other languages or with file I/O: use 'code_executor' with {"code": "...", "language": "python"}
- For reading/writing files: use 'file_operations'
- For shell commands: use 'bash'

IMPORTANT RULES:
1. When writing code, put ONLY the raw code in the Action Input — do NOT wrap it in markdown fences (no ```python).
2. Always include print() statements so you can see the output.
3. If you get an error, read the error message carefully, fix the code, and try again.
4. When done, provide the final answer with the output.

EXAMPLE:
Thought: I need to check if 17 is prime. I'll write a function and test it.
Action: python_repl
Action Input: {"code": "def is_prime(n):\\n    if n < 2:\\n        return False\\n    for i in range(2, int(n**0.5) + 1):\\n        if n % i == 0:\\n            return False\\n    return True\\n\\nresult = is_prime(17)\\nprint(f'Is 17 prime? {result}')"}
"""


def run_test(agent, test_id, description, question, expected_keywords=None,
             check_fn=None, expected_tool=None):
    """Run a single test and return result dict."""
    print(f"\n{'='*60}")
    print(f"Test: {test_id} — {description}")
    print(f"Q: {question}")

    t0 = time.time()
    resp = agent.run(question)
    dt = time.time() - t0

    output = resp.output if resp.output else ""
    print(f"A: {output[:600]}")
    print(f"Tool calls: {resp.tool_calls}, Iterations: {resp.iterations}, Time: {dt:.1f}s")

    # Check keywords in output
    keyword_pass = True
    if expected_keywords:
        for kw in expected_keywords:
            if kw.lower() not in output.lower():
                keyword_pass = False
                print(f"  MISSING keyword: '{kw}'")

    # Check custom function
    custom_pass = True
    if check_fn:
        custom_pass = check_fn(output, resp)
        if not custom_pass:
            print("  Custom check FAILED")

    # Check tool was used
    tool_pass = True
    if expected_tool:
        trace_tools = []
        for trace in resp.execution_trace:
            if isinstance(trace, dict):
                data = trace.get("data", {})
                if data.get("tool_name"):
                    trace_tools.append(data["tool_name"])
        if expected_tool not in trace_tools:
            tool_pass = False
            print(f"  Expected tool '{expected_tool}' not found in trace: {trace_tools}")
        else:
            print(f"  Tool used: {expected_tool}")

    passed = keyword_pass and custom_pass and tool_pass
    status = "PASS" if passed else "FAIL"
    print(f"Result: {status}")

    return {
        "test_id": test_id,
        "description": description,
        "passed": passed,
        "output": output[:400],
        "tool_calls": resp.tool_calls,
        "iterations": resp.iterations,
        "time": dt,
        "status": status,
    }


def run_all_tests(agent, model_name="unknown"):
    """Run all code execution tests."""
    results = []

    # T1: Simple code — write and run a prime checker
    results.append(run_test(
        agent, "T1", "Simple code execution — prime check",
        "Write and run a Python function that checks if a number is prime. Test it with 17 and print the result.",
        expected_keywords=["17", "prime"],
        expected_tool="python_repl",
        check_fn=lambda out, resp: resp.tool_calls >= 1 and ("true" in out.lower() or "yes" in out.lower() or "prime" in out.lower()),
    ))

    # T2: Code with error — division by zero
    results.append(run_test(
        agent, "T2", "Error handling — division by zero",
        "Write Python code that divides 10 by 0 and run it. Report what happens.",
        expected_keywords=["zero"],
        expected_tool="python_repl",
        check_fn=lambda out, resp: any(kw in out.lower() for kw in ["error", "exception", "zerodivision", "cannot", "undefined", "infinity"]),
    ))

    # T3: Iterative fix — sort a list
    results.append(run_test(
        agent, "T3", "Iterative fix — list sorting",
        "Write a Python function to sort a list of numbers [5, 2, 8, 1, 9, 3] in ascending order. Run it and print the sorted result.",
        expected_keywords=["1", "2", "3", "5", "8", "9"],
        expected_tool="python_repl",
    ))

    # T4: Markdown fence stripping test — explicitly use fenced code
    results.append(run_test(
        agent, "T4", "Markdown fence stripping",
        "Run this Python code using the python_repl tool: print('Markdown fence test passed!')",
        expected_keywords=["fence test passed"],
        expected_tool="python_repl",
    ))

    # T5: Multi-file — create module and run code that uses it
    # Use code_executor (subprocess) so we can import from /tmp
    results.append(run_test(
        agent, "T5", "Multi-file — create and import module",
        ("First use bash to write a file /tmp/effgen_codegen_utils.py with this exact content:\n"
         "def greet(name):\n"
         "    return f'Hello, {name}!'\n\n"
         "Then use the code_executor tool with language='python' to run this code:\n"
         "import sys; sys.path.insert(0, '/tmp')\n"
         "from effgen_codegen_utils import greet\n"
         "print(greet('effgen'))"),
        expected_keywords=["hello", "effgen"],
        check_fn=lambda out, resp: resp.tool_calls >= 1,
    ))

    # T6: Data processing — random numbers, mean, stdev
    results.append(run_test(
        agent, "T6", "Data processing — statistics",
        "Write Python code to create a list of 10 numbers [1, 4, 7, 2, 5, 8, 3, 6, 9, 10], calculate their mean and standard deviation using the statistics module, and print both values.",
        expected_keywords=["mean", "5.5"],
        expected_tool="python_repl",
    ))

    # Summary
    print(f"\n{'='*60}")
    print(f"Code Execution Results — Model: {model_name}")
    print(f"{'='*60}")
    pass_count = 0
    for r in results:
        status = r["status"]
        if r["passed"]:
            pass_count += 1
        print(f"  {status:5s} — {r['test_id']}: {r['description']} ({r['time']:.1f}s)")
    print(f"\n{pass_count}/{len(results)} passed")
    return results


def run_regression(agent, model_name="unknown"):
    """Run regression tests to verify previous examples still work with the coding agent."""
    results = []

    # Basic Q&A (no tools)
    results.append(run_test(
        agent, "REG-1", "Q&A: capital of France",
        "What is the capital of France?",
        expected_keywords=["paris"],
    ))

    # Calculator (math)
    results.append(run_test(
        agent, "REG-2", "Calculator: 15 + 27",
        "What is 15 + 27? Use a tool to calculate it.",
        expected_keywords=["42"],
    ))

    # Multi-tool — bash
    results.append(run_test(
        agent, "REG-3", "Multi-tool: run bash pwd",
        "Run the command 'echo hello_regression' using bash and tell me the output.",
        expected_keywords=["hello_regression"],
        expected_tool="bash",
    ))

    # File ops — read
    results.append(run_test(
        agent, "REG-4", "File ops: read file",
        "Use bash to run: cat /etc/hostname",
        check_fn=lambda out, resp: resp.tool_calls >= 1,
    ))

    print(f"\n{'='*60}")
    print(f"Regression Results — Model: {model_name}")
    print(f"{'='*60}")
    pass_count = 0
    for r in results:
        if r["passed"]:
            pass_count += 1
        print(f"  {r['status']:5s} — {r['test_id']}: {r['description']} ({r['time']:.1f}s)")
    print(f"\n{pass_count}/{len(results)} regression passed")
    return results


def cleanup_generated_files():
    """Remove all generated .py files from code execution tests."""
    patterns = [
        "/tmp/effgen_codegen_*",
    ]
    count = 0
    for pattern in patterns:
        for f in glob_mod.glob(pattern):
            try:
                os.remove(f)
                count += 1
                print(f"  Cleaned up: {f}")
            except Exception as e:
                print(f"  Failed to clean {f}: {e}")
    if count == 0:
        print("  No generated files to clean up.")
    return count


def interactive_mode(agent):
    """Interactive coding chat."""
    print("\n--- Interactive Coding Agent (type 'quit' to exit) ---")
    print("Available tools: code_executor, python_repl, file_operations, bash")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            t0 = time.time()
            resp = agent.run(user_input)
            dt = time.time() - t0
            print(f"Agent: {resp.output}")
            print(f"   (tools={resp.tool_calls}, iters={resp.iterations}, {dt:.1f}s)")
        except (KeyboardInterrupt, EOFError):
            break
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="effGen Code Execution Agent Example")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--regression", action="store_true", help="Run regression tests only")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of generated files")
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print("effGen — Code Execution Agent")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    agent = create_agent(
        "coding", model,
        system_prompt=CODING_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=12,
    )
    print(f"Coding agent created with tools: {list(agent.tools.keys())}")

    if args.interactive:
        interactive_mode(agent)
    elif args.regression:
        run_regression(agent, model_name=args.model)
    else:
        # Run regression first
        print("\n--- Regression Tests ---")
        run_regression(agent, model_name=args.model)

        # Run code execution tests
        print("\n--- Code Execution Tests ---")
        run_all_tests(agent, model_name=args.model)

        # Cleanup
        if not args.no_cleanup:
            print("\n--- Cleanup ---")
            cleanup_generated_files()

    model.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
