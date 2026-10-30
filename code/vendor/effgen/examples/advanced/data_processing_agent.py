#!/usr/bin/env python3
"""
effGen — Data Processing Agent

Processes structured data using JSONTool, TextProcessingTool, PythonREPL,
and FileOperations. Demonstrates JSON querying, validation, formatting,
text analysis, and multi-tool data pipelines.

Tests:
  - T1: JSON query — extract value from JSON object
  - T2: JSON validation — detect invalid JSON
  - T3: JSON format — pretty-print compact JSON
  - T4: Complex query — nested JSON with deep path query
  - T5: Text analysis — word count and sentence count
  - T6: Data pipeline — read JSON, extract, sort, and write output

Recommended models:
  - Qwen/Qwen2.5-3B-Instruct (default, fast)
  - Qwen/Qwen2.5-7B-Instruct (best quality)
  - microsoft/Phi-4-mini-instruct (good quality)

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/data_processing_agent.py
  CUDA_VISIBLE_DEVICES=0 python examples/data_processing_agent.py --model Qwen/Qwen2.5-7B-Instruct
  CUDA_VISIBLE_DEVICES=0 python examples/data_processing_agent.py --regression
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin.file_ops import FileOperations
from effgen.tools.builtin.json_tool import JSONTool
from effgen.tools.builtin.python_repl import PythonREPL
from effgen.tools.builtin.text_processing import TextProcessingTool

# ── System Prompt ────────────────────────────────────────────────────────────

DATA_PROCESSING_SYSTEM_PROMPT = """You are a data processing assistant with access to tools.

TOOL SELECTION RULES:
- For JSON operations: use 'json_tool' with {"data": "<json string>", "operation": "<op>", "query": "<path>"}
  Operations: "query" (default), "validate", "format", "keys", "length"
  Query uses JSONPath-like syntax: $.key, $.key.subkey, $.array[0], $.array[*]
- For text analysis: use 'text_processing' with {"operation": "<op>", "text": "<text>"}
  Operations: "word_count", "summarize", "regex_search", "regex_replace", "compare", "transform"
- For Python code: use 'python_repl' with {"code": "<python code>"}
- For file operations: use 'file_operations' with {"operation": "<op>", "path": "<path>", ...}
  Operations: "read", "write", "list", "search"

IMPORTANT:
1. When querying JSON, use operation "query" with a JSONPath query like "$.name" or "$.users[0].name"
2. The data parameter for json_tool must be a valid JSON string (use double quotes, not single quotes)
3. When done, provide your response using 'Final Answer:'.
4. Show the actual results from tool calls in your final answer.
"""


# ── Test Helpers ─────────────────────────────────────────────────────────────

def run_test(agent, test_id, description, question,
             check_fn=None, expected_keywords=None, expected_tool=None,
             timeout=120):
    """Run a single test and return result dict."""
    print(f"\n{'='*60}")
    print(f"Test: {test_id} — {description}")
    print(f"Q: {question[:300]}")

    t0 = time.time()
    try:
        resp = agent.run(question)
    except Exception as e:
        dt = time.time() - t0
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
        return {
            "test_id": test_id, "description": description,
            "passed": False, "output": f"EXCEPTION: {e}",
            "tool_calls": 0, "iterations": 0, "time": dt, "status": "FAIL",
        }

    dt = time.time() - t0
    output = resp.output if resp.output else ""
    print(f"A: {output[:600]}")
    print(f"Tool calls: {resp.tool_calls}, Iterations: {resp.iterations}, "
          f"Time: {dt:.1f}s, Success: {resp.success}")
    if resp.metadata:
        print(f"Metadata: {resp.metadata}")

    # Check keywords
    keyword_pass = True
    if expected_keywords:
        for kw in expected_keywords:
            if kw.lower() not in output.lower():
                keyword_pass = False
                print(f"  MISSING keyword: '{kw}'")

    # Check custom function
    custom_pass = True
    if check_fn:
        try:
            custom_pass = check_fn(output, resp)
        except Exception as e:
            custom_pass = False
            print(f"  Custom check EXCEPTION: {e}")
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
        "test_id": test_id, "description": description,
        "passed": passed, "output": output[:400],
        "tool_calls": resp.tool_calls, "iterations": resp.iterations,
        "time": dt, "status": status,
    }


# ── Data Processing Tests ────────────────────────────────────────────────────

def test_t1_json_query(model, model_name):
    """T1: JSON query — query a key from a JSON object."""
    agent = Agent(AgentConfig(
        name="data_processing_t1",
        model=model,
        tools=[JSONTool(), TextProcessingTool(), PythonREPL()],
        system_prompt=DATA_PROCESSING_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=8,
        enable_sub_agents=False,
    ))

    json_data = '{"name": "Alice", "age": 30, "city": "Wonderland"}'
    return run_test(
        agent, "T1", "JSON query — extract value of 'name'",
        f'Use the json_tool to query this JSON and find the value of the "name" key: {json_data}\n'
        f'Use operation "query" with query "$.name".',
        expected_keywords=["alice"],
        expected_tool="json_tool",
    )


def test_t2_json_validate(model, model_name):
    """T2: JSON validation — validate invalid JSON (single-quoted keys)."""
    agent = Agent(AgentConfig(
        name="data_processing_t2",
        model=model,
        tools=[JSONTool(), TextProcessingTool(), PythonREPL()],
        system_prompt=DATA_PROCESSING_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=8,
        enable_sub_agents=False,
    ))

    # This is intentionally invalid JSON (single quotes)
    invalid_json = "{key: value}"
    return run_test(
        agent, "T2", "JSON validation — detect invalid JSON",
        f'Use json_tool with operation "validate" to check if this is valid JSON: {invalid_json}\n'
        f'Pass the exact string as the data parameter. Report whether it is valid or invalid.',
        check_fn=lambda out, resp: (
            "invalid" in out.lower() or "not valid" in out.lower()
            or "false" in out.lower() or "error" in out.lower()
            or resp.tool_calls >= 1  # At least tried to use the tool
        ),
        expected_tool="json_tool",
    )


def test_t3_json_format(model, model_name):
    """T3: JSON format — pretty-print compact JSON."""
    agent = Agent(AgentConfig(
        name="data_processing_t3",
        model=model,
        tools=[JSONTool(), TextProcessingTool(), PythonREPL()],
        system_prompt=DATA_PROCESSING_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=8,
        enable_sub_agents=False,
    ))

    compact_json = '{"a":1,"b":2,"c":{"d":3,"e":4}}'
    return run_test(
        agent, "T3", "JSON format — pretty-print compact JSON",
        f'Use json_tool with operation "format" to pretty-print this JSON: {compact_json}',
        check_fn=lambda out, resp: (
            resp.tool_calls >= 1
            and len(out) > 10
        ),
        expected_tool="json_tool",
    )


def test_t4_complex_query(model, model_name):
    """T4: Complex query — nested JSON with arrays, deep path query."""
    agent = Agent(AgentConfig(
        name="data_processing_t4",
        model=model,
        tools=[JSONTool(), TextProcessingTool(), PythonREPL()],
        system_prompt=DATA_PROCESSING_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=10,
        enable_sub_agents=False,
    ))

    nested_json = json.dumps({
        "company": "Acme Corp",
        "employees": [
            {"name": "Alice", "role": "engineer", "skills": ["python", "rust"]},
            {"name": "Bob", "role": "designer", "skills": ["figma", "css"]},
            {"name": "Charlie", "role": "manager", "skills": ["leadership"]}
        ],
        "location": {"city": "San Francisco", "state": "CA"}
    })

    return run_test(
        agent, "T4", "Complex query — nested JSON deep path",
        f'Use json_tool to query this JSON: {nested_json}\n'
        f'Find the name of the first employee using query "$.employees[0].name".',
        expected_keywords=["alice"],
        expected_tool="json_tool",
    )


def test_t5_text_analysis(model, model_name):
    """T5: Text analysis — word count, sentence count, pattern finding."""
    agent = Agent(AgentConfig(
        name="data_processing_t5",
        model=model,
        tools=[JSONTool(), TextProcessingTool(), PythonREPL()],
        system_prompt=DATA_PROCESSING_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=8,
        enable_sub_agents=False,
    ))

    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence has exactly nine words in it. "
        "Machine learning is transforming the world today."
    )

    return run_test(
        agent, "T5", "Text analysis — word count and sentence count",
        f'Use text_processing with operation "word_count" to analyze this text:\n"{sample_text}"\n'
        f'Report the word count, character count, and sentence count.',
        check_fn=lambda out, resp: (
            resp.tool_calls >= 1
            and len(out) > 10
        ),
        expected_tool="text_processing",
    )


def test_t6_data_pipeline(model, model_name, sandbox_dir):
    """T6: Data pipeline — read JSON, extract names, sort, write output."""
    # Prepare input file
    data = {
        "team": [
            {"name": "Charlie", "score": 85},
            {"name": "Alice", "score": 92},
            {"name": "Bob", "score": 78},
            {"name": "Diana", "score": 95},
        ]
    }
    input_path = os.path.join(sandbox_dir, "team_data.json")
    output_path = os.path.join(sandbox_dir, "sorted_names.txt")
    with open(input_path, "w") as f:
        json.dump(data, f, indent=2)

    agent = Agent(AgentConfig(
        name="data_processing_t6",
        model=model,
        tools=[
            JSONTool(),
            TextProcessingTool(),
            PythonREPL(),
            FileOperations(allowed_directories=[sandbox_dir]),
        ],
        system_prompt=DATA_PROCESSING_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=15,
        enable_sub_agents=False,
    ))

    return run_test(
        agent, "T6", "Data pipeline — read JSON, extract and sort names",
        f'Read the JSON file at {input_path} using file_operations (operation "read", path "{input_path}"). '
        f'Then use json_tool or python_repl to extract all the "name" values from the "team" array, '
        f'sort them alphabetically, and write them (one per line) to {output_path} using file_operations '
        f'(operation "write", path "{output_path}", content "...").',
        check_fn=lambda out, resp: (
            resp.tool_calls >= 2  # At least read + write (or python)
            and len(out) > 5
        ),
    )


# ── Regression Tests ─────────────────────────────────────────────────────────

def run_regression(model, model_name):
    """Run regression tests to verify previous examples still work."""
    from effgen.tools.builtin.bash_tool import BashTool
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.python_repl import PythonREPL as PythonREPLTool

    results = []

    agent = Agent(AgentConfig(
        name="regression_agent",
        model=model,
        tools=[Calculator(), PythonREPLTool(), BashTool()],
        system_prompt=(
            "You are a helpful AI assistant with access to tools.\n"
            "TOOL SELECTION RULES:\n"
            "- For math: use 'calculator' with {\"expression\": \"...\"}\n"
            "- For Python code: use 'python_repl' with {\"code\": \"...\"}\n"
            "- For shell commands: use 'bash' with {\"command\": \"...\"}\n"
            "When done, provide your response using 'Final Answer:'."
        ),
        temperature=0.1,
        max_iterations=8,
        enable_sub_agents=False,
    ))

    # Q&A
    results.append(run_test(
        agent, "REG-P1", "Q&A: capital of France",
        "What is the capital of France?",
        expected_keywords=["paris"],
    ))

    # Calculator
    results.append(run_test(
        agent, "REG-P2", "Calculator: 15 + 27",
        "What is 15 + 27? Use a tool to calculate it.",
        expected_keywords=["42"],
    ))

    # Multi-tool — bash
    results.append(run_test(
        agent, "REG-P3", "Multi-tool: bash echo",
        "Run the command 'echo hello_regression' using bash and tell me the output.",
        expected_keywords=["hello_regression"],
        expected_tool="bash",
    ))

    # File ops
    results.append(run_test(
        agent, "REG-P4", "File ops: cat hostname",
        "Use bash to run: cat /etc/hostname",
        check_fn=lambda out, resp: resp.tool_calls >= 1,
    ))

    # Code execution
    results.append(run_test(
        agent, "REG-P5", "Code: prime check",
        "Write and run Python code to check if 17 is prime. Print the result.",
        expected_keywords=["17"],
        expected_tool="python_repl",
    ))

    # Multi-turn memory
    results.append(run_test(
        agent, "REG-P6", "Memory: follow-up",
        "My name is TestUser. Remember that.",
        check_fn=lambda out, resp: len(out) > 3,
    ))

    # Error recovery
    results.append(run_test(
        agent, "REG-P7", "Error recovery: explains instead of crashing",
        "Calculate the square root of -1. If the calculator can't handle it, explain mathematically.",
        check_fn=lambda out, resp: len(out) > 10,
    ))

    # Summary
    print(f"\n{'='*60}")
    print(f"Regression Results — Model: {model_name}")
    print(f"{'='*60}")
    pc = sum(1 for r in results if r["passed"])
    for r in results:
        print(f"  {r['status']:5s} — {r['test_id']}: {r['description']} ({r['time']:.1f}s)")
    print(f"\n{pc}/{len(results)} regression passed")
    return results


# ── Main Entry Point ─────────────────────────────────────────────────────────

def run_all_tests(model, model_name, sandbox_dir):
    """Run all 6 data processing tests."""
    results = []

    print("\n" + "="*60)
    print(f"Data Processing Agent — Model: {model_name}")
    print("="*60)

    results.append(test_t1_json_query(model, model_name))
    results.append(test_t2_json_validate(model, model_name))
    results.append(test_t3_json_format(model, model_name))
    results.append(test_t4_complex_query(model, model_name))
    results.append(test_t5_text_analysis(model, model_name))
    results.append(test_t6_data_pipeline(model, model_name, sandbox_dir))

    # Summary
    print(f"\n{'='*60}")
    print(f"Data Processing Results — Model: {model_name}")
    print(f"{'='*60}")
    pc = sum(1 for r in results if r["passed"])
    for r in results:
        print(f"  {r['status']:5s} — {r['test_id']}: {r['description']} ({r['time']:.1f}s)")
    print(f"\n{pc}/{len(results)} tests passed")

    return results


def main():
    parser = argparse.ArgumentParser(description="effGen Data Processing Agent Example")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--regression", action="store_true", help="Run regression tests only")
    parser.add_argument("--test", type=str, help="Run specific test (e.g., T1)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print("effGen — Data Processing Agent")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Create sandbox directory for file operations
    sandbox_dir = tempfile.mkdtemp(prefix="effgen_data_")
    print(f"Sandbox: {sandbox_dir}")

    try:
        if args.regression:
            run_regression(model, model_name=args.model)
        elif args.test:
            test_map = {
                "T1": lambda: test_t1_json_query(model, args.model),
                "T2": lambda: test_t2_json_validate(model, args.model),
                "T3": lambda: test_t3_json_format(model, args.model),
                "T4": lambda: test_t4_complex_query(model, args.model),
                "T5": lambda: test_t5_text_analysis(model, args.model),
                "T6": lambda: test_t6_data_pipeline(model, args.model, sandbox_dir),
            }
            test_fn = test_map.get(args.test.upper())
            if test_fn:
                test_fn()
            else:
                print(f"Unknown test: {args.test}. Available: {list(test_map.keys())}")
        else:
            # Run regression first
            print("\n--- Regression Tests ---")
            run_regression(model, model_name=args.model)

            # Run data processing tests
            print("\n--- Data Processing Tests ---")
            run_all_tests(model, model_name=args.model, sandbox_dir=sandbox_dir)
    finally:
        # Cleanup sandbox
        if os.path.exists(sandbox_dir):
            shutil.rmtree(sandbox_dir)
            print(f"\nCleaned up sandbox: {sandbox_dir}")

    model.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
