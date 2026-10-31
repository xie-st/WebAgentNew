#!/usr/bin/env python3
"""
effGen — File Operations Agent (Read/Write/Search)

A file management agent with FileOperations, BashTool, and TextProcessingTool.
Demonstrates reading files, writing files, listing directories, and searching
file contents. The FileOperations tool is sandboxed to allowed directories.

Recommended models:
  - Qwen/Qwen2.5-3B-Instruct (3B)      — excellent quality (default)
  - Qwen/Qwen2.5-7B-Instruct (7B)      — best quality
  - microsoft/Phi-4-mini-instruct       — very accurate (slower)
  - Qwen/Qwen2.5-1.5B-Instruct (1.5B)  — good quality, fast

Tools used: FileOperations, BashTool, TextProcessingTool

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/file_operations_agent.py
  CUDA_VISIBLE_DEVICES=0 python examples/file_operations_agent.py --model Qwen/Qwen2.5-7B-Instruct
  CUDA_VISIBLE_DEVICES=0 python examples/file_operations_agent.py --interactive
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin.bash_tool import BashTool
from effgen.tools.builtin.file_ops import FileOperations
from effgen.tools.builtin.text_processing import TextProcessingTool

FILE_AGENT_SYSTEM_PROMPT = """You are a file management assistant with access to tools for file operations.

TOOL SELECTION RULES:
- For reading, writing, listing, or searching files: use the 'file_operations' tool
- For running shell commands (ls, pwd, find, grep): use the 'bash' tool
- For text analysis (word count, regex, etc.): use the 'text_processing' tool
- If no tool fits, answer directly using your knowledge

FILE OPERATIONS TOOL USAGE:
The file_operations tool requires JSON input with these fields:
- "operation": one of "read", "write", "list", "search", "metadata"
- "path": the file or directory path
- "content": (for write) the text content to write
- "pattern": (for search) the search pattern
- "format": (optional) "text", "json", "csv", or "xml"

EXAMPLES:
- To read a file: {"operation": "read", "path": "test.txt"}
- To write a file: {"operation": "write", "path": "test.txt", "content": "Hello world"}
- To list a directory: {"operation": "list", "path": "."}
- To search files: {"operation": "search", "path": ".", "pattern": "*.txt"}

Always provide the operation and path as JSON. Use absolute paths when provided."""


def create_file_agent(model, test_dir, system_prompt=None):
    """Create a file operations agent with sandboxed directory."""
    tools = [
        FileOperations(allowed_directories=[test_dir]),
        BashTool(working_directory=test_dir),
        TextProcessingTool(),
    ]
    config = AgentConfig(
        name="file_agent",
        model=model,
        tools=tools,
        system_prompt=system_prompt or FILE_AGENT_SYSTEM_PROMPT,
        max_iterations=10,
        temperature=0.1,
        enable_fallback=True,
    )
    return Agent(config)


def run_test(agent, test_id, description, question, expected_keywords=None,
             expected_tool=None, check_fn=None, setup_fn=None, test_dir=None):
    """Run a single test and return result dict."""
    print(f"\n{'='*60}")
    print(f"Test: {test_id} — {description}")
    print(f"Q: {question}")
    if expected_keywords:
        print(f"Expected keywords: {expected_keywords}")
    if expected_tool:
        print(f"Expected tool: {expected_tool}")

    # Run optional setup
    if setup_fn:
        setup_fn()

    t0 = time.time()
    resp = agent.run(question)
    dt = time.time() - t0

    output = resp.output if resp.output else ""
    print(f"A: {output[:500]}")
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
        custom_pass = check_fn(output, resp, test_dir)
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
            print(f"  Tool used: {expected_tool} ✓")

    passed = keyword_pass and custom_pass and tool_pass
    status = "PASS" if passed else "FAIL"
    print(f"Result: {status}")

    return {
        "test_id": test_id,
        "description": description,
        "passed": passed,
        "output": output[:300],
        "tool_calls": resp.tool_calls,
        "iterations": resp.iterations,
        "time": dt,
        "status": status,
    }


def run_all_tests(agent, model_name="unknown", test_dir="/tmp/effgen_file_ops_test"):
    """Run all file operations tests."""
    results = []

    # T1: Write file
    write_path = os.path.join(test_dir, "test_output.txt")
    results.append(run_test(
        agent, "T1", "Write file",
        f'Create a file called {write_path} with the content \'Hello from effgen\'',
        expected_keywords=["hello", "effgen"],
        expected_tool="file_operations",
        check_fn=lambda out, resp, td: os.path.exists(os.path.join(td, "test_output.txt")) if td else True,
        test_dir=test_dir,
    ))

    # T2: Read file — create the file first to ensure it exists
    read_path = os.path.join(test_dir, "test_output.txt")
    if not os.path.exists(read_path):
        with open(read_path, "w") as f:
            f.write("Hello from effgen")

    results.append(run_test(
        agent, "T2", "Read file",
        f'Read the file {read_path}',
        expected_keywords=["hello", "effgen"],
        expected_tool="file_operations",
        test_dir=test_dir,
    ))

    # T3: List directory
    # Create some test files for listing
    for fname in ["alpha.txt", "beta.txt", "gamma.log"]:
        fpath = os.path.join(test_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                f.write(f"Content of {fname}")

    results.append(run_test(
        agent, "T3", "List directory",
        f'List the files in the directory {test_dir}',
        expected_keywords=[],
        expected_tool="file_operations",
        check_fn=lambda out, resp, td: resp.tool_calls >= 1,
        test_dir=test_dir,
    ))

    # T4: Search for content in a file using bash grep
    search_file = os.path.join(test_dir, "search_test.txt")
    with open(search_file, "w") as f:
        f.write("import os\nimport sys\nimport json\nprint('hello')\n")

    results.append(run_test(
        agent, "T4", "Search file content",
        f'Search for the word "import" in the file {search_file}. Use the bash tool with grep.',
        expected_keywords=["import"],
        expected_tool="bash",
        test_dir=test_dir,
    ))

    # T5: Path handling — file:/// prefix
    file_uri_path = os.path.join(test_dir, "test_output.txt")
    results.append(run_test(
        agent, "T5", "Path handling (absolute path)",
        f'Read the contents of the file at {file_uri_path}',
        expected_keywords=["hello", "effgen"],
        expected_tool="file_operations",
        test_dir=test_dir,
    ))

    # T6: Non-existent file — clear error
    nonexistent = os.path.join(test_dir, "nonexistent_file_xyz.txt")
    results.append(run_test(
        agent, "T6", "Non-existent file error",
        f'Read the file {nonexistent}',
        expected_keywords=[],
        expected_tool="file_operations",
        check_fn=lambda out, resp, td: any(kw in out.lower() for kw in ["not found", "error", "does not exist", "no such", "fail", "cannot", "could not"]),
        test_dir=test_dir,
    ))

    # T7: Multi-step — create, write, read back
    multi_path = os.path.join(test_dir, "multi_step.txt")
    # Clean up if exists
    if os.path.exists(multi_path):
        os.remove(multi_path)

    results.append(run_test(
        agent, "T7", "Multi-step: write then read",
        f'First, create a file at {multi_path} with 3 lines: "Line 1: Alpha", "Line 2: Beta", "Line 3: Gamma". Then read the file back and tell me what it contains.',
        expected_keywords=["alpha", "beta", "gamma"],
        check_fn=lambda out, resp, td: resp.tool_calls >= 2,
        test_dir=test_dir,
    ))

    # Summary
    print(f"\n{'='*60}")
    print(f"File Operations Results — Model: {model_name}")
    print(f"{'='*60}")
    pass_count = 0
    for r in results:
        status = r["status"]
        if r["passed"]:
            pass_count += 1
        print(f"  {status:5s} — {r['test_id']}: {r['description']} ({r['time']:.1f}s)")
    print(f"\n{pass_count}/{len(results)} passed")
    return results


def interactive_mode(agent):
    """Interactive file operations chat."""
    print("\n--- Interactive File Agent (type 'quit' to exit) ---")
    print("Available tools: file_operations, bash, text_processing")
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
    parser = argparse.ArgumentParser(description="effGen File Operations Agent Example")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--test", type=str, help="Run a single test (e.g., T1)")
    parser.add_argument(
        "--test-dir",
        default="/tmp/effgen_file_ops_test",
        help="Directory for test files (default: /tmp/effgen_file_ops_test)",
    )
    args = parser.parse_args()

    # Ensure test directory exists
    os.makedirs(args.test_dir, exist_ok=True)

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print("effGen — File Operations Agent")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")
    print(f"Test directory: {args.test_dir}")

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    agent = create_file_agent(model, args.test_dir)
    print(f"File agent created with tools: {list(agent.tools.keys())}")

    if args.interactive:
        interactive_mode(agent)
    else:
        run_all_tests(agent, model_name=args.model, test_dir=args.test_dir)

    model.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
