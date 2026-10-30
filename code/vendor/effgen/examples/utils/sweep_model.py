#!/usr/bin/env python3
"""
effGen — Cross-Model Compatibility Sweep (Single Model Runner)

Runs all 10 example agents on a single model and outputs structured results.
Designed to be called in parallel across GPUs for different models.

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/sweep_model.py \
    --model Qwen/Qwen2.5-3B-Instruct --output /tmp/sweep_results/qwen2.5-3b.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_agent_test(agent_name, model, model_name, gpu_id):
    """Run a single agent's tests and return structured results.

    Returns dict with:
      - agent: agent name
      - status: PASS / PARTIAL / FAIL
      - passed: int
      - total: int
      - details: list of individual test results
      - time: float
      - error: str or None
    """
    result = {
        "agent": agent_name,
        "model": model_name,
        "status": "FAIL",
        "passed": 0,
        "total": 0,
        "details": [],
        "time": 0.0,
        "error": None,
    }

    t0 = time.time()

    try:
        if agent_name == "qa_agent":
            result = _test_qa(model, model_name)
        elif agent_name == "calculator_agent":
            result = _test_calculator(model, model_name)
        elif agent_name == "advanced_multi_tool_agent":
            result = _test_multi_tool(model, model_name)
        elif agent_name == "file_operations_agent":
            result = _test_file_ops(model, model_name)
        elif agent_name == "coding_agent":
            result = _test_coding(model, model_name)
        elif agent_name == "conversational_agent":
            result = _test_conversational(model, model_name)
        elif agent_name == "error_recovery_agent":
            result = _test_error_recovery(model, model_name)
        elif agent_name == "data_processing_agent":
            result = _test_data_processing(model, model_name)
        elif agent_name == "advanced_streaming_agent":
            result = _test_streaming(model, model_name)
        elif agent_name == "multi_agent_pipeline":
            result = _test_multi_agent(model, model_name)
        else:
            result["error"] = f"Unknown agent: {agent_name}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        result["status"] = "FAIL"
        traceback.print_exc()

    result["time"] = time.time() - t0
    result["agent"] = agent_name
    result["model"] = model_name

    # Determine status
    if result.get("error"):
        result["status"] = "FAIL"
    elif result["total"] > 0:
        ratio = result["passed"] / result["total"]
        if ratio >= 0.8:
            result["status"] = "PASS"
        elif ratio >= 0.4:
            result["status"] = "PARTIAL"
        else:
            result["status"] = "FAIL"

    return result


def _quick_test(agent, question, expected_keywords=None, check_fn=None):
    """Run a single quick test, return (passed, detail_str)."""
    try:
        resp = agent.run(question)
        output = resp.output if resp.output else ""

        kw_pass = True
        if expected_keywords:
            for kw in expected_keywords:
                if kw.lower() not in output.lower():
                    kw_pass = False

        custom_pass = True
        if check_fn:
            try:
                custom_pass = check_fn(output, resp)
            except Exception:
                custom_pass = False

        passed = kw_pass and custom_pass
        return passed, output[:200]
    except Exception as e:
        return False, f"ERROR: {e}"


# ── Agent Test Implementations ──────────────────────────────────────────

def _test_qa(model, model_name):
    """Test Q&A agent — 3 questions, no tools."""
    from effgen.presets import create_agent

    agent = create_agent(
        "minimal", model,
        system_prompt="You are a helpful AI assistant. Answer questions directly and concisely.",
    )

    tests = [
        ("What is the capital of France?", ["paris"]),
        ("Explain photosynthesis in 3 sentences.", ["light", "plant"]),
        ("What are the three laws of thermodynamics?", ["energy", "entropy"]),
    ]

    passed = 0
    details = []
    for q, kws in tests:
        ok, detail = _quick_test(agent, q, expected_keywords=kws)
        if ok:
            passed += 1
        details.append({"question": q[:60], "passed": ok, "output": detail[:150]})
        print(f"  {'PASS' if ok else 'FAIL'} — {q[:50]}... -> {detail[:80]}")

    return {"passed": passed, "total": len(tests), "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_calculator(model, model_name):
    """Test calculator agent — 5 math tasks."""
    from effgen.presets import create_agent

    agent = create_agent("math", model)

    tests = [
        ("What is 247 * 83?", ["20501"]),
        ("What is the square root of 144?", ["12"]),
        ("What is (15 + 27) * 3 - 10?", ["116"]),
        ("Add 5 and 3, then multiply the result by 2.", ["16"]),
        ("Use the calculator to compute 2+2.", ["4"]),
    ]

    passed = 0
    details = []
    for q, kws in tests:
        ok, detail = _quick_test(agent, q, expected_keywords=kws)
        if ok:
            passed += 1
        details.append({"question": q[:60], "passed": ok, "output": detail[:150]})
        print(f"  {'PASS' if ok else 'FAIL'} — {q[:50]}... -> {detail[:80]}")

    return {"passed": passed, "total": len(tests), "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_multi_tool(model, model_name):
    """Test multi-tool agent — 5 tool-selection tasks (skip T6/T7 circuit breaker programmatic tests)."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.tools.builtin.bash_tool import BashTool
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.datetime_tool import DateTimeTool
    from effgen.tools.builtin.python_repl import PythonREPL
    from effgen.tools.builtin.text_processing import TextProcessingTool

    tools = [Calculator(), PythonREPL(), DateTimeTool(), BashTool(), TextProcessingTool()]
    config = AgentConfig(
        name="multi_tool_agent", model=model, tools=tools,
        system_prompt=(
            "You are a helpful AI assistant with access to multiple tools.\n"
            "TOOL SELECTION RULES:\n"
            "- For math: use 'calculator'\n"
            "- For date/time: use 'datetime' tool\n"
            "- For text: use 'text_processing'\n"
            "- For shell commands: use 'bash'\n"
            "- For Python code: use 'python_repl'\n"
            "Always select the MOST APPROPRIATE tool."
        ),
        max_iterations=10, temperature=0.1, enable_fallback=True,
    )
    agent = Agent(config)

    tests = [
        ("What is 347 * 29?", ["10063"]),
        ("What day of the week is March 15, 2026?", ["sunday"]),
        ("Count the words in this text: 'The quick brown fox jumps over the lazy dog'", ["9"]),
        ("What is the current working directory? Use the bash tool with the pwd command.", ["/"]),
        ("Search the web for Python tutorials.", None),  # Just check non-empty
    ]

    passed = 0
    details = []
    for q, kws in tests:
        check = None
        if kws is None:
            def check(out, resp):
                return len(out) > 10
            kws_arg = None
        else:
            kws_arg = kws
        ok, detail = _quick_test(agent, q, expected_keywords=kws_arg, check_fn=check)
        if ok:
            passed += 1
        details.append({"question": q[:60], "passed": ok, "output": detail[:150]})
        print(f"  {'PASS' if ok else 'FAIL'} — {q[:50]}... -> {detail[:80]}")

    return {"passed": passed, "total": len(tests), "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_file_ops(model, model_name):
    """Test file operations agent — 4 tests (write, read, list, search)."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.tools.builtin.bash_tool import BashTool
    from effgen.tools.builtin.file_ops import FileOperations
    from effgen.tools.builtin.text_processing import TextProcessingTool

    test_dir = tempfile.mkdtemp(prefix="effgen_sweep_fileops_")

    tools = [
        FileOperations(allowed_directories=[test_dir]),
        BashTool(working_directory=test_dir),
        TextProcessingTool(),
    ]
    config = AgentConfig(
        name="file_agent", model=model, tools=tools,
        system_prompt=(
            "You are a file management assistant.\n"
            "- For file ops: use 'file_operations' with JSON: {\"operation\": \"read/write/list\", \"path\": \"...\", \"content\": \"...\"}\n"
            "- For shell: use 'bash'\n"
            "- For text: use 'text_processing'\n"
            "Always provide operation and path as JSON."
        ),
        max_iterations=10, temperature=0.1,
    )
    agent = Agent(config)

    try:
        # T1: Write file
        write_path = os.path.join(test_dir, "test.txt")
        ok1, d1 = _quick_test(agent, f"Create a file at {write_path} with the content 'Hello from effgen'",
                              check_fn=lambda out, resp: resp.tool_calls >= 1)

        # Ensure file exists for read test
        if not os.path.exists(write_path):
            with open(write_path, "w") as f:
                f.write("Hello from effgen")

        # T2: Read file
        ok2, d2 = _quick_test(agent, f"Read the file {write_path}", expected_keywords=["hello", "effgen"])

        # T3: List directory
        for fn in ["alpha.txt", "beta.txt"]:
            with open(os.path.join(test_dir, fn), "w") as f:
                f.write(f"content of {fn}")
        ok3, d3 = _quick_test(agent, f"List the files in {test_dir}",
                              check_fn=lambda out, resp: resp.tool_calls >= 1)

        # T4: Read non-existent file
        ok4, d4 = _quick_test(agent, f"Read the file {test_dir}/nonexistent.txt",
                              check_fn=lambda out, resp: any(kw in out.lower() for kw in ["not found", "error", "does not exist", "no such", "fail"]))

        tests_passed = sum([ok1, ok2, ok3, ok4])
        details = [
            {"test": "write", "passed": ok1}, {"test": "read", "passed": ok2},
            {"test": "list", "passed": ok3}, {"test": "error", "passed": ok4},
        ]
        for d in details:
            print(f"  {'PASS' if d['passed'] else 'FAIL'} — {d['test']}")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    return {"passed": tests_passed, "total": 4, "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_coding(model, model_name):
    """Test coding agent — 4 code execution tests."""
    from effgen.presets import create_agent

    agent = create_agent(
        "coding", model,
        system_prompt=(
            "You are an expert Python coding agent.\n"
            "- For Python code: use 'python_repl' with {\"code\": \"...\"}\n"
            "- Do NOT wrap code in markdown fences.\n"
            "- Always include print() to see output.\n"
            "When done, provide Final Answer: with the result."
        ),
        temperature=0.1, max_iterations=12,
    )

    tests = [
        ("Write and run Python code to check if 17 is prime. Print the result.",
         None, lambda out, resp: resp.tool_calls >= 1 and ("true" in out.lower() or "prime" in out.lower())),
        ("Write Python code that divides 10 by 0 and run it. Report what happens.",
         ["zero"], None),
        ("Write a Python function to sort [5,2,8,1,9,3]. Run it and print the sorted list.",
         ["1", "2", "3"], None),
        ("Run this Python code using python_repl: print('fence test passed!')",
         ["fence test passed"], None),
    ]

    passed = 0
    details = []
    for q, kws, check in tests:
        ok, detail = _quick_test(agent, q, expected_keywords=kws, check_fn=check)
        if ok:
            passed += 1
        details.append({"question": q[:60], "passed": ok, "output": detail[:150]})
        print(f"  {'PASS' if ok else 'FAIL'} — {q[:50]}... -> {detail[:80]}")

    return {"passed": passed, "total": len(tests), "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_conversational(model, model_name):
    """Test conversational agent — 3 multi-turn memory tests."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.datetime_tool import DateTimeTool

    def make_agent():
        config = AgentConfig(
            name="conv_agent", model=model,
            tools=[Calculator(), DateTimeTool()],
            system_prompt=(
                "You are a conversational assistant with memory. Remember what the user tells you.\n"
                "When asked to recall, look in the conversation context.\n"
                "- calculator: for math\n- datetime: for date/time"
            ),
            max_iterations=8, temperature=0.1,
            enable_memory=True,
            memory_config={"short_term_max_tokens": 4096, "short_term_max_messages": 50, "auto_summarize": True},
        )
        return Agent(config)

    tests_passed = 0
    details = []

    # T1: Name recall
    a = make_agent()
    a.run("My name is Alice and I'm a data scientist.")
    resp = a.run("What's my name and what do I do?")
    out = (resp.output or "").lower()
    ok1 = "alice" in out and "data scientist" in out
    tests_passed += ok1
    details.append({"test": "name_recall", "passed": ok1})
    print(f"  {'PASS' if ok1 else 'FAIL'} — name recall: {out[:80]}")

    # T2: Preference recall
    a = make_agent()
    a.run("I prefer Python over JavaScript for backend development.")
    resp = a.run("What language should I use for a backend project?")
    out = (resp.output or "").lower()
    ok2 = "python" in out
    tests_passed += ok2
    details.append({"test": "preference_recall", "passed": ok2})
    print(f"  {'PASS' if ok2 else 'FAIL'} — preference recall: {out[:80]}")

    # T3: Contextual tool use
    a = make_agent()
    a.run("My office temperature is 22 degrees Celsius.")
    resp = a.run("Convert my office temperature to Fahrenheit using the calculator: 22 * 9/5 + 32")
    out = (resp.output or "").lower()
    ok3 = "71.6" in out
    tests_passed += ok3
    details.append({"test": "contextual_tool", "passed": ok3})
    print(f"  {'PASS' if ok3 else 'FAIL'} — contextual tool: {out[:80]}")

    return {"passed": tests_passed, "total": 3, "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_error_recovery(model, model_name):
    """Test error recovery — 5 key tests (T1-T5 from the error-recovery suite)."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.tools.base_tool import (
        BaseTool,
        ParameterSpec,
        ParameterType,
        ToolCategory,
        ToolMetadata,
    )
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.python_repl import PythonREPL

    class BrokenTool(BaseTool):
        def __init__(self):
            super().__init__(metadata=ToolMetadata(
                name="broken_tool", description="Always crashes",
                category=ToolCategory.COMPUTATION,
                parameters=[ParameterSpec(name="input", type=ParameterType.STRING, description="Input", required=True)],
            ))
        async def _execute(self, input="", **kw):
            raise RuntimeError("BrokenTool crash!")

    tests_passed = 0
    details = []

    # T1: Invalid input handling
    a1 = Agent(AgentConfig(name="t1", model=model, tools=[Calculator(), PythonREPL()],
                           system_prompt="You are a math helper. If calculator fails, explain why. Use Final Answer:",
                           temperature=0.1, max_iterations=8))
    ok1, d1 = _quick_test(a1, "Calculate 'not_a_number + abc'. If calculator can't handle it, explain why.",
                          check_fn=lambda out, resp: len(out) > 10)
    tests_passed += ok1
    details.append({"test": "invalid_input", "passed": ok1})
    print(f"  {'PASS' if ok1 else 'FAIL'} — invalid input: {d1[:80]}")

    # T2: Tool crash
    a2 = Agent(AgentConfig(name="t2", model=model, tools=[BrokenTool(), Calculator()],
                           system_prompt="Use tools to help. If one fails, try another. Final Answer:",
                           temperature=0.1, max_iterations=8))
    ok2, d2 = _quick_test(a2, "Use broken_tool to process 'test'. If it fails, use calculator for 10+5.",
                          check_fn=lambda out, resp: len(out) > 3)
    tests_passed += ok2
    details.append({"test": "tool_crash", "passed": ok2})
    print(f"  {'PASS' if ok2 else 'FAIL'} — tool crash: {d2[:80]}")

    # T3: All tools fail
    a3 = Agent(AgentConfig(name="t3", model=model, tools=[BrokenTool()],
                           system_prompt="Use tools first. If all fail, answer from knowledge. Final Answer:",
                           temperature=0.1, max_iterations=5))
    ok3, d3 = _quick_test(a3, "What is 25 * 4? Try using a tool, but answer from knowledge if it fails.",
                          check_fn=lambda out, resp: len(out) >= 3)
    tests_passed += ok3
    details.append({"test": "all_fail", "passed": ok3})
    print(f"  {'PASS' if ok3 else 'FAIL'} — all tools fail: {d3[:80]}")

    # T4: Max iterations
    a4 = Agent(AgentConfig(name="t4", model=model, tools=[Calculator(), PythonREPL()],
                           system_prompt="Solve math step by step. Final Answer:",
                           temperature=0.1, max_iterations=2))
    ok4, d4 = _quick_test(a4, "Calculate 15+27, then multiply by 3.",
                          check_fn=lambda out, resp: len(out) > 5 and resp.iterations <= 2)
    tests_passed += ok4
    details.append({"test": "max_iters", "passed": ok4})
    print(f"  {'PASS' if ok4 else 'FAIL'} — max iterations: {d4[:80]}")

    # T5: Control character sanitization (non-model, always passes if framework works)
    from effgen.core.agent import Agent as AgentClass
    sanitized = AgentClass._sanitize_tool_input("\x00\x01hello\x03world")
    ok5 = sanitized == "helloworld"
    tests_passed += ok5
    details.append({"test": "sanitize", "passed": ok5})
    print(f"  {'PASS' if ok5 else 'FAIL'} — sanitization")

    return {"passed": tests_passed, "total": 5, "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_data_processing(model, model_name):
    """Test data processing — 4 JSON/text tests."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.tools.builtin.json_tool import JSONTool
    from effgen.tools.builtin.python_repl import PythonREPL
    from effgen.tools.builtin.text_processing import TextProcessingTool

    tools = [JSONTool(), TextProcessingTool(), PythonREPL()]

    def make_agent():
        return Agent(AgentConfig(
            name="data_agent", model=model, tools=tools,
            system_prompt=(
                "You are a data processing assistant.\n"
                "- json_tool: {\"data\": \"<json>\", \"operation\": \"query/validate/format/keys/length\", \"query\": \"$.path\"}\n"
                "- text_processing: {\"operation\": \"word_count\", \"text\": \"...\"}\n"
                "- python_repl: {\"code\": \"...\"}\n"
                "Show actual results. Use Final Answer:"
            ),
            max_iterations=8, temperature=0.1,
        ))

    tests_passed = 0
    details = []

    # T1: JSON query
    a = make_agent()
    ok1, d1 = _quick_test(a,
        'Use json_tool to query this JSON for the "name" key: {"name": "Alice", "age": 30}. Use operation "query" with query "$.name".',
        expected_keywords=["alice"])
    tests_passed += ok1
    details.append({"test": "json_query", "passed": ok1})
    print(f"  {'PASS' if ok1 else 'FAIL'} — json query: {d1[:80]}")

    # T2: JSON validate
    a = make_agent()
    ok2, d2 = _quick_test(a,
        'Use json_tool with operation "validate" to check if this is valid JSON: {key: value}',
        check_fn=lambda out, resp: ("invalid" in out.lower() or "not valid" in out.lower() or "false" in out.lower() or "error" in out.lower() or resp.tool_calls >= 1))
    tests_passed += ok2
    details.append({"test": "json_validate", "passed": ok2})
    print(f"  {'PASS' if ok2 else 'FAIL'} — json validate: {d2[:80]}")

    # T3: Text word count
    a = make_agent()
    ok3, d3 = _quick_test(a,
        'Use text_processing with operation "word_count" on: "The quick brown fox jumps over the lazy dog"',
        check_fn=lambda out, resp: resp.tool_calls >= 1 and len(out) > 5)
    tests_passed += ok3
    details.append({"test": "text_wordcount", "passed": ok3})
    print(f"  {'PASS' if ok3 else 'FAIL'} — text word count: {d3[:80]}")

    # T4: JSON format
    a = make_agent()
    ok4, d4 = _quick_test(a,
        'Use json_tool with operation "format" to pretty-print: {"a":1,"b":2,"c":3}',
        check_fn=lambda out, resp: resp.tool_calls >= 1 and len(out) > 10)
    tests_passed += ok4
    details.append({"test": "json_format", "passed": ok4})
    print(f"  {'PASS' if ok4 else 'FAIL'} — json format: {d4[:80]}")

    return {"passed": tests_passed, "total": 4, "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_streaming(model, model_name):
    """Test streaming agent — 3 streaming tests."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.datetime_tool import DateTimeTool

    agent = Agent(AgentConfig(
        name="stream_agent", model=model,
        tools=[Calculator(), DateTimeTool()],
        max_iterations=5, temperature=0.1,
    ))

    tests_passed = 0
    details = []

    # T1: Streaming with tool call
    tokens = []
    answers = []
    tool_calls = []
    try:
        for tok in agent.stream(
            "What is 15 * 7?",
            on_answer=lambda a: answers.append(a),
            on_tool_call=lambda tn, ti: tool_calls.append(tn),
        ):
            tokens.append(tok)
        ok1 = len(tokens) > 0 and any("105" in a for a in answers)
    except Exception as e:
        ok1 = False
        print(f"    Stream error: {e}")
    tests_passed += ok1
    details.append({"test": "stream_tool", "passed": ok1})
    print(f"  {'PASS' if ok1 else 'FAIL'} — streaming with tool ({len(tokens)} tokens)")

    # T2: No-tool streaming
    tokens2 = []
    answers2 = []
    tc2 = []
    try:
        for tok in agent.stream(
            "Tell me a short joke.",
            on_answer=lambda a: answers2.append(a),
            on_tool_call=lambda tn, ti: tc2.append(tn),
        ):
            tokens2.append(tok)
        ok2 = len(tokens2) > 0 and len(answers2) > 0
    except Exception:
        ok2 = False
    tests_passed += ok2
    details.append({"test": "stream_no_tool", "passed": ok2})
    print(f"  {'PASS' if ok2 else 'FAIL'} — no-tool streaming ({len(tokens2)} tokens)")

    # T3: Token accumulation
    tokens3 = []
    answers3 = []
    try:
        for tok in agent.stream(
            "What is 8 * 9?",
            on_answer=lambda a: answers3.append(a),
        ):
            tokens3.append(tok)
        full = "".join(tokens3)
        ok3 = len(tokens3) > 0 and ("72" in full or any("72" in a for a in answers3))
    except Exception:
        ok3 = False
    tests_passed += ok3
    details.append({"test": "token_accumulation", "passed": ok3})
    print(f"  {'PASS' if ok3 else 'FAIL'} — token accumulation")

    return {"passed": tests_passed, "total": 3, "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


def _test_multi_agent(model, model_name):
    """Test multi-agent pipeline — 3 tests (manual pipeline, routing, synthesis)."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.core.router import SubAgentRouter
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.python_repl import PythonREPL

    tests_passed = 0
    details = []

    # T1: Manual pipeline (A→B→C)
    try:
        agent_a = Agent(AgentConfig(
            name="QGen", model=model, tools=[],
            system_prompt="Create ONE specific math word problem about compound interest with numbers. Output ONLY the question.",
            max_iterations=1, enable_sub_agents=False, enable_memory=False,
        ))
        resp_a = agent_a.run("Create a math problem about compound interest.")
        question = resp_a.output.strip() if resp_a.output else ""

        agent_b = Agent(AgentConfig(
            name="Solver", model=model, tools=[Calculator(), PythonREPL()],
            system_prompt="Solve the math problem using calculator. Show your work.",
            max_iterations=8, enable_sub_agents=False, enable_memory=False,
        ))
        resp_b = agent_b.run(f"Solve: {question}")
        solution = resp_b.output.strip() if resp_b.output else ""

        agent_c = Agent(AgentConfig(
            name="Summary", model=model, tools=[],
            system_prompt="Summarize the Q&A in 2-3 sentences.",
            max_iterations=1, enable_sub_agents=False, enable_memory=False,
        ))
        resp_c = agent_c.run(f"Q: {question}\nA: {solution}")
        summary = resp_c.output.strip() if resp_c.output else ""

        ok1 = (resp_a.success and len(question) > 10 and
               resp_b.success and len(solution) > 10 and
               resp_c.success and len(summary) > 10)
    except Exception as e:
        ok1 = False
        print(f"    Pipeline error: {e}")
    tests_passed += ok1
    details.append({"test": "manual_pipeline", "passed": ok1})
    print(f"  {'PASS' if ok1 else 'FAIL'} — manual pipeline")

    # T2: Router decision — use user-explicit trigger to test sub-agent routing
    try:
        router = SubAgentRouter()
        simple = router.route("What is 2 + 2?")
        # User-explicit trigger should always activate sub-agents
        explicit_task = router.route(
            "Use sub-agents to calculate compound interest for $10k at 5% for 10 years "
            "and compare with simple interest."
        )
        ok2 = (not simple.use_sub_agents and explicit_task.use_sub_agents)
    except Exception:
        ok2 = False
    tests_passed += ok2
    details.append({"test": "routing", "passed": ok2})
    print(f"  {'PASS' if ok2 else 'FAIL'} — routing decision")

    # T3: Synthesis quality
    try:
        synth_agent = Agent(AgentConfig(
            name="Synth", model=model, tools=[],
            system_prompt="Synthesize information into a clear summary.",
            max_iterations=1, enable_sub_agents=False, enable_memory=False,
        ))
        resp = synth_agent.run(
            "Synthesize: Compound interest on $10k at 5% for 10yr = $16,288.95. "
            "Simple interest = $15,000. Compound earns $1,288.95 more."
        )
        out = (resp.output or "").lower()
        ok3 = resp.success and ("compound" in out or "16,288" in out or "16288" in out) and len(resp.output) > 30
    except Exception:
        ok3 = False
    tests_passed += ok3
    details.append({"test": "synthesis", "passed": ok3})
    print(f"  {'PASS' if ok3 else 'FAIL'} — synthesis quality")

    return {"passed": tests_passed, "total": 3, "details": details, "status": "", "error": None, "time": 0, "agent": "", "model": ""}


# ── Main ────────────────────────────────────────────────────────────────

AGENTS = [
    "qa_agent",
    "calculator_agent",
    "advanced_multi_tool_agent",
    "file_operations_agent",
    "coding_agent",
    "conversational_agent",
    "error_recovery_agent",
    "data_processing_agent",
    "advanced_streaming_agent",
    "multi_agent_pipeline",
]


def main():
    parser = argparse.ArgumentParser(description="Sweep a single model across all example agents")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--agents", nargs="*", default=None, help="Specific agents to test (default: all)")
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    model_name = args.model

    print(f"\n{'='*70}")
    print("Model Compatibility Sweep")
    print(f"Model: {model_name}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")
    print(f"{'='*70}")

    # Load model once
    print(f"\nLoading model {model_name}...")
    t0 = time.time()
    from effgen import load_model
    model = load_model(model_name)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    agents_to_test = args.agents if args.agents else AGENTS
    all_results = []

    for agent_name in agents_to_test:
        print(f"\n{'─'*70}")
        print(f"Testing: {agent_name} on {model_name}")
        print(f"{'─'*70}")

        result = run_agent_test(agent_name, model, model_name, gpu)
        all_results.append(result)

        print(f"\n  => {agent_name}: {result['status']} ({result['passed']}/{result['total']}) in {result['time']:.1f}s")
        if result.get("error"):
            print(f"  => ERROR: {result['error'][:200]}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY — {model_name}")
    print(f"{'='*70}")

    pass_count = sum(1 for r in all_results if r["status"] == "PASS")
    partial_count = sum(1 for r in all_results if r["status"] == "PARTIAL")
    fail_count = sum(1 for r in all_results if r["status"] == "FAIL")

    for r in all_results:
        print(f"  {r['status']:8s} — {r['agent']} ({r['passed']}/{r['total']}, {r['time']:.1f}s)")

    print(f"\nTotal: {pass_count} PASS, {partial_count} PARTIAL, {fail_count} FAIL")

    # Save results
    output_data = {
        "model": model_name,
        "gpu": gpu,
        "load_time": load_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {"pass": pass_count, "partial": partial_count, "fail": fail_count},
        "results": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")

    # Unload
    model.unload()
    print("Model unloaded. Done.")


if __name__ == "__main__":
    main()
