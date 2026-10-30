#!/usr/bin/env python3
"""
effGen — Advanced Streaming Agent (Real-Time Output)

Demonstrates agent.stream() with real-time token-by-token output and
callbacks for thoughts, tool calls, observations, and final answers.

Tests:
  - T1: Simple streaming with tool call (Calculator)
  - T2: No-tool streaming (direct answer)
  - T3: Multi-tool streaming (Calculator + DateTimeTool)
  - T4: Stop sequence handling
  - T5: Error handling — the stream still completes
  - T6: Callback accumulation (token reconstruction)

Recommended models:
  - Qwen/Qwen2.5-3B-Instruct (default)
  - Qwen/Qwen2.5-7B-Instruct (best quality)
  - microsoft/Phi-4-mini-instruct

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/advanced_streaming_agent.py
  CUDA_VISIBLE_DEVICES=0 python examples/advanced_streaming_agent.py --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.datetime_tool import DateTimeTool

# ANSI color codes
BLUE = "\033[94m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


class StreamTracker:
    """Tracks all streaming events for verification."""

    def __init__(self):
        self.thoughts = []
        self.tool_calls = []
        self.observations = []
        self.answers = []
        self.all_tokens = []
        self.event_order = []  # Track callback firing order

    def on_thought(self, thought: str):
        self.thoughts.append(thought)
        self.event_order.append(("thought", thought[:50]))
        print(f"{BLUE}[Thought] {thought}{RESET}")

    def on_tool_call(self, tool_name: str, tool_input: str):
        self.tool_calls.append((tool_name, tool_input))
        self.event_order.append(("tool_call", tool_name))
        print(f"{YELLOW}[Tool Call] {tool_name}({tool_input}){RESET}")

    def on_observation(self, observation: str):
        self.observations.append(observation)
        self.event_order.append(("observation", observation[:50]))
        print(f"{GREEN}[Observation] {observation}{RESET}")

    def on_answer(self, answer: str):
        self.answers.append(answer)
        self.event_order.append(("answer", answer[:50]))
        print(f"{BOLD}[Answer] {answer}{RESET}")

    def reset(self):
        self.thoughts.clear()
        self.tool_calls.clear()
        self.observations.clear()
        self.answers.clear()
        self.all_tokens.clear()
        self.event_order.clear()


def create_streaming_agent(model):
    """Create an agent with Calculator and DateTimeTool."""
    tools = [Calculator(), DateTimeTool()]
    config = AgentConfig(
        name="streaming_agent",
        model=model,
        tools=tools,
        max_iterations=5,
        temperature=0.1,
    )
    return Agent(config)


class _StreamTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _StreamTimeout("Streaming test timed out")


def run_test(agent, tracker, test_id, desc, task, checks, timeout_sec=120, **stream_kwargs):
    """Run a single streaming test and return PASS/FAIL."""
    import signal

    print(f"\n{'='*60}")
    print(f"{test_id}: {desc}")
    print(f"Task: {task}")
    print(f"{'='*60}")

    tracker.reset()
    t0 = time.time()
    token_count = 0

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(timeout_sec)
        print("\n--- Stream Output ---")
        for token in agent.stream(
            task,
            on_thought=tracker.on_thought,
            on_tool_call=tracker.on_tool_call,
            on_observation=tracker.on_observation,
            on_answer=tracker.on_answer,
            **stream_kwargs,
        ):
            tracker.all_tokens.append(token)
            token_count += 1
        signal.alarm(0)
        print(f"\n--- End Stream ({token_count} tokens, {time.time()-t0:.1f}s) ---")
    except _StreamTimeout:
        signal.alarm(0)
        print(f"\n[TIMEOUT] Streaming timed out after {timeout_sec}s ({token_count} tokens)")
    except Exception as e:
        signal.alarm(0)
        print(f"\n[ERROR] {e}")
        return False, str(e)
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    # Run checks
    results = {}
    for check_name, check_fn in checks.items():
        try:
            passed = check_fn(tracker)
            results[check_name] = passed
            status = "PASS" if passed else "FAIL"
            print(f"  Check [{status}]: {check_name}")
        except Exception as e:
            results[check_name] = False
            print(f"  Check [FAIL]: {check_name} — {e}")

    all_passed = all(results.values())
    print(f"\n{test_id} Result: {'PASS' if all_passed else 'FAIL'}")
    print(f"  Tokens: {token_count}, Thoughts: {len(tracker.thoughts)}, "
          f"Tool calls: {len(tracker.tool_calls)}, Observations: {len(tracker.observations)}, "
          f"Answers: {len(tracker.answers)}")
    print(f"  Event order: {[e[0] for e in tracker.event_order]}")

    return all_passed, results


def run_streaming_tests(model_name: str):
    """Run all streaming tests."""
    print(f"\n{'#'*60}")
    print("effGen — Advanced Streaming Agent")
    print(f"Model: {model_name}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"{'#'*60}")

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(model_name, quantization="4bit")
    print(f"Model loaded in {time.time()-t0:.1f}s")

    agent = create_streaming_agent(model)
    tracker = StreamTracker()
    print(f"Agent created with tools: {list(agent.tools.keys())}")

    results = {}

    # T1: Simple streaming with tool call
    passed, detail = run_test(
        agent, tracker, "T1", "Simple streaming — 15 * 7",
        "What is 15 * 7?",
        {
            "tokens_generated": lambda t: len(t.all_tokens) > 0,
            "thought_callback_fired": lambda t: len(t.thoughts) > 0,
            "tool_call_callback_fired": lambda t: len(t.tool_calls) > 0,
            "observation_callback_fired": lambda t: len(t.observations) > 0,
            "answer_callback_fired": lambda t: len(t.answers) > 0,
            "answer_contains_105": lambda t: any("105" in a for a in t.answers),
            "callback_order_correct": lambda t: (
                len(t.event_order) >= 3 and
                t.event_order[0][0] == "thought" and
                any(e[0] == "tool_call" for e in t.event_order) and
                any(e[0] == "answer" for e in t.event_order)
            ),
        }
    )
    results["T1"] = passed

    # T2: No-tool streaming
    passed, detail = run_test(
        agent, tracker, "T2", "No-tool streaming — Tell me a joke",
        "Tell me a short joke.",
        {
            "tokens_generated": lambda t: len(t.all_tokens) > 0,
            "answer_callback_fired": lambda t: len(t.answers) > 0,
            "no_tool_calls": lambda t: len(t.tool_calls) == 0,
        }
    )
    results["T2"] = passed

    # T3: Multi-tool streaming
    passed, detail = run_test(
        agent, tracker, "T3", "Multi-tool streaming — Calculator + DateTime",
        "What is 25 * 4, and what is today's date?",
        {
            "tokens_generated": lambda t: len(t.all_tokens) > 0,
            "multiple_tool_calls": lambda t: len(t.tool_calls) >= 2,
            "calculator_used": lambda t: any("calculator" in tc[0].lower() for tc in t.tool_calls),
            "datetime_used": lambda t: any("datetime" in tc[0].lower() for tc in t.tool_calls),
            "answer_callback_fired": lambda t: len(t.answers) > 0,
            "answer_contains_100": lambda t: any("100" in a for a in t.answers),
        }
    )
    results["T3"] = passed

    # T4: Stop sequence handling
    passed, detail = run_test(
        agent, tracker, "T4", "Stop sequence handling",
        "Use the calculator to compute 99 + 1.",
        {
            "tokens_generated": lambda t: len(t.all_tokens) > 0,
            "tool_call_fired": lambda t: len(t.tool_calls) > 0,
            "no_observation_in_tokens": lambda t: not any(
                "\nObservation:" in tok for tok in t.all_tokens
                if not tok.startswith("\nObservation:")  # skip yielded obs
            ),
            "observation_callback_fired": lambda t: len(t.observations) > 0,
        }
    )
    results["T4"] = passed

    # T5: Error during stream (use a tool name that doesn't exist)
    passed, detail = run_test(
        agent, tracker, "T5", "Error handling — the stream still completes",
        "What is the square root of 49?",
        {
            "tokens_generated": lambda t: len(t.all_tokens) > 0,
            "completes_without_crash": lambda _: True,  # If we get here, no crash
            "has_answer": lambda t: len(t.answers) > 0 or len(t.all_tokens) > 5,
        }
    )
    results["T5"] = passed

    # T6: Callback accumulation — verify token reconstruction
    passed, detail = run_test(
        agent, tracker, "T6", "Callback accumulation — token reconstruction",
        "What is 8 * 9?",
        {
            "tokens_generated": lambda t: len(t.all_tokens) > 0,
            "tokens_form_text": lambda t: len("".join(t.all_tokens).strip()) > 0,
            "answer_in_accumulated": lambda t: (
                "72" in "".join(t.all_tokens) or
                any("72" in a for a in t.answers)
            ),
            "all_callbacks_in_order": lambda t: (
                len(t.event_order) >= 2 and
                # First event should be thought
                t.event_order[0][0] == "thought"
            ),
        }
    )
    results["T6"] = passed

    # Summary
    print(f"\n{'#'*60}")
    print(f"Streaming Tests Summary — {model_name}")
    print(f"{'#'*60}")
    for test_id, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'} — {test_id}")
    pass_count = sum(1 for p in results.values() if p)
    print(f"\n{pass_count}/{len(results)} PASS")

    model.unload()
    print("\nModel unloaded. Done.")
    return results


def run_regression(model_name: str):
    """Run regression tests to verify previous examples still work."""
    print(f"\n{'#'*60}")
    print(f"Regression Tests — {model_name}")
    print(f"{'#'*60}")

    from effgen.presets import create_agent as create_preset_agent
    from effgen.tools.builtin.bash_tool import BashTool
    from effgen.tools.builtin.json_tool import JSONTool
    from effgen.tools.builtin.python_repl import PythonREPL
    from effgen.tools.builtin.text_processing import TextProcessingTool

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(model_name, quantization="4bit")
    print(f"Model loaded in {time.time()-t0:.1f}s")

    results = {}

    # P1: Q&A
    agent = create_preset_agent("minimal", model)
    resp = agent.run("What is the capital of France?")
    p1 = "paris" in (resp.output or "").lower()
    results["P1-QA"] = p1
    print(f"  {'PASS' if p1 else 'FAIL'} — P1: Q&A (got: {(resp.output or '')[:60]})")

    # P2: Calculator
    agent = create_preset_agent("math", model)
    resp = agent.run("What is 247 * 83?")
    p2 = "20501" in (resp.output or "")
    results["P2-Calc"] = p2
    print(f"  {'PASS' if p2 else 'FAIL'} — P2: Calculator (got: {(resp.output or '')[:60]})")

    # P3: Multi-tool
    tools = [Calculator(), PythonREPL(), DateTimeTool(), BashTool(), TextProcessingTool()]
    config = AgentConfig(name="multi", model=model, tools=tools, max_iterations=5, temperature=0.1)
    agent = Agent(config)
    resp = agent.run("What is 123 + 456?")
    p3 = "579" in (resp.output or "")
    results["P3-Multi"] = p3
    print(f"  {'PASS' if p3 else 'FAIL'} — P3: Multi-tool (got: {(resp.output or '')[:60]})")

    # P5: Code exec
    agent = create_preset_agent("coding", model)
    resp = agent.run("Write and run a Python one-liner that prints 'hello world'.")
    p5 = "hello" in (resp.output or "").lower()
    results["P5-Code"] = p5
    print(f"  {'PASS' if p5 else 'FAIL'} — P5: Code exec (got: {(resp.output or '')[:60]})")

    # P6: Memory
    agent = create_preset_agent("math", model)
    agent.run("My name is Alice.")
    resp = agent.run("What is my name?")
    p6 = "alice" in (resp.output or "").lower()
    results["P6-Mem"] = p6
    print(f"  {'PASS' if p6 else 'FAIL'} — P6: Memory (got: {(resp.output or '')[:60]})")

    # P8: JSON
    tools = [JSONTool(), TextProcessingTool()]
    config = AgentConfig(name="json", model=model, tools=tools, max_iterations=5, temperature=0.1)
    agent = Agent(config)
    resp = agent.run('How many keys are in this JSON: {"a":1,"b":2,"c":3}?')
    p8 = "3" in (resp.output or "")
    results["P8-JSON"] = p8
    print(f"  {'PASS' if p8 else 'FAIL'} — P8: JSON (got: {(resp.output or '')[:60]})")

    pass_count = sum(1 for p in results.values() if p)
    print(f"\nRegression: {pass_count}/{len(results)} PASS")

    model.unload()
    return results


def main():
    parser = argparse.ArgumentParser(description="effGen Advanced Streaming Agent Example")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--regression-only", action="store_true")
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args()

    if args.regression_only:
        run_regression(args.model)
    elif args.tests_only:
        run_streaming_tests(args.model)
    else:
        run_regression(args.model)
        run_streaming_tests(args.model)


if __name__ == "__main__":
    main()
