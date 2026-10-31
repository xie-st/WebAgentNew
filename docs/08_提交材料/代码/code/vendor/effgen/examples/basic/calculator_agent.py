#!/usr/bin/env python3
"""
effGen — Calculator Agent (Single Tool)

A math agent using the Calculator and PythonREPL tools via the ReAct loop.
Demonstrates the core tool-calling pipeline: model produces Thought/Action/Action Input,
framework parses them, tool executes, observation returned.

Recommended models:
  - Qwen/Qwen2.5-3B-Instruct (3B)      — excellent quality (default)
  - Qwen/Qwen2.5-7B-Instruct (7B)      — best quality
  - microsoft/Phi-4-mini-instruct       — very accurate
  - google/gemma-3-4b-it                — good quality
  - Qwen/Qwen2.5-1.5B-Instruct (1.5B)  — good quality, fast
  - meta-llama/Llama-3.2-3B-Instruct    — fast, concise

Tools used: Calculator (primary), PythonREPL (fallback)

Usage:
  # Run calculator demo (default model: Qwen2.5-3B-Instruct)
  CUDA_VISIBLE_DEVICES=0 python examples/calculator_agent.py

  # Specify a model
  CUDA_VISIBLE_DEVICES=0 python examples/calculator_agent.py --model Qwen/Qwen2.5-7B-Instruct

  # Interactive mode
  CUDA_VISIBLE_DEVICES=0 python examples/calculator_agent.py --interactive
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure effgen is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.presets import create_agent


def demo_calculator(agent):
    """Demonstrate calculator tool usage with various math tasks."""
    tasks = [
        ("Simple arithmetic", "What is 247 * 83?", "20501"),
        ("Math function", "What is the square root of 144?", "12"),
        ("Multi-step expression", "What is (15 + 27) * 3 - 10?", "116"),
        ("Natural language math", "Add 5 and 3, then multiply the result by 2.", "16"),
        ("Direct computation", "Use the calculator to compute 2+2.", "4"),
    ]

    results = []
    for desc, question, expected in tasks:
        print(f"\n{'='*50}")
        print(f"Test: {desc}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")

        t0 = time.time()
        resp = agent.run(question)
        dt = time.time() - t0

        output = resp.output if resp.output else ""
        passed = expected.lower() in output.lower()

        print(f"A: {output}")
        print(f"Tool calls: {resp.tool_calls}, Iterations: {resp.iterations}, Time: {dt:.1f}s")
        print(f"Result: {'PASS' if passed else 'CHECK'}")
        results.append((desc, passed, dt))

    print(f"\n{'='*50}")
    print("Summary:")
    for desc, passed, dt in results:
        print(f"  {'PASS' if passed else 'CHECK':5s} — {desc} ({dt:.1f}s)")
    pass_count = sum(1 for _, p, _ in results if p)
    print(f"\n{pass_count}/{len(results)} passed")


def interactive_mode(agent):
    """Interactive math calculator chat."""
    print("\n--- Interactive Calculator (type 'quit' to exit) ---")
    print("Ask any math question. The agent will use Calculator or PythonREPL.")
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
    parser = argparse.ArgumentParser(description="effGen Calculator Agent Example")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive math chat")
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print("effGen — Calculator Agent")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Create math agent (Calculator + PythonREPL)
    agent = create_agent("math", model)
    print(f"Math agent created with tools: {list(agent.tools.keys())}")

    try:
        if args.interactive:
            interactive_mode(agent)
        else:
            demo_calculator(agent)
    finally:
        agent.close()
        model.unload()
        print("\nDone.")


if __name__ == "__main__":
    main()
