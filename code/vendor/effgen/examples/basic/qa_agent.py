#!/usr/bin/env python3
"""
effGen — Q&A Agent (No Tools)

A minimal Q&A agent using direct inference — no tools, no ReAct loop.
The simplest agent path: model loads, prompt goes in, answer comes out.

Recommended models:
  - Qwen/Qwen2.5-0.5B-Instruct (0.5B)  — fast, concise answers
  - Qwen/Qwen2.5-1.5B-Instruct (1.5B)  — good quality, fast
  - Qwen/Qwen2.5-3B-Instruct (3B)      — excellent quality (default)
  - Qwen/Qwen2.5-7B-Instruct (7B)      — best quality in Qwen2.5 family
  - meta-llama/Llama-3.2-3B-Instruct    — fast, concise
  - microsoft/Phi-4-mini-instruct       — good quality, slower (~14s/query)
  - google/gemma-3-4b-it                — good quality, slowest (~25s/query)

Usage:
  # Single question (default model: Qwen2.5-3B-Instruct)
  CUDA_VISIBLE_DEVICES=0 python examples/qa_agent.py

  # Specify a model
  CUDA_VISIBLE_DEVICES=0 python examples/qa_agent.py --model Qwen/Qwen2.5-7B-Instruct

  # Interactive mode
  CUDA_VISIBLE_DEVICES=0 python examples/qa_agent.py --interactive

  # Multi-turn demo
  CUDA_VISIBLE_DEVICES=0 python examples/qa_agent.py --multi-turn
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


def demo_single_qa(agent):
    """Demonstrate single Q&A."""
    questions = [
        "What is the capital of France?",
        "Explain photosynthesis in 3 sentences.",
        "What are the three laws of thermodynamics?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        t0 = time.time()
        resp = agent.run(q)
        dt = time.time() - t0
        print(f"A: {resp.output}")
        print(f"   ({dt:.1f}s, success={resp.success})")


def demo_multi_turn(agent):
    """Demonstrate multi-turn memory."""
    turns = [
        "My name is Alice and I work as a data scientist at a tech startup.",
        "What is my name and what do I do?",
        "I prefer Python over R for data analysis.",
        "Which programming language should I use for my next project?",
    ]

    print("\n--- Multi-Turn Memory Demo ---")
    for i, msg in enumerate(turns, 1):
        print(f"\n[Turn {i}] User: {msg}")
        t0 = time.time()
        resp = agent.run(msg)
        dt = time.time() - t0
        print(f"[Turn {i}] Agent: {resp.output}")
        print(f"   ({dt:.1f}s)")


def interactive_mode(agent):
    """Interactive chat loop."""
    print("\n--- Interactive Mode (type 'quit' to exit) ---")
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
            print(f"   ({dt:.1f}s)")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="effGen Q&A Agent Example")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--multi-turn", action="store_true", help="Multi-turn memory demo")
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print("effGen — Q&A Agent")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Create minimal agent (no tools)
    agent = create_agent(
        "minimal",
        model,
        system_prompt="You are a helpful AI assistant. Answer questions directly and concisely.",
    )

    if args.interactive:
        interactive_mode(agent)
    elif args.multi_turn:
        demo_multi_turn(agent)
    else:
        demo_single_qa(agent)

    # Cleanup
    model.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
