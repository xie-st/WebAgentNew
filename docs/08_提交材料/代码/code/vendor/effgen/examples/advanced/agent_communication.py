#!/usr/bin/env python3
"""
Agent-to-Agent Communication Example
=====================================

Demonstrates two agents communicating through a simple orchestrator:
1. A "researcher" agent answers factual questions
2. A "writer" agent takes the researcher's output and rewrites it

This pattern is useful for:
- Chain-of-expertise pipelines (research -> write -> review)
- Divide-and-conquer task decomposition
- Quality-control loops (generate -> verify)

Requires: a local model (e.g. Qwen2.5-1.5B-Instruct)
"""

from __future__ import annotations

from effgen import Agent, AgentConfig
from effgen.models import load_model
from effgen.tools.builtin import Calculator


def create_agent(name: str, model, system_prompt: str) -> Agent:
    """Helper to create an agent with a custom system prompt."""
    config = AgentConfig(
        name=name,
        model=model,
        tools=[Calculator()],
        max_iterations=5,
        system_prompt=system_prompt,
        enable_sub_agents=False,
    )
    return Agent(config=config)


def orchestrate(task: str):
    """
    Two-agent pipeline:
      User question -> Researcher -> Writer -> Final output
    """
    model = load_model("Qwen/Qwen2.5-1.5B-Instruct")

    researcher = create_agent(
        "researcher",
        model,
        "You are a research assistant. Answer questions with precise facts and numbers.",
    )
    writer = create_agent(
        "writer",
        model,
        "You are a skilled writer. Take the research provided and rewrite it "
        "as a clear, engaging paragraph for a general audience.",
    )

    # Step 1: Researcher answers the question
    print(f"[Orchestrator] Sending to researcher: {task}")
    research_result = researcher.run(task)
    print(f"[Researcher]   {research_result.output}\n")

    # Step 2: Writer rewrites the research
    writer_prompt = (
        f"Rewrite the following research into a clear paragraph:\n\n"
        f"{research_result.output}"
    )
    print("[Orchestrator] Sending to writer...")
    writer_result = writer.run(writer_prompt)
    print(f"[Writer]       {writer_result.output}\n")

    # Clean up
    researcher.close()
    writer.close()

    return writer_result.output


if __name__ == "__main__":
    question = "What is the population of Tokyo and how does it compare to New York?"
    print("=" * 60)
    print(f"Task: {question}")
    print("=" * 60 + "\n")
    final = orchestrate(question)
    print("=" * 60)
    print(f"Final output:\n{final}")
