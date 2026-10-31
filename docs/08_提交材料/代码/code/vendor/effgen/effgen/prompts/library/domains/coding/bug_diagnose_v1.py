"""
Coding prompt: Bug Diagnosis — chain-of-thought variant.

Inputs: code (str), error_message (str), repro_steps (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "minLength": 1},
        "error_message": {"type": "string", "minLength": 1},
        "repro_steps": {"type": "string"},
    },
    "required": ["code", "error_message"],
}

_FIXTURE = {
    "code": (
        "def fibonacci(n):\n"
        "    if n == 0:\n"
        "        return 0\n"
        "    memo = {0: 0, 1: 1}\n"
        "    for i in range(2, n):\n"
        "        memo[i] = memo[i-1] + memo[i-2]\n"
        "    return memo[n]\n"
    ),
    "error_message": "KeyError: 2 when calling fibonacci(2)",
    "repro_steps": "Call fibonacci(2) — returns KeyError instead of 1.",
}


def _bug_diagnose_cot(
    code: str,
    error_message: str,
    repro_steps: str = "",
) -> str:
    repro_block = (
        f"Reproduction steps:\n{repro_steps.strip()}\n\n"
        if repro_steps.strip()
        else ""
    )
    return (
        "You are an expert software debugger. Use chain-of-thought reasoning to diagnose the bug below.\n\n"
        f"Error:\n{error_message.strip()}\n\n"
        f"{repro_block}"
        "Code under investigation:\n"
        f"```\n{code}\n```\n\n"
        "Think step by step:\n"
        "1. Read the error message carefully — what type of error is it and what does it say literally?\n"
        "2. Trace execution: walk through the code path that leads to the error condition.\n"
        "3. Identify the root cause — the exact line or expression that is wrong.\n"
        "4. Explain WHY the bug exists (off-by-one, wrong assumption, missing guard, etc.).\n"
        "5. Propose a minimal, correct fix with the corrected code snippet.\n"
        "6. Verify your fix: explain why the error will no longer occur after the change.\n\n"
        "Show your step-by-step reasoning before presenting the final fix."
    )


bug_diagnose_v1 = LibraryPrompt(
    name="coding.bug_diagnose.v1",
    domain="coding",
    variant="cot",
    description=(
        "Chain-of-thought bug diagnosis: traces execution, identifies root cause, "
        "and proposes a minimal fix."
    ),
    template=_bug_diagnose_cot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(root cause|fix|bug|error|line|step)",
    },
    tags=["coding", "debugging", "cot", "bug"],
)

registry.register(bug_diagnose_v1)
