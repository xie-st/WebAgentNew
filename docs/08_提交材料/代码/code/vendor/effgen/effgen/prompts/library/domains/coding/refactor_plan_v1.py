"""
Coding prompt: Refactor Plan — tool-augmented variant.

Instructs an agent to read source files before producing a refactoring plan.

Inputs: file_path (str), goals (str), language (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "minLength": 1},
        "goals": {"type": "string", "minLength": 1},
        "language": {"type": "string"},
        "code_snippet": {"type": "string"},
    },
    "required": ["file_path", "goals"],
}

_FIXTURE = {
    "file_path": "src/utils.py",
    "goals": "Improve readability and reduce code duplication",
    "language": "Python",
    "code_snippet": (
        "def calc_area_circle(r):\n"
        "    return 3.14159 * r * r\n\n"
        "def calc_area_square(s):\n"
        "    return s * s\n\n"
        "def calc_perimeter_circle(r):\n"
        "    return 2 * 3.14159 * r\n\n"
        "def calc_perimeter_square(s):\n"
        "    return 4 * s\n"
    ),
}


def _refactor_plan_tool(
    file_path: str,
    goals: str,
    language: str = "Python",
    code_snippet: str = "",
) -> str:
    lang = language or "Python"
    if code_snippet.strip():
        file_block = (
            f"Current code from {file_path}:\n"
            f"```{lang.lower()}\n{code_snippet.strip()}\n```\n\n"
        )
        tool_instruction = (
            "If you have access to the file_operations tool, call it with "
            f"operation='read' and path='{file_path}' to catch any recent changes.\n\n"
        )
    else:
        file_block = ""
        tool_instruction = (
            "Tool use required before planning:\n"
            f"- Use the file_operations tool with operation='read' and path='{file_path}'.\n"
            "- If the tool returns an error, ask the user to paste the relevant code.\n\n"
        )

    return (
        f"You are a senior {lang} engineer tasked with producing a detailed refactoring plan.\n\n"
        f"{tool_instruction}"
        f"{file_block}"
        f"Refactoring goals:\n{goals.strip()}\n\n"
        "Produce a structured refactoring plan with these sections:\n"
        "1. Summary of current issues — what makes the existing code hard to maintain or extend?\n"
        "2. Proposed changes — list each specific refactoring step (e.g., extract function, rename, introduce abstraction).\n"
        "3. Code sketch — show the key before/after snippets for the most impactful changes.\n"
        "4. Risk assessment — what could break and how to mitigate it?\n"
        "5. Suggested test plan — which tests to run or write to verify the refactoring is safe?\n\n"
        "Be concrete and precise. Reference line numbers or function names from the actual code."
    )


refactor_plan_v1 = LibraryPrompt(
    name="coding.refactor_plan.v1",
    domain="coding",
    variant="tool",
    description=(
        "Tool-augmented refactoring plan: reads the source file then produces a structured "
        "plan with risk assessment and test strategy."
    ),
    template=_refactor_plan_tool,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(refactor|plan|risk|test|change|improve)",
    },
    tags=["coding", "refactor", "tool", "plan"],
)

registry.register(refactor_plan_v1)
