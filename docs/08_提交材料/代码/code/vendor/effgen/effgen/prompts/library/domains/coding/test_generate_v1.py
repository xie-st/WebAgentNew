"""
Coding prompt: Test Generation — few-shot variant.

Two exemplar tests are included to guide output style.
Live eval asserts that ast.parse() succeeds on the generated Python code.

Inputs: code (str), framework (str), function_name (str)
"""

from __future__ import annotations

import ast

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "minLength": 1},
        "framework": {
            "type": "string",
            "enum": ["pytest", "unittest"],
        },
        "function_name": {"type": "string"},
    },
    "required": ["code", "framework"],
}

_FIXTURE = {
    "code": (
        "def clamp(value: float, lo: float, hi: float) -> float:\n"
        "    \"\"\"Return value clamped to [lo, hi].\"\"\"\n"
        "    if lo > hi:\n"
        "        raise ValueError(f'lo ({lo}) must be <= hi ({hi})')\n"
        "    return max(lo, min(value, hi))\n"
    ),
    "framework": "pytest",
    "function_name": "clamp",
}

_EXEMPLAR_1 = '''\
# Example: testing a simple add() function with pytest
def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

def test_add_zero():
    assert add(0, 0) == 0
'''

_EXEMPLAR_2 = '''\
# Example: testing a safe_divide() function with pytest
import pytest

def test_safe_divide_normal():
    assert safe_divide(10, 2) == 5.0

def test_safe_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        safe_divide(1, 0)

def test_safe_divide_negative():
    assert safe_divide(-6, 3) == -2.0
'''


def _ast_parse_check(output: str) -> bool | str:
    """Assert that the model output is syntactically valid Python 3.11."""
    code = output.strip()
    # Strip markdown fences if present
    if code.startswith("```"):
        lines = code.splitlines()
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        code = "\n".join(lines[start:end])
    try:
        ast.parse(code)
        return True
    except SyntaxError as exc:
        return f"SyntaxError in generated code: {exc}"


def _test_generate_few_shot(
    code: str,
    framework: str = "pytest",
    function_name: str = "",
) -> str:
    fn_clause = (
        f"Focus on the function `{function_name}`."
        if function_name.strip()
        else "Cover all public functions."
    )
    fw = framework.lower()
    import_line = "import pytest" if fw == "pytest" else "import unittest"
    return (
        f"You are an expert Python test engineer writing {framework} tests.\n\n"
        "Here are two examples of well-written test suites to follow:\n\n"
        f"--- Example 1 ---\n{_EXEMPLAR_1}\n"
        f"--- Example 2 ---\n{_EXEMPLAR_2}\n"
        "--- End of examples ---\n\n"
        f"Now write a complete test suite for the code below. {fn_clause}\n\n"
        "Requirements:\n"
        f"  - Use {framework}.\n"
        f"  - Start with `{import_line}` if needed.\n"
        "  - Test normal cases, boundary conditions, and error/exception paths.\n"
        "  - Each test function must have a descriptive name starting with `test_`.\n"
        "  - IMPORTANT: pytest.raises() requires parentheses — write `with pytest.raises(ValueError):` NOT `with pytest.raises ValueError:`.\n"
        "  - Do NOT include the source function — only the test code.\n"
        "  - Output raw Python only, no markdown fences, no explanatory text.\n\n"
        f"Code to test:\n```python\n{code}\n```"
    )


test_generate_v1 = LibraryPrompt(
    name="coding.test_generate.v1",
    domain="coding",
    variant="few_shot",
    description=(
        "Few-shot test generation: two exemplar pytest suites guide output style. "
        "Live eval asserts ast.parse() passes on generated Python."
    ),
    template=_test_generate_few_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _ast_parse_check,
    },
    tags=["coding", "testing", "few_shot", "pytest", "ast"],
)

registry.register(test_generate_v1)
