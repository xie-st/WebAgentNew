"""
Coding prompt: Docstring Fill — zero-shot variant.

Generates Google-style docstrings for undocumented Python functions.

Inputs: code (str), style (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "minLength": 1},
        "style": {
            "type": "string",
            "enum": ["google", "numpy", "sphinx"],
        },
    },
    "required": ["code"],
}

_FIXTURE = {
    "code": (
        "def merge_sorted(a, b):\n"
        "    result = []\n"
        "    i = j = 0\n"
        "    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n"
        "            result.append(a[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            result.append(b[j])\n"
        "            j += 1\n"
        "    result.extend(a[i:])\n"
        "    result.extend(b[j:])\n"
        "    return result\n"
    ),
    "style": "google",
}

_STYLE_EXAMPLES = {
    "google": (
        '"""Short one-line summary.\n\n'
        "Args:\n"
        "    param_name (type): Description.\n\n"
        "Returns:\n"
        "    type: Description.\n\n"
        "Raises:\n"
        '    ExceptionType: When and why it is raised.\n"""'
    ),
    "numpy": (
        '"""Short one-line summary.\n\n'
        "Parameters\n"
        "----------\n"
        "param_name : type\n"
        "    Description.\n\n"
        "Returns\n"
        "-------\n"
        "type\n"
        '    Description.\n"""'
    ),
    "sphinx": (
        '"""Short one-line summary.\n\n'
        ":param param_name: Description.\n"
        ":type param_name: type\n"
        ":returns: Description.\n"
        ":rtype: type\n"
        ':raises ExceptionType: When raised.\n"""'
    ),
}


def _docstring_fill_zero_shot(code: str, style: str = "google") -> str:
    style = style.lower() if style else "google"
    if style not in _STYLE_EXAMPLES:
        style = "google"
    example = _STYLE_EXAMPLES[style]
    return (
        "You are an expert Python technical writer.\n\n"
        f"Add {style.capitalize()}-style docstrings to every function and class in the code below "
        "that is currently missing one.\n\n"
        f"Use this docstring format:\n{example}\n\n"
        "Rules:\n"
        "  - Infer parameter types and return types from usage patterns and names.\n"
        "  - Include an 'Args' (or equivalent) section only if the function has parameters.\n"
        "  - Include a 'Returns' section only if the function returns a non-None value.\n"
        "  - Include a 'Raises' section only if exceptions are explicitly raised.\n"
        "  - Preserve all existing code exactly — only add docstrings, change nothing else.\n"
        "  - Output the full updated code, no markdown fences, no extra commentary.\n\n"
        f"Code:\n```python\n{code}\n```"
    )


docstring_fill_v1 = LibraryPrompt(
    name="coding.docstring_fill.v1",
    domain="coding",
    variant="zero_shot",
    description=(
        "Zero-shot docstring generator: adds Google/NumPy/Sphinx-style docstrings "
        "to undocumented Python functions."
    ),
    template=_docstring_fill_zero_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r'(Args:|Parameters|:param|""")',
    },
    tags=["coding", "docstring", "zero_shot", "documentation"],
)

registry.register(docstring_fill_v1)
