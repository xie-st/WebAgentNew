"""
OpenAI native tool calling example.

Demonstrates:
- Using chat() with tools for multi-turn tool loops
- Parsing tool_calls from the response
- A complete tool loop (call → execute → return result → final answer)

Run:
    python examples/openai/tool_calling.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

from effgen.models.base import GenerationConfig
from effgen.models.openai_adapter import OpenAIAdapter

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A safe Python math expression using math module functions. "
                            "E.g. '2 ** 16', 'math.sqrt(12321)', '17 * 23 + math.log(100)'"
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": "Convert a value between common units.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit (km, miles, kg, lbs, celsius, fahrenheit)"},
                    "to_unit": {"type": "string", "description": "Target unit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool locally and return the result as a string."""
    if name == "calculate":
        try:
            # math module available, builtins restricted
            result = eval(args["expression"], {"math": math, "__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    elif name == "unit_convert":
        value = args["value"]
        conversions = {
            ("km", "miles"): lambda v: v * 0.621371,
            ("miles", "km"): lambda v: v * 1.60934,
            ("kg", "lbs"): lambda v: v * 2.20462,
            ("lbs", "kg"): lambda v: v / 2.20462,
            ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
            ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        }
        fn = conversions.get((args["from_unit"], args["to_unit"]))
        if fn:
            return f"{fn(value):.4f} {args['to_unit']}"
        return f"Unsupported conversion: {args['from_unit']} -> {args['to_unit']}"
    return f"Unknown tool: {name}"


def run_tool_loop(adapter: OpenAIAdapter, user_prompt: str) -> str:
    """Run a complete tool-calling loop until the model gives a final text answer."""
    messages = [{"role": "user", "content": user_prompt}]
    print(f"User: {user_prompt}")
    config = GenerationConfig(max_tokens=512)

    while True:
        result = adapter.chat(messages=messages, tools=TOOLS, config=config)

        tool_calls = result.metadata.get("tool_calls", [])
        if not tool_calls:
            # Model gave a final text answer
            print(f"Assistant: {result.text}")
            return result.text

        # Re-submit the assistant turn, in whatever shape this provider takes.
        messages.append(adapter.build_assistant_message(result))

        # Execute each tool and append its result. Every effGen adapter reports
        # a call in this shape and builds its own provider's result message, so
        # the whole loop is provider-agnostic: swap OpenAIAdapter for the Gemini
        # or Anthropic one and nothing here changes, even though their wire
        # formats differ (functionResponse parts, tool_result blocks).
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            print(f"  [Tool call] {fn_name}({fn_args})")
            tool_result = execute_tool(fn_name, fn_args)
            print(f"  [Tool result] {tool_result}")
            messages.append(
                adapter.build_tool_result_message(tc["id"], fn_name, tool_result)
            )


def main():
    adapter = OpenAIAdapter("gpt-5.4-nano")
    adapter.load()
    print(f"Model: {adapter.model_name}")
    print(f"Supports tools: {adapter.supports_tool_calling()}\n")

    questions = [
        "What is 2^16 + math.sqrt(12321)?",
        "Convert 42 km to miles, then convert 98.6 fahrenheit to celsius.",
        "What is (17 * 23) + math.sqrt(144) + 2**8?",
    ]

    for q in questions:
        print("-" * 60)
        run_tool_loop(adapter, q)
        print()

    print(f"Total cost: ${adapter.get_total_cost():.6f}")
    adapter.unload()


if __name__ == "__main__":
    main()
