"""Canonical parity tasks for effGen backend interchangeability tests.

Two complementary tasks let the parity suite check different guarantees without
conflating them:

* **Answer correctness** — ``run_canonical_task`` asks ``(17 * 23) + sqrt(144)``
  (expected ``403``). Every backend must reach the right answer, *by whatever
  path it chooses*. A small model leans on the Calculator; a capable reasoning
  model may do the trivial arithmetic in its head. Both are correct, so this
  task does not require a tool call.

* **Tool calling actually works** — ``run_tool_required_task`` asks for a value
  that is impossible to know without calling the tool (an opaque inventory count
  behind a fictitious SKU). The only way to answer is to invoke the tool, so a
  passing run is direct proof that the backend's tool-calling path is wired up —
  even for a model clever enough to shortcut arithmetic.
"""

from __future__ import annotations

CANONICAL_QUESTION = "What is (17 * 23) + sqrt(144)?"
CANONICAL_ANSWER = "403"

# A value that cannot be derived, guessed, or computed — it only exists inside
# the tool below, so reaching it in the answer means the tool was really called.
TOOL_REQUIRED_SENTINEL = 58219
TOOL_REQUIRED_QUESTION = (
    "Use the lookup_inventory_count tool to find the current inventory count for "
    "SKU 'QX-9000', then reply with only that number."
)

# Calculator tool schema in OpenAI function-calling format
CALCULATOR_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Supports +, -, *, /, **, sqrt(), and standard Python math."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '17 * 23' or 'math.sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
}


def _run_agent_task(adapter, question: str, tools: list, strategy: str) -> dict:
    """Run *question* against *adapter* with *tools* and return a result dict.

    Returns a dict with keys ``answer_text``, ``tool_calls``, ``latency_ms`` and
    ``error`` (``answer_correct`` is left to the caller, which knows the task's
    success criterion).
    """
    import time

    from effgen import Agent
    from effgen.core.agent import AgentConfig

    t0 = time.monotonic()
    error = None
    answer_text = ""
    tool_calls_count = 0

    agent = None
    try:
        agent = Agent(
            config=AgentConfig(
                name="parity_agent",
                model=adapter,
                tools=tools,
                max_iterations=8,
                temperature=0.0,
                enable_memory=False,
                enable_sub_agents=False,
                tool_calling_mode=strategy if strategy != "react" else "react",
            )
        )
        result = agent.run(question)
        answer_text = result.output or ""
        tool_calls_count = result.tool_calls
        # Propagate agent-level errors (e.g., Replicate billing) from metadata
        if not result.success and result.output:
            error = result.output[:500]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if agent is not None:
            agent.close()

    return {
        "answer_text": answer_text,
        "tool_calls": tool_calls_count,
        "latency_ms": (time.monotonic() - t0) * 1000,
        "error": error,
    }


def run_canonical_task(adapter, strategy: str = "react") -> dict:
    """Run the arithmetic answer-correctness task against *adapter*.

    Args:
        adapter: A loaded effGen BaseModel adapter instance.
        strategy: "react" or "native".

    Returns:
        dict with keys ``answer_correct`` (``403`` present), ``answer_text``,
        ``tool_calls``, ``latency_ms`` and ``error``.
    """
    from effgen.tools.builtin import Calculator

    out = _run_agent_task(adapter, CANONICAL_QUESTION, [Calculator()], strategy)
    out["answer_correct"] = CANONICAL_ANSWER in out["answer_text"]
    return out


def _make_inventory_tool():
    """Build a tool whose return value is unknowable without calling it."""
    from effgen import tool

    @tool
    def lookup_inventory_count(sku: str) -> int:
        """Look up the current warehouse inventory count for a product SKU.

        Args:
            sku: The product SKU to look up.
        """
        return TOOL_REQUIRED_SENTINEL

    return lookup_inventory_count


def _score_tool_required(out: dict) -> dict:
    # Tolerate digit grouping like "58,219" when checking the sentinel.
    normalized = out["answer_text"].replace(",", "").replace(" ", "")
    out["answer_correct"] = str(TOOL_REQUIRED_SENTINEL) in normalized
    return out


def run_tool_required_task(adapter, strategies: tuple[str, ...] = ("native", "react")) -> dict:
    """Run the must-call-a-tool task against *adapter*.

    The answer (an opaque inventory count) only exists inside the tool, so a
    correct answer is direct proof the backend invoked the tool. A backend
    passes if it calls the tool through **any** of its tool-calling modes: a
    capable reasoning model tends to use native function-calling reliably while
    a small instruct model leans on the scaffolded react loop, so trying the
    structured path first and falling back to react keeps the proof fair across
    the whole model-capability range.

    Returns:
        dict with keys ``answer_correct`` (the sentinel reached the answer),
        ``answer_text``, ``tool_calls``, ``latency_ms``, ``error`` and
        ``strategy`` (the mode whose result is returned).
    """
    last: dict = {}
    for strategy in strategies:
        out = _run_agent_task(
            adapter, TOOL_REQUIRED_QUESTION, [_make_inventory_tool()], strategy
        )
        out["strategy"] = strategy
        last = _score_tool_required(out)
        # Stop on a clean win, or on a provider-side error the caller should skip on.
        if out["error"] is None and out["tool_calls"] >= 1 and out["answer_correct"]:
            return out
        if out["error"] is not None:
            return out
    return last
