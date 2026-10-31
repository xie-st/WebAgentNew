"""What a run's tool calls were, not just how many there were.

:attr:`~effgen.core.agent_response.AgentResponse.tool_calls` reports the calls a
run made: which tool, with what arguments, what came back, how long it took and
whether it failed. Error analysis and debugging both start from *which* call
went wrong, which a bare count cannot answer.

The field reads as a sequence and compares as a number, so both of these work
on the same object::

    for call in result.tool_calls:
        print(call.name, call.arguments, call.error)

    assert result.tool_calls == 2        # still compares like the old count
    if result.tool_calls > 0: ...

:attr:`ToolCallList.total` is the authoritative number of calls the run made.
It equals ``len()`` whenever the records were captured, and exceeds it only on
a path that reported a number without them — reading ``total`` rather than
``len()`` is therefore the safe way to ask "how many". It is spelled ``total``
rather than ``count`` because ``list.count(value)`` is a method this class
inherits and must keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: How much of a tool result to keep on the record. Whole results can be very
#: large (a fetched page, a file read), and a record is for reading back, not
#: for storing the payload again — the full text is in the execution trace.
MAX_RESULT_CHARS = 4000


@dataclass
class ToolCall:
    """One tool invocation a run made.

    Attributes:
        name: The tool's registered name.
        arguments: The input as the model supplied it. A string for the ReAct
            path, whose action input is text; a dict for native tool calling,
            whose arguments arrive parsed.
        result: What the tool returned, truncated to :data:`MAX_RESULT_CHARS`.
            None when the call failed before producing one.
        duration: Wall-clock seconds the call took, when it was measured.
        error: The failure message when the call did not succeed, else None.
        iteration: The 1-based loop iteration the call was made on, when known.
    """

    name: str
    arguments: Any = None
    result: str | None = None
    duration: float | None = None
    error: str | None = None
    iteration: int | None = None

    @property
    def ok(self) -> bool:
        """True when the call returned without an error."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return the record as plain data, for JSON output and saved runs."""
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "duration": self.duration,
            "error": self.error,
            "iteration": self.iteration,
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Rebuild a record from :meth:`to_dict` output, ignoring derived keys."""
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments"),
            result=data.get("result"),
            duration=data.get("duration"),
            error=data.get("error"),
            iteration=data.get("iteration"),
        )

    def __str__(self) -> str:
        state = "error" if self.error else "ok"
        return f"{self.name}({self.arguments!r}) -> {state}"


class ToolCallList(list):
    """The calls a run made, which also compares and casts like their count.

    A plain list of :class:`ToolCall`, with the number-like behaviour the field
    had before it carried records: ``== 3``, ``> 0``, ``int(...)`` and
    truthiness all read :attr:`count`. Code written against the old integer
    keeps working; code that iterates now gets the calls.

    Attributes:
        total: Calls the run made. Equals ``len(self)`` when the records were
            captured, and is larger only on a path that counted without them.
            Named ``total`` and not ``count`` so that ``list.count(value)``,
            which this class inherits, is not shadowed by an integer.
    """

    def __init__(self, records: "list[ToolCall] | None" = None, total: int | None = None) -> None:
        """
        Args:
            records: The captured calls, in the order they were made.
            total: How many calls the run made. Defaults to the number of
                records, and is only given separately by a path that counted
                calls without recording them.
        """
        super().__init__(records or [])
        self.total: int = len(self) if total is None else int(total)

    # ------------------------------------------------------------------
    # Number-like behaviour, so the pre-1.0 integer contract still holds
    # ------------------------------------------------------------------
    def __int__(self) -> int:
        return self.total

    def __index__(self) -> int:
        return self.total

    def __bool__(self) -> bool:
        return self.total > 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        if isinstance(other, int):
            return self.total == other
        return bool(list(self) == other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        return self.total < other if isinstance(other, int) else bool(list(self) < other)

    def __le__(self, other: Any) -> bool:
        return self.total <= other if isinstance(other, int) else bool(list(self) <= other)

    def __gt__(self, other: Any) -> bool:
        return self.total > other if isinstance(other, int) else bool(list(self) > other)

    def __ge__(self, other: Any) -> bool:
        return self.total >= other if isinstance(other, int) else bool(list(self) >= other)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, int):
            return self.total + other
        return ToolCallList(list(self) + list(other))

    __radd__ = __add__

    def __repr__(self) -> str:
        if not len(self):
            return f"ToolCallList(total={self.total})"
        return f"ToolCallList({[c.name for c in self]!r}, total={self.total})"

    # ------------------------------------------------------------------
    # Reading the calls
    # ------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """The tool names in call order, with repeats kept."""
        return [call.name for call in self]

    @property
    def failed(self) -> "ToolCallList":
        """Only the calls that reported an error."""
        return ToolCallList([call for call in self if not call.ok])

    def by_name(self, name: str) -> "ToolCallList":
        """Only the calls made to *name*."""
        return ToolCallList([call for call in self if call.name == name])

    def to_list(self) -> list[dict[str, Any]]:
        """Return the records as plain data, for JSON output and saved runs."""
        return [call.to_dict() for call in self]


def coerce_tool_calls(value: Any) -> ToolCallList:
    """Return *value* as a :class:`ToolCallList`, whatever shape it arrives in.

    Accepts an existing list, a count from a path that reports one without
    records, records already built, or the plain dicts a saved run reads back.
    """
    if isinstance(value, ToolCallList):
        return value
    if value is None:
        return ToolCallList()
    if isinstance(value, bool):
        return ToolCallList(total=int(value))
    if isinstance(value, int):
        return ToolCallList(total=value)
    if isinstance(value, list | tuple):
        records = [
            item if isinstance(item, ToolCall)
            else ToolCall.from_dict(item) if isinstance(item, dict)
            else ToolCall(name=str(item))
            for item in value
        ]
        return ToolCallList(records)
    return ToolCallList()


def truncate_result(result: Any) -> str | None:
    """Return *result* as text short enough to keep on a record."""
    if result is None:
        return None
    text = result if isinstance(result, str) else str(result)
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + f"… ({len(text)} chars)"
    return text
