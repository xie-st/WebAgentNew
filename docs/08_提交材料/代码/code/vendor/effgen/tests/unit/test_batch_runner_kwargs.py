"""BatchRunner forwards per-run kwargs (e.g. max_tokens) to ``agent.run``.

The ``effgen batch`` CLI grew a ``--max-tokens`` flag so token-heavy / reasoning
models are not silently emptied out; that value rides through BatchRunner into
each ``agent.run`` call. This guards that plumbing without any live model.
"""

from __future__ import annotations

from effgen.core.agent import AgentResponse
from effgen.core.batch import BatchConfig, BatchRunner


class _RecordingAgent:
    """Stands in for an Agent: records the kwargs each run() received."""

    def __init__(self) -> None:
        self.seen_kwargs: list[dict] = []
        self.closed = False

    def run(self, query: str, **kwargs) -> AgentResponse:
        self.seen_kwargs.append(kwargs)
        return AgentResponse(output=f"answer:{query}", success=True)

    def close(self) -> None:
        self.closed = True


def test_run_kwargs_forwarded_to_agent():
    agent = _RecordingAgent()
    runner = BatchRunner(agent)
    result = runner.run(
        ["a", "b"], config=BatchConfig(max_concurrency=1), max_tokens=4096,
    )
    assert result.succeeded == 2
    assert agent.seen_kwargs == [{"max_tokens": 4096}, {"max_tokens": 4096}]


def test_no_kwargs_when_none_supplied():
    agent = _RecordingAgent()
    BatchRunner(agent).run(["x"], config=BatchConfig(max_concurrency=1))
    assert agent.seen_kwargs == [{}]
