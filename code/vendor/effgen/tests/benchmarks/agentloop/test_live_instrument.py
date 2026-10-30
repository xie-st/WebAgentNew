"""Planting numbers in the live instrument, and checking the clock recovers them.

The endpoint here answers from a script with a stated delay and a stated usage
block. That is the only way to know what the right answer is: a real model
cannot be told to take exactly 250 ms or to report exactly 137 prompt tokens.
Nothing here claims anything about how an agent behaves — those claims are
checked against a real model, on a card.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from . import timing
from .live import (
    ENDPOINT_ENV_VARS,
    EndpointEnvironmentSet,
    LiveRun,
    check_endpoint_env,
    probe_endpoint,
    run_cell,
)
from .records import Cell, CellKey, Record
from .stub_endpoint import Reply, StubConfig, StubEndpoint

SAMPLES = [
    {
        "sample_id": f"arc_e-{i}",
        "question": f"Question number {i}?",
        "answer": "A",
        "context": "A. first\nB. second\nC. third\nD. fourth",
        "meta": {"labels": ["A", "B", "C", "D"], "options": ["first", "second", "third", "fourth"]},
    }
    for i in range(3)
]


@pytest.fixture
def samples_file(tmp_path: Path) -> Path:
    path = tmp_path / "arc_e.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in SAMPLES))
    return path


def _plan(stub: StubEndpoint, tmp_path: Path, samples_file: Path, **kwargs) -> LiveRun:
    return LiveRun(
        bench="arc_e",
        model=stub.config.model,
        base_url=stub.base_url,
        out_dir=tmp_path / "run",
        samples_path=samples_file,
        concurrency=1,
        max_steps=3,
        max_tokens=64,
        context_length=4096,
        timeout_s=30,
        **kwargs,
    )


# --------------------------------------------- 11: the delay lands on the model


def test_a_delay_in_the_transport_is_charged_to_the_model(tmp_path, samples_file):
    delay = 0.25
    with StubEndpoint(
        StubConfig(delay_s=delay, replies=[Reply(content="Answer: A")])
    ) as stub:
        outcome = run_cell(_plan(stub, tmp_path, samples_file))

    attempts = [row["attempt"] for row in outcome.records]
    assert len(attempts) == len(SAMPLES)
    for attempt in attempts:
        calls = attempt["llm_calls"]
        assert calls >= 1
        assert attempt["model_wall_s"] == pytest.approx(delay * calls, abs=0.08)
        # Whatever the framework's own cost turns out to be, it is on the other
        # side of the split and it is not negative.
        assert attempt["framework_wall_s"] > 0
        assert attempt["framework_wall_s"] == pytest.approx(
            attempt["latency_s"] - attempt["model_wall_s"], abs=0.01
        )
    assert outcome.summary["avg_model_wall_s"] == pytest.approx(delay, abs=0.1)


# ------------------------------ 12: a delay in a tool is charged to the framework


def test_a_delay_in_a_tool_is_charged_to_the_framework_not_the_model():
    """The discriminating test for the split.

    A tool is the framework's own work. Time spent in one has to land on the
    framework side, or the split is measuring the wrong boundary. It is read as
    the difference between two runs of the same code with a slower tool, so
    whatever the framework costs on this host cancels.
    """
    from .agent_binding import build_tools
    from .harness.tools.spec import ToolSpec, one_string_arg

    sleep_for = {"s": 0.0}
    calls = {"n": 0}
    tool_delay = 1.0

    def slow_tool(expression: str) -> str:
        calls["n"] += 1
        time.sleep(sleep_for["s"])
        return "42"

    spec = ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        parameters=one_string_arg("expression", "an expression"),
        func=slow_tool,
    )
    replies = [
        Reply(content="", tool_calls=[{"name": "calculator", "arguments": {"expression": "6*7"}}]),
        Reply(content="Answer: 42"),
    ]

    timing.install()
    with StubEndpoint(StubConfig(delay_s=0.0, replies=replies)) as stub:
        from effgen.core.agent import Agent, AgentConfig
        from effgen.models import load_model

        model = load_model(
            stub.config.model,
            provider="openai_compatible",
            base_url=stub.base_url,
            api_key="EMPTY",
            context_length=4096,
        )
        tools = build_tools([spec])

        def once() -> timing.Timed:
            agent = Agent(
                config=AgentConfig(
                    name="measured_agent",
                    model=model,
                    tools=tools,
                    system_prompt="use the tool",
                    max_iterations=3,
                    raise_on_error=False,
                    enable_memory=False,
                    temperature=0.1,
                    max_tokens=64,
                )
            )
            stub.reset()
            with timing.measure() as timed:
                agent.run("What is 6 times 7?")
            return timed

        once()  # the first run pays for imports; it is not the measurement
        sleep_for["s"] = 0.0
        calls["n"] = 0
        fast = once()
        assert calls["n"] >= 1, "the tool never ran, so there is nothing to measure"

        sleep_for["s"] = tool_delay
        calls["n"] = 0
        slow = once()
        assert calls["n"] >= 1

    assert slow.framework_wall_s - fast.framework_wall_s == pytest.approx(
        tool_delay, abs=0.3
    )
    assert slow.model_wall_s < 0.2
    assert fast.model_wall_s < 0.2


# ------------------------------------------------ 13: a known usage block, k times


def test_the_token_counts_are_the_ones_the_endpoint_reported():
    import httpx

    reply = Reply(content="Answer: A", prompt_tokens=137, completion_tokens=11)
    k = 4
    timing.install()
    with StubEndpoint(StubConfig(replies=[reply])) as stub:
        with timing.measure() as timed:
            with httpx.Client() as client:
                for _ in range(k):
                    client.post(
                        f"{stub.base_url}/chat/completions",
                        json={"model": stub.config.model, "messages": []},
                    )
    assert timed.usage.calls == k
    assert timed.usage.prompt_tokens == reply.prompt_tokens * k
    assert timed.usage.completion_tokens == reply.completion_tokens * k
    assert timed.usage.total_tokens == (reply.prompt_tokens + reply.completion_tokens) * k


# --------------------------------------- 14: the other name for the same numbers


def test_a_response_shaped_endpoint_is_not_recorded_as_free():
    """``/v1/responses`` names the same two numbers differently.

    Reading only the Chat Completions names once recorded a whole run as making
    no calls and costing nothing, in a measurement that is about cost.
    """
    import httpx

    reply = Reply(content="Answer: A", prompt_tokens=91, completion_tokens=7)
    timing.install()
    with StubEndpoint(StubConfig(responses_api=True, replies=[reply])) as stub:
        with timing.measure() as timed:
            with httpx.Client() as client:
                client.post(
                    f"{stub.base_url}/responses",
                    json={"model": stub.config.model, "input": "hello"},
                )
    assert timed.usage.calls == 1
    assert timed.usage.prompt_tokens == reply.prompt_tokens
    assert timed.usage.completion_tokens == reply.completion_tokens


# ------------------------------------- 15: a call made on a thread of its own


def test_a_call_made_on_a_thread_the_framework_started_is_still_counted():
    """A framework is free to generate on a thread of its own.

    The counters are per thread so concurrent samples do not mix, so without
    inheritance such a call lands on a thread with no counter and is recorded as
    never having happened.
    """
    import httpx

    reply = Reply(prompt_tokens=50, completion_tokens=5)
    timing.install()
    with StubEndpoint(StubConfig(replies=[reply])) as stub:

        def generate() -> None:
            with httpx.Client() as client:
                client.post(
                    f"{stub.base_url}/chat/completions",
                    json={"model": stub.config.model, "messages": []},
                )

        with timing.measure() as timed:
            worker = threading.Thread(target=generate)
            worker.start()
            worker.join()

    assert timed.usage.calls == 1
    assert timed.usage.prompt_tokens == reply.prompt_tokens


# --------------------------------- 16, 17: the endpoint is an argument, always


@pytest.mark.parametrize("name", ENDPOINT_ENV_VARS)
@pytest.mark.parametrize("value", ["https://example.invalid/v1", ""])
def test_the_runner_refuses_to_start_while_an_endpoint_variable_is_set(name, value):
    """Including when it is set to the empty string.

    That is the shape that does the damage without looking set at all: it
    redirects every call in the process and the failures read as an outage at
    the provider.
    """
    with pytest.raises(EndpointEnvironmentSet) as raised:
        check_endpoint_env({name: value})
    message = str(raised.value)
    assert name in message
    assert f"unset {name}" in message


def test_a_clean_environment_is_accepted():
    check_endpoint_env({"PATH": "/usr/bin"})


def test_there_is_no_default_endpoint_and_no_environment_fallback(tmp_path, samples_file):
    plan = LiveRun(
        bench="arc_e",
        model="stub-model",
        base_url="",
        out_dir=tmp_path / "run",
        samples_path=samples_file,
    )
    with pytest.raises(ValueError) as raised:
        run_cell(plan)
    assert "base_url" in str(raised.value)


def test_the_run_refuses_an_endpoint_that_serves_a_different_model():
    with StubEndpoint(StubConfig(model="something-else")) as stub:
        with pytest.raises(RuntimeError) as raised:
            probe_endpoint(stub.base_url, "the-model-we-asked-for")
    assert "the-model-we-asked-for" in str(raised.value)


# --------------------------------------------------- 18: a streamed response


def test_a_streamed_response_is_flagged_rather_than_reported_as_a_number():
    """``send`` returns before a streamed body is read.

    The model side of the split is then short by however long the body took, so
    the split is marked unreliable instead of being printed as a number that is
    quietly wrong.
    """
    import httpx

    timing.install()
    with StubEndpoint(StubConfig(stream=True)) as stub:
        with timing.measure() as timed:
            with httpx.Client() as client:
                client.post(
                    f"{stub.base_url}/chat/completions",
                    json={"model": stub.config.model, "messages": [], "stream": True},
                )
    assert timed.usage.streaming_seen is True
    assert timed.usage.calls == 0
    assert timed.usage.uncounted == 1

    record = Record.from_json(
        {
            "sample_id": "arc_e-0",
            "question": "q",
            "ground_truth": "A",
            "prediction": "A",
            "score": 1.0,
            "correct": True,
            "output": "A",
            "meta": {"labels": ["A", "B"], "options": ["x", "y"]},
            "attempt": {
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": 1.0,
                "model_wall_s": 0.1,
                "framework_wall_s": 0.9,
                "streaming_seen": True,
                "stop_reason": "final_answer",
                "tool_calls": [],
            },
        }
    )
    cell = Cell(
        CellKey("stub", "arc_e", "live"),
        [record],
        {"state": "done", "completed": 1},
    )
    assert cell.stats().streaming_seen is True
    assert cell.stats().split_is_reliable is False


# ------------------------------------------- 19: a sample that raises is a record


def test_a_sample_that_raises_is_recorded_and_the_run_still_finishes(
    tmp_path, samples_file
):
    class Exploding:
        def run(self, task: str):
            raise RuntimeError("the agent could not be built")

    with StubEndpoint(StubConfig()) as stub:
        plan = _plan(
            stub, tmp_path, samples_file, agent_factory=lambda builder: Exploding()
        )
        outcome = run_cell(plan)

    assert outcome.summary["state"] == "done"
    assert outcome.summary["completed"] == len(SAMPLES)
    assert outcome.summary["errors"] == len(SAMPLES)
    assert outcome.summary["accuracy"] == 0.0
    for row in outcome.records:
        assert "the agent could not be built" in row["attempt"]["error"]
        assert row["score"] == 0.0


# ------------------------------------ 20: a live run reads back as a recorded one


def test_a_live_run_replays_through_the_recorded_reader_with_no_special_case(
    tmp_path, samples_file
):
    with StubEndpoint(
        StubConfig(delay_s=0.05, replies=[Reply(content="Answer: A", prompt_tokens=77, completion_tokens=9)])
    ) as stub:
        outcome = run_cell(_plan(stub, tmp_path, samples_file, capture_logs=True))

    cell = Cell.load(outcome.directory, CellKey("stub", "arc_e", "live"))
    stats = cell.stats()

    assert stats.n == outcome.summary["completed"]
    assert stats.accuracy == pytest.approx(outcome.summary["accuracy"], abs=1e-6)
    assert stats.mean_prompt_tokens == pytest.approx(
        outcome.summary["avg_prompt_tokens"], abs=0.05
    )
    assert stats.mean_completion_tokens == pytest.approx(
        outcome.summary["avg_completion_tokens"], abs=0.05
    )
    assert stats.roundtrip_mismatches == 0
    assert stats.mean_model_wall_s is not None
    assert stats.split_is_reliable
    reported = {str(row.reason): row.n for row in stats.stop_reasons}
    assert reported == outcome.summary["stop_reasons"]


def test_the_run_records_where_it_went(tmp_path, samples_file):
    with StubEndpoint(StubConfig()) as stub:
        outcome = run_cell(_plan(stub, tmp_path, samples_file))
    manifest = json.loads((outcome.directory / "manifest.json").read_text())
    assert manifest["base_url"] == stub.base_url
    assert manifest["model"] == stub.config.model
    assert manifest["served_models"] == [stub.config.model]
    assert manifest["instrument_version"] >= 1
    assert manifest["revision"]
    assert manifest["endpoint"]["seed"] == 42
