from __future__ import annotations

import pytest

from webagent.budget import (
    Budget,
    BudgetLimits,
    ContextPacker,
    Deadline,
    HeuristicTokenCounter,
    HistoryEntry,
    ProvidedTokenCounter,
)
from webagent.clock import FakeClock
from webagent.errors import BudgetExceeded, ContextOverflow, DeadlineExceeded, RequestLimitExceeded


def test_token_counters_record_their_source() -> None:
    heuristic = HeuristicTokenCounter()
    assert heuristic.count_text("abcdefgh") == 2
    assert heuristic.source == "estimate:chars_div_4"
    exact = ProvidedTokenCounter(lambda text: len(text.split()), source="fake-provider:exact")
    assert exact.count_text("one two three") == 3
    assert exact.source == "fake-provider:exact"


def test_pack_keeps_mandatory_content_and_compresses_history(make_observation) -> None:
    observation = make_observation(ax_text="The current result is Item 42.")
    packer = ContextPacker(
        BudgetLimits(input_tokens=260, planned_context_tokens=512),
        counter=HeuristicTokenCounter(),
    )
    history = [
        HistoryEntry(f"old action {index} with a long description " + "x" * 40, kind="step")
        for index in range(20)
    ]
    packed = packer.pack(
        task="Find Item 42 and open its detail page.",
        observation=observation,
        pending="confirm the detail page title",
        history=history,
    )
    combined = "\n".join(message["content"] for message in packed.messages)
    assert "Find Item 42" in combined
    assert "The current result is Item 42." in combined
    assert "confirm the detail page title" in combined
    assert packed.dropped_history_entries > 0
    assert packed.compressed is True
    assert packed.token_count <= 260
    assert packed.token_count_source == "estimate:chars_div_4"
    assert packed.request_bytes > 0
    assert len(packed.messages) <= 256


def test_pack_reports_when_mandatory_content_cannot_fit(make_observation) -> None:
    observation = make_observation(ax_text="AX " * 100)
    packer = ContextPacker(
        BudgetLimits(input_tokens=20, planned_context_tokens=64),
        counter=HeuristicTokenCounter(),
    )
    with pytest.raises(ContextOverflow) as excinfo:
        packer.pack(task="a very long task description " * 50, observation=observation)
    error = excinfo.value
    assert error.required_tokens > error.available_tokens
    assert error.available_tokens == 20
    assert "not truncated" in str(error)
    assert error.token_count_source == "estimate:chars_div_4"


def test_oversized_current_observation_is_not_silently_truncated(make_observation) -> None:
    observation = make_observation(ax_text="x" * 10_000)
    packer = ContextPacker(
        BudgetLimits(input_tokens=100, planned_context_tokens=256),
        counter=HeuristicTokenCounter(),
    )
    with pytest.raises(ContextOverflow) as excinfo:
        packer.pack(task="small task", observation=observation)
    assert excinfo.value.required_tokens > 100
    assert "current AX text are not truncated" in str(excinfo.value)


def test_message_and_byte_limits_are_explicit(make_observation) -> None:
    observation = make_observation()
    message_limited = ContextPacker(
        BudgetLimits(input_tokens=10_000, planned_context_tokens=16_000, max_messages=2),
        counter=HeuristicTokenCounter(),
    )
    with pytest.raises(RequestLimitExceeded) as excinfo:
        message_limited.pack(task="task", observation=observation, pending="pending")
    assert excinfo.value.limit_name == "max_messages"
    assert excinfo.value.limit == 2

    byte_limited = ContextPacker(
        BudgetLimits(input_tokens=10_000, planned_context_tokens=16_000, max_request_bytes=80),
        counter=HeuristicTokenCounter(),
    )
    with pytest.raises(RequestLimitExceeded) as excinfo:
        byte_limited.pack(task="task " * 50, observation=observation)
    assert excinfo.value.limit_name == "max_request_bytes"
    assert excinfo.value.limit == 80


def test_exact_counter_source_is_preserved(make_observation) -> None:
    counter = ProvidedTokenCounter(lambda text: len(text), source="provided:exact")
    packer = ContextPacker(BudgetLimits(input_tokens=100_000, planned_context_tokens=100_000), counter=counter)
    packed = packer.pack(task="task", observation=make_observation())
    assert packed.token_count_source == "provided:exact"
    assert packed.token_count > 0


def test_deadline_and_model_call_checks(clock: FakeClock) -> None:
    deadline = Deadline(clock.now() + 10, clock=clock)
    assert deadline.remaining_seconds() == pytest.approx(10)
    budget = Budget(BudgetLimits(), deadline=deadline, clock=clock)
    assert budget.limits.max_model_calls is None
    for index in range(15):
        assert budget.begin_model_call() == index + 1
    clock.advance(11)
    assert deadline.expired()
    with pytest.raises(DeadlineExceeded):
        budget.begin_model_call()

    limited_clock = FakeClock()
    limited = Budget(
        BudgetLimits(max_model_calls=2),
        deadline=Deadline(limited_clock.now() + 100, clock=limited_clock),
        clock=limited_clock,
    )
    limited.begin_model_call()
    limited.begin_model_call()
    with pytest.raises(BudgetExceeded):
        limited.begin_model_call()
