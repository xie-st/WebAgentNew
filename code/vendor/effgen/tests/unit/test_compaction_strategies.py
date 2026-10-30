"""What survives when a conversation outgrows the context window.

Compression for small models is one of effGen's claims, so which turns are kept
is a choice a caller should be able to make rather than one baked into the
memory. The default reproduces what effGen has always done.
"""

from __future__ import annotations

import pytest

from effgen.core.agent_config import AgentConfig
from effgen.memory.compaction import (
    STRATEGIES_BY_NAME,
    CompactionStrategy,
    DropOldest,
    KeepFirstAndLast,
    KeepToolResults,
    SummarizeOldest,
    resolve_strategy,
)
from effgen.memory.short_term import ShortTermMemory

LONG = "word " * 40


def _filled(strategy=None, *, count=30, keep_recent=2, label=False, **kwargs):
    memory = ShortTermMemory(
        max_tokens=100, keep_recent_messages=keep_recent,
        compaction_strategy=strategy, **kwargs,
    )
    for n in range(count):
        memory.add_user_message(f"M{n} {LONG}" if label else LONG)
    return memory


def _labels(memory) -> list[str]:
    return [m.content.split()[0] for m in memory.messages]


class TestResolution:
    """Naming a strategy."""

    @pytest.mark.parametrize("name", sorted(STRATEGIES_BY_NAME))
    def test_every_shipped_strategy_resolves_by_name(self, name):
        assert isinstance(resolve_strategy(name), CompactionStrategy)

    def test_none_means_the_default(self):
        assert isinstance(resolve_strategy(None), SummarizeOldest)

    def test_a_class_is_instantiated(self):
        assert isinstance(resolve_strategy(DropOldest), DropOldest)

    def test_an_instance_is_used_as_given(self):
        strategy = KeepFirstAndLast(first=1, last=1)
        assert resolve_strategy(strategy) is strategy

    def test_an_unknown_name_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="drop_oldest"):
            resolve_strategy("keep_everything_forever")

    def test_a_nonsense_value_says_what_is_accepted(self):
        with pytest.raises(ValueError, match="CompactionStrategy"):
            resolve_strategy(42)


class TestTheDefaultIsUnchanged:
    """The behaviour effGen has always had."""

    def test_a_memory_with_no_strategy_gets_the_default(self):
        assert isinstance(ShortTermMemory().compaction_strategy, SummarizeOldest)

    def test_older_turns_become_a_summary(self):
        memory = _filled()
        assert memory.summaries, "nothing was summarized"
        assert len(memory.messages) <= 30

    def test_the_most_recent_turns_stay_verbatim(self):
        memory = _filled(count=30, keep_recent=2, label=True)
        assert _labels(memory)[-1] == "M29"


class TestDropOldest:
    """No model call, no summary."""

    def test_nothing_is_summarized(self):
        assert _filled(DropOldest()).summaries == []

    def test_the_history_is_still_bounded(self):
        assert len(_filled(DropOldest(), keep_recent=2).messages) <= 3

    def test_the_recent_turns_are_the_ones_kept(self):
        memory = _filled(DropOldest(), count=30, keep_recent=2, label=True)
        assert _labels(memory) == ["M28", "M29"]


class TestKeepFirstAndLast:
    """The task at the top and the recent turns; the middle goes."""

    def test_both_ends_survive_and_the_middle_does_not(self):
        memory = _filled(
            KeepFirstAndLast(first=2, last=2, summarize_middle=False),
            count=30, label=True,
        )
        assert _labels(memory) == ["M0", "M1", "M28", "M29"]

    def test_the_middle_can_be_summarized_instead_of_dropped(self):
        memory = _filled(
            KeepFirstAndLast(first=2, last=2, summarize_middle=True),
            count=30, label=True,
        )
        assert memory.summaries
        assert _labels(memory)[:2] == ["M0", "M1"]

    def test_a_short_conversation_is_left_alone(self):
        memory = _filled(KeepFirstAndLast(first=2, last=6), count=3, label=True)
        assert _labels(memory) == ["M0", "M1", "M2"]


class TestKeepToolResults:
    """The evidence stays, the reasoning goes."""

    def test_a_tool_result_is_not_compacted_away(self):
        memory = ShortTermMemory(
            max_tokens=100, keep_recent_messages=2,
            compaction_strategy=KeepToolResults(summarize_dropped=False),
        )
        for n in range(20):
            memory.add_user_message(f"U{n} {LONG}")
            if n % 5 == 0:
                memory.add_tool_message(f"T{n} {LONG}")
        kept = _labels(memory)
        assert any(label.startswith("T") for label in kept), kept


class TestTheTokenizer:
    """Measuring the history the way the window is measured."""

    def test_a_supplied_tokenizer_is_used(self):
        class WordTokenizer:
            def encode(self, text):
                return text.split()

        memory = ShortTermMemory(tokenizer=WordTokenizer())
        assert memory._count_tokens("a b c d e") == 5

    def test_a_count_tokens_tokenizer_is_used(self):
        class Counter:
            def count_tokens(self, text):
                return len(text.split())

        assert ShortTermMemory(tokenizer=Counter())._count_tokens("a b c") == 3

    def test_a_broken_tokenizer_falls_back_rather_than_failing_the_run(self):
        class Broken:
            def encode(self, text):
                raise RuntimeError("tokenizer unavailable")

        # The character estimate is what the memory used before tokenizers
        # were accepted, so falling back to it is the old behaviour.
        assert ShortTermMemory(tokenizer=Broken())._count_tokens("x" * 40) == 10

    def test_no_tokenizer_keeps_the_character_estimate(self):
        assert ShortTermMemory()._count_tokens("x" * 40) == 10


class TestOnTheAgentConfig:
    """Reaching it from where an agent is built."""

    def test_a_strategy_can_be_named_on_the_config(self):
        config = AgentConfig(
            model="x", require_model=False, compaction_strategy="drop_oldest",
        )
        assert config.compaction_strategy == "drop_oldest"

    def test_a_tokenizer_can_be_given_on_the_config(self):
        marker = object()
        config = AgentConfig(model="x", require_model=False, tokenizer=marker)
        assert config.tokenizer is marker

    def test_the_default_config_names_no_strategy(self):
        config = AgentConfig(model="x", require_model=False)
        assert config.compaction_strategy is None
        assert config.tokenizer is None


class TestACustomStrategy:
    """The interface a caller subclasses."""

    def test_a_subclass_decides_when_to_compact(self):
        class Never(CompactionStrategy):
            def should_compact(self, memory):
                return False

        memory = _filled(Never(), count=20)
        assert memory.summaries == []
        assert memory.total_summarizations == 0

    def test_a_subclass_decides_what_leaves(self):
        class OnlyTheOldest(CompactionStrategy):
            def messages_to_compact(self, memory):
                return list(memory.messages)[:1]

            def summarize(self, memory, messages):
                return None

        memory = _filled(OnlyTheOldest(), count=30, label=True)
        assert "M0" not in _labels(memory)

    def test_a_subclass_decides_what_replaces_them(self):
        class FixedNote(CompactionStrategy):
            def summarize(self, memory, messages):
                return "the earlier turns covered setup"

        memory = _filled(FixedNote(), count=30)
        # Several rounds merge into one retained summary, so the strategy's
        # text is what that summary is built from rather than all of it.
        assert "the earlier turns covered setup" in memory.summaries[0].summary

    def test_the_strategy_is_recorded_on_the_summary(self):
        memory = _filled(SummarizeOldest(), count=30)
        assert memory.summaries[0].metadata["strategy"] == "SummarizeOldest"
