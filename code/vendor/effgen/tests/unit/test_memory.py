"""Unit tests for memory systems."""

from effgen.memory.short_term import Message, MessageRole, ShortTermMemory


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.timestamp > 0

    def test_message_to_dict(self):
        msg = Message(role=MessageRole.ASSISTANT, content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"
        assert "timestamp" in d

    def test_message_from_dict(self):
        d = {"role": "user", "content": "Test message"}
        msg = Message.from_dict(d)
        assert msg.role == MessageRole.USER
        assert msg.content == "Test message"

    def test_message_estimate_tokens(self):
        msg = Message(role=MessageRole.USER, content="Hello world test message here")
        tokens = msg.estimate_tokens()
        assert tokens > 0
        assert tokens < 100

    def test_message_roles(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"


class TestShortTermMemory:
    """Tests for ShortTermMemory."""

    def test_create_empty(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        assert len(mem.messages) == 0

    def test_add_message(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "Hello")
        assert len(mem.messages) == 1

    def test_add_multiple_messages(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "Hello")
        mem.add_message(MessageRole.ASSISTANT, "Hi!")
        mem.add_message(MessageRole.USER, "How are you?")
        assert len(mem.messages) == 3

    def test_get_recent_messages(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "Hello")
        mem.add_message(MessageRole.ASSISTANT, "Hi!")
        messages = mem.get_recent_messages()
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi!"

    def test_get_recent_messages_limited(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "A")
        mem.add_message(MessageRole.ASSISTANT, "B")
        mem.add_message(MessageRole.USER, "C")
        messages = mem.get_recent_messages(n=2)
        assert len(messages) == 2
        assert messages[0].content == "B"
        assert messages[1].content == "C"

    def test_clear(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "Hello")
        mem.add_message(MessageRole.ASSISTANT, "Hi!")
        mem.clear()
        assert len(mem.messages) == 0

    def test_total_messages_counter(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "A")
        mem.add_message(MessageRole.USER, "B")
        assert mem.total_messages_added == 2


class TestSummariesStayBounded:
    """A long session must not grow the stored context without limit.

    Summarizing turns older messages into a summary, and every summary is
    replayed into the next prompt. The message deque is capped but the summary
    list was not, so a session that ran long enough grew its prompt past the
    model's context window and every further call was refused.
    """

    @staticmethod
    def _run(turns: int, **kwargs) -> ShortTermMemory:
        memory = ShortTermMemory(max_tokens=4096, max_messages=100, **kwargs)
        for turn in range(turns):
            memory.add_user_message(f"Question {turn}: " + "context padding words " * 12)
            memory.add_assistant_message(f"Answer {turn}: " + "response words here " * 12)
        return memory

    def test_stored_tokens_stay_within_the_configured_maximum(self):
        memory = self._run(1_200)
        assert memory.get_token_count() <= memory.max_tokens, (
            f"{memory.get_token_count()} tokens stored against a "
            f"max_tokens of {memory.max_tokens} after 1200 turns"
        )

    def test_summaries_do_not_grow_with_the_conversation(self):
        """Three times the turns must not mean three times the summaries."""
        short = self._run(400)
        long = self._run(1_200)
        assert len(long.summaries) <= len(short.summaries) + 2, (
            f"{len(short.summaries)} summaries after 400 turns, "
            f"{len(long.summaries)} after 1200 — the list grows with the session"
        )
        assert long.summary_token_count() <= long._summary_budget()

    def test_older_context_is_folded_rather_than_dropped(self):
        """A merged entry records how many summaries it stands for."""
        memory = self._run(1_200)
        assert memory.summaries, "expected at least one retained summary"
        folded = max(s.metadata.get("folded_summaries", 1) for s in memory.summaries)
        assert folded > 1, "no summary was merged; nothing was compacted"
        # The messages each summary accounts for are still counted.
        assert sum(s.message_count for s in memory.summaries) > 0

    def test_the_summary_budget_is_configurable(self):
        tight = self._run(600, summary_budget_ratio=0.1)
        assert tight.summary_token_count() <= tight._summary_budget()
        assert tight._summary_budget() < ShortTermMemory(max_tokens=4096)._summary_budget()

    def test_recent_messages_survive_compaction(self):
        """Compacting summaries does not disturb the recent-message window."""
        memory = self._run(1_200)
        messages = memory.get_messages()
        assert messages, "recent messages were lost"
        assert "1199" in messages[-1].content or "1199" in messages[-2].content

    def test_the_budget_holds_for_a_dense_tokenizer(self):
        """The budget is measured in the model's tokens, not in characters.

        An agent counts tokens with its model's tokenizer. Scripts that
        tokenize close to one token per character count several times what a
        characters-per-token estimate predicts, so a cut taken on that estimate
        alone leaves the summaries above the budget.
        """
        class DenseCounter:
            def count_tokens(self, text: str) -> int:
                return len(text)

        memory = ShortTermMemory(
            max_tokens=4096, max_messages=100, keep_recent_messages=4,
            model=DenseCounter(),
        )
        for turn in range(400):
            memory.add_user_message(f"Question {turn}: " + "context padding words " * 12)
            memory.add_assistant_message(f"Answer {turn}: " + "response words " * 12)
        assert memory.summary_token_count() <= memory._summary_budget(), (
            f"{memory.summary_token_count()} tokens of summary against a budget "
            f"of {memory._summary_budget()}"
        )
        assert memory.get_token_count() <= memory.max_tokens

    def test_the_configured_budget_survives_a_save_and_reload(self):
        """A stored session reloads with the budget it was configured with."""
        memory = ShortTermMemory(max_tokens=4096, summary_budget_ratio=0.1)
        restored = ShortTermMemory.from_dict(memory.to_dict())
        assert restored.summary_budget_ratio == 0.1
        assert restored._summary_budget() == memory._summary_budget()

    def test_memory_config_sets_the_budget_on_an_agent(self):
        """The knob is reachable from ``AgentConfig.memory_config``."""
        from effgen.core.agent import Agent, AgentConfig

        agent = Agent(config=AgentConfig(
            name="budget", model=None, require_model=False,
            memory_config={"short_term_max_tokens": 2048,
                           "summary_budget_ratio": 0.25},
        ))
        try:
            assert agent.short_term_memory.summary_budget_ratio == 0.25
            assert agent.short_term_memory._summary_budget() == 512
        finally:
            agent.close()
