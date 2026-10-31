"""One agent object, many independent conversations.

``run(..., session=...)`` builds the prompt from that conversation's history and
appends the turn to it, so a server handling many users does not need an agent
per user, nor its own history bookkeeping outside the framework.
"""

from __future__ import annotations

import pytest

from effgen import Agent, AgentConfig
from effgen.core.session import Session
from tests.fixtures.mock_models import MockModel

ANSWERS = [f"Final Answer: answer {n}" for n in range(1, 21)]


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    """Keep saved sessions out of the developer's real session store."""
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def agent():
    a = Agent(AgentConfig(model=MockModel(list(ANSWERS)), raise_on_error=False))
    yield a
    a.close()


class TestTheTurnIsRecorded:
    """What a session holds after a run."""

    def test_the_task_and_the_answer_are_appended(self, agent, sessions_dir):
        session = Session(session_id="s1")
        agent.run("what is the capital of France?", session=session)
        roles = [m["role"] for m in session.messages]
        assert roles == ["user", "assistant"]
        assert session.messages[0]["content"] == "what is the capital of France?"

    def test_successive_turns_accumulate(self, agent, sessions_dir):
        session = Session(session_id="s2")
        agent.run("first", session=session)
        agent.run("second", session=session)
        assert [m["content"] for m in session.messages if m["role"] == "user"] == [
            "first", "second",
        ]

    def test_a_session_id_is_loaded_or_created(self, agent, sessions_dir):
        agent.run("hello", session="by-id")
        stored = Session.load("by-id", str(sessions_dir))
        assert [m["role"] for m in stored.messages] == ["user", "assistant"]

    def test_a_session_given_by_id_is_reopened_on_the_next_run(self, agent, sessions_dir):
        agent.run("first", session="resumed")
        agent.run("second", session="resumed")
        stored = Session.load("resumed", str(sessions_dir))
        assert [m["content"] for m in stored.messages if m["role"] == "user"] == [
            "first", "second",
        ]


class TestConversationsStayApart:
    """The point of the handle."""

    def test_two_conversations_on_one_agent_do_not_mix(self, agent, sessions_dir):
        alice, bob = Session(session_id="alice"), Session(session_id="bob")
        agent.run("my favourite colour is blue", session=alice)
        agent.run("my favourite animal is dogs", session=bob)
        assert len(alice.messages) == 2
        assert len(bob.messages) == 2
        assert "dogs" not in str(alice.messages)
        assert "blue" not in str(bob.messages)

    def test_one_conversation_s_history_reaches_its_own_prompt(self, sessions_dir):
        model = MockModel(list(ANSWERS))
        a = Agent(AgentConfig(model=model, raise_on_error=False))
        try:
            alice = Session(session_id="alice2")
            a.run("my favourite colour is blue", session=alice)
            a.run("what did I say?", session=alice)
            second_prompt = model._generate_calls[-1]["prompt"]
        finally:
            a.close()
        assert "blue" in second_prompt

    def test_another_conversation_s_history_does_not(self, sessions_dir):
        model = MockModel(list(ANSWERS))
        a = Agent(AgentConfig(model=model, raise_on_error=False))
        try:
            a.run("my favourite colour is blue", session=Session(session_id="alice3"))
            a.run("what did I say?", session=Session(session_id="bob3"))
            bob_prompt = model._generate_calls[-1]["prompt"]
        finally:
            a.close()
        assert "blue" not in bob_prompt

    def test_the_agent_s_own_memory_is_left_alone(self, agent, sessions_dir):
        agent.run("into a session", session=Session(session_id="s3"))
        assert agent.short_term_memory.get_recent_messages(n=10) == []

    def test_a_run_without_a_handle_still_uses_the_agent_s_memory(self, agent, sessions_dir):
        agent.run("no session here")
        assert agent.short_term_memory.get_recent_messages(n=10) != []


class TestRestoration:
    """The handle applies to one call and no more."""

    def test_the_agent_s_session_is_restored_afterwards(self, agent, sessions_dir):
        before = agent.session
        agent.run("one call", session=Session(session_id="s4"))
        assert agent.session is before

    def test_it_is_restored_even_when_the_run_fails(self, sessions_dir):
        class Boom(MockModel):
            def generate(self, prompt, config=None, **kwargs):
                raise RuntimeError("provider exploded")

        a = Agent(AgentConfig(model=Boom(["x"]), raise_on_error=False))
        try:
            before = a.session
            a.run("this fails", session=Session(session_id="s5"))
            assert a.session is before
        finally:
            a.close()

    def test_the_next_run_without_a_handle_is_unaffected(self, agent, sessions_dir):
        agent.run("in a session", session=Session(session_id="s6"))
        agent.run("plain run")
        assert agent.short_term_memory.get_recent_messages(n=10) != []
