"""Performance benchmarks for effGen."""

import time

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin import Calculator, DateTimeTool, JSONTool, TextProcessingTool
from tests.fixtures.mock_models import MockModel


class TestAgentInitPerformance:
    """Benchmark agent initialization time."""

    def test_basic_init_time(self):
        """Agent with no tools should init in < 100ms."""
        model = MockModel(responses=["Thought: done\nFinal Answer: ok"])
        start = time.perf_counter()
        for _ in range(10):
            Agent(config=AgentConfig(
                name="bench",
                model=model,
                tools=[],
                enable_memory=False,
                enable_sub_agents=False,
            ))
        elapsed = (time.perf_counter() - start) / 10
        assert elapsed < 0.1, f"Agent init took {elapsed:.3f}s (should be < 100ms)"

    def test_init_with_tools_time(self):
        """Agent with tools should init in < 200ms."""
        model = MockModel(responses=["Thought: done\nFinal Answer: ok"])
        tools = [Calculator(), DateTimeTool(), JSONTool(), TextProcessingTool()]
        start = time.perf_counter()
        for _ in range(10):
            Agent(config=AgentConfig(
                name="bench",
                model=model,
                tools=tools,
                enable_memory=False,
                enable_sub_agents=False,
            ))
        elapsed = (time.perf_counter() - start) / 10
        assert elapsed < 0.2, f"Agent init with tools took {elapsed:.3f}s (should be < 200ms)"


class TestToolExecutionPerformance:
    """Benchmark tool execution latency."""

    @pytest.mark.asyncio
    async def test_calculator_latency(self):
        """Calculator should execute in < 10ms."""
        calc = Calculator()
        start = time.perf_counter()
        for _ in range(100):
            await calc.execute(expression="2 + 2")
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.01, f"Calculator took {elapsed*1000:.1f}ms (should be < 10ms)"

    @pytest.mark.asyncio
    async def test_datetime_latency(self):
        """DateTimeTool should execute in < 10ms."""
        dt = DateTimeTool()
        start = time.perf_counter()
        for _ in range(100):
            await dt.execute(operation="now")
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.01, f"DateTimeTool took {elapsed*1000:.1f}ms (should be < 10ms)"

    @pytest.mark.asyncio
    async def test_json_tool_latency(self):
        """JSONTool should execute in < 10ms."""
        jt = JSONTool()
        start = time.perf_counter()
        for _ in range(100):
            await jt.execute(operation="keys", data='{"a": 1, "b": 2}')
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.01, f"JSONTool took {elapsed*1000:.1f}ms (should be < 10ms)"


class TestEndToEndPerformance:
    """Benchmark end-to-end response time with mock model."""

    def test_simple_query_time(self):
        """Simple query with mock model should complete in < 50ms."""
        model = MockModel(responses=["Thought: I know this.\nFinal Answer: 42"])
        agent = Agent(config=AgentConfig(
            name="bench",
            model=model,
            tools=[],
            max_iterations=3,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        start = time.perf_counter()
        for _ in range(10):
            model._idx = 0  # Reset
            agent.run("What is the answer?")
        elapsed = (time.perf_counter() - start) / 10
        assert elapsed < 0.05, f"Simple query took {elapsed*1000:.1f}ms (should be < 50ms)"

    def test_tool_query_time(self):
        """Query with tool call (mock model) should complete in < 100ms."""
        model = MockModel(responses=[
            'Thought: Need calculator.\nAction: calculator\nAction Input: {"expression": "2+2"}',
            "Thought: Got it.\nFinal Answer: 4",
        ])
        agent = Agent(config=AgentConfig(
            name="bench",
            model=model,
            tools=[Calculator()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        start = time.perf_counter()
        for _ in range(10):
            model._idx = 0
            agent.run("What is 2+2?")
        elapsed = (time.perf_counter() - start) / 10
        assert elapsed < 0.1, f"Tool query took {elapsed*1000:.1f}ms (should be < 100ms)"


class TestMemoryPerformance:
    """Benchmark memory operations."""

    def test_short_term_memory_add(self):
        """Adding messages should be fast."""
        from effgen.memory.short_term import MessageRole, ShortTermMemory
        mem = ShortTermMemory(max_tokens=8192, max_messages=1000)
        start = time.perf_counter()
        for i in range(1000):
            mem.add_message(MessageRole.USER, f"Message {i}")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Adding 1000 messages took {elapsed:.3f}s (should be < 500ms)"
