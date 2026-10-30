"""
Mock model implementations for unit testing.

These models return predetermined responses, allowing tests to run
without GPU or real model inference.
"""

from __future__ import annotations

from collections.abc import Iterator

from effgen.models.base import BaseModel, GenerationConfig, GenerationResult, ModelType, TokenCount


class MockModel(BaseModel):
    """Model that returns predetermined responses for testing."""

    def __init__(self, responses: list[str], model_name: str = "mock-model"):
        super().__init__(model_name=model_name, model_type=ModelType.TRANSFORMERS)
        self.responses = responses
        self._idx = 0
        self._is_loaded = True
        self._generate_calls = []

    def load(self) -> None:
        self._is_loaded = True

    def generate(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs
    ) -> GenerationResult:
        response = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        self._generate_calls.append({"prompt": prompt, "config": config})
        return GenerationResult(
            text=response,
            tokens_used=len(response.split()),
            finish_reason="stop",
            model_name=self.model_name,
        )

    def generate_stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs
    ) -> Iterator[str]:
        result = self.generate(prompt, config, **kwargs)
        for token in result.text.split():
            yield token + " "

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def unload(self) -> None:
        self._is_loaded = False

    @property
    def call_count(self) -> int:
        return self._idx

    @property
    def last_prompt(self) -> str | None:
        if self._generate_calls:
            return self._generate_calls[-1]["prompt"]
        return None


class MockToolCallingModel(MockModel):
    """Mock model that produces ReAct-formatted tool call responses."""

    def __init__(self, tool_sequence: list[dict[str, str]], model_name: str = "mock-tool-model"):
        """
        Args:
            tool_sequence: List of dicts with keys:
                - "thought": The thought text
                - "action": Tool name (or "Final Answer")
                - "action_input": Input to the tool
        """
        responses = []
        for step in tool_sequence:
            thought = step.get("thought", "Let me think about this.")
            action = step.get("action", "Final Answer")
            action_input = step.get("action_input", "")
            responses.append(
                f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}"
            )
        super().__init__(responses=responses, model_name=model_name)


class MockStreamingModel(MockModel):
    """Mock model that simulates token-by-token streaming."""

    def generate_stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs
    ) -> Iterator[str]:
        result = self.generate(prompt, config, **kwargs)
        yield from result.text
