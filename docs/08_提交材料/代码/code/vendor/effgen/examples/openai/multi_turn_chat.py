"""
Multi-turn conversation with OpenAI models via the chat() method.

Demonstrates:
- Full conversation history management
- System prompts
- Multi-turn context retention
- Cost tracking across turns

Run:
    python examples/openai/multi_turn_chat.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

from effgen.models.base import GenerationConfig
from effgen.models.openai_adapter import OpenAIAdapter


def main():
    adapter = OpenAIAdapter("gpt-5.4-nano")
    adapter.load()
    print(f"Model: {adapter.model_name}\n")

    # Conversation history
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise, helpful assistant. "
                "Keep all responses under 3 sentences."
            ),
        }
    ]

    conversation = [
        "My name is Alex. I'm learning Python.",
        "What's the most important concept I should master first?",
        "Can you give me a one-line example of that concept?",
        "What did I tell you my name was?",  # tests context retention
    ]

    config = GenerationConfig(max_tokens=200, temperature=0.7)

    for user_turn in conversation:
        messages.append({"role": "user", "content": user_turn})
        print(f"User: {user_turn}")

        result = adapter.chat(messages=messages, config=config)
        assistant_reply = result.text
        messages.append({"role": "assistant", "content": assistant_reply})

        print(f"Assistant: {assistant_reply}")
        print(f"  (tokens: {result.tokens_used}, cost: ${result.metadata['cost']:.6f})\n")

    print(f"Total conversation cost: ${adapter.get_total_cost():.6f}")
    print(f"Total tokens: {adapter.get_total_tokens()}")
    adapter.unload()


if __name__ == "__main__":
    main()
