"""
OpenAI structured outputs v2 example.

Structured outputs guarantee that the model returns valid JSON that matches
your Pydantic schema.  This is different from asking the model to "respond
in JSON" — OpenAI enforces the schema at the token level so you always get
parseable output.

This example shows:
  1. Defining a Pydantic response model.
  2. Converting it to an OpenAI-compatible JSON Schema with to_openai_schema().
  3. Calling generate_structured() to get a type-safe response.
  4. Handling ModelRefusalError if the model refuses to answer.
  5. A more complex nested schema example (structured document analysis).

Run:
    python examples/openai/structured_outputs.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

logging.basicConfig(level=logging.WARNING)

from pydantic import BaseModel, Field

from effgen.models.errors import ModelRefusalError
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_schema import to_openai_schema

MODEL = "gpt-5.4-nano"

adapter = OpenAIAdapter(model_name=MODEL)
adapter.load()

# ---------------------------------------------------------------------------
# Example 1: Sentiment analysis
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


print("=" * 60)
print("Example 1: Sentiment Analysis with Strict JSON Schema")
print("=" * 60)

sentiment_rf = {
    "type": "json_schema",
    "json_schema": {
        "name": "SentimentResult",
        "schema": to_openai_schema(SentimentResult),
        "strict": True,
    },
}

reviews = [
    "This product exceeded all my expectations! Absolutely love it.",
    "Worst purchase of my life. Complete waste of money.",
    "It arrived on time and works as described.",
]

for review in reviews:
    result = adapter.generate_structured(
        prompt=f"Classify the sentiment of this review:\n\n{review!r}",
        response_format=sentiment_rf,
    )
    parsed = SentimentResult.model_validate_json(result.text)
    print(f"\nReview: {review[:50]}...")
    print(f"  Sentiment:  {parsed.sentiment} (confidence={parsed.confidence:.2f})")
    print(f"  Reasoning:  {parsed.reasoning[:80]}")

# ---------------------------------------------------------------------------
# Example 2: Structured document extraction
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    name: str
    entity_type: Literal["person", "organization", "location", "product", "date"]


class DocumentAnalysis(BaseModel):
    summary: str
    entities: list[Entity]
    key_topics: list[str]
    sentiment: Literal["positive", "negative", "neutral", "mixed"]


print("\n" + "=" * 60)
print("Example 2: Nested Schema — Document Entity Extraction")
print("=" * 60)

doc_rf = {
    "type": "json_schema",
    "json_schema": {
        "name": "DocumentAnalysis",
        "schema": to_openai_schema(DocumentAnalysis),
        "strict": True,
    },
}

article = """
Apple Inc. announced on March 15, 2026 that their new M5 chip has set a
record in benchmark performance. CEO Tim Cook called it "the most
significant leap in computing we've made at Apple." The chips will be
manufactured in partnership with TSMC in Taiwan and are expected to
power the next generation of MacBooks shipping later this year.
"""

result = adapter.generate_structured(
    prompt=f"Analyze this article:\n\n{article}",
    response_format=doc_rf,
)
doc = DocumentAnalysis.model_validate_json(result.text)

print(f"\nSummary: {doc.summary}")
print(f"Sentiment: {doc.sentiment}")
print(f"Key topics: {', '.join(doc.key_topics)}")
print(f"Entities ({len(doc.entities)}):")
for entity in doc.entities:
    print(f"  [{entity.entity_type:12}] {entity.name}")

# ---------------------------------------------------------------------------
# Example 3: Refusal handling
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Example 3: ModelRefusalError Handling")
print("=" * 60)


class SafetyRating(BaseModel):
    is_safe: bool
    reason: str


safety_rf = {
    "type": "json_schema",
    "json_schema": {
        "name": "SafetyRating",
        "schema": to_openai_schema(SafetyRating),
        "strict": True,
    },
}

test_prompts = [
    "Rate the safety of: 'Always wear a seatbelt when driving.'",
    "Rate the safety of: 'Exercise regularly and eat a balanced diet.'",
]

for prompt in test_prompts:
    try:
        result = adapter.generate_structured(prompt, response_format=safety_rf)
        rating = SafetyRating.model_validate_json(result.text)
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"  is_safe: {rating.is_safe}")
        print(f"  reason:  {rating.reason[:80]}")
    except ModelRefusalError as e:
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"  ModelRefusalError: {e.refusal_message[:100]}")
        print("  (handled without raising — application can show a safe fallback)")

adapter.unload()
print("\n\nAll structured output examples completed successfully.")
print()
print("Key takeaways:")
print("  • to_openai_schema() converts Pydantic models → OpenAI JSON Schema")
print("  • generate_structured() guarantees schema-valid JSON responses")
print("  • ModelRefusalError is raised if the model declines to answer")
print("  • Nested Pydantic models are fully supported ($refs inlined)")
