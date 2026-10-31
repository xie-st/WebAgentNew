"""
Text processing tool for text analysis and transformation.

Provides word count, text summarization (extractive), regex operations,
text comparison, and other NLP-lite operations using only the standard library.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


class TextProcessingTool(BaseTool):
    """
    Text analysis and transformation tool.

    Features:
    - Word count, character count, sentence count
    - Extractive summarization (sentence ranking)
    - Regex search and replace
    - Text comparison (similarity)
    - Case conversion
    - No external dependencies
    """

    # Natural-name aliases for the ``operation`` selector.
    operation_aliases = {
        "count": "word_count",
        "wordcount": "word_count",
        "words": "word_count",
        "summary": "summarize",
        "summarise": "summarize",
        "search": "regex_search",
        "find": "regex_search",
        "match": "regex_search",
        "regex": "regex_search",
        "replace": "regex_replace",
        "sub": "regex_replace",
        "substitute": "regex_replace",
        "diff": "compare",
        "comparison": "compare",
    }

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="text_processing",
                description=(
                    "Process and analyze text: word count, summarize, regex search/replace, "
                    "compare texts, and transform text. No API key needed."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform",
                        required=True,
                        enum=["word_count", "summarize", "regex_search", "regex_replace", "compare", "transform"],
                    ),
                    ParameterSpec(
                        name="text",
                        type=ParameterType.STRING,
                        description="Input text to process",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="pattern",
                        type=ParameterType.STRING,
                        description="Regex pattern (for regex_search/regex_replace)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="replacement",
                        type=ParameterType.STRING,
                        description="Replacement string (for regex_replace)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="text2",
                        type=ParameterType.STRING,
                        description="Second text (for compare operation)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="transform_type",
                        type=ParameterType.STRING,
                        description="Transformation type (for transform operation)",
                        required=False,
                        default="lowercase",
                        enum=["lowercase", "uppercase", "title", "reverse", "strip"],
                    ),
                    ParameterSpec(
                        name="num_sentences",
                        type=ParameterType.INTEGER,
                        description="Number of sentences for summarization (default: 3)",
                        required=False,
                        default=3,
                        min_value=1,
                        max_value=20,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "result": {"type": "any"},
                    },
                },
                timeout_seconds=10,
                tags=["text", "nlp", "processing", "regex", "summarize"],
                examples=[
                    {
                        "operation": "word_count",
                        "text": "Hello world this is a test",
                        "output": {"word_count": 6, "char_count": 26},
                    },
                    {
                        "operation": "regex_search",
                        "text": "Call 555-1234 or 555-5678",
                        "pattern": r"\d{3}-\d{4}",
                        "output": {"matches": ["555-1234", "555-5678"]},
                    },
                ],
            )
        )

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences."""
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extractive_summarize(self, text: str, num_sentences: int) -> str:
        """
        Simple extractive summarization by sentence scoring.

        Scores sentences based on word frequency and position.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text

        # Word frequency scoring
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "and", "or", "but", "not", "this", "that",
            "it", "its", "as", "if", "so", "no", "can", "may",
        }
        word_freq = Counter(w for w in words if w not in stop_words and len(w) > 2)

        # Score each sentence
        scored = []
        for i, sent in enumerate(sentences):
            sent_words = re.findall(r'\b\w+\b', sent.lower())
            score = sum(word_freq.get(w, 0) for w in sent_words)
            # Position bonus: first and last sentences get boost
            if i == 0:
                score *= 1.5
            elif i == len(sentences) - 1:
                score *= 1.2
            # Normalize by length
            if sent_words:
                score /= len(sent_words)
            scored.append((i, score, sent))

        # Select top sentences, maintaining original order
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = sorted(scored[:num_sentences], key=lambda x: x[0])

        return " ".join(s[2] for s in selected)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word-overlap similarity (Jaccard)."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return round(len(intersection) / len(union), 4)

    async def _execute(
        self,
        operation: str,
        text: str,
        pattern: str | None = None,
        replacement: str | None = None,
        text2: str | None = None,
        transform_type: str = "lowercase",
        num_sentences: int = 3,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute text processing operation."""

        if operation == "word_count":
            words = re.findall(r'\b\w+\b', text)
            sentences = self._split_sentences(text)
            return {
                "word_count": len(words),
                "char_count": len(text),
                "sentence_count": len(sentences),
                "line_count": len(text.splitlines()),
                "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
            }

        elif operation == "summarize":
            summary = self._extractive_summarize(text, num_sentences)
            return {
                "summary": summary,
                "original_sentences": len(self._split_sentences(text)),
                "summary_sentences": len(self._split_sentences(summary)),
            }

        elif operation == "regex_search":
            if not pattern:
                raise ValueError("'pattern' required for regex_search")
            try:
                matches = re.findall(pattern, text)
                return {
                    "matches": matches,
                    "count": len(matches),
                    "pattern": pattern,
                }
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        elif operation == "regex_replace":
            if not pattern:
                raise ValueError("'pattern' required for regex_replace")
            replacement = replacement or ""
            try:
                result = re.sub(pattern, replacement, text)
                return {
                    "result": result,
                    "pattern": pattern,
                    "replacement": replacement,
                }
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        elif operation == "compare":
            if not text2:
                raise ValueError("'text2' required for compare operation")
            similarity = self._text_similarity(text, text2)
            return {
                "similarity": similarity,
                "text1_words": len(re.findall(r'\b\w+\b', text)),
                "text2_words": len(re.findall(r'\b\w+\b', text2)),
            }

        elif operation == "transform":
            transforms = {
                "lowercase": text.lower(),
                "uppercase": text.upper(),
                "title": text.title(),
                "reverse": text[::-1],
                "strip": text.strip(),
            }
            result = transforms.get(transform_type, text)
            return {"result": result, "transform": transform_type}

        raise ValueError(f"Unknown operation: {operation}")
