"""LoCoMo and LongMemEval: answering questions about a long conversation.

Both are scored with partial credit, not exact match, exactly as the paper's
scripts did. LoCoMo uses a stemmed token F1 and LongMemEval a graded overlap
score, so `score()` returns a float between 0 and 1 rather than a bool. The
reported accuracy is the mean of those scores.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from functools import lru_cache
from typing import Any

from ..config import DATA_DIR
from ..types import Sample
from .base import Benchmark

# The conversation is pasted into the prompt. The paper capped it here so it
# fits a small model's context, and the cap has to stay for the numbers to line up.
MAX_CONTEXT_CHARS = 12000

LOCOMO_SYSTEM = """You are a helpful AI assistant with excellent memory capabilities. You have been given a conversation history between two people. Your task is to answer questions about this conversation accurately.

IMPORTANT GUIDELINES:
1. Answer based ONLY on the information provided in the conversation
2. Be concise and precise in your answers
3. If the information is not in the conversation, say "The information is not available in the conversation"
4. For temporal questions, provide specific dates or time references mentioned
5. For multi-part questions, ensure you cover all parts
6. State your final answer clearly

Always end your response with: "The answer is: [YOUR ANSWER]"
"""

LOCOMO_EXAMPLES = """Here are examples of how to answer questions:

Example 1:
Question: When did Alice visit the museum?
Based on the conversation, Alice mentioned visiting the museum on June 15, 2023.
The answer is: June 15, 2023

Example 2:
Question: What hobbies does Bob have?
According to the conversation, Bob mentioned enjoying painting and hiking.
The answer is: painting, hiking

Example 3:
Question: What did Alice realize after the meeting?
This specific information is not mentioned in the conversation.
The answer is: The information is not available in the conversation

Now answer the following question based on the conversation:
"""

LONGMEM_SYSTEM = """You are a helpful AI assistant with excellent long-term memory. You have been given a history of previous conversations. Your task is to answer questions about these conversations accurately.

IMPORTANT GUIDELINES:
1. Answer based ONLY on information from the conversation history
2. Be concise and direct in your answers
3. For temporal questions, calculate time differences carefully
4. If information is NOT in the history, clearly state that
5. Do not make up information

Always end your response with: "The answer is: [YOUR ANSWER]"
"""

ABSTENTION_PHRASES = [
    "not available", "not mentioned", "no information", "cannot find",
    "not in the conversation", "not stated", "doesn't mention",
    "does not mention", "no record", "unknown", "cannot determine",
    "not specified", "no evidence",
]


# ------------------------------------------------------------------ scoring


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def _normalize(text: str) -> str:
    text = str(text).lower().replace(",", "")
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _f1(prediction: str, truth: str) -> float:
    stem = _stemmer().stem
    pred_tokens = [stem(w) for w in _normalize(prediction).split()]
    gold_tokens = [stem(w) for w in _normalize(truth).split()]
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _f1_multi(prediction: str, truth: str) -> float:
    """Comma separated answers count as a set: each gold part scores its best match."""
    preds = [p.strip() for p in prediction.split(",")]
    golds = [g.strip() for g in str(truth).split(",")]
    if not golds:
        return 0.0
    return sum(max(_f1(p, g) for p in preds) for g in golds) / len(golds)


def _abstained(prediction: str) -> bool:
    low = prediction.lower()
    return any(phrase in low for phrase in ABSTENTION_PHRASES)


def _graded_overlap(prediction: str, truth: str) -> float:
    pred = _normalize(prediction)
    gold = _normalize(truth)
    if not pred or not gold:
        return 0.0
    if pred == gold:
        return 1.0
    if gold in pred:
        return 0.8
    gold_words = set(gold.split())
    if not gold_words:
        return 0.0
    recall = len(set(pred.split()) & gold_words) / len(gold_words)
    return recall if recall > 0.7 else 0.0


def _final_answer(response: str) -> str:
    """The answer a response is offering, marker or no marker.

    Two things here were getting the memory columns wrong for every framework.

    The markers are searched for the *last* match, not the first. The system
    prompt asks the model to close with "The answer is: X", so a model that
    mentions the phrase while reasoning and then closes with it was scored on
    the first mention.

    The no-marker fallback used to be ``response.strip().split(".")[-1]``,
    which is the empty string for any response ending in a full stop — most of
    them. That made 50-95% of predictions empty, so both benchmarks were
    measuring whether the model typed the marker rather than whether it knew
    the answer. Frameworks that rewrite the closing line (effGen's ReAct loop,
    Smolagents) lost the marker and scored near zero on answers that were
    right. The fallback is now the last non-empty line, and its last sentence
    only if that line runs long.
    """
    if not response:
        return ""
    for pattern in [
        r"[Tt]he\s+answer\s+is\s*:?\s*(.+?)(?:\n|$)",
        r"[Ff]inal\s+[Aa]nswer\s*:?\s*(.+?)(?:\n|$)",
        r"[Aa]nswer\s*:?\s*(.+?)(?:\n|$)",
    ]:
        matches = re.findall(pattern, response)
        if matches:
            answer = str(matches[-1]).strip().rstrip(".").strip()
            if answer:
                return answer

    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    if len(last) <= 200:
        return last.rstrip(".").strip()
    sentences = [s.strip() for s in last.split(".") if s.strip()]
    return sentences[-1] if sentences else last.strip()


def _clip(text: str) -> str:
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    return "...[earlier conversation truncated]...\n" + text[-MAX_CONTEXT_CHARS:]


# ------------------------------------------------------------------- LoCoMo


class LoCoMo(Benchmark):
    key = "locomo"
    label = "LoCoMo"
    category = "Memory"
    tools = ()

    def system_prompt(self, with_tools: bool) -> str:
        return LOCOMO_SYSTEM

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return (
            "Here is a conversation between two people:\n\n"
            f"{sample.context}\n\n"
            f"{LOCOMO_EXAMPLES}\n"
            f"Question: {sample.question}"
        )

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        with open(DATA_DIR / "locomo" / "locomo10.json") as f:
            data = json.load(f)

        samples: list[Sample] = []
        for conv_idx, entry in enumerate(data):
            conv_id = entry.get("sample_id", f"conv-{conv_idx}")
            text = _clip(self._format_conversation(entry.get("conversation", {})))
            for qa_idx, qa in enumerate(entry.get("qa", [])):
                samples.append(
                    Sample(
                        sample_id=f"locomo-{conv_id}-{qa_idx}",
                        question=qa.get("question", ""),
                        answer=str(qa.get("answer", "")),
                        context=text,
                        meta={"category": qa.get("category", 1), "conversation": conv_id},
                    )
                )
        return self._slice(samples, limit, offset)

    @staticmethod
    def _format_conversation(conversation: dict) -> str:
        parts = []
        index = 1
        while f"session_{index}" in conversation:
            date = conversation.get(f"session_{index}_date_time", f"Session {index}")
            parts.append(f"\n--- {date} ---")
            for turn in conversation[f"session_{index}"]:
                parts.append(f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}")
            index += 1
        return "\n".join(parts)

    def score(self, sample: Sample, output: str) -> tuple[float, Any]:
        prediction = _final_answer(output)
        # Category 5 questions have no answer in the conversation. The model is
        # right when it says so.
        if sample.meta.get("category") == 5:
            return (1.0 if _abstained(prediction) else 0.0), prediction
        return _f1_multi(prediction, sample.answer), prediction


# -------------------------------------------------------------- LongMemEval


class LongMemEval(Benchmark):
    key = "longmemeval"
    label = "LongMemEval"
    category = "Memory"
    tools = ()

    def system_prompt(self, with_tools: bool) -> str:
        return LONGMEM_SYSTEM

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return (
            "Here is the history of your previous conversations:\n\n"
            f"{sample.context}\n\n"
            f"Question: {sample.question}"
        )

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        with open(DATA_DIR / "longmemeval" / "longmemeval_oracle.json") as f:
            data = json.load(f)

        samples = []
        for idx, item in enumerate(data):
            samples.append(
                Sample(
                    sample_id=f"longmemeval-{item.get('question_id', idx)}",
                    question=item.get("question", ""),
                    answer=str(item.get("answer", "")),
                    context=_clip(self._format_history(item)),
                    meta={"question_type": item.get("question_type", "unknown")},
                )
            )
        return self._slice(samples, limit, offset)

    @staticmethod
    def _format_history(item: dict) -> str:
        parts = []
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        for i, session in enumerate(sessions):
            date = dates[i] if i < len(dates) else f"Session {i + 1}"
            parts.append(f"\n--- Conversation on {date} ---")
            for turn in session:
                who = "User" if turn.get("role") == "user" else "Assistant"
                parts.append(f"{who}: {turn.get('content', '')}")
        return "\n".join(parts)

    def score(self, sample: Sample, output: str) -> tuple[float, Any]:
        prediction = _final_answer(output)
        # Types ending in _abs have no answer in the history on purpose.
        if str(sample.meta.get("question_type", "")).endswith("_abs"):
            return (1.0 if _abstained(prediction) else 0.0), prediction
        return _graded_overlap(prediction, sample.answer), prediction
