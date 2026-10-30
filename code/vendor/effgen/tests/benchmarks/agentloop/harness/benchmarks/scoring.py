"""Answer extraction and matching.

These are ports of the parsers the paper's scripts used. Keeping them identical
matters more than making them better: if the parser changes, the new numbers
stop being comparable to the old ones.
"""

from __future__ import annotations

import re

NUMBER = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"


def _to_float(text: str) -> float | None:
    text = text.replace(",", "").replace("$", "").strip()
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            return float(num) / float(den)
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def extract_number(response: str) -> float | None:
    """Pull a numeric answer out of a model response.

    Tries \\boxed{} first, then the usual "the answer is" phrasings, then falls
    back to the last number in the text.
    """
    if not response:
        return None
    response = response.strip()

    boxed = [
        rf"\\boxed\{{({NUMBER}(?:/\d+)?)\}}",
        r"\\boxed\{\\frac\{(\d+)\}\{(\d+)\}\}",
        rf"boxed\{{({NUMBER})\}}",
    ]
    for pattern in boxed:
        matches = re.findall(pattern, response)
        if matches:
            last = matches[-1]
            if isinstance(last, tuple):
                try:
                    return float(last[0]) / float(last[1])
                except (ValueError, ZeroDivisionError):
                    continue
            value = _to_float(last)
            if value is not None:
                return value

    phrases = [
        rf"[Tt]he\s+answer\s+is\s*:?\s*\$?\s*({NUMBER})",
        rf"[Ff]inal\s+[Aa]nswer\s*:?\s*\$?\s*({NUMBER})",
        rf"[Aa]nswer\s*:?\s*\$?\s*({NUMBER})",
        rf"####\s*\$?\s*({NUMBER})",
        rf"=\s*\$?\s*({NUMBER})\s*$",
    ]
    for pattern in phrases:
        matches = re.findall(pattern, response)
        if matches:
            value = _to_float(matches[-1])
            if value is not None:
                return value

    numbers = re.findall(rf"({NUMBER})", response)
    if numbers:
        return _to_float(numbers[-1])
    return None


def numbers_match(predicted: float | None, truth: float | None, tol: float = 1e-5) -> bool:
    if predicted is None or truth is None:
        return False
    return abs(predicted - truth) < tol


def extract_gsm8k_truth(answer: str) -> float | None:
    """GSM8K keeps the final answer after '####'."""
    if "####" in answer:
        return _to_float(answer.split("####")[-1])
    return None


def extract_choice(response: str, valid: str = "ABCDE") -> str | None:
    """Pull a multiple-choice letter out of a response."""
    if not response:
        return None
    text = response.strip().upper()
    letters = f"[{valid}]"
    patterns = [
        # The text is upper-cased above, so the marker is too.
        rf"\\?BOXED\{{\s*({letters})\s*\}}",
        rf"(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s*(?:IS)?\s*[:\-]?\s*\(?({letters})\)?\b",
        rf"(?:OPTION\s+)?\(?({letters})\)?\s*(?:IS\s+)?(?:CORRECT|THE\s+ANSWER)",
        rf"^\(?({letters})\)?\s*[:\.\)]",
        rf"^\(?({letters})\)?$",
        rf"\b({letters})\b\s*$",
    ]
    for pattern in patterns:
        # Last match, not first. A response that lists the options and then
        # settles on one ("... B. ... The answer is D") was scored on the first
        # phrase it happened to match, which is usually part of the reasoning.
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return matches[-1]
    return None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    return re.sub(r"[.,!?;:'\"]+$", "", text)


def extract_short_answer(response: str) -> str | None:
    """Pull a short free-text answer out of a response."""
    if not response:
        return None
    patterns = [
        r"\\boxed\{([^{}]+)\}",
        r"Final Answer:\s*(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    # No marker: use the last non-empty line, which is where models put it.
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


def short_answers_match(predicted: str | None, truth: str) -> bool:
    """Loose string match, the same rule the paper's agentic scripts used."""
    if not predicted or not truth:
        return False
    pred = normalize_text(predicted)
    gold = normalize_text(truth)
    if not pred or not gold:
        return False
    if pred == gold or gold in pred:
        return True
    pred_nums = re.findall(r"-?\d+\.?\d*", pred)
    gold_nums = re.findall(r"-?\d+\.?\d*", gold)
    if pred_nums and gold_nums:
        try:
            return abs(float(pred_nums[0]) - float(gold_nums[0])) < 0.1
        except ValueError:
            return False
    return False


# --------------------------------------------------------------- MATH-500

_LATEX_STRIP = [
    (r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""), (r"\\ ", " "),
    (r"\\\$", ""), (r"\$", ""), (r"\\%", ""), (r"%", ""), (r"\\text\{[^}]*\}", ""),
    (r"\\mbox\{[^}]*\}", ""), (r"\s+", ""),
]


def normalize_math(expr: str) -> str:
    """Flatten a LaTeX answer enough that equivalent forms compare equal."""
    if expr is None:
        return ""
    out = str(expr).strip()
    out = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", out)
    out = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", out)
    for pattern, repl in _LATEX_STRIP:
        out = re.sub(pattern, repl, out)
    out = out.replace("\\", "").rstrip(".")
    if out.endswith("^\\circ") or out.endswith("^circ"):
        out = out.split("^")[0]
    return out.lower()


def extract_math_answer(response: str) -> str | None:
    """Pull the contents of the last \\boxed{} out of a MATH-500 response."""
    if not response:
        return None
    idx = response.rfind("\\boxed")
    if idx == -1:
        idx = response.rfind("boxed")
    if idx != -1:
        brace = response.find("{", idx)
        if brace != -1:
            depth = 0
            for pos in range(brace, len(response)):
                if response[pos] == "{":
                    depth += 1
                elif response[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return response[brace + 1 : pos].strip()
    return extract_short_answer(response)


def math_answers_match(predicted: str | None, truth: str) -> bool:
    if predicted is None:
        return False
    left, right = normalize_math(predicted), normalize_math(truth)
    if not left or not right:
        return False
    if left == right:
        return True
    left_num, right_num = _to_float(left), _to_float(right)
    if left_num is not None and right_num is not None:
        return abs(left_num - right_num) < 1e-6
    return False
