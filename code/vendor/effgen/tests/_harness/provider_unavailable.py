"""Telling "the provider would not serve this" apart from "effGen is wrong".

A live test asserts something about effGen's behaviour, and it can only do that
if the provider ran the call. When a free tier is spent, a queue is full, or the
provider is having an outage, the run reports the *account's* state or the
*provider's* state — neither is a defect in the tree, and reading it as one
makes a suite that cannot be green for reasons nobody can fix.

So a failure whose text carries an unambiguous provider-capacity or
provider-outage statement is converted to a skip, the same way an absent
optional dependency is (see ``optional_deps``).

**The rule is deliberately narrow, and the exclusion matters more than the
inclusions.** A bare connection error is NOT here. "Connection error" is what
effGen reports when it cannot reach the endpoint at all, and the commonest cause
of that is local: a wrong or blank ``OPENAI_BASE_URL``, a proxy, a firewall. On
14 Aug 2026 exactly that misconfiguration turned 45 live cells red, and every
one of them said "Transient provider error — retry shortly; check the provider
status page". Converting that shape to a skip would have hidden the defect
instead of surfacing it. A provider that is genuinely down says so with a status
code and a sentence; that is what this matches.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

#: Statements that mean "your account has no capacity right now". Each is a
#: quota, a rate, or a billing state — recoverable by waiting or by paying, and
#: unaffected by any change to effGen.
_EXHAUSTED = (
    "[rate_limited]",
    "resource_exhausted",
    "rate_limit_exceeded",
    "rate limit reached",
    "rate limit hit",
    "quota exceeded",
    "exceeded your current quota",
    "too_many_requests",
    "queue_exceeded",
    "tokens per minute",
    "tokens per day",
    "requests per minute",
    "requests per day",
    "insufficient credit",
    "insufficient_quota",
    "out of credits",
    "billing hard limit",
)

#: Statements that mean "the provider is not serving right now" — its own
#: infrastructure, in its own words.
_PROVIDER_DOWN = (
    "internal server error",
    "service unavailable",
    "is currently unavailable",
    "temporarily unavailable",
    # Covers "overloaded_error", "service overloaded", "is currently
    # overloaded" and Anthropic's bare "Overloaded" in one word.
    "overloaded",
    "bad gateway",
    "gateway timeout",
    "deadline exceeded",
    "server had an error",
    "no healthy upstream",
    "upstream connect error",
    "model_not_ready",
    # Hugging Face keeps a cold model behind a 503 while it warms up.
    "currently loading",
    "model is loading",
)

#: Exception types an SDK raises for the two states above. A provider that
#: reports an outage inside an ``invalid_request_error`` envelope — Fireworks
#: does — leaves its exception class as the only signal in the text.
_EXHAUSTED_TYPES = (
    "ratelimiterror",
    "rate_limit_error",
    "resourceexhausted",
    "toomanyrequests",
    "insufficientquota",
)
_DOWN_TYPES = (
    "internalservererror",
    "serviceunavailableerror",
    "serviceunavailable",
    "overloadederror",
    "badgatewayerror",
    # Google's is spelled plainly enough to collide with an effGen class of the
    # same name, so it is matched with its module.
    "genai.errors.servererror",
)

#: The statuses that mean each state, and the spellings a status arrives in.
#: One pattern rather than one string per spelling: ``Error code: 429``,
#: ``429 RESOURCE_EXHAUSTED``, ``status_code=503``, ``'code': 503`` and
#: ``HTTP 502`` are the same fact, and matching them one at a time is what left
#: three consecutive runs red on three different providers.
_NO_CAPACITY_STATUS = frozenset({"402", "429"})
_NOT_SERVING_STATUS = frozenset({"500", "502", "503", "504", "529"})
_STATUS_RE = re.compile(
    r"(?:error\s*code|status(?:[_\s]*code)?|http)\s*[:=]?\s*(\d{3})\b"
    r"|['\"]code['\"]\s*:\s*(\d{3})\b"
    r"|\b(\d{3})\s+(?:resource_exhausted|unavailable|internal|too_many_requests|"
    r"deadline_exceeded|bad\s+gateway|gateway\s+timeout|service\s+unavailable)\b",
    re.I,
)

#: A provider has to be named for the text to be about a provider call at all.
_PROVIDERS = (
    "openai", "anthropic", "claude", "gemini", "google", "groq", "cerebras",
    "together", "fireworks", "replicate", "huggingface", "hugging face",
    "mistral", "cohere", "ollama", "vertex", "bedrock",
)


def _status_in(line: str) -> str | None:
    """Return the first HTTP status in *line*, however the provider spelled it."""
    for match in _STATUS_RE.finditer(line):
        for group in match.groups():
            if group:
                return group
    return None


def provider_unavailable_reason(text: str) -> str | None:
    """Return why the provider could not serve the call, or None.

    Two passes, because providers report the same fact two different ways:

    * the provider's **own words** or its SDK's exception type, anywhere in the
      failure — this is the narrow, curated half;
    * a **status code** that means no-capacity or not-serving, on a line that
      also names the provider. The locality requirement is what keeps an
      offline test asserting its own 503 out of this rule: that assertion says
      nothing about a provider on the line it fails on.

    Args:
        text: The rendered failure text of a test.

    Returns:
        A short reason naming what was matched, or None when the failure is not
        an unambiguous provider-capacity or provider-outage statement.
    """
    if not text:
        return None
    low = text.lower()
    if not any(name in low for name in _PROVIDERS):
        return None

    for needle in _EXHAUSTED:
        if needle in low:
            return f"the provider has no capacity for this call ({needle.strip('[]')})"
    for needle in _EXHAUSTED_TYPES:
        if needle in low:
            return f"the provider has no capacity for this call ({needle})"
    for needle in _PROVIDER_DOWN:
        if needle in low:
            return f"the provider is not serving right now ({needle})"
    for needle in _DOWN_TYPES:
        if needle and needle in low:
            return f"the provider is not serving right now ({needle})"

    for raw_line in low.splitlines():
        if not any(name in raw_line for name in _PROVIDERS):
            continue
        status = _status_in(raw_line)
        if status in _NO_CAPACITY_STATUS:
            return f"the provider has no capacity for this call (status {status})"
        if status in _NOT_SERVING_STATUS:
            return f"the provider is not serving right now (status {status})"
    return None


def skip_if_provider_refused(*parts: Any, what: str = "the call") -> None:
    """Skip when a *result* carries a provider refusal instead of an answer.

    The report hook classifies a failure by the text of that failure, and an API
    that reports errors in its return value leaves that text nowhere the hook
    can reach: a workflow whose node was rate limited comes back
    ``success=False`` and the assertion reads ``assert False is True``, with the
    provider's sentence inside the result object. A live test that asserts on a
    returned result therefore has to look for itself, before it asserts.

    Args:
        *parts: Result objects or strings; each is rendered with ``str`` and the
            rule is applied to all of them together.
        what: What was being attempted, for the skip reason.
    """
    text = " ".join(str(part) for part in parts)
    reason = provider_unavailable_reason(text)
    if reason:
        pytest.skip(f"{what}: {reason} — {first_provider_sentence(text)}")


def first_provider_sentence(text: str, limit: int = 240) -> str:
    """Return the provider's own sentence from *text*, for the skip reason."""
    for line in text.splitlines():
        stripped = line.lstrip("E ").strip()
        if re.search(
            r"error code: \d{3}|\[rate_limited\]|RESOURCE_EXHAUSTED"
            r"|service overloaded|ServiceUnavailableError",
            stripped,
            re.I,
        ):
            return stripped[:limit]
    return text.strip().splitlines()[-1][:limit] if text.strip() else ""


def assert_cli_succeeded(proc: Any, what: str = "the command") -> None:
    """Assert a CLI subprocess exited 0, keeping both of its streams.

    The rule above classifies a failure by reading its text, and across a
    subprocess boundary the provider's sentence only reaches the parent through
    what the child printed. A CLI that reports per-item outcomes prints them on
    **stdout** — with ``--json``, the refusal is a field in that document — while
    the process-level summary goes to stderr. So a message built from stderr
    alone drops exactly the sentence the rule needs, and a throttled run is
    reported as a defect in the tree.

    Both streams therefore go into the message. Use this instead of
    ``assert proc.returncode == 0, proc.stderr`` wherever the subprocess calls a
    provider.
    """
    if proc.returncode == 0:
        return
    raise AssertionError(
        f"{what} exited {proc.returncode}\n"
        f"--- stderr ---\n{proc.stderr}\n"
        f"--- stdout ---\n{proc.stdout}"
    )
