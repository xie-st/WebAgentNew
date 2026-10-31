"""
Secret redaction for effGen structured logging.

All log attributes are passed through a ``Redactor`` before being serialised
to JSON.  The redactor applies a set of regex patterns that match common API
keys, bearer tokens, and webhook URLs, replacing the matched text with a
labelled placeholder like ``<REDACTED:openai_key>``.

Built-in patterns
-----------------
- ``openai_key``     — ``sk-[a-zA-Z0-9_-]{20,}``
- ``anthropic_key``  — ``sk-ant-[a-zA-Z0-9_-]{20,}``
- ``cerebras_key``   — ``csk-[a-zA-Z0-9_-]{20,}``
- ``google_key``     — ``AIza[0-9A-Za-z_-]{35,}``
- ``hf_key``         — ``hf_[a-zA-Z0-9_-]{20,}``
- ``groq_key``       — ``gsk_[a-zA-Z0-9_-]{20,}``
- ``replicate_key``  — ``r8_[A-Za-z0-9_-]{30,}``
- ``fireworks_key``  — ``fw_[A-Za-z0-9_-]{20,}``
- ``github_token``   — ``github_pat_…`` / ``ghp_…`` and the other ``gh?_`` prefixes
- ``slack_token``    — ``xox[baprs]-…``
- ``aws_access_key`` — ``(AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}``
- ``bearer_token``   — ``Bearer [^\\s]+`` (case-insensitive keyword)
- ``slack_webhook``  — Slack incoming webhook URLs (host kept, path replaced)
- ``discord_webhook`` — Discord webhook path
- ``env_secret``     — generic ``*_API_KEY=…`` / ``*SECRET…=…`` / ``*TOKEN=…`` /
  ``*PASSWORD=…`` env-style assignments (the variable name is kept, only the
  value is replaced).  This is the catch-all that covers providers without a
  distinctive key prefix — Together, Azure, AWS secret keys, etc. — whenever
  they appear as ``VAR=value`` (e.g. an environment dump in a stack trace).

User-extensible via ``Redactor.add_pattern(name, regex)``.

Usage
-----
    from effgen.observability.redact import get_redactor, Redactor

    r = get_redactor()
    r.scrub("my key is sk-abc123xyz9876543210")
    # → "my key is <REDACTED:openai_key>"

    # Redact a dict (recursive, values only — keys are never secrets)
    clean = r.scrub_dict({"Authorization": "Bearer sk-abc123xyz9876543210"})
    # → {"Authorization": "<REDACTED:bearer_token>"}
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

# Each entry is (name, regex, replacement-template).  The template is fed to
# ``re.Pattern.sub`` so it may reference capture groups (``\g<1>`` etc.).  Most
# patterns redact the whole match; ``env_secret`` keeps the variable name and
# separator and replaces only the value.
#
# Ordered so that more-specific patterns are applied before less-specific ones
# (e.g. anthropic_key before openai_key because both start with "sk-"; every
# prefixed provider key before the generic ``env_secret`` catch-all).
_REDACT = "<REDACTED:{}>"

_BUILTIN_PATTERNS: list[tuple[str, str, str]] = [
    # More-specific first
    # Every key body accepts ``_`` and ``-`` and is open-ended, so a token that
    # is longer than the canonical length or that contains an underscore is
    # replaced whole instead of leaving a usable tail in the log line.
    ("anthropic_key", r"sk-ant-[a-zA-Z0-9_\-]{20,}", _REDACT.format("anthropic_key")),
    ("cerebras_key",  r"csk-[a-zA-Z0-9_\-]{20,}",    _REDACT.format("cerebras_key")),
    ("google_key",    r"AIza[0-9A-Za-z_\-]{35,}",    _REDACT.format("google_key")),
    ("hf_key",        r"hf_[a-zA-Z0-9_\-]{20,}",      _REDACT.format("hf_key")),
    ("groq_key",      r"gsk_[a-zA-Z0-9_\-]{20,}",     _REDACT.format("groq_key")),
    # Replicate API tokens: ``r8_`` + 30+ url-safe chars.
    ("replicate_key", r"r8_[A-Za-z0-9_\-]{30,}",      _REDACT.format("replicate_key")),
    # Fireworks API keys: ``fw_`` + 20+ chars.
    ("fireworks_key", r"fw_[A-Za-z0-9_\-]{20,}",      _REDACT.format("fireworks_key")),
    # GitHub tokens: classic ``gh?_`` prefixes and fine-grained ``github_pat_``.
    ("github_token",  r"(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9_]{30,})",
     _REDACT.format("github_token")),
    # Slack bot/user/app tokens.
    ("slack_token",   r"xox[baprs]-[A-Za-z0-9\-]{10,}", _REDACT.format("slack_token")),
    # AWS access key IDs: a fixed 4-letter type prefix + 16 upper/digits.
    ("aws_access_key", r"(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}",
     _REDACT.format("aws_access_key")),
    # openai key AFTER more-specific sk-ant- / csk- patterns
    ("openai_key",    r"sk-[a-zA-Z0-9_\-]{20,}",      _REDACT.format("openai_key")),
    # Bearer token — matches "Bearer <anything non-whitespace>", in any case
    # (an HTTP header dump may carry "bearer" or "BEARER").
    ("bearer_token",  r"(?i:Bearer) [^\s]{6,}",       _REDACT.format("bearer_token")),
    # Slack incoming webhook: keep host, mask path
    ("slack_webhook", r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+",
     _REDACT.format("slack_webhook")),
    # Discord webhook: keep host, mask path
    ("discord_webhook", r"https://discord(?:app)?\.com/api/webhooks/[0-9/A-Za-z_\-]+",
     _REDACT.format("discord_webhook")),
]

# Last-resort generic patterns, applied AFTER every specific pattern *and* every
# user-added pattern. The env-style secret assignment is the catch-all for
# providers without a distinctive key prefix (Together, Azure, AWS secret access
# keys, …) and for accidental environment dumps. It keeps the variable name +
# separator and redacts only the value. The value lookahead avoids re-redacting
# a value a more-specific pattern already replaced. Running it last means a
# user's own ``add_pattern`` still wins on the same span.
_FALLBACK_PATTERNS: list[tuple[str, str, str]] = [
    ("env_secret",
     (r"(?i)(\b[A-Za-z0-9_]*"
     r"(?:API[_\-]?KEY|APIKEY|SECRET[_\-]?ACCESS[_\-]?KEY|ACCESS[_\-]?KEY|"
     r"SECRET[_\-]?KEY|AUTH[_\-]?TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|TOKEN))"
     r"([\"']?\s*[:=]\s*[\"']?)(?!<REDACTED)[^\s\"',]{4,}"),
     r"\g<1>\g<2>" + _REDACT.format("env_secret")),
]


class Redactor:
    """
    Apply regex-based secret redaction to strings and nested data structures.

    Thread-safe: compiled pattern list is built once at construction; later
    ``add_pattern`` calls append to the list atomically enough for logging
    use (GIL-protected list append in CPython).
    """

    def __init__(self) -> None:
        # List of (name, compiled_regex, replacement) triples — order matters
        # (first match wins when the same span is covered by multiple patterns,
        # though that is rare).  ``replacement`` is a ``re.sub`` template and may
        # reference capture groups (used by ``env_secret`` to keep the var name).
        self._patterns: list[tuple[str, re.Pattern[str], str]] = []
        for name, pattern, replacement in _BUILTIN_PATTERNS:
            self._patterns.append((name, re.compile(pattern), replacement))
        # Generic last-resort patterns, always applied after self._patterns
        # (including any user-added ones) so a caller's own pattern wins.
        self._fallback_patterns: list[tuple[str, re.Pattern[str], str]] = [
            (name, re.compile(pattern), replacement)
            for name, pattern, replacement in _FALLBACK_PATTERNS
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_pattern(self, name: str, regex: str, replacement: str | None = None) -> None:
        """
        Register a custom redaction pattern.

        Args:
            name:  Label used in the placeholder, e.g. ``"my_secret"``.
            regex: Python regex string.
            replacement: Optional ``re.sub`` template.  When omitted the entire
                   match is replaced by ``<REDACTED:{name}>``.  Provide a custom
                   template (with ``\\g<1>`` group references) to keep part of
                   the match, e.g. an assignment's variable name.

        Example::

            r = get_redactor()
            r.add_pattern("custom_token", r"tok-[A-Za-z0-9]{32}")
        """
        repl = replacement if replacement is not None else _REDACT.format(name)
        self._patterns.append((name, re.compile(regex), repl))

    def scrub(self, text: str) -> str:
        """
        Redact all secrets in *text* and return the sanitised string.

        Args:
            text: Arbitrary string that may contain secrets.

        Returns:
            Copy of *text* with every matched secret replaced by a labelled
            placeholder (e.g. ``<REDACTED:openai_key>``).
        """
        for _name, pattern, replacement in self._patterns:
            text = pattern.sub(replacement, text)
        # Generic catch-all patterns run last so a user's add_pattern wins.
        for _name, pattern, replacement in self._fallback_patterns:
            text = pattern.sub(replacement, text)
        return text

    def scrub_value(self, value: Any) -> Any:
        """
        Redact a single log-field value.

        - ``str`` → scrub and return new string
        - ``dict`` → recursively scrub values (not keys)
        - ``list`` / ``tuple`` → recursively scrub elements
        - Anything else → return unchanged (cannot contain raw secret tokens)
        """
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, dict):
            return self.scrub_dict(value)
        if isinstance(value, list | tuple):
            scrubbed = [self.scrub_value(v) for v in value]
            return type(value)(scrubbed)
        return value

    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Return a new dict with all string values redacted.

        Only values are scrubbed; keys are never treated as secrets.

        Args:
            data: Dictionary to sanitise (may be nested).

        Returns:
            New dictionary with all string values scrubbed.
        """
        return {k: self.scrub_value(v) for k, v in data.items()}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def pattern_names(self) -> list[str]:
        """Return the names of all registered redaction patterns."""
        return [name for name, _pat, _repl in (*self._patterns, *self._fallback_patterns)]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_redactor: Redactor | None = None


def get_redactor() -> Redactor:
    """
    Return the process-wide default :class:`Redactor` instance.

    The instance is created lazily on first call.  The same instance is
    returned on subsequent calls — add custom patterns to it once at
    application startup.

    Returns:
        The default :class:`Redactor`.
    """
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = Redactor()
    return _default_redactor


__all__ = [
    "Redactor",
    "get_redactor",
]
