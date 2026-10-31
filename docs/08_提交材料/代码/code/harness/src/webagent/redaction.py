"""Redaction for logs, configuration summaries and exception messages.

Credentials, platform account values and connection tokens never belong in a
prompt, an event, an exception message or a committed config. The redactor is
the single filter applied by :mod:`webagent.events` and configuration summary
code.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "api_key",
        "apikey",
        "api_secret",
        "client_secret",
        "token",
        "auth_token",
        "access_token",
        "refresh_token",
        "connection_token",
        "authorization",
        "credential",
        "credentials",
    }
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s\"',;}]+)"),
    re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._~+/=-]{8,})"),
    re.compile(r"(?i)\b(sk-[A-Za-z0-9_-]{6,})\b"),
    re.compile(
        r"(?i)\b((?:password|passwd|secret|api[_-]?key|api[_-]?secret|"
        r"client[_-]?secret|access[_-]?token|refresh[_-]?token|"
        r"connection[_-]?token|auth[_-]?token|token)\s*[:=]\s*)"
        r"(\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
    ),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
)


class Redactor:
    """Apply literal secrets and common credential shapes to strings."""

    def __init__(
        self,
        secrets: Iterable[str] = (),
        extra_patterns: Iterable[str] = (),
    ) -> None:
        self._secrets = tuple(
            sorted(
                {str(secret) for secret in secrets if isinstance(secret, str) and len(secret) >= 4},
                key=len,
                reverse=True,
            )
        )
        self._patterns = list(_PATTERNS)
        for pattern in extra_patterns:
            self._patterns.append(re.compile(str(pattern)))

    def redact_text(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        result = text
        for secret in self._secrets:
            result = result.replace(secret, REDACTED)
        for pattern in self._patterns:
            if pattern.groups >= 2:
                result = pattern.sub(lambda match: match.group(1) + REDACTED, result)
            else:
                result = pattern.sub(REDACTED, result)
        return result

    def redact(self, value: Any) -> Any:
        """Recursively redact strings and values under sensitive mapping keys."""

        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if key_text.lower() in _SENSITIVE_KEYS:
                    result[key_text] = REDACTED
                else:
                    result[key_text] = self.redact(item)
            return result
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self.redact(item) for item in value]
        if isinstance(value, MappingProxyType):
            return {str(key): self.redact(item) for key, item in value.items()}
        return value

    def redact_exception(self, exc: BaseException) -> str:
        return f"{type(exc).__name__}: {self.redact_text(str(exc))}"


DEFAULT_REDACTOR = Redactor()
