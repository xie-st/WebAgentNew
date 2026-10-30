"""
LibraryPrompt — base dataclass for all prompt library entries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LibraryPrompt:
    """A named, versioned prompt template with evaluation metadata."""

    name: str
    """Dot-qualified identifier, e.g. 'research.literature_review.v1'."""

    domain: str
    """Top-level domain, e.g. 'research', 'coding', 'data', 'legal', 'medical'."""

    variant: str
    """Style: 'zero_shot' | 'cot' | 'few_shot' | 'tool' | 'structured'."""

    description: str

    template: Callable[..., str]
    """Deterministic render function.  Called as template(**inputs) → str."""

    input_schema: dict[str, Any]
    """JSON-Schema dict describing accepted inputs."""

    fixture: dict[str, Any]
    """Default input values used by the eval harness."""

    expected_shape: dict[str, Any] | None = None
    """Optional output shape spec:
       {'type': 'json', 'schema': {...}}
       {'type': 'regex', 'pattern': '...'}
       {'type': 'callable', 'fn': <callable>}
    """

    tags: list[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> str:
        """Render with given inputs (falls back to fixture for missing keys)."""
        merged = {**self.fixture, **kwargs}
        return self.template(**merged)

    def render_fixture(self) -> str:
        """Render with fixture inputs."""
        return self.template(**self.fixture)

    def validate_input(self, inputs: dict[str, Any]) -> list[str]:
        """Validate a caller-supplied input object against ``input_schema``.

        Checks exactly what was supplied — it does not fall back to
        ``fixture`` for missing fields. Returns a field-named error message
        per schema violation (empty list when *inputs* is valid).
        """
        from jsonschema import Draft202012Validator

        validator = Draft202012Validator(self.input_schema)
        errors = sorted(validator.iter_errors(inputs), key=lambda exc: list(exc.absolute_path))
        messages = []
        for exc in errors:
            path = ".".join(str(p) for p in exc.absolute_path)
            messages.append(f"'{path}': {exc.message}" if path else exc.message)
        return messages
