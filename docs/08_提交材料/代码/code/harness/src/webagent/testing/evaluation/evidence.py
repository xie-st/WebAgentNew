"""External evidence records supplied to the local judge.

The site cannot know that a browser screenshot happened, so the judge accepts
an explicit evidence bundle from the run. Records are local file references;
the judge verifies existence, PNG signature and an optional SHA-256. Hidden
site truth remains the primary source for draft/cancel/submission checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping


class EvidenceError(ValueError):
    """Malformed evidence record."""


@dataclass(frozen=True)
class EvidenceRecord:
    object_type: str
    object_id: str
    kind: str = "screenshot"
    reference: str = ""
    sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.object_type not in ("product", "post"):
            raise EvidenceError("evidence.object_type must be product or post")
        if not str(self.object_id).strip():
            raise EvidenceError("evidence.object_id is required")
        if not str(self.kind).strip():
            raise EvidenceError("evidence.kind is required")
        if not str(self.reference).strip():
            raise EvidenceError("evidence.reference is required")
        if self.sha256 is not None and len(str(self.sha256)) != 64:
            raise EvidenceError("evidence.sha256 must be a 64-character hex digest")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvidenceRecord":
        if not isinstance(raw, Mapping):
            raise EvidenceError("evidence record must be a mapping")
        return cls(
            object_type=str(raw.get("object_type", "")),
            object_id=str(raw.get("object_id", "")),
            kind=str(raw.get("kind", "screenshot")),
            reference=str(raw.get("reference", "")),
            sha256=raw.get("sha256"),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": self.object_type,
            "object_id": self.object_id,
            "kind": self.kind,
            "reference": self.reference,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
        }


class EvidenceBundle:
    """Ordered evidence records for one judged run."""

    def __init__(self, records: Iterable[EvidenceRecord | Mapping[str, Any]] = ()) -> None:
        normalized = [
            record if isinstance(record, EvidenceRecord) else EvidenceRecord.from_mapping(record)
            for record in records
        ]
        self.records: tuple[EvidenceRecord, ...] = tuple(normalized)

    @classmethod
    def coerce(cls, value: Any) -> "EvidenceBundle":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, EvidenceRecord):
            return cls([value])
        if isinstance(value, Mapping):
            return cls([value])
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return cls(value)
        raise EvidenceError("evidence must be a bundle, a record or an iterable of records")

    def for_object(
        self, object_type: str, object_id: str, *, kind: str = "screenshot"
    ) -> list[EvidenceRecord]:
        return [
            record
            for record in self.records
            if record.object_type == object_type
            and record.object_id == object_id
            and record.kind == kind
        ]

    def __iter__(self) -> Iterator[EvidenceRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def to_list(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]


__all__ = ["EvidenceBundle", "EvidenceError", "EvidenceRecord"]
