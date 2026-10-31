"""Run evidence directory and artifact integrity.

Layout::

    runtime/runs/<run_id>/
      events.jsonl
      result.json
      observations/<observation_id>.json
      screenshots/<...>.png
      html/<...>.html
      dom/<...>.json

Each saved artifact gets a unique reference and its actual SHA-256/byte size.
A failed save raises instead of silently pointing at an older artifact.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

from .errors import EvidenceError
from .protocol import Artifact, Observation, json_dumps, validate_run_id

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")
_SUFFIX_BY_KIND = {"screenshot": ".png", "html": ".html", "dom": ".json"}


def _safe_token(value: str, fallback: str = "obs") -> str:
    cleaned = _UNSAFE.sub("_", str(value)).strip("_")[:60]
    return cleaned or fallback


class EvidenceStore:
    def __init__(self, runtime_dir: str | Path, run_id: str) -> None:
        validate_run_id(run_id)
        self.run_id = run_id
        self.runtime_dir = Path(runtime_dir)
        self.runs_root = (self.runtime_dir / "runs").resolve()
        self.run_dir = (self.runs_root / run_id).resolve()
        if self.run_dir.parent != self.runs_root:
            raise EvidenceError("run directory escapes the runtime root")
        self.observations_dir = self.run_dir / "observations"
        self.screenshots_dir = self.run_dir / "screenshots"
        self.html_dir = self.run_dir / "html"
        self.dom_dir = self.run_dir / "dom"
        for directory in (
            self.observations_dir,
            self.screenshots_dir,
            self.html_dir,
            self.dom_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._closed = False

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def result_path(self) -> Path:
        return self.run_dir / "result.json"

    def _directory_for(self, kind: str) -> Path:
        if kind == "screenshot":
            return self.screenshots_dir
        if kind == "html":
            return self.html_dir
        if kind == "dom":
            return self.dom_dir
        raise EvidenceError(f"unknown artifact kind {kind!r}")

    def resolve_reference(self, reference: str) -> Path:
        """Resolve a stored reference, rejecting absolute paths and traversal."""

        if not isinstance(reference, str) or not reference.strip():
            raise EvidenceError("artifact reference must be a non-empty relative path")
        candidate = Path(reference)
        if candidate.is_absolute():
            raise EvidenceError("artifact reference must be relative")
        resolved = (self.run_dir / candidate).resolve()
        if resolved == self.run_dir or not resolved.is_relative_to(self.run_dir):
            raise EvidenceError("artifact reference escapes the run directory")
        return resolved

    def _atomic_write(self, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                import os

                os.fsync(handle.fileno())
            temporary.replace(target)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise EvidenceError(f"failed to write evidence {target.name}: {exc}") from exc

    def save_bytes(
        self,
        kind: str,
        data: bytes,
        *,
        observation_id: str | None = None,
        suffix: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Artifact:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise EvidenceError("evidence data must be bytes")
        if kind not in _SUFFIX_BY_KIND:
            raise EvidenceError(f"unknown artifact kind {kind!r}")
        raw = bytes(data)
        digest = hashlib.sha256(raw).hexdigest()
        chosen_suffix = suffix if suffix is not None else _SUFFIX_BY_KIND[kind]
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", chosen_suffix):
            raise EvidenceError("artifact suffix must be a short extension such as .png")
        stem = _safe_token(observation_id or "artifact")
        filename = f"{stem}-{digest[:12]}-{uuid.uuid4().hex[:8]}{chosen_suffix}"
        target = self._directory_for(kind) / filename
        self._atomic_write(target, raw)
        try:
            written = target.read_bytes()
        except OSError as exc:
            raise EvidenceError(f"failed to read back evidence {target.name}: {exc}") from exc
        if written != raw:
            raise EvidenceError("evidence read-back does not match the written bytes")
        reference = target.relative_to(self.run_dir).as_posix()
        artifact_metadata = dict(metadata or {})
        if observation_id is not None:
            artifact_metadata.setdefault("observation_id", observation_id)
        return Artifact(
            kind=kind,
            available=True,
            reference=reference,
            sha256=digest,
            byte_size=len(raw),
            metadata=artifact_metadata,
        )

    def save_text(
        self,
        kind: str,
        text: str,
        *,
        observation_id: str | None = None,
        suffix: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Artifact:
        if not isinstance(text, str):
            raise EvidenceError("evidence text must be a string")
        return self.save_bytes(
            kind,
            text.encode("utf-8"),
            observation_id=observation_id,
            suffix=suffix,
            metadata=metadata,
        )

    def save_observation(self, observation: Observation) -> str:
        if not isinstance(observation, Observation):
            raise EvidenceError("save_observation expects an Observation")
        filename = _safe_token(observation.observation_id) + ".json"
        target = self.observations_dir / filename
        self._atomic_write(target, json_dumps(observation.to_local_dict()).encode("utf-8"))
        return target.relative_to(self.run_dir).as_posix()

    def write_result(self, payload: Mapping[str, Any]) -> Path:
        if not isinstance(payload, Mapping):
            raise EvidenceError("result payload must be a mapping")
        self._atomic_write(
            self.result_path,
            json_dumps(payload).encode("utf-8"),
        )
        return self.result_path

    def verify(self, artifact: Artifact) -> bool:
        try:
            data = self.resolve_reference(artifact.reference or "").read_bytes()
        except (EvidenceError, OSError):
            return False
        return (
            hashlib.sha256(data).hexdigest() == artifact.sha256
            and len(data) == artifact.byte_size
        )

    def assert_valid(self, artifact: Artifact) -> None:
        if not artifact.available:
            raise EvidenceError(
                f"artifact {artifact.kind} is unavailable: {artifact.reason or 'unknown reason'}"
            )
        if not self.verify(artifact):
            raise EvidenceError(f"artifact {artifact.kind} failed integrity verification")

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
