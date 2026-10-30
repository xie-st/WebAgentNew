from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from webagent.errors import EvidenceError, ProtocolError
from webagent.evidence import EvidenceStore
from webagent.protocol import Artifact


def test_save_bytes_records_integrity_and_verifies(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "runtime", "run-a")
    artifact = store.save_bytes(
        "screenshot", b"\x89PNG\r\n\x1a\nfake", observation_id="obs-1", metadata={"url": "x"}
    )
    assert artifact.available is True
    assert artifact.kind == "screenshot"
    assert artifact.reference.startswith("screenshots/")
    assert artifact.byte_size == len(b"\x89PNG\r\n\x1a\nfake")
    assert artifact.sha256 == hashlib.sha256(b"\x89PNG\r\n\x1a\nfake").hexdigest()
    assert artifact.metadata["observation_id"] == "obs-1"
    assert store.verify(artifact) is True
    store.assert_valid(artifact)
    path = store.resolve_reference(artifact.reference)
    path.write_bytes(b"tampered")
    assert store.verify(artifact) is False
    with pytest.raises(EvidenceError):
        store.assert_valid(artifact)


def test_two_saves_get_distinct_references(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "runtime", "run-a")
    first = store.save_bytes("screenshot", b"same", observation_id="obs-1")
    second = store.save_bytes("screenshot", b"same", observation_id="obs-1")
    assert first.reference != second.reference
    assert store.verify(first) and store.verify(second)


def test_unavailable_artifact_is_not_usable(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "runtime", "run-a")
    unavailable = Artifact(kind="html", available=False, reason="size_limit")
    assert store.verify(unavailable) is False
    with pytest.raises(EvidenceError):
        store.assert_valid(unavailable)


def test_unsafe_run_id_and_reference_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError):
        EvidenceStore(tmp_path / "runtime", "../escape")
    store = EvidenceStore(tmp_path / "runtime", "run-a")
    with pytest.raises(EvidenceError):
        store.resolve_reference("../outside.png")
    with pytest.raises(EvidenceError):
        store.resolve_reference(str(tmp_path / "absolute.png"))
    fake_artifact = Artifact(
        kind="screenshot",
        available=True,
        reference="../outside.png",
        sha256="a" * 64,
        byte_size=1,
    )
    assert store.verify(fake_artifact) is False


def test_observation_and_result_files(tmp_path: Path, make_observation) -> None:
    store = EvidenceStore(tmp_path / "runtime", "run-a")
    observation = make_observation()
    reference = store.save_observation(observation)
    assert reference == "observations/obs-1.json"
    payload = json.loads((store.run_dir / reference).read_text(encoding="utf-8"))
    assert payload["elements"]["1"]["localization"] == {"fake_index": 1}
    path = store.write_result({"run_id": "run-a", "state": "finished"})
    assert path == store.result_path
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "finished"


def test_failed_save_is_visible(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "runtime", "run-a")
    with pytest.raises(EvidenceError):
        store.save_bytes("screenshot", "not bytes")  # type: ignore[arg-type]
    with pytest.raises(EvidenceError):
        store.save_bytes("unknown-kind", b"x")
    with pytest.raises(EvidenceError):
        store.save_text("screenshot", 12)  # type: ignore[arg-type]
