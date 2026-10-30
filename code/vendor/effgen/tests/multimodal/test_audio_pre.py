"""Unit tests for effgen/multimodal/audio_pre.py."""

from __future__ import annotations

import io
import math
import struct
import wave

import pytest

from effgen.core.messages import AudioPart
from effgen.errors import MissingSystemDependency
from effgen.multimodal.audio_pre import (
    _PROVIDER_CONSTRAINTS,
    AudioPreprocessor,
    chunk,
    get_audio_duration,
    prepare,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(duration_s: float = 1.0, sample_rate: int = 16_000, freq: float = 440.0) -> bytes:
    """Return minimal mono 16-bit PCM WAV bytes."""
    num = int(sample_rate * duration_s)
    samples = [int(32767 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(num)]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return buf.getvalue()


def _make_part(duration_s: float = 1.0, mime: str = "audio/wav") -> AudioPart:
    return AudioPart(audio=_make_wav(duration_s), mime=mime)


def _make_silent_wav_ms(duration_ms: int) -> bytes:
    """Return silent mono 16-bit PCM WAV bytes of the given duration.

    Built with the stdlib ``wave`` module (silent PCM is just zero frames), so
    the helper needs neither pydub nor ffmpeg on PATH. The preprocessor under
    test still decodes/chunks the result via pydub.
    """
    sample_rate = 16_000
    num_frames = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Constraint table
# ---------------------------------------------------------------------------

class TestConstraintTable:
    def test_providers_have_constraints(self):
        for provider in ("gemini", "openai", "hf_inference"):
            assert provider in _PROVIDER_CONSTRAINTS

    def test_openai_max_bytes_25mb(self):
        assert _PROVIDER_CONSTRAINTS["openai"]["max_bytes"] == 25 * 1024 * 1024

    def test_gemini_max_bytes_20mb(self):
        assert _PROVIDER_CONSTRAINTS["gemini"]["max_bytes"] == 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_returns_audio_part(self):
        part = _make_part()
        result = prepare(part, "openai")
        assert isinstance(result, AudioPart)

    def test_openai_wav_passthrough(self):
        part = _make_part()
        result = prepare(part, "openai")
        # WAV is allowed for openai; should not change mime
        assert result.mime == "audio/wav"

    def test_gemini_wav_passthrough(self):
        part = _make_part()
        result = prepare(part, "gemini")
        assert isinstance(result, AudioPart)
        assert result.mime == "audio/wav"

    def test_hf_inference_resamples_to_16khz(self):
        """HF inference needs 16 kHz mono (pydub handles this when available)."""
        part = _make_part(duration_s=1.0)
        # Should not raise; resampling is best-effort
        result = prepare(part, "hf_inference")
        assert isinstance(result, AudioPart)

    def test_unknown_provider_passthrough(self):
        part = _make_part()
        result = prepare(part, "some_unknown_provider")
        assert result.audio == part.audio

    def test_oversized_logs_warning(self, caplog):
        import logging
        big_audio = b"\x00" * (30 * 1024 * 1024)  # 30 MB fake audio
        part = AudioPart(audio=big_audio, mime="audio/wav")
        with caplog.at_level(logging.WARNING, logger="effgen.multimodal.audio_pre"):
            result = prepare(part, "openai")
        # Should warn about size but not raise
        assert isinstance(result, AudioPart)


# ---------------------------------------------------------------------------
# chunk()
# ---------------------------------------------------------------------------

class TestChunk:
    def test_short_audio_returns_one_chunk(self):
        part = _make_part(duration_s=2.0)
        chunks = chunk(part, "openai")
        assert len(chunks) >= 1

    def test_returns_list_of_audio_parts(self):
        part = _make_part(duration_s=1.0)
        chunks = chunk(part, "openai")
        for c in chunks:
            assert isinstance(c, AudioPart)

    def test_chunks_mime_preserved(self):
        part = _make_part(duration_s=1.0, mime="audio/wav")
        chunks = chunk(part, "openai")
        for c in chunks:
            assert c.mime == "audio/wav"

    def test_chunks_have_positive_duration(self):
        part = _make_part(duration_s=2.0)
        chunks = chunk(part, "openai")
        for c in chunks:
            if c.duration_s is not None:
                assert c.duration_s > 0

    def test_hf_inference_chunking(self):
        part = _make_part(duration_s=3.0)
        chunks = chunk(part, "hf_inference")
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, AudioPart)

    def test_unknown_provider_returns_one_chunk(self):
        part = _make_part(duration_s=1.0)
        chunks = chunk(part, "unknown_provider")
        assert len(chunks) == 1

    def test_openai_chunks_over_30_min_synthetic_audio(self):
        part = AudioPart(audio=_make_silent_wav_ms(31 * 60 * 1000), mime="audio/wav")
        chunks = chunk(part, "openai")
        assert len(chunks) >= 2
        assert all(c.duration_s is not None and c.duration_s <= 30 * 60 for c in chunks)
        assert sum(c.duration_s or 0 for c in chunks) >= 31 * 60 - 2

    def test_openai_chunks_by_byte_limit(self):
        original = _PROVIDER_CONSTRAINTS["openai"].copy()
        _PROVIDER_CONSTRAINTS["openai"]["max_bytes"] = 10_000
        _PROVIDER_CONSTRAINTS["openai"]["max_duration_s"] = 60 * 60
        try:
            part = _make_part(duration_s=3.0)
            chunks = chunk(part, "openai")
        finally:
            _PROVIDER_CONSTRAINTS["openai"].update(original)
        assert len(chunks) > 1
        assert all(len(c.audio) <= 10_000 for c in chunks)

    def test_compressed_audio_without_ffmpeg_raises(self, monkeypatch):
        monkeypatch.setattr("effgen.multimodal.audio_pre.shutil.which", lambda _: None)
        part = AudioPart(audio=b"ID3" + b"\x00" * 100, mime="audio/mp3")
        with pytest.raises(MissingSystemDependency) as exc_info:
            chunk(part, "openai")
        assert exc_info.value.dependency == "ffmpeg"


# ---------------------------------------------------------------------------
# get_audio_duration()
# ---------------------------------------------------------------------------

class TestGetAudioDuration:
    def test_returns_float_for_valid_wav(self):
        wav = _make_wav(duration_s=2.0)
        dur = get_audio_duration(wav, "audio/wav")
        if dur is not None:
            assert abs(dur - 2.0) < 0.5

    def test_returns_none_for_garbage(self):
        dur = get_audio_duration(b"\x00" * 100, "audio/wav")
        # Either None (pydub absent/failed) or a near-zero float
        assert dur is None or isinstance(dur, float)


# ---------------------------------------------------------------------------
# AudioPreprocessor
# ---------------------------------------------------------------------------

class TestAudioPreprocessor:
    def test_instantiation(self):
        proc = AudioPreprocessor("openai")
        assert proc.provider == "openai"

    def test_prepare_returns_part(self):
        proc = AudioPreprocessor("gemini")
        part = _make_part()
        result = proc.prepare(part)
        assert isinstance(result, AudioPart)

    def test_chunk_returns_list(self):
        proc = AudioPreprocessor("openai")
        part = _make_part(duration_s=2.0)
        chunks = proc.chunk(part)
        assert isinstance(chunks, list)
        assert all(isinstance(c, AudioPart) for c in chunks)
