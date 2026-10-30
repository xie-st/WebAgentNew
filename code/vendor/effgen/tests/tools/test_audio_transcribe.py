"""Tests for AudioTranscribeTool — unit (backend selection) + integration (local Whisper)."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

_FASTER_WHISPER_AVAILABLE = importlib.util.find_spec("faster_whisper") is not None
_HF_TOKEN_SET = bool(
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_API_KEY")
    or os.environ.get("HUGGINGFACE_TOKEN")
)

# Path to the fixture audio (gTTS-generated speech)
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "audio"
_SAMPLE_AUDIO = _FIXTURE_DIR / "sample_audio.mp3"
_FIXTURE_AVAILABLE = _SAMPLE_AUDIO.exists()


def _make_silent_wav(duration_secs: float = 1.0) -> bytes:
    """Create a minimal silent WAV file in memory."""
    import io
    import wave

    sample_rate = 16000
    num_channels = 1
    sample_width = 2  # 16-bit
    num_frames = int(sample_rate * duration_secs)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * num_frames * num_channels * sample_width)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests — error classes and backend selection
# ---------------------------------------------------------------------------

class TestAudioTranscribeToolUnit:
    def test_import(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool  # noqa: F401

    def test_metadata(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        assert tool.metadata.name == "audio_transcribe"
        param_names = [p.name for p in tool.metadata.parameters]
        assert "operation" in param_names
        assert "audio_path" in param_names
        assert "audio_bytes" in param_names
        assert "model_size" in param_names
        assert "language" in param_names
        assert "word_timestamps" in param_names
        assert "backend" in param_names

    def test_resolve_source_path(self, tmp_path):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        f = tmp_path / "test.wav"
        f.write_bytes(b"RIFF....")
        src = tool._resolve_source(str(f), None)
        assert src == str(f)

    def test_resolve_source_bytes(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        raw = b"\x00\x01\x02\x03"
        src = tool._resolve_source(None, raw)
        assert src == raw

    def test_resolve_source_both_raises(self, tmp_path):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        f = tmp_path / "test.wav"
        f.write_bytes(b"x")
        with pytest.raises(ValueError, match="not both"):
            tool._resolve_source(str(f), b"bytes")

    def test_resolve_source_none_raises(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        with pytest.raises(ValueError, match="required"):
            tool._resolve_source(None, None)

    def test_resolve_source_missing_file(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        with pytest.raises(FileNotFoundError):
            tool._resolve_source("/nonexistent/audio.mp3", None)

    def test_unknown_operation_raises(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        wav = _make_silent_wav()
        with pytest.raises(ValueError, match="Unknown operation"):
            run(tool._execute(operation="detect", audio_bytes=wav))

    def test_no_backend_raises_missing_dep(self):
        from effgen.errors import AudioBackendUnavailable
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        wav = _make_silent_wav()

        with patch("effgen.tools.builtin.audio_transcribe._check_faster_whisper", return_value=False), \
             patch("effgen.tools.builtin.audio_transcribe._hf_token", return_value=None):
            with pytest.raises(AudioBackendUnavailable):
                run(tool._execute(operation="transcribe", audio_bytes=wav))

    def test_backend_auto_prefers_faster_whisper(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        with patch("effgen.tools.builtin.audio_transcribe._check_faster_whisper", return_value=True):
            chosen = tool._pick_backend("auto")
        assert chosen == "faster-whisper"

    def test_backend_auto_falls_back_to_hf(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
        tool = AudioTranscribeTool()
        with patch("effgen.tools.builtin.audio_transcribe._check_faster_whisper", return_value=False), \
             patch("effgen.tools.builtin.audio_transcribe._hf_token", return_value="fake-token"):
            chosen = tool._pick_backend("auto")
        assert chosen == "hf-inference"

    def test_gpu_detection_returns_valid_tuple(self):
        from effgen.tools.builtin.audio_transcribe import _detect_device
        device, compute_type = _detect_device()
        assert device in ("cpu", "cuda")
        assert compute_type in ("int8", "float16")

    def test_missing_dep_error_contains_install_instructions(self):
        from effgen.errors import AudioBackendUnavailable
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        wav = _make_silent_wav()

        with patch("effgen.tools.builtin.audio_transcribe._check_faster_whisper", return_value=False), \
             patch("effgen.tools.builtin.audio_transcribe._hf_token", return_value=None):
            with pytest.raises(AudioBackendUnavailable) as exc_info:
                run(tool._execute(operation="transcribe", audio_bytes=wav))
        msg = str(exc_info.value)
        assert "effgen[audio]" in msg
        assert "ffmpeg" in msg

    def test_execute_accepts_audio_bytes(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        wav = _make_silent_wav()

        with patch("effgen.tools.builtin.audio_transcribe._check_faster_whisper", return_value=False), \
             patch("effgen.tools.builtin.audio_transcribe._hf_token", return_value=None):
            result = run(tool.execute(operation="transcribe", audio_bytes=wav))
        assert result.success is False
        assert result.metadata["error_type"] == "AudioBackendUnavailable"

    def test_ffmpeg_missing_for_m4a_raises_clean_error(self, tmp_path):
        from effgen.errors import MissingSystemDependency
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        audio = tmp_path / "sample.m4a"
        audio.write_bytes(b"not really m4a")
        tool = AudioTranscribeTool()

        with patch("effgen.tools.builtin.audio_transcribe._check_faster_whisper", return_value=True), \
             patch("effgen.tools.builtin.audio_transcribe._check_ffmpeg", return_value=False):
            with pytest.raises(MissingSystemDependency) as exc_info:
                run(
                    tool._execute(
                        operation="transcribe",
                        audio_path=str(audio),
                        backend="faster-whisper",
                    )
                )
        assert "ffmpeg" in str(exc_info.value)
        assert "apt install ffmpeg" in str(exc_info.value)

    def test_hf_fallback_uses_effgen_adapter(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        wav = _make_silent_wav()
        adapter = MagicMock()
        adapter._client = MagicMock()
        adapter._client.automatic_speech_recognition.return_value = types.SimpleNamespace(
            text="adapter transcript",
            chunks=[],
        )

        with patch("effgen.tools.builtin.audio_transcribe._check_faster_whisper", return_value=False), \
             patch("effgen.tools.builtin.audio_transcribe._hf_token", return_value="fake-token"), \
             patch("effgen.models.hf_inference_adapter.HFInferenceAdapter", return_value=adapter):
            result = run(tool._execute(operation="transcribe", audio_bytes=wav))

        assert result["success"] is True
        assert result["text"] == "adapter transcript"
        assert result["backend"] == "hf-inference"
        adapter.load.assert_called_once()
        adapter.unload.assert_called_once()

    def test_media_preset_registered(self):
        from effgen.presets import PRESETS
        assert "media" in PRESETS
        preset = PRESETS["media"]
        assert "audio_transcribe" in preset.tool_names

    def test_general_preset_includes_audio_transcribe(self):
        from effgen.presets import PRESETS
        assert "audio_transcribe" in PRESETS["general"].tool_names


# ---------------------------------------------------------------------------
# Integration tests — faster-whisper local (skipped if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed",
)
@pytest.mark.skipif(
    not _FIXTURE_AVAILABLE,
    reason="sample_audio.mp3 fixture not present",
)
class TestAudioTranscribeToolFasterWhisper:
    def test_transcribe_fixture_mp3(self):
        """Transcribe the gTTS-generated speech fixture."""
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        result = run(
            tool._execute(
                operation="transcribe",
                audio_path=str(_SAMPLE_AUDIO),
                model_size="tiny",
                backend="faster-whisper",
            )
        )
        assert result["success"] is True
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert result["backend"] == "faster-whisper"
        assert isinstance(result["segments"], list)
        assert isinstance(result["language"], str)

    def test_transcribe_returns_text_field(self):
        """Result dict must contain top-level text, segments, language keys."""
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        result = run(
            tool._execute(
                operation="transcribe",
                audio_path=str(_SAMPLE_AUDIO),
                model_size="tiny",
            )
        )
        for key in ("success", "text", "segments", "language", "backend"):
            assert key in result, f"Missing key: {key}"

    def test_transcribe_with_word_timestamps(self):
        """Word timestamps should appear in segment words list."""
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        result = run(
            tool._execute(
                operation="transcribe",
                audio_path=str(_SAMPLE_AUDIO),
                model_size="tiny",
                word_timestamps=True,
                backend="faster-whisper",
            )
        )
        assert result["success"] is True
        # At least one segment should have word-level data if there's speech
        if result["segments"]:
            first_seg = result["segments"][0]
            if "words" in first_seg:
                for w in first_seg["words"]:
                    assert "word" in w
                    assert "start" in w
                    assert "end" in w
                    assert "probability" in w

    def test_transcribe_from_bytes(self):
        """Test that raw bytes input works."""
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        audio_bytes = _SAMPLE_AUDIO.read_bytes()
        result = run(
            tool._execute(
                operation="transcribe",
                audio_bytes=audio_bytes,
                model_size="tiny",
                backend="faster-whisper",
            )
        )
        assert result["success"] is True
        assert isinstance(result["text"], str)

    def test_transcribe_fixture_contains_expected_words(self):
        """The LibriVox fixture should contain expected public-domain intro words."""
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        result = run(
            tool._execute(
                operation="transcribe",
                audio_path=str(_SAMPLE_AUDIO),
                model_size="tiny",
                language="en",
            )
        )
        assert result["success"] is True
        text_lower = result["text"].lower()
        expected_words = ["librivox", "domain", "recording", "volunteer"]
        matches = sum(1 for w in expected_words if w in text_lower)
        assert matches >= 1, f"Expected at least one of {expected_words} in: {text_lower!r}"

    def test_transcribe_via_execute_wrapper(self):
        """Test the public execute() wrapper returns a ToolResult."""
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        tool_result = run(
            tool.execute(
                operation="transcribe",
                audio_path=str(_SAMPLE_AUDIO),
                model_size="tiny",
            )
        )
        assert tool_result.success is True
        assert "text" in tool_result.output

    def test_cpu_warning_for_large_model(self, recwarn):
        """CPU with large model should emit a RuntimeWarning."""
        from effgen.tools.builtin import audio_transcribe

        with patch.object(audio_transcribe, "_detect_device", return_value=("cpu", "int8")):
            with pytest.warns(RuntimeWarning, match="CPU"):
                # We mock the actual WhisperModel to avoid downloading
                mock_seg = MagicMock()
                mock_seg.id = 0
                mock_seg.start = 0.0
                mock_seg.end = 1.0
                mock_seg.text = "test"
                mock_seg.avg_logprob = -0.5
                mock_seg.no_speech_prob = 0.1
                mock_seg.words = None

                mock_info = MagicMock()
                mock_info.language = "en"
                mock_info.duration = 1.0

                mock_model = MagicMock()
                mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)

                with patch("faster_whisper.WhisperModel", return_value=mock_model):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(_make_silent_wav())
                        tmp = f.name
                    try:
                        audio_transcribe._whisper_transcribe(tmp, model_size="large-v3")
                    finally:
                        Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Integration tests — HuggingFace Inference fallback (skipped if no token)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _HF_TOKEN_SET,
    reason="HF_TOKEN not set",
)
@pytest.mark.skipif(
    not _FIXTURE_AVAILABLE,
    reason="sample_audio.mp3 fixture not present",
)
class TestAudioTranscribeToolHFInference:
    def test_transcribe_via_hf_inference(self):
        from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

        tool = AudioTranscribeTool()
        try:
            result = run(
                tool._execute(
                    operation="transcribe",
                    audio_path=str(_SAMPLE_AUDIO),
                    backend="hf-inference",
                )
            )
            assert result["success"] is True
            assert isinstance(result["text"], str)
            assert result["backend"] == "hf-inference"
        except RuntimeError as exc:
            msg = str(exc)
            # Skip if credits are exhausted or the model is gated / unavailable
            if any(code in msg for code in ("402", "403", "Payment Required", "credits")):
                pytest.skip(f"HF Inference API unavailable (credits/quota): {msg[:120]}")
            raise
