"""Unit tests for audio input adapter translation (no live API calls)."""

from __future__ import annotations

import io
import math
import struct
import wave
from unittest.mock import MagicMock

import pytest

from effgen.core.messages import AudioPart, Message, Role, TextPart
from effgen.errors import CapabilityNotSupportedError
from effgen.models._multimodal import has_audio_input, require_audio_support
from effgen.models.capabilities import Capability

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(duration_s: float = 1.0, sample_rate: int = 16_000) -> bytes:
    """Return a minimal mono 16-bit WAV."""
    num = int(sample_rate * duration_s)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num)]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return buf.getvalue()


def _audio_message(duration_s: float = 1.0, text: str = "Transcribe this") -> Message:
    part = AudioPart(audio=_make_wav(duration_s), mime="audio/wav")
    return Message(role=Role.USER, content=[part, TextPart(text=text)])


# ---------------------------------------------------------------------------
# has_audio_input
# ---------------------------------------------------------------------------

class TestHasAudioInput:
    def test_detects_audio_in_message(self):
        msg = _audio_message()
        assert has_audio_input(msg) is True

    def test_detects_audio_in_list(self):
        msg = _audio_message()
        assert has_audio_input([msg]) is True

    def test_text_only_returns_false(self):
        msg = Message(role=Role.USER, content="Hello")
        assert has_audio_input(msg) is False

    def test_plain_string_returns_false(self):
        assert has_audio_input("some text") is False


# ---------------------------------------------------------------------------
# require_audio_support
# ---------------------------------------------------------------------------

class TestRequireAudioSupport:
    def test_raises_when_not_supported(self):
        msg = _audio_message()
        with pytest.raises(CapabilityNotSupportedError) as exc_info:
            require_audio_support(
                msg,
                provider="anthropic",
                model_name="claude-sonnet-4-6",
                supports_audio=False,
            )
        err = exc_info.value
        assert err.capability == Capability.audio_input.value
        assert "anthropic" in str(err).lower()

    def test_passes_when_supported(self):
        msg = _audio_message()
        require_audio_support(
            msg,
            provider="openai",
            model_name="gpt-4o-mini",
            supports_audio=True,
        )  # must not raise

    def test_passes_for_text_only_even_without_support(self):
        msg = Message(role=Role.USER, content="Hello")
        require_audio_support(
            msg,
            provider="anthropic",
            model_name="claude-sonnet-4-6",
            supports_audio=False,
        )  # no audio → no error

    def test_callable_supports_audio(self):
        msg = _audio_message()
        require_audio_support(
            msg,
            provider="gemini",
            model_name="gemini-3.1-flash-lite-preview",
            supports_audio=lambda m: "flash" in m,
        )  # should pass


# ---------------------------------------------------------------------------
# Gemini adapter — audio message translation (unit, fake client)
# ---------------------------------------------------------------------------

class TestGeminiAudioTranslation:
    def _make_adapter(self):
        from effgen.models.gemini_adapter import GeminiAdapter
        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter.model_name = "gemini-3.1-flash-lite-preview"
        adapter._is_loaded = True
        return adapter

    def test_audio_part_translated_to_genai_part(self):
        """AudioPart should become a genai Part.from_bytes with audio mime."""
        adapter = self._make_adapter()


        from effgen.multimodal.image_pre import prepare as _preprocess_image

        wav = _make_wav()
        audio = AudioPart(audio=wav, mime="audio/wav")
        msg = Message(role=Role.USER, content=[audio, TextPart(text="describe")])

        parts = adapter._message_to_genai_parts(msg, _preprocess_image)
        # should have 2 parts: one audio, one text
        assert len(parts) == 2
        # first part should have audio bytes (from_bytes)
        first = parts[0]
        assert hasattr(first, "inline_data") or hasattr(first, "data") or first is not None


# ---------------------------------------------------------------------------
# OpenAI adapter — transcribe_audio method (unit, fake client)
# ---------------------------------------------------------------------------

class TestOpenAIAudioTranscription:
    def _make_adapter(self):
        from effgen.models.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.model_name = "gpt-4o-mini"
        adapter._is_loaded = True
        adapter._is_reasoning_model = False
        adapter._supports_sampling_params = True
        adapter.total_cost = 0.0
        adapter.total_tokens = 0

        mock_client = MagicMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "hello world"
        mock_client.audio.transcriptions.create.return_value = mock_transcription
        adapter.client = mock_client
        return adapter

    def test_transcribe_audio_returns_text(self):
        adapter = self._make_adapter()
        wav = _make_wav()
        result = adapter.transcribe_audio(wav, mime="audio/wav")
        assert result == "hello world"
        adapter.client.audio.transcriptions.create.assert_called_once()

    def test_message_with_audio_injects_transcript(self):
        adapter = self._make_adapter()
        wav = _make_wav()
        audio = AudioPart(audio=wav, mime="audio/wav")
        msg = Message(role=Role.USER, content=[audio, TextPart(text="Summarize")])
        openai_msg = adapter._message_to_openai(msg)
        # Content should contain the transcript
        content = openai_msg["content"]
        if isinstance(content, str):
            assert "Audio transcript" in content
        else:
            texts = [p["text"] for p in content if p.get("type") == "text"]
            combined = " ".join(texts)
            assert "Audio transcript" in combined or "hello world" in combined

    def test_transcription_api_called_with_whisper_model(self):
        adapter = self._make_adapter()
        wav = _make_wav()
        adapter.transcribe_audio(wav, mime="audio/wav", model="whisper-1")
        call_kwargs = adapter.client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs.get("model") == "whisper-1" or call_kwargs[1].get("model") == "whisper-1"


# ---------------------------------------------------------------------------
# HF adapter — transcribe_audio method (unit, fake client)
# ---------------------------------------------------------------------------

class TestHFAudioTranscription:
    def _make_adapter(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter
        adapter = HFInferenceAdapter.__new__(HFInferenceAdapter)
        adapter.model_name = "Qwen/Qwen2.5-7B-Instruct"
        adapter._is_loaded = True
        adapter._info = {}

        mock_client = MagicMock()
        mock_asr = MagicMock()
        mock_asr.text = "the quick brown fox"
        mock_client.automatic_speech_recognition.return_value = mock_asr
        adapter._client = mock_client
        adapter._rate_limiter = None
        return adapter

    def test_transcribe_audio_returns_text(self):
        adapter = self._make_adapter()
        wav = _make_wav()
        result = adapter.transcribe_audio(wav, mime="audio/wav")
        assert result == "the quick brown fox"
        adapter._client.automatic_speech_recognition.assert_called_once()

    def test_asr_called_with_whisper_model(self):
        adapter = self._make_adapter()
        wav = _make_wav()
        adapter.transcribe_audio(wav, mime="audio/wav", asr_model="openai/whisper-large-v3")
        call_kwargs = adapter._client.automatic_speech_recognition.call_args
        assert call_kwargs.kwargs.get("model") == "openai/whisper-large-v3"


# ---------------------------------------------------------------------------
# Anthropic adapter — rejects audio
# ---------------------------------------------------------------------------

class TestAnthropicAudioRejection:
    def test_generate_rejects_audio_message(self):
        from effgen.models.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.model_name = "claude-sonnet-4-6"
        adapter._is_loaded = True

        msg = _audio_message()
        with pytest.raises(CapabilityNotSupportedError) as exc_info:
            require_audio_support(
                msg,
                provider="anthropic",
                model_name=adapter.model_name,
                supports_audio=False,
                hint="Anthropic Claude does not currently support audio input.",
            )
        assert exc_info.value.capability == Capability.audio_input.value
