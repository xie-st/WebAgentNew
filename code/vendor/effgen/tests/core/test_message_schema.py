"""
Tests for effgen.core.messages — construction, validation, back-compat.
"""

from __future__ import annotations

import time

import pytest

from effgen.core.messages import (
    AudioPart,
    ImagePart,
    Message,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    VideoPart,
)
from effgen.errors import InvalidMultimodalContent

# ---------------------------------------------------------------------------
# Minimal fixture bytes
# ---------------------------------------------------------------------------

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20  # minimal PNG magic
JPEG_HEADER = b"\xff\xd8\xff" + b"\x00" * 20
WAV_HEADER = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 8
MP3_HEADER = b"ID3" + b"\x00" * 20


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------

class TestRole:
    def test_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"


# ---------------------------------------------------------------------------
# TextPart
# ---------------------------------------------------------------------------

class TestTextPart:
    def test_basic(self):
        p = TextPart(text="hello")
        assert p.text == "hello"
        assert p.type == "text"

    def test_empty_string(self):
        p = TextPart(text="")
        assert p.text == ""


# ---------------------------------------------------------------------------
# ImagePart
# ---------------------------------------------------------------------------

class TestImagePart:
    def test_valid_png(self):
        p = ImagePart(image=PNG_HEADER, mime="image/png")
        assert p.type == "image"
        assert p.mime == "image/png"
        assert p.image == PNG_HEADER

    def test_valid_jpeg(self):
        p = ImagePart(image=JPEG_HEADER, mime="image/jpeg")
        assert p.mime == "image/jpeg"

    def test_valid_gif(self):
        p = ImagePart(image=b"GIF89a" + b"\x00" * 10, mime="image/gif")
        assert p.mime == "image/gif"

    def test_valid_webp(self):
        p = ImagePart(image=b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4, mime="image/webp")
        assert p.mime == "image/webp"

    def test_invalid_mime_raises(self):
        with pytest.raises(InvalidMultimodalContent) as exc:
            ImagePart(image=PNG_HEADER, mime="image/tiff")
        assert "image/tiff" in str(exc.value)

    def test_invalid_data_type_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            ImagePart(image="not bytes", mime="image/png")  # type: ignore[arg-type]

    def test_bytearray_coerced_to_bytes(self):
        p = ImagePart(image=bytearray(PNG_HEADER), mime="image/png")
        assert isinstance(p.image, bytes)

    def test_meta_defaults_empty(self):
        p = ImagePart(image=PNG_HEADER, mime="image/png")
        assert p.meta == {}

    def test_meta_stored(self):
        p = ImagePart(image=PNG_HEADER, mime="image/png", meta={"width": 100})
        assert p.meta["width"] == 100


# ---------------------------------------------------------------------------
# AudioPart
# ---------------------------------------------------------------------------

class TestAudioPart:
    def test_valid_wav(self):
        p = AudioPart(audio=WAV_HEADER, mime="audio/wav")
        assert p.type == "audio"
        assert p.mime == "audio/wav"

    def test_valid_mp3(self):
        p = AudioPart(audio=MP3_HEADER, mime="audio/mp3")
        assert p.mime == "audio/mp3"

    def test_valid_flac(self):
        p = AudioPart(audio=b"fLaC" + b"\x00" * 20, mime="audio/flac")
        assert p.mime == "audio/flac"

    def test_valid_ogg(self):
        p = AudioPart(audio=b"OggS" + b"\x00" * 20, mime="audio/ogg")
        assert p.mime == "audio/ogg"

    def test_valid_m4a(self):
        p = AudioPart(audio=b"\x00" * 20, mime="audio/m4a")
        assert p.mime == "audio/m4a"

    def test_invalid_mime_raises(self):
        with pytest.raises(InvalidMultimodalContent) as exc:
            AudioPart(audio=WAV_HEADER, mime="audio/aac")
        assert "audio/aac" in str(exc.value)

    def test_invalid_data_type_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            AudioPart(audio="not bytes", mime="audio/wav")  # type: ignore[arg-type]

    def test_negative_duration_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            AudioPart(audio=WAV_HEADER, mime="audio/wav", duration_s=-1)

    def test_duration_stored(self):
        p = AudioPart(audio=WAV_HEADER, mime="audio/wav", duration_s=3.14)
        assert p.duration_s == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# VideoPart
# ---------------------------------------------------------------------------

class TestVideoPart:
    def test_valid(self):
        frames = [JPEG_HEADER, JPEG_HEADER]
        p = VideoPart(frames=frames, fps=1.0, mime="image/jpeg")
        assert p.type == "video_frames"
        assert len(p.frames) == 2
        assert p.fps == 1.0

    def test_invalid_frame_mime_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            VideoPart(frames=[JPEG_HEADER], fps=1.0, mime="image/tiff")

    def test_non_positive_fps_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            VideoPart(frames=[JPEG_HEADER], fps=0, mime="image/jpeg")

    def test_empty_frames_raises(self):
        with pytest.raises(InvalidMultimodalContent) as exc:
            VideoPart(frames=[], fps=1.0, mime="image/jpeg")
        assert "non-empty" in str(exc.value)

    def test_non_bytes_frame_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            VideoPart(frames=["bad"], fps=1.0, mime="image/jpeg")  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# ToolCallPart / ToolResultPart
# ---------------------------------------------------------------------------

class TestToolParts:
    def test_tool_call(self):
        p = ToolCallPart(tool_call_id="id1", name="weather", arguments={"city": "NY"})
        assert p.type == "tool_call"
        assert p.name == "weather"
        assert p.arguments["city"] == "NY"

    def test_tool_result(self):
        p = ToolResultPart(tool_call_id="id1", result={"temp": 72})
        assert p.type == "tool_result"
        assert p.result["temp"] == 72
        assert not p.is_error

    def test_tool_result_error(self):
        p = ToolResultPart(tool_call_id="id1", result="err msg", is_error=True)
        assert p.is_error


# ---------------------------------------------------------------------------
# Message — construction
# ---------------------------------------------------------------------------

class TestMessageConstruction:
    def test_text_parts_list(self):
        msg = Message(role=Role.USER, content=[TextPart("hello")])
        assert msg.role == Role.USER
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextPart)

    def test_back_compat_string_content(self):
        msg = Message(role=Role.USER, content="hello world")  # type: ignore[arg-type]
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextPart)
        assert msg.content[0].text == "hello world"

    def test_back_compat_string_role(self):
        msg = Message(role="user", content="hi")  # type: ignore[arg-type]
        assert msg.role == Role.USER

    def test_invalid_string_role_raises_invalid_multimodal_content(self):
        with pytest.raises(InvalidMultimodalContent):
            Message(role="developer", content="hi")  # type: ignore[arg-type]

    def test_from_str(self):
        msg = Message.from_str(Role.ASSISTANT, "reply")
        assert msg.text == "reply"
        assert msg.role == Role.ASSISTANT

    def test_user_classmethod(self):
        msg = Message.user("question?")
        assert msg.role == Role.USER
        assert msg.text == "question?"

    def test_assistant_classmethod(self):
        msg = Message.assistant("answer")
        assert msg.role == Role.ASSISTANT

    def test_system_classmethod(self):
        msg = Message.system("you are helpful")
        assert msg.role == Role.SYSTEM

    def test_metadata_default_empty(self):
        msg = Message(role=Role.USER, content="hi")
        assert msg.metadata == {}

    def test_metadata_stored(self):
        msg = Message(role=Role.USER, content="hi", metadata={"key": "val"})
        assert msg.metadata["key"] == "val"

    def test_timestamp_set(self):
        before = time.time()
        msg = Message(role=Role.USER, content="hi")
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_invalid_content_type_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            Message(role=Role.USER, content=12345)  # type: ignore[arg-type]

    def test_bare_str_content_list_item_is_wrapped(self):
        # A bare string in the content list is auto-wrapped in a TextPart so
        # Message(content=[img, "describe this"]) works as documented.
        msg = Message(role=Role.USER, content=["not a part"])  # type: ignore[list-item]
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextPart)
        assert msg.content[0].text == "not a part"

    def test_invalid_content_list_item_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            Message(role=Role.USER, content=[123])  # type: ignore[list-item]

    def test_invalid_metadata_type_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            Message(role=Role.USER, content="hi", metadata=[])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Message.text property
# ---------------------------------------------------------------------------

class TestMessageTextProperty:
    def test_single_text_part(self):
        msg = Message(role=Role.USER, content=[TextPart("hello")])
        assert msg.text == "hello"

    def test_multiple_text_parts_joined(self):
        msg = Message(role=Role.USER, content=[TextPart("foo"), TextPart(" bar")])
        assert msg.text == "foo bar"

    def test_mixed_parts_joins_only_text(self):
        msg = Message(
            role=Role.USER,
            content=[TextPart("desc: "), ImagePart(image=PNG_HEADER, mime="image/png")],
        )
        assert msg.text == "desc: "

    def test_no_text_parts_empty_string(self):
        msg = Message(role=Role.USER, content=[ImagePart(image=PNG_HEADER, mime="image/png")])
        assert msg.text == ""

    def test_back_compat_text_property(self):
        msg = Message(role=Role.USER, content="simple text")  # type: ignore[arg-type]
        assert msg.text == "simple text"


# ---------------------------------------------------------------------------
# Message.has_* properties
# ---------------------------------------------------------------------------

class TestMessageHasProperties:
    def test_has_image_true(self):
        msg = Message(role=Role.USER, content=[ImagePart(image=PNG_HEADER, mime="image/png")])
        assert msg.has_image

    def test_has_image_false(self):
        msg = Message.user("no image here")
        assert not msg.has_image

    def test_has_audio_true(self):
        msg = Message(role=Role.USER, content=[AudioPart(audio=WAV_HEADER, mime="audio/wav")])
        assert msg.has_audio

    def test_has_audio_false(self):
        msg = Message.user("text only")
        assert not msg.has_audio

    def test_has_video_true(self):
        msg = Message(
            role=Role.USER,
            content=[VideoPart(frames=[PNG_HEADER], fps=1.0, mime="image/jpeg")],
        )
        assert msg.has_video

    def test_has_video_false(self):
        msg = Message.user("text only")
        assert not msg.has_video


# ---------------------------------------------------------------------------
# Message.to_dict (serialisation)
# ---------------------------------------------------------------------------

class TestMessageToDict:
    def test_text_only(self):
        msg = Message.user("hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == [{"type": "text", "text": "hello"}]

    def test_image_part_base64(self):
        import base64
        msg = Message(role=Role.USER, content=[ImagePart(image=PNG_HEADER, mime="image/png")])
        d = msg.to_dict()
        assert d["content"][0]["type"] == "image"
        assert d["content"][0]["mime"] == "image/png"
        # base64-encoded roundtrip
        decoded = base64.b64decode(d["content"][0]["image"])
        assert decoded == PNG_HEADER

    def test_audio_part_base64(self):
        import base64
        msg = Message(role=Role.USER, content=[AudioPart(audio=WAV_HEADER, mime="audio/wav")])
        d = msg.to_dict()
        assert d["content"][0]["type"] == "audio"
        decoded = base64.b64decode(d["content"][0]["audio"])
        assert decoded == WAV_HEADER

    def test_tool_call_part(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[ToolCallPart("id1", "calc", {"x": 1})],
        )
        d = msg.to_dict()
        assert d["content"][0]["type"] == "tool_call"
        assert d["content"][0]["name"] == "calc"

    def test_tool_result_part(self):
        msg = Message(
            role=Role.TOOL,
            content=[ToolResultPart("id1", "result text")],
        )
        d = msg.to_dict()
        assert d["content"][0]["type"] == "tool_result"
        assert d["content"][0]["result"] == "result text"
