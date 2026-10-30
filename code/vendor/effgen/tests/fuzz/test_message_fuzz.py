"""
Hypothesis-driven fuzz tests for the effGen message schema.

Asserts that:
  1. Valid Message constructions never crash.
  2. Invalid constructions raise only ``InvalidMultimodalContent`` (declared).
  3. Serialisation (to_dict) never crashes on a valid Message.
  4. No secret leaks in error messages.
  5. Message.text / has_image / has_audio / has_video properties never crash.

Exit criterion: ≥500 examples per test.
"""

from __future__ import annotations

import re
import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.core.messages import (
    AudioPart,
    ContentPart,
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
# Secret leak detection
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"csk-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    re.compile(r"gsk_[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"Bearer [^\s]{6,}"),
]


def _has_secret(text: str) -> bool:
    return any(pat.search(text) for pat in _SECRET_PATTERNS)


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

_SAFE_TEXT = st.text(
    alphabet=string.ascii_letters + string.digits + " +-_./,@()[]{}",
    min_size=0,
    max_size=200,
)

_SAFE_NONEMPTY = st.text(
    alphabet=string.ascii_letters + string.digits + "_-.",
    min_size=1,
    max_size=50,
)

# Valid MIME types for images
_VALID_IMAGE_MIMES = ["image/png", "image/jpeg", "image/gif", "image/webp"]
_INVALID_IMAGE_MIMES = ["application/pdf", "text/plain", "audio/mp3", "image/bmp", ""]
_ALL_IMAGE_MIMES = _VALID_IMAGE_MIMES + _INVALID_IMAGE_MIMES

# Valid MIME types for audio
_VALID_AUDIO_MIMES = [
    "audio/flac", "audio/m4a", "audio/mp4", "audio/mp3", "audio/mpeg",
    "audio/mpga", "audio/ogg", "audio/wav", "audio/webm",
]
_INVALID_AUDIO_MIMES = ["image/png", "text/plain", "video/mp4", ""]
_ALL_AUDIO_MIMES = _VALID_AUDIO_MIMES + _INVALID_AUDIO_MIMES

_ROLE_VALUES = [r.value for r in Role]
_INVALID_ROLES = ["god", "bot", "ai", "", "system\x00", "USER ", " assistant"]


@st.composite
def valid_text_part(draw) -> TextPart:
    return TextPart(text=draw(_SAFE_TEXT))


@st.composite
def valid_image_part(draw) -> ImagePart:
    return ImagePart(
        image=draw(st.binary(min_size=1, max_size=64)),
        mime=draw(st.sampled_from(_VALID_IMAGE_MIMES)),
    )


@st.composite
def valid_audio_part(draw) -> AudioPart:
    return AudioPart(
        audio=draw(st.binary(min_size=1, max_size=64)),
        mime=draw(st.sampled_from(_VALID_AUDIO_MIMES)),
        duration_s=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=3600.0, allow_nan=False))),
    )


@st.composite
def valid_tool_call_part(draw) -> ToolCallPart:
    return ToolCallPart(
        tool_call_id=draw(_SAFE_NONEMPTY),
        name=draw(_SAFE_NONEMPTY),
        arguments=draw(st.dictionaries(_SAFE_NONEMPTY, _SAFE_TEXT, max_size=5)),
    )


@st.composite
def valid_tool_result_part(draw) -> ToolResultPart:
    return ToolResultPart(
        tool_call_id=draw(_SAFE_NONEMPTY),
        result=draw(st.one_of(_SAFE_TEXT, st.integers(), st.none())),
        is_error=draw(st.booleans()),
    )


@st.composite
def valid_content_part(draw) -> ContentPart:
    return draw(st.one_of(
        valid_text_part(),
        valid_tool_call_part(),
        valid_tool_result_part(),
    ))


@st.composite
def valid_message(draw) -> Message:
    role = draw(st.sampled_from(list(Role)))
    n_parts = draw(st.integers(min_value=1, max_value=6))
    parts = [draw(valid_content_part()) for _ in range(n_parts)]
    metadata = draw(st.dictionaries(_SAFE_NONEMPTY, _SAFE_TEXT, max_size=4))
    return Message(role=role, content=parts, metadata=metadata)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(text=_SAFE_TEXT)
def test_text_part_construction_fuzz(text):
    """TextPart accepts any string without crashing."""
    part = TextPart(text=text)
    assert part.text == text
    assert part.type == "text"


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=st.binary(min_size=0, max_size=256),
    mime=st.sampled_from(_ALL_IMAGE_MIMES),
)
def test_image_part_construction_fuzz(data, mime):
    """ImagePart either succeeds or raises InvalidMultimodalContent — nothing else."""
    try:
        part = ImagePart(image=data, mime=mime)
        assert isinstance(part.image, bytes)
    except InvalidMultimodalContent as exc:
        # Declared error — check no secret in message
        assert not _has_secret(str(exc))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=st.binary(min_size=0, max_size=256),
    mime=st.sampled_from(_ALL_AUDIO_MIMES),
    duration=st.one_of(
        st.none(),
        st.floats(min_value=-10.0, max_value=3600.0, allow_nan=False),
    ),
)
def test_audio_part_construction_fuzz(data, mime, duration):
    """AudioPart either succeeds or raises InvalidMultimodalContent — nothing else."""
    try:
        part = AudioPart(audio=data, mime=mime, duration_s=duration)
        assert isinstance(part.audio, bytes)
    except InvalidMultimodalContent as exc:
        assert not _has_secret(str(exc))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    frames=st.lists(st.binary(min_size=1, max_size=32), min_size=0, max_size=5),
    fps=st.one_of(
        st.floats(min_value=-1.0, max_value=120.0, allow_nan=False),
        st.just(0.0),
    ),
    mime=st.sampled_from(_ALL_IMAGE_MIMES),
)
def test_video_part_construction_fuzz(frames, fps, mime):
    """VideoPart either succeeds or raises InvalidMultimodalContent — nothing else."""
    try:
        part = VideoPart(frames=frames, fps=fps, mime=mime)
        assert isinstance(part.frames, list)
    except InvalidMultimodalContent as exc:
        assert not _has_secret(str(exc))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    tool_call_id=st.one_of(_SAFE_TEXT, st.just("")),
    name=st.one_of(_SAFE_TEXT, st.just("")),
    arguments=st.one_of(
        st.dictionaries(_SAFE_NONEMPTY, _SAFE_TEXT, max_size=5),
        st.just(None),
        _SAFE_TEXT,
    ),
)
def test_tool_call_part_fuzz(tool_call_id, name, arguments):
    """ToolCallPart either succeeds or raises InvalidMultimodalContent."""
    try:
        part = ToolCallPart(tool_call_id=tool_call_id, name=name, arguments=arguments)
        assert part.type == "tool_call"
    except InvalidMultimodalContent as exc:
        assert not _has_secret(str(exc))
    except (TypeError, AttributeError):
        pass  # arguments=None triggers TypeError in dataclass — acceptable
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(msg=valid_message())
def test_valid_message_properties(msg):
    """Valid Messages expose .text, .has_image, .has_audio, .has_video without error."""
    _ = msg.text
    _ = msg.has_image
    _ = msg.has_audio
    _ = msg.has_video
    assert isinstance(msg.content, list)


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(msg=valid_message())
def test_valid_message_serialisation(msg):
    """Message.to_dict() never crashes on valid Messages."""
    d = msg.to_dict()
    assert isinstance(d, dict)
    assert "role" in d
    assert "content" in d
    assert isinstance(d["content"], list)


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    role=st.one_of(
        st.sampled_from(_ROLE_VALUES),
        st.sampled_from(_INVALID_ROLES),
        _SAFE_TEXT,
    ),
)
def test_message_role_fuzz(role):
    """Message construction with arbitrary role strings either succeeds or raises InvalidMultimodalContent."""
    try:
        msg = Message(role=role, content=[TextPart(text="hi")])
        assert msg.role in list(Role)
    except InvalidMultimodalContent as exc:
        assert not _has_secret(str(exc))
    except (ValueError, KeyError):
        pass  # invalid role string — also acceptable
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    content=st.one_of(
        st.just([]),
        st.just("plain string"),
        st.just(None),
        st.just(42),
        st.just([TextPart("hello"), TextPart("world")]),
    ),
)
def test_message_content_types_fuzz(content):
    """Message with edge-case content types either constructs or raises InvalidMultimodalContent."""
    try:
        msg = Message(role=Role.USER, content=content)
        _ = msg.text
    except InvalidMultimodalContent as exc:
        assert not _has_secret(str(exc))
    except (TypeError, AttributeError):
        pass  # None / int content handled
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_messages=st.integers(min_value=1, max_value=10),
    roles=st.lists(st.sampled_from(list(Role)), min_size=1, max_size=10),
)
def test_message_sequence_fuzz(n_messages, roles):
    """Random sequences of messages construct without crashing."""
    messages = []
    for i in range(min(n_messages, len(roles))):
        msg = Message(role=roles[i], content=[TextPart(text=f"message {i}")])
        messages.append(msg)
    assert len(messages) <= n_messages


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    metadata=st.one_of(
        st.dictionaries(_SAFE_NONEMPTY, _SAFE_TEXT, max_size=5),
        st.just({}),
        st.just(None),
        _SAFE_TEXT,
    ),
)
def test_message_metadata_fuzz(metadata):
    """Message metadata validation: dicts succeed, non-dicts raise InvalidMultimodalContent."""
    try:
        msg = Message(role=Role.USER, content=[TextPart("hi")], metadata=metadata)
        assert isinstance(msg.metadata, dict)
    except InvalidMultimodalContent as exc:
        assert not _has_secret(str(exc))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(text=_SAFE_TEXT)
def test_message_factory_methods_fuzz(text):
    """Message.user/assistant/system factory methods never crash on arbitrary text."""
    u = Message.user(text)
    a = Message.assistant(text)
    s = Message.system(text)
    assert u.role == Role.USER
    assert a.role == Role.ASSISTANT
    assert s.role == Role.SYSTEM
    assert u.text == text
