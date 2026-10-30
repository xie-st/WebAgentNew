"""
Multimodal message schema for effGen.

Defines the canonical content-part union used across all provider adapters.
Each adapter is responsible for translating these parts to its own wire format.

Back-compat: Message(role, "text") still works; .text joins all TextParts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Literal

from effgen.errors import InvalidMultimodalContent

# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

class Role(Enum):
    """Author of a message: system, user, assistant, or tool."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------

_VALID_IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_VALID_AUDIO_MIMES = {
    "audio/flac",
    "audio/m4a",
    "audio/mp4",
    "audio/mp3",
    "audio/mpeg",
    "audio/mpga",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
}


@dataclass
class TextPart:
    """Plain-text content within a message."""

    text: str
    type: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InvalidMultimodalContent("text", "text must be a string")


@dataclass
class ImagePart:
    """Image content as raw bytes with its MIME type."""

    image: bytes
    mime: str
    meta: dict[str, Any] = field(default_factory=dict)
    type: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image, bytes | bytearray):
            raise InvalidMultimodalContent("image", "image data must be bytes")
        self.image = bytes(self.image)
        if self.mime not in _VALID_IMAGE_MIMES:
            raise InvalidMultimodalContent(
                "image",
                f"MIME type '{self.mime}' is not supported. "
                f"Allowed: {sorted(_VALID_IMAGE_MIMES)}",
            )
        if not isinstance(self.meta, dict):
            raise InvalidMultimodalContent("image", "meta must be a dict")


@dataclass
class AudioPart:
    """Audio content as raw bytes with its MIME type and optional duration."""

    audio: bytes
    mime: str
    duration_s: float | None = None
    type: Literal["audio"] = field(default="audio", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.audio, bytes | bytearray):
            raise InvalidMultimodalContent("audio", "audio data must be bytes")
        self.audio = bytes(self.audio)
        if self.mime not in _VALID_AUDIO_MIMES:
            raise InvalidMultimodalContent(
                "audio",
                f"MIME type '{self.mime}' is not supported. "
                f"Allowed: {sorted(_VALID_AUDIO_MIMES)}",
            )
        if self.duration_s is not None and self.duration_s < 0:
            raise InvalidMultimodalContent("audio", "duration_s must be non-negative")


@dataclass
class VideoPart:
    """Video content as sampled frames (image bytes) at a given FPS."""

    frames: list[bytes]
    fps: float
    mime: str
    meta: dict[str, Any] = field(default_factory=dict)
    type: Literal["video_frames"] = field(default="video_frames", init=False)

    def __post_init__(self) -> None:
        if self.mime not in _VALID_IMAGE_MIMES:
            raise InvalidMultimodalContent(
                "video_frames",
                f"Frame MIME type '{self.mime}' is not supported. "
                f"Allowed: {sorted(_VALID_IMAGE_MIMES)}",
            )
        if self.fps <= 0:
            raise InvalidMultimodalContent("video_frames", "fps must be positive")
        if not self.frames:
            raise InvalidMultimodalContent("video_frames", "frames list must be non-empty")
        if not isinstance(self.meta, dict):
            raise InvalidMultimodalContent("video_frames", "meta must be a dict")
        normalised_frames = []
        for index, frame in enumerate(self.frames):
            if not isinstance(frame, bytes | bytearray):
                raise InvalidMultimodalContent(
                    "video_frames",
                    f"frame[{index}] must be bytes",
                )
            normalised_frames.append(bytes(frame))
        self.frames = normalised_frames


@dataclass
class ToolCallPart:
    """A tool invocation requested by the assistant (name plus arguments)."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = field(default="tool_call", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
            raise InvalidMultimodalContent("tool_call", "tool_call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise InvalidMultimodalContent("tool_call", "name must be a non-empty string")
        if not isinstance(self.arguments, dict):
            raise InvalidMultimodalContent("tool_call", "arguments must be a dict")


@dataclass
class ToolResultPart:
    """The result returned for an earlier tool call."""

    tool_call_id: str
    result: Any
    is_error: bool = False
    type: Literal["tool_result"] = field(default="tool_result", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
            raise InvalidMultimodalContent("tool_result", "tool_call_id must be a non-empty string")
        if not isinstance(self.is_error, bool):
            raise InvalidMultimodalContent("tool_result", "is_error must be a bool")


ContentPart = TextPart | ImagePart | AudioPart | VideoPart | ToolCallPart | ToolResultPart
_CONTENT_PART_TYPES = (
    TextPart,
    ImagePart,
    AudioPart,
    VideoPart,
    ToolCallPart,
    ToolResultPart,
)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """Unified multimodal message.

    Back-compat: ``Message(role, "some text")`` auto-wraps the string in a
    ``TextPart``. Use ``Message.text`` to join all TextParts back to a string.
    """

    role: Role
    content: list[ContentPart]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)

    def __post_init__(self) -> None:
        # Accept Role enum or its string value for back-compat
        if isinstance(self.role, str):
            try:
                self.role = Role(self.role)
            except ValueError as exc:
                raise InvalidMultimodalContent(
                    "role",
                    f"role must be one of {[role.value for role in Role]}",
                ) from exc
        # Back-compat: if content was passed as a plain string, wrap it
        if isinstance(self.content, str):
            self.content = [TextPart(text=self.content)]
        elif not isinstance(self.content, list):
            raise InvalidMultimodalContent(
                "content",
                f"content must be str or list[ContentPart], got {type(self.content)}",
            )
        else:
            normalised: list[ContentPart] = []
            for index, part in enumerate(self.content):
                # Back-compat / ergonomics: a bare string in the content list is
                # auto-wrapped in a TextPart, so Message(content=[img, "describe"])
                # works as the docs show.
                if isinstance(part, str):
                    normalised.append(TextPart(text=part))
                elif isinstance(part, _CONTENT_PART_TYPES):
                    normalised.append(part)
                else:
                    raise InvalidMultimodalContent(
                        "content",
                        f"content[{index}] must be a ContentPart or str, "
                        f"got {type(part).__name__}",
                    )
            self.content = normalised
        if not isinstance(self.metadata, dict):
            raise InvalidMultimodalContent("metadata", "metadata must be a dict")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Join all TextPart texts into a single string."""
        return "".join(
            part.text for part in self.content if isinstance(part, TextPart)
        )

    @property
    def has_image(self) -> bool:
        """True when any content part is an image."""
        return any(isinstance(p, ImagePart) for p in self.content)

    @property
    def has_audio(self) -> bool:
        """True when any content part is audio."""
        return any(isinstance(p, AudioPart) for p in self.content)

    @property
    def has_video(self) -> bool:
        """True when any content part is video frames."""
        return any(isinstance(p, VideoPart) for p in self.content)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_str(cls, role: Role | str, text: str, **kwargs: Any) -> "Message":
        """Create a text-only message from a plain string.

        Args:
            role: Who the message is from.
            text: The message text.
            **kwargs: Extra message fields, such as a name or tool call id.

        Returns:
            The constructed message.
        """
        if isinstance(role, str):
            role = Role(role)
        return cls(role=role, content=[TextPart(text=text)], **kwargs)

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> "Message":
        """Create a text-only user message."""
        return cls.from_str(Role.USER, text, **kwargs)

    @classmethod
    def assistant(cls, text: str, **kwargs: Any) -> "Message":
        """Create a text-only assistant message."""
        return cls.from_str(Role.ASSISTANT, text, **kwargs)

    @classmethod
    def system(cls, text: str, **kwargs: Any) -> "Message":
        """Create a text-only system message."""
        return cls.from_str(Role.SYSTEM, text, **kwargs)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the message as a JSON-serializable dict (binary parts base64-encoded)."""
        parts = []
        for part in self.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                import base64
                parts.append({
                    "type": "image",
                    "image": base64.b64encode(part.image).decode(),
                    "mime": part.mime,
                    "meta": part.meta,
                })
            elif isinstance(part, AudioPart):
                import base64
                parts.append({
                    "type": "audio",
                    "audio": base64.b64encode(part.audio).decode(),
                    "mime": part.mime,
                    "duration_s": part.duration_s,
                })
            elif isinstance(part, VideoPart):
                import base64
                parts.append({
                    "type": "video_frames",
                    "frames": [base64.b64encode(f).decode() for f in part.frames],
                    "fps": part.fps,
                    "mime": part.mime,
                    "meta": part.meta,
                })
            elif isinstance(part, ToolCallPart):
                parts.append({
                    "type": "tool_call",
                    "tool_call_id": part.tool_call_id,
                    "name": part.name,
                    "arguments": part.arguments,
                })
            elif isinstance(part, ToolResultPart):
                parts.append({
                    "type": "tool_result",
                    "tool_call_id": part.tool_call_id,
                    "result": part.result,
                    "is_error": part.is_error,
                })
        return {
            "role": self.role.value,
            "content": parts,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
